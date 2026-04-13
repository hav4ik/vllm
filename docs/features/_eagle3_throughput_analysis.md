# Eagle3 throughput analysis on NemotronH (H100)

## Architecture

NemotronH has 52 layers:
- 23 mamba layers (recurrent: O(1) per token, no seq_len scaling)
- ~6 real attention layers (O(K × seq_len) per token)
- Remaining layers are MoE MLP-only
- Active parameters per token much less than 30B due to MoE

## Benchmark: pure generation, no tools, no PC reuse

AIME-25 problems, temperature 1.0, max-model-len 131072,
single H100, `--no-enable-chunked-prefill`.

### Per-request throughput (tok/s)

| Concurrency | Eagle3 (K=4) | No-Eagle | Ratio |
|-------------|-------------|----------|-------|
| p=1         | 158         | 294      | 0.54x |
| p=2         | 143         | 266      | 0.54x |

### Aggregate throughput (tok/s)

| Concurrency | Eagle3 (K=4) | No-Eagle | Ratio |
|-------------|-------------|----------|-------|
| p=1         | 151         | 295      | 0.51x |
| p=2         | 229         | 462      | 0.50x |

Eagle3 is ~2x SLOWER than no-Eagle at all tested concurrency levels.

## Spec decode acceptance

Mean acceptance length: 3.11 tokens/step (out of K+1=5).
Per-position acceptance: 74.2%, 52.6%, 37.7%, 27.2%, 19.6%.
Overall acceptance rate: ~42%.

## Why Eagle3 doesn't help

The speculative decoding premise is: verifying K tokens costs roughly
the same as generating 1, because the forward pass is weight-reading
dominated (memory-bandwidth bound). This should hold well for
NemotronH because:

- Mamba layers are recurrent (O(1) per token)
- Only ~6 attention layers scale with seq_len
- MoE means small active parameter set per token

Yet Eagle3 is 2x slower. Possible explanations:

1. **Drafter overhead**: Eagle3 drafter runs K=4 sequential forward
   passes per step to propose tokens. Even at ~400MB, 4 sequential
   calls add up relative to the fast MoE target forward.

2. **Spec decode pipeline overhead**: Rejection sampling, token
   management, metadata construction for K+1 tokens per request.

3. **Low acceptance rate**: 42% at T=1.0 means 58% of drafted
   tokens are wasted compute. At T=0.6 or greedy, acceptance would
   be higher.

4. **CUDA graph overhead**: The captured graph processes K+1=5
   tokens per request. Even if the compute is similar, the graph
   replay overhead for larger batches may not scale linearly.

## Recommendation

For Kaggle AIME (8 parallel sessions, long-context tool use):
- **Use PC-only** (no Eagle3): 1170 tok/s aggregate, 99.2% accuracy
- Eagle3 is a net loss at all concurrency levels tested

Eagle3 might help with:
- Lower temperature (higher acceptance)
- Short-context generation (KV cache small → attention cheap)
- Different drafter with higher acceptance rate
- K=1 or K=2 (lower overhead, higher per-position acceptance)

## Profiling results (H100, p=1, wall-clock + CUDA events)

### No-Eagle baseline
- **3.4ms/step** (296 steps/s, 296 tok/s)
- GPU forward: ~2ms, CPU overhead: ~1.4ms

### Eagle3 step breakdown
- **18.0ms/step** (55 steps/s × 3.11 tok/step = 171 tok/s)

| Component        | GPU (ms) | CPU (ms) | Serial? |
|------------------|----------|----------|---------|
| Target forward   | 4.2      | —        | no      |
| Rejection sample | 0.5      | —        | no      |
| Drafter iter ×4  | 4×0.8=3.2| 4×2.0=8.0| **YES** |
| Other overhead   | —        | ~2.0     | —       |
| **Total**        | **7.9**  | **10.0** | —       |

### Root cause: CPU↔GPU ping-pong in drafter loop

Each drafter iteration needs the previous token → serial:
```
iter1: CPU metadata(2ms) → GPU fwd(0.8ms) → wait for token
iter2: CPU metadata(2ms) → GPU fwd(0.8ms) → wait for token
iter3: CPU metadata(2ms) → GPU fwd(0.8ms) → wait for token
iter4: CPU metadata(2ms) → GPU fwd(0.8ms) → done
= 4 × 2.8ms = 11.2ms serial drafter time
```

GPU is idle 71% of drafter phase, waiting for Python metadata
rebuilds (`build_per_group_and_layer_attn_metadata`,
`eagle_step_update_slot_mapping_and_metadata`).

### Theoretical speedup if CPU overhead eliminated

GPU-only step time: 7.9ms → 126 steps/s × 3.11 = 394 tok/s
That's 1.33x over no-Eagle (296 tok/s). Eagle3 WOULD win if the
CPU overhead were eliminated.

### Potential optimizations (vLLM-level)

1. **Precompute drafter metadata**: Build all 4 iterations' metadata
   upfront before the first drafter forward, eliminating per-iteration
   Python overhead.

2. **Fuse drafter iterations into a CUDA graph**: Capture all 4
   sequential drafter forwards as one graph. Metadata in persistent
   buffers updated once before replay.

3. **Reduce K**: K=2 instead of K=4 would halve drafter overhead
   (2×2.8 = 5.6ms) while keeping the higher per-position acceptance
   rates (74%, 53%). Expected: ~11ms/step → ~2.05 tok/step → 186 tok/s.
   Still below no-Eagle.

4. **K=1**: Single drafter iteration, 2.8ms drafter overhead.
   ~8.5ms/step → ~1.74 tok/step → 205 tok/s. Getting closer but
   still below no-Eagle 296.

## Detailed profiling: execute_model breakdown (p=1, H100)

| Component        | Eagle3 (ms) | No-Eagle (ms) | Delta    |
|------------------|-------------|---------------|----------|
| update_states    | 0.16        | 0.11          | +0.05    |
| prepare_inputs   | 1.68        | 0.65          | +1.03    |
| **attn_meta**    | **5.68**    | **0.80**      | **+4.88**|
| other_pre        | 0.79        | 1.05          | -0.26    |
| **execute_model**| **8.3**     | **2.6**       | **+5.7** |
| unaccounted*     | 9.5         | 0.8           | +8.7     |
| **Total step**   | **17.8**    | **3.4**       | **+14.4**|

*unaccounted = sample_tokens + drafter + engine scheduling

### Root cause 1: _build_attention_metadata (5.7ms vs 0.8ms)

7x slower with Eagle3. Builds per-layer metadata for 52 layers
with K+1=5 tokens per request + spec_decode_common_attn_metadata.

### Root cause 2: sample_tokens overhead (9.5ms vs 0.8ms)

Includes rejection sampling bookkeeping, drafter (1.7ms),
_copy_draft_token_ids_to_cpu D2H sync, engine scheduling.

### Drafter loop is NOT the bottleneck (1.7ms total)

Metadata caching optimization confirmed: build_per_group dropped
from ~2ms to 0.02ms, but zero impact on step time.

## Why Eagle3 cannot win on fast MoE models

The spec decode pipeline has ~9ms of **fixed per-step CPU overhead**
that does not scale with model size:

| Component              | Time (ms) | Scales with model? |
|------------------------|-----------|-------------------|
| Target forward (5 tok) | 4.2       | Yes               |
| Drafter forward (4×)   | 2.9       | Yes (tiny model)  |
| attn_meta build        | 2.4       | No — Python/CPU   |
| prepare_inputs         | 1.7       | No — Python/CPU   |
| commit_boundary (23L)  | 3.9       | No — kernel launches |
| sample + bookkeep      | 1.1       | No — fixed         |
| **Fixed overhead**     | **~9**    | **No**            |

For a 70B dense model (~20ms forward), 9ms overhead = 45%. Spec
decode can break even with ~60% acceptance.

For NemotronH (3B active MoE, ~2ms forward), 9ms overhead = 4.5x
the actual compute. **Spec decode can never win** because the
scheduling tax exceeds the model compute.

Our optimizations reduced total step from 18.0ms to 13.2ms (+42%),
but the ~9ms CPU floor requires architectural changes to break.

## Does this problem affect MTP and other speculators?

**Yes, partially.** The overhead has two components:

### 1. Per-step metadata overhead (~6ms) — affects ALL speculators

- `_build_attention_metadata` for K+1 tokens: +4.9ms (mitigated to
  +1.6ms with update_block_table, but still nonzero)
- `_prepare_inputs` for K+1 tokens: +1.0ms
- `commit_boundary_states` for hybrid mamba: +3.9ms (mamba-specific,
  doesn't affect pure transformer models)

Any speculator that processes K+1 tokens per target forward step
pays this cost. MTP also processes multiple tokens per step, so it
has the same `_build_attention_metadata` and `_prepare_inputs`
overhead. The commit_boundary cost is specific to hybrid mamba
models with prefix caching.

### 2. Sequential drafter loop (~3ms) — Eagle3/MTP specific

Eagle3 runs K=4 sequential drafter forward passes (each depends on
the previous token). MTP similarly runs sequential draft heads.
DFlash/Medusa use parallel drafting (one forward, multiple heads)
which avoids this sequential cost.

### Speculator comparison for fast MoE models:

| Speculator    | Sequential drafts? | Metadata overhead? | Mamba commit? |
|---------------|-------------------|-------------------|---------------|
| Eagle3        | Yes (K passes)    | Yes (+4.9ms)      | Yes (+3.9ms)  |
| MTP           | Yes (K passes)    | Yes (+4.9ms)      | Yes (+3.9ms)  |
| DFlash/Medusa | No (1 pass)       | Yes (+4.9ms)      | Yes (+3.9ms)  |
| Ngram         | No (CPU lookup)   | Yes (+4.9ms)      | Yes (+3.9ms)  |

**DFlash/Medusa would eliminate the 3ms drafter loop** but still pay
the metadata and commit overhead. For NemotronH, that reduces the
overhead from ~9ms to ~6ms — still 3x the model forward.

### When does spec decode help?

Spec decode breaks even when:
  mean_acceptance × (1 - overhead_fraction) > 1

For NemotronH: overhead = 9ms, forward = 4.2ms, step = 13.2ms
  overhead_fraction = 9/13.2 = 68%
  need: mean_acceptance > 1/(1-0.68) = 3.13

Current mean_acceptance = 3.11 — exactly at breakeven. This is why
Eagle3 is marginally slower (214 tok/s) than no-Eagle (294 tok/s)
instead of dramatically slower.

For a 70B dense model: overhead = 9ms, forward = 20ms, step = 29ms
  overhead_fraction = 9/29 = 31%
  need: mean_acceptance > 1/(1-0.31) = 1.45

Much easier to achieve. Eagle3 would give ~2x speedup there.

## Optimization roadmap (upstream vLLM)

## 3-way throughput comparison (p=1, H100, no tools)

| Config                   | Per-req tok/s | vs No-Eagle |
|--------------------------|---------------|-------------|
| Eagle3 no-PC             | 349           | +15%        |
| Eagle3 + PC (optimized)  | 319           | +6%         |
| No-Eagle no-PC           | 302           | baseline    |

Eagle3 spec decode WORKS when overhead is minimized. PC integration
adds ~30 tok/s tax from spec-slot infrastructure (init checks, stash
cloning, deferred commit) running every step even when PC isn't used.

### Quick wins (implemented in this branch)
- [x] Re-enable update_block_table for mamba spec decode (-3ms)
- [x] Batch commit_boundary_states indices (-1.5ms)
- [x] Hoist pool_slot/src_slot computation (-0.3ms)
- [x] Merge replace() calls in update_block_table (-0.1ms)

### Medium effort
- [ ] Batched Triton kernel for 23-layer commit (-2ms estimated)
- [ ] Pre-create ForwardContext for drafter loop (-0.1ms)
- [ ] Cache attn_meta across steps for decode-only batches (-2ms)

### High effort (architectural)
- [ ] Fuse drafter loop into single CUDA graph capture (-2ms)
- [ ] Overlap CPU metadata build with GPU forward via pipelining
- [ ] Move metadata construction to GPU (Triton kernels)
