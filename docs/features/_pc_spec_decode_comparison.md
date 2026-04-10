# PC + Eagle3 Speculative Decoding: V1 vs V2 Comparison

> Date: 2026-04-10
> Model: NemotronH 30B FP8 (chankhavu/Nemotron-Cascade-2-30B-A3B-FP8)
> Drafter: Eagle3 K=4 (chankhavu/c2.eagle3-test)
> Hardware: 1x RTX PRO 6000 Blackwell (98GB)
> Benchmark: AIME-25 (30 problems, multi-turn tool-calling)

## Problem Statement

When prefix caching (`mamba_cache_mode=all`) AND speculative decoding
(Eagle3) are both enabled on NemotronH hybrid models:

1. **Without any fix**: server crashes with `illegal memory access` —
   the SSM kernel indexes into the block table using `init_token_idx`
   (a candidate index 0..K), but the block table has block entries,
   not candidate slots.

2. **Prefix cache hits are 0%**: the Eagle block drop mechanism removes
   the last matched block for hidden-state recomputation, but with
   LCM-aligned block_size=4608, this wipes all cache hits.

3. **Conv state contamination**: speculative candidate conv states
   are written directly to pool block slots, corrupting boundary
   states for prefix cache reuse.

## Two Approaches

### V1: Segregated Scratch Buffer (branch: `nemotron3-eagle3-support`)

**Core idea**: Allocate separate tensors for both SSM and conv state,
fully isolated from the prefix cache pool. Kernels read/write ONLY
to scratch during spec decode.

**Implementation**:
1. One-time init: pool → scratch[base+0] on first decode step per request
2. SSM kernel: K+1 scratch slots with native `init_token_idx` rollback
3. Conv kernel: 1 widened scratch slot with `conv_state_token_offset` rollback
4. Boundary commit: scratch → pool only when crossing block boundaries
5. Condense hook: move scratch data + `_spec_inited` flags when batch
   positions shift

**Files changed**: 5 files, 191 insertions
- `mamba_mixer2.py`: scratch allocation, decode path routing, init/commit
- `gpu_model_runner.py`: finish hook, condense hook, scratch init
- `kv_cache_coordinator.py`: Eagle block drop fix
- `scheduler.py`: boundary disable
- `eagle.py`: seq_lens clone

**Memory overhead**: ~2.8 GB (fp32) / ~1.4 GB (fp16) for page-padded
scratch tensors across 31 mamba layers.

### V2: Reuse Kernel's Native Spec Slots (branch: `pc-spec-v2-nonpc-slots`)

**Core idea**: The SSM kernel already has an `IS_SPEC_DECODING` mode
with K+1 slots and `init_token_idx` rollback. It was just disabled
when prefix caching was active. Re-enable it.

**Implementation**:
1. Spec slot allocation: K+1 dedicated slots per request (same as non-PC path)
2. SSM kernel: uses spec slots with native rollback (identical to non-PC)
3. Conv kernel: stays on pool with APC mode (rollback via `conv_state_token_offset`)
4. Boundary commit: spec slot → pool with improved `at_boundary` detection
5. Eagle block drop: SlidingWindow-aware, no cascade to target model groups

**Files changed**: 3 files, 38 insertions
- `mamba_mixer2.py`: boundary commit detection fix
- `kv_cache_coordinator.py`: Eagle block drop fix
- `scheduler.py`: boundary disable removed (commit handles all cases)

**Memory overhead**: None — slots carved from existing pool infrastructure.

## Root Causes Found

### Bugs fixed in both V1 and V2:

1. **Eagle block drop cascade** (`kv_cache_coordinator.py`):
   `HybridKVCacheCoordinator` passed `use_eagle=True` to ALL managers,
   causing the FullAttention manager to drop its last 4608-token block.
   Fix: skip drop for FullAttention+Mamba, keep for SlidingWindow (drafter
   needs it), prevent SlidingWindow's reduced hit length from cascading.

2. **seq_lens corruption** (`eagle.py`): Eagle's `propose()` modifies
   `common_attn_metadata.seq_lens` in-place (`seq_lens -= num_rejected`
   and `seq_lens += 1` in Triton kernel). Since `seq_lens` is a VIEW of
   `self.seq_lens` in `gpu_model_runner`, this corrupts the next step's
   `_prepare_inputs()`. Fix: clone before modification.

### Bugs specific to V1:

3. **Per-step pre-copy from stale pool** (`mamba_mixer2.py`): The original
   V1 code pre-copied pool → scratch[base+0] every decode step. But the
   pool is only updated at block boundaries (~every 4608 tokens), so the
   pre-copy read data that was potentially thousands of steps stale. This
   clobbered the accepted state when `init_token_idx=0` (~50% of steps),
   causing cumulative drift: 92% → 50% accuracy.
   Fix: one-time init via `_spec_inited` flag.

4. **Batch condensation desync** (`gpu_model_runner.py`): When
   `InputBatch.condense()` moved a request from batch_idx=5 to
   batch_idx=2, the `_spec_inited` flag and scratch data were not moved.
   The request got spurious re-init from stale pool state.
   Fix: condense hook that moves scratch SSM/conv data + init flags.

5. **Conv scratch stride mismatch** (`mamba_mixer2.py`): Conv scratch was
   allocated with contiguous strides, but the Triton kernel reads
   `stride[0]` from the tensor for pointer arithmetic. With contiguous
   strides, the kernel computed wrong memory offsets. This caused 50% →
   25% spec decode acceptance rate and garbage output.
   Fix: page-padded allocation matching pool's stride.

### Bugs specific to V2:

6. **Boundary commit detection gap** (`mamba_mixer2.py`): The detection
   `blk_before != blk_after` missed the case where accepted tokens land
   exactly AT a boundary position without crossing past it (e.g., position
   4607 with block_size=4608). The boundary state sat in spec slots and was
   overwritten by the next step's kernel.
   Fix: added `at_boundary = (last_accepted_pos % block_size == block_size - 1)`.

## Experimental Results

### Server Configuration (all tests)

```
--model chankhavu/Nemotron-Cascade-2-30B-A3B-FP8
--max-model-len 65536
--mamba-ssm-cache-dtype float32
--max-num-seqs 4
--kv-cache-dtype fp8
--enable-chunked-prefill
--max-num-batched-tokens 8192
--enforce-eager
--trust-remote-code
--download-dir /workspace/models
```

PC configs add: `--enable-prefix-caching --mamba-cache-mode all --mamba-block-size 512`
Eagle3 configs add: `--speculative-config '{"model":"chankhavu/c2.eagle3-test","method":"eagle3","num_speculative_tokens":4}'`

Eval: `--n_sessions 2 --max_parallel 2`

### Results: max_tokens=32768

| Config | Per-session | Majority | Problems | Notes |
|--------|-------------|----------|----------|-------|
| V2 PC+Eagle3 | 23/24 = 96% | 11/13 = 85% | 14 | 1 miss from token limit |
| V1 PC+Eagle3 | 28/30 = 93% | 12/14 = 86% | 16 | 2 misses from token limit |
| PC-only (no Eagle3) | 21/22 = 95% | - | 11 | 1 miss from token limit |

### Results: max_tokens=60000 (ongoing)

| Config | Per-session | Majority | Problems | Notes |
|--------|-------------|----------|----------|-------|
| V2 PC+Eagle3 | 41/44 = 93% | - | 22 | Misses: model gave up or sandbox error |
| PC-only (no Eagle3) | 22/23 = 96% | - | 12 | 1 miss: model gave up |

### Key Metrics (V2 PC+Eagle3, max_tokens=60000)

| Metric | Value |
|--------|-------|
| Spec decode acceptance | 42-56% |
| Prefix cache hit rate | 53-78% |
| Generation throughput | 150-340 tok/s |
| Server errors (CUDA) | 0 |
| Max parallel tested | 2 (stable), 8 (crashed after 10/10 correct) |

### Stress Test (V1, fp16 mamba, max_num_seqs=16, max_parallel=8)

| Metric | Value |
|--------|-------|
| Accuracy before crash | 10/10 = 100% |
| Block size | 2560 (vs 4608 with fp32) |
| Acceptance | 46% |
| Crashed after | ~5 problems (OOM on engine restart) |

## Architecture Comparison

| | V1 (Scratch) | V2 (Spec Slots) |
|---|---|---|
| SSM isolation | Full — separate tensor | Full — dedicated slots |
| Conv isolation | Full — separate tensor | None — stays on pool (APC rollback) |
| Extra memory | ~2.8 GB (31 layers × page-padded) | None |
| Code complexity | 5 files, 191 lines | 3 files, 38 lines |
| Condense hook | Required (move scratch + flags) | Not needed |
| Boundary disable | Required (scheduler workaround) | Removed (commit handles all cases) |
| Kernel changes | None — redirects tensor pointers | None — re-enables existing mode |
| Generality | Works with any kernel | Relies on kernel's IS_SPEC_DECODING mode |

## Technical Insights

### Why SSM and Conv need different strategies

- **SSM**: K+1 candidates stored in **K+1 separate slots**. Kernel reads
  `state[indices[seq, init_token_idx]]` and writes K+1 different slots.
- **Conv**: K+1 candidates stored as **consecutive token offsets within ONE
  widened slot** (state_len = conv_kernel-1 + num_spec). Kernel reads/writes
  one slot only.

### Why the Triton stride bug was so subtle

Triton kernels read `stride[0]` from the tensor to compute memory offsets.
`index_copy_` between tensors with different strides works correctly
(element-by-element copy), but passing a contiguously-allocated scratch to
a Triton kernel that expects page-padded strides causes the kernel to compute
wrong pointer offsets — silent data corruption, no crash.

### Why the Eagle block drop cascades

`HybridKVCacheCoordinator` iterates over ALL attention groups (FullAttention,
Mamba, SlidingWindow) and converges on a common cache hit length. When
SlidingWindow drops 1 block (16 tokens), `curr_hit_length` drops to
`4608 - 16 = 4592`. On the next iteration, FullAttention needs
`4592 / 4608 = 0` blocks → all cache hits wiped.

Fix: don't update `curr_hit_length` from SlidingWindow's result.

### Why the scheduler boundary-disable is unnecessary in V2

With the improved boundary detection (`at_boundary | crossed_past`), the
commit logic handles all cases:
- **Crossed past boundary**: boundary_pos = `(blk_before+1) * block_size - 1`,
  cand_idx identifies the boundary candidate
- **At boundary (not crossed)**: same formula produces the same result because
  `last_accepted_pos = (blk_before+1) * block_size - 1`

The scheduler disable was a workaround for the detection gap.

## Recommendations

**For production**: Use V2. Simpler, no memory overhead, same accuracy.

**For upstreaming**: V2's changes are minimal (38 lines across 3 files) and
self-contained. The key insight to communicate: the kernel-level
IS_SPEC_DECODING support already exists — the fix is metadata plumbing only.

**For future work**:
- Remove `sample_complete_event` syncs (may be unnecessary with SlidingWindow fix)
- Test with CUDA graphs (currently enforce-eager)
- Test at TP=2 for larger context (131K)
- Investigate max_parallel=8 crash (may need more tensor clones in Eagle drafter)
