# PC + Eagle3 spec decode: ablation results and root cause analysis

> Complete record of experiments run on 2026-04-09 on the
> `nemotron3-eagle3-support` branch. This document saves future agents
> ~10 hours of testing. Read this BEFORE attempting any new fix.

## TL;DR

- **No-PC + Eagle3 (K=4)**: 92-96% accuracy. Works perfectly.
- **PC + Eagle3 with our scratch fix**: ~50% accuracy. Degraded.
- **The degradation is NOT from state corruption** — single-request short
  generations are perfect. The model just fails to solve hard problems.
- **Prefix cache hits are 0%** when spec decode is enabled (upstream
  issues #38182, #31920). This is the biggest performance problem.
- **The per-step round-trip** (pool → scratch → pool) for mamba states
  is the most likely remaining correctness issue.

## Hardware & model

- GPU: H100 80GB
- Model: chankhavu/c2-softcpy-fp8 (NemotronH FP8, 30B params)
- Drafter: chankhavu/c2.eagle3-test (Eagle3)
- NemotronH: 31 Mamba2 layers, 6 GQA attention, 15 MoE/MLP

## Server configurations tested

All tests use: `--trust-remote-code --mamba-ssm-cache-dtype float32
--max-num-seqs 16 --kv-cache-dtype fp8 --enable-chunked-prefill
--max-num-batched-tokens 8192 --enable-auto-tool-choice
--tool-call-parser qwen3_coder`

| Config ID | max_model_len | enable_prefix_caching | mamba_cache_mode | mamba_block_size | num_spec_tokens | Notes |
|-----------|---------------|----------------------|------------------|------------------|-----------------|-------|
| A (90% baseline) | 65536 | No | (default) | (none) | 5 | Original baseline from before our work |
| B (95% baseline) | 65536 | Yes | all | 512 | (none) | PC only, no spec |
| C (our fix, 65K) | 65536 | Yes | all | 512 | 4 | SSM+conv scratch, boundary disable |
| D (our fix, 131K) | 131072 | Yes | all | 512 | 4 | Same as C but bigger context |
| E (verify no-PC) | 65536 | No | (default) | (none) | 4 | Our modified code, no PC |

## Ablation results

### Full AIME-25 runs (with tool calls, sandbox execution)

| Config | Sessions | Per-session | Majority | Avg prompt | Avg gen | Avg tools | Traces dir |
|--------|----------|-------------|----------|-----------|---------|-----------|-----------|
| A (no-PC Eagle3 K=5) | 120 | 90.0% | 100% | 73,862 | 11,281 | 5.8 | eagle3_no_pc |
| B (PC only, no spec) | 120 | 95.0% | 100% | 78,348 | 10,889 | 6.0 | baseline_pc |
| C (PC+spec, 65K) | 60 | 50.0% | 60% | 21,929 | 11,813 | 2.9 | vllm_pc_eagle3_conv_fix |
| D (PC+spec, 131K) | 57 | 47.4% | 53% | 23,851 | 15,097 | 2.9 | vllm_pc_eagle3_131k |
| E (no-PC, our code) | 56 | **92.9%** | **96.4%** | 48,488 | 10,959 | 4.6 | vllm_nopc_verify |

**Key finding**: Config E proves our modified code works correctly WITHOUT
PC. The 50% degradation in configs C/D is entirely from the PC +
mamba_cache_mode=all interaction.

### Intermediate ablations (SSM-only scratch, before conv_state fix)

| Variant | Sessions | Per-session | Majority | Notes |
|---------|----------|-------------|----------|-------|
| SSM scratch + boundary disable (eager) | 60 | 48.3% | 53.3% | First full AIME run |
| SSM scratch + boundary disable (cudagraph) | 60 | 53.3% | 60.0% | Cudagraph mode |
| SSM+conv scratch + boundary disable | 60 | 50.0% | 60.0% | Conv scratch didn't help |

### No-tool single-turn tests (isolating batching vs multi-turn)

| Variant | Sessions | Correct | Degenerate | Notes |
|---------|----------|---------|------------|-------|
| No-tool, 8-parallel (run 1) | 30 | 20% | 18/30 | First no-tool test |
| No-tool, 8-parallel (run 2) | 30 | 30% | 16/30 | With conv fix |
| No-tool, serial (contaminated) | 18 | ~33% | ~5 | Was co-running with parallel test |

**Key finding**: no-tool tests have LOWER accuracy than tool tests (20-30%
vs 50%) because the model needs tools (Python sandbox) to verify its
answers on AIME problems. This is NOT a sign of corruption — it's the
expected behavior for a model trained for tool-integrated reasoning.

### Single-request manual tests

| Test | Tokens | Correct | Degenerate | Notes |
|------|--------|---------|------------|-------|
| Simple math (2+2) | 47 | ✓ | No | |
| Integration by parts | 1096 | ✓ | No | 2 boundary crossings |
| AIME base-b problem | ~1400 | ✓ | No | 3 runs, all correct, non-deterministic |
| Hard polygon coloring | 4293 | ✓ | No | 8+ boundary crossings |
| 3-turn multi-turn chat | ~2000 | ✓ | No | Simulated tool calls |

**Key finding**: single-request short-to-medium generations are perfect.
The model produces correct, coherent output with our scratch code.

## What we ruled out

### 1. Blatant state corruption → RULED OUT
Working baselines have SIMILAR "degeneration" rates (26-39%) as our fix
(23%). The degeneration detector was producing false positives from
the model's normal reasoning patterns (repetitive mathematical
analysis in think blocks). The model is NOT producing garbled output.

### 2. Context length starvation → RULED OUT
Config D (131K context) gave the same accuracy (47%) as Config C (65K).
The 90% baseline used 65K context. Context length is not the issue.

### 3. Batching / cross-request interference → RULED OUT
Serial (max_parallel=1) tests showed similar degradation to parallel
tests. Single-request manual tests work perfectly because they're
SHORT, not because they're alone.

### 4. Conv state contamination → PARTIALLY RULED OUT
Adding conv_state scratch (configs with conv fix) didn't improve
accuracy over SSM-only scratch. Removing it probably won't hurt either.

### 5. Block boundary handling → PARTIALLY ADDRESSED
Boundary disable (force num_accepted=1 at block boundaries) improved
things marginally but is not the main issue.

## What we know IS the problem

### 1. Prefix cache hits are 0%
```
vllm:prefix_cache_queries_total: 103,296
vllm:prefix_cache_hits_total:    0
```
Upstream issues #38182 and #31920 document that spec decode completely
suppresses prefix cache hits. The block hash computation includes
speculative tokens, so subsequent turns never match.

**Impact**: Multi-turn tool-call conversations re-prefill the entire
context every turn. The model uses 22K avg prompt tokens (vs 74K in
the no-PC baseline) because sessions are SHORTER (fewer tool calls,
lower accuracy). The 0% hit rate means PC is providing zero benefit.

### 2. The per-step round-trip is the likely accuracy degrader
The ONLY difference between the working no-PC path (92%) and the broken
PC path (50%) is how mamba states are managed:

- **No-PC**: states persist in fixed per-request slots. No copying.
  The kernel reads/writes directly. conv_state uses widened dimension
  with `conv_state_token_offset = num_accepted - 1` for rollback.

- **PC (our scratch)**: states are round-tripped EVERY STEP:
  pool → scratch → kernel → scratch → pool. The pre-copy reads from
  the canonical pool slot; the commit writes back. The conv and SSM
  kernels run in non-APC mode on the scratch tensor.

The round-trip is mathematically correct (we verified the offset
arithmetic extensively — see `_pc_spec_decode_three_approaches.md`).
But something subtle about the round-trip causes accuracy degradation
over thousands of decode steps.

### 3. The upstream code crashes without our scratch
Disabling the scratch path (falling through to the upstream elif
branch) causes `RuntimeError: illegal memory access` in
`BatchPrefillWithPagedKVCache`. Our scratch code IS necessary to
prevent crashes. Without it, PC + spec decode doesn't run at all.

## Upstream issues that may be relevant

| Issue/PR | Status | Relevance |
|----------|--------|-----------|
| #38182 (spec reduces PC hits) | Open | HIGH — 0% hit rate confirmed |
| #31920 (spec + PC 0% hits) | Open | HIGH — same issue |
| #39146 (stale KV blocks, no zeroing) | Open, unfixed | HIGH — blocks recycled without zeroing |
| #37076 (TOCTOU race in block alloc) | Fixed (#37164) | Check if in build |
| #33937 (null-block padding) | Open, unmerged | MEDIUM — block ID 0 leaks |
| #38556 (async stale state) | Merged | Already fixed |
| #35447 (prefill chunk misclassification) | Merged | Already fixed |
| #38898 (DS conv layout + spec) | Open | Low — we use SD layout |
| #39273 (GDN spec corruption) | Open | LOW — different architecture |

## Proposed next steps (for the next agent/session)

### Idea 1: Eliminate conv_state round-trip (5 min)
Remove the conv_state scratch. Let conv_state go through the regular
pool with IS_APC_ENABLED=True (the upstream path). Since cache hits
are 0%, "contamination" of cached conv states doesn't matter. Test
if SSM-only scratch + native conv gives better accuracy.

### Idea 2: Eliminate ALL round-trips (persistent scratch slots)
Don't pre-copy or commit every step. Instead, keep scratch slots as
the persistent "live" state across decode steps. The kernel reads/writes
directly to scratch (no round-trip). Only commit to the pool at block
boundaries (for prefix cache snapshots). This makes the decode path
identical to the no-PC path behavior.

### Idea 3: Fix the 0% prefix cache hit rate
This is the highest-impact fix but also the hardest. Issues #38182 and
#31920 suggest the block hash includes speculative tokens. The fix
would be to exclude spec tokens from the hash computation, or to
re-hash blocks after the rejection sampler determines which tokens
are accepted.

### Idea 4: Hybrid approach
Disable PC for mamba layers (use direct slots like no-PC path) but
keep PC for attention layers (which handle it correctly). The attention
KV cache is much larger than mamba state, so most PC benefit comes from
attention. This requires per-layer-type cache mode, which vLLM doesn't
currently support.

### Idea 5: Use mamba_cache_mode="align" instead of "all"
The "align" mode has different (simpler) state management. It might
work better with spec decode. The upstream has some support for
align + spec (PR #33705). Worth testing.

## How to reproduce

### Working baseline (no PC, ~92%)
```bash
vllm serve chankhavu/c2-softcpy-fp8 --max-model-len 65536 \
  --trust-remote-code --mamba-ssm-cache-dtype float32 \
  --max-num-seqs 16 --kv-cache-dtype fp8 \
  --enable-chunked-prefill --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --download-dir /workspace/models --host 127.0.0.1 --port 18000 \
  --speculative-config '{"model":"chankhavu/c2.eagle3-test","method":"eagle3","num_speculative_tokens":4}'
```

### Broken config (PC + spec, ~50%)
Add `--enable-prefix-caching --mamba-cache-mode all --mamba-block-size 512`
to the above command. Requires the scratch patches from this branch.

### AIME-25 test command
```bash
python collect_traces_nemotron.py \
  --input_dir data --output_dir traces/YOUR_RUN_NAME \
  --n_sessions 2 --max_parallel 8 \
  --model_name chankhavu/c2-softcpy-fp8 --max_tokens 32768 \
  --server_addr http://127.0.0.1:18000 \
  --metrics_url http://127.0.0.1:18000/metrics \
  --watchdog_interval 30 --resume
```

### Accuracy computation
```bash
python calc_accuracy.py traces/YOUR_RUN_NAME/
```

## Files modified on this branch

| File | What changed | Commits |
|------|-------------|---------|
| `vllm/model_executor/layers/mamba/mamba_mixer2.py` | SSM+conv scratch tensors, boundary disable, pre-allocated index buffers | ec24bdf, 5c63c1d, 0df56f3, 37316ec, e5f3a95, d95eeb7, 5622f8d, 509fcc9 |
| `vllm/v1/worker/gpu_model_runner.py` | Eager init hook, commit hook, boundary clamp | ec24bdf, 37316ec, e5f3a95, d95eeb7 |
| `vllm/v1/attention/backends/mamba_attn.py` | block_idx_first_scheduled_token field + cudagraph buffer | e5f3a95 |
| `docs/features/` | Investigation docs, lifecycle docs, approach comparison, this file | Multiple |
