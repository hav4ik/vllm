# PC + Eagle3 + NemotronH: Evaluation Results

> Branch: `pc-spec-v2-nonpc-slots`
> Date: 2026-04-10
> Model: chankhavu/Nemotron-Cascade-2-30B-A3B-FP8 + Eagle3 drafter
> Single GPU: NVIDIA RTX PRO 6000 Blackwell (98 GiB)

## Summary of changes on the branch

1. **`kv_cache_coordinator.py`** — Skip Eagle block drop for ALL managers
   in `HybridKVCacheCoordinator`. The LCM-aligned block size (4608 tokens)
   makes the drop too expensive.

2. **`scheduler.py`** — Disable spec decode for requests approaching a
   mamba block boundary. Forces normal decode at boundaries so the kernel
   writes correct state directly to the pool.

3. **`gpu_model_runner.py`** — `torch.cuda.synchronize()` before
   `commit_boundary_states()` (from previous agent's v2 spec slot code).

## Eval configurations tested

### Config A: max_parallel=1, enforce-eager, fp32 mamba

```
--enforce-eager --mamba-ssm-cache-dtype float32 --max-num-seqs 8
--max-model-len 65536 --kv-cache-dtype fp8
--enable-prefix-caching --mamba-cache-mode all --mamba-block-size 512
--speculative-config '{"model":"chankhavu/c2.eagle3-test","method":"eagle3","num_speculative_tokens":4}'
```

| Metric | Value |
|--------|-------|
| Per-session accuracy | **100%** (21/21) |
| Majority-vote accuracy | **100%** (11/11) |
| Cache hit rate | **49%** (235K/478K tokens) |
| Spec decode acceptance | **48%** (88K/183K drafted) |
| Acceptance pos 0/1/2/3 | 72% / 53% / 40% / 30% |
| Server stability | Stable through 500K+ gen tokens |
| Block size | 4608 tokens |
| Num blocks | 80 |

**Status: WORKING.** All three objectives met (accuracy, cache hits > 0%, spec decode).
Run incomplete (11/30 problems) but no crashes observed.

### Config B: max_parallel=4, cudagraphs, fp32 mamba (targeted fix)

This used the WRONG targeted fix (kept Eagle drop for SlidingWindow).
Cache hits were 0% because SlidingWindow drop cascaded to reduce all hits.

| Metric | Value |
|--------|-------|
| Per-session accuracy | **94.8%** (55/58) |
| Majority-vote accuracy | **100%** (30/30) |
| Cache hit rate | **0%** (no cache hits) |
| Spec decode acceptance | **44%** (491K/1.12M) |
| Server stability | Stable, all 30 problems completed |

**Status: Accuracy validated but cache hits broken by targeted fix bug.**

### Config C: max_parallel=2, enforce-eager, fp32 mamba (blanket fix)

| Metric | Value |
|--------|-------|
| Per-session accuracy | 2/2 (crashed early) |
| Cache hit rate | N/A (server crashed) |
| Server stability | **CRASHED** — device-side assert |

**Status: CRASHES.** Same crash as all max_parallel>=2 attempts.

### Config D: max_parallel=1, enforce-eager, CUDA_LAUNCH_BLOCKING=1

Earlier test with `torch.cuda.synchronize()` fix. Confirmed cache hits work.

| Metric | Value |
|--------|-------|
| Per-session accuracy | **100%** (21/21) |
| Majority-vote accuracy | **100%** (11/11) |
| Cache hit rate | **~50%** |
| Spec decode acceptance | ~48% |
| Server stability | Stable |

## Baseline comparison (previous agent's v2 fix, no cache hit fix)

| Metric | Previous agent | Our fix (Config A) |
|--------|---------------|-------------------|
| Per-session accuracy | 91.5% | **100%** (21/21) |
| Majority-vote accuracy | 96.7% (29/30) | **100%** (11/11) |
| Cache hit rate | **0%** | **49%** |
| Spec decode acceptance | ~46% | 48% |
| Setup | TP=2, max_parallel=8 | TP=1, max_parallel=1, enforce-eager |

## Outstanding issue: max_parallel >= 2 crashes

All attempts with 2+ concurrent requests crash with
`CUDA error: device-side assert triggered`. This happens:
- With and without CUDA graphs
- With and without `torch.cuda.synchronize()`
- With and without the boundary disable
- Even when cache hits are 0% (pool too busy for caching)

The crash does NOT happen:
- With `max_parallel=1` (serial sessions)
- With `CUDA_LAUNCH_BLOCKING=1` (serializes all GPU ops)
- Without our cache hit fix (baseline code)

Root cause under investigation. The crash is related to two requests
being processed in the same model forward pass when the Eagle block
drop is disabled in the coordinator.

## Key block size info

```
mamba_cache_mode=all → block_size alignment:
  mamba_state_bytes / attn_bytes_per_token ≈ 4100
  block_size = lcm(512, 16) * ceil(4100 / 512) = 512 * 9 = 4608

With fp16 mamba (--mamba-ssm-cache-dtype float16):
  block_size = 2560 (mamba state halved)
  BUT: cache hits also 0% with fp16 (tested, same issue)

num_gpu_blocks = 80 (override from cudagraph profiling, not from memory)
Available KV memory: 52 GiB (could fit ~400 blocks)
```
