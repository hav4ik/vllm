# PC + Eagle3 + NemotronH: Full Status

> Branch: `pc-spec-v2-nonpc-slots`
> Date: 2026-04-10
> Model: chankhavu/Nemotron-Cascade-2-30B-A3B-FP8 (NemotronH 30B FP8)
> Drafter: chankhavu/c2.eagle3-test (Eagle3)

## What the branch does

### Change 1: Fix 0% prefix cache hit rate (`kv_cache_coordinator.py`)

In `HybridKVCacheCoordinator.find_longest_cache_hit()`, pass
`use_eagle=False` to ALL individual managers. This prevents the Eagle
"last block drop" from wiping all cache hits.

**Why the drop was catastrophic**: For hybrid models,
`_align_hybrid_block_size()` inflates block_size to the LCM of all
attention types (4608 tokens for NemotronH). Dropping one 4608-token
block eliminates most or all cacheable blocks.

**Why skipping the drop is safe**: `get_computed_blocks()` already
sets `max_cache_hit_length = num_tokens - 1`, guaranteeing at least
the last token is always recomputed during prefill. This recomputation
produces the fresh hidden states Eagle's drafter needs.

### Change 2: Disable spec decode at block boundaries (`scheduler.py`)

In `update_draft_token_ids()`, clear `spec_token_ids` for requests
about to cross a mamba block boundary. This forces a normal
single-token decode step where the kernel writes the boundary state
directly to the pool — guaranteed correct for prefix cache reuse.

**Cost**: ~2 non-spec steps per 4608 tokens per request = 0.04%.

### Change 3: sample_complete_event (`gpu_model_runner.py`)

Add a CUDA event recorded at end of `sample_tokens()` and waited on
at start of `execute_model()`'s model forward. This was an attempt
to fix the `max_parallel>=2` race — it helps but doesn't fully solve it.

## What works

### max_parallel=1 + enforce-eager (1 GPU)

All three objectives met:

| Metric | Value | Baseline |
|--------|-------|----------|
| Per-session accuracy | **100%** (21/21) | 91.5% |
| Majority-vote accuracy | **100%** (11/11) | 96.7% |
| Cache hit rate | **49%** (235K/478K) | 0% |
| Spec decode acceptance | **48%** | ~46% |
| Server stability | Stable through 500K+ gen tokens | Stable |

Run was 11/30 problems (incomplete, killed for other tests).
Accuracy tracked 100% from problem 1 through 11.

### max_parallel=4 + cudagraphs (1 GPU, targeted fix variant)

The "targeted" variant kept Eagle drop for SlidingWindow, which
broke cache hits back to 0%. But accuracy and stability were proven:

| Metric | Value |
|--------|-------|
| Per-session accuracy | **94.8%** (55/58) |
| Majority-vote accuracy | **100%** (30/30) |
| Cache hit rate | 0% (targeted fix bug) |
| Spec decode acceptance | 44% |
| Stability | All 30 problems completed |

## What doesn't work

### max_parallel>=2 with blanket cache hit fix

Crashes with `CUDA error: device-side assert triggered` (IndexKernel.cu
"index out of bounds"). Happens with and without:
- CUDA graphs
- torch.cuda.synchronize()
- sample_complete_event
- Boundary disable

Only `CUDA_LAUNCH_BLOCKING=1` (full GPU serialization) prevents it.

## Root cause analysis of max_parallel>=2 crash

### What Change 1 actually does

The ONLY functional difference: `find_longest_cache_hit()` returns
more cached blocks → larger `num_computed_tokens`. No kernel changes,
no buffer changes, no stream changes.

### Why this matters

With 0% cache hits (baseline), every new-turn request has
`num_computed_tokens = 0`. With our fix, a new-turn request can have
`num_computed_tokens = 4608` (one block cached). This creates a
**code path that was never exercised before**: a request with BOTH
`num_computed_tokens > 0` AND active spec decode in the same batch.

### Investigation timeline

1. **Hypothesis: Eagle block drop for SlidingWindow** — the blanket
   `use_eagle=False` disabled the drafter's own cache drop. Fixed by
   keeping `use_eagle=True` for SlidingWindow. But this broke cache
   hits to 0%.

2. **Hypothesis: CUDA stream race** — `torch.cuda.synchronize()`
   before commit prevented crash with `max_parallel=1` but not `>=2`.
   The event-sync attempt broke async scheduling.

3. **Hypothesis: Request reclassification** — cache-hit requests
   reclassified from decode→prefill, shrinking `state_indices_tensor_d`.
   But Python OOB checks showed no bounds violation.

4. **DSA debug** — `IndexKernel.cu:111: "index out of bounds"` in
   PyTorch's IndexKernel. The assert is from a `gather()` or
   `index_select()` on an undersized tensor.

5. **Systematic sync isolation** — Added `torch.cuda.synchronize()`
   after each stage (target forward, sample, commit, drafter, end of
   sample_tokens). None caught the error. Only CUDA_LAUNCH_BLOCKING
   prevents it.

6. **sample_complete_event** — CUDA event between sample_tokens(N)
   and execute_model(N+1). Doesn't help.

### Current understanding

The crash is a CPU-GPU buffer race that only manifests when:
- Our cache hit fix changes `num_computed_tokens` (non-zero for new turns)
- Multiple requests are batched (max_parallel>=2)
- GPU operations execute asynchronously (no CUDA_LAUNCH_BLOCKING)

The race is NOT between specific pipeline stages (all inter-stage
syncs were tried). It's likely in the `non_blocking=True` CPU→GPU
copies in `_prepare_inputs()` — the CPU modifies a buffer while the
GPU is still reading the previous copy. With our fix changing
`num_computed_tokens`, the async copy timing changes just enough to
trigger the race.

This may be a latent vLLM bug in hybrid model async scheduling that
was never triggered because `num_computed_tokens` was always 0 with
the baseline's 0% cache hits.

## Block size and KV cache

```
mamba_cache_mode=all → block alignment:
  mamba_state_bytes / attn_bytes_per_token ≈ 4100
  block_size = 512 * ceil(4100/512) = 512 * 9 = 4608
  With fp16 mamba: block_size = 2560 (state halved)

num_gpu_blocks = 80 (override from cudagraph profiling)
Available KV memory: 52 GiB
Max possible blocks: ~400 (limited by profiler bug, not memory)
```

## Files changed on the branch

| File | Lines | Description |
|------|-------|-------------|
| `vllm/v1/core/kv_cache_coordinator.py` | +22 | Skip Eagle drop for FullAttn/Mamba in hybrid coordinator |
| `vllm/v1/core/sched/scheduler.py` | +20 | Disable spec decode at mamba block boundaries |
| `vllm/v1/worker/gpu_model_runner.py` | +21 | sample_complete_event for async batch queue |
| `docs/features/*.md` | +300 | Investigation docs |
| `eval/collect_traces_nemotron.py` | +30 | Add Acc% and Spec% to watchdog |

## Next steps

1. **Investigate the async copy race** — add assertions in
   `_prepare_inputs()` to check if CPU buffers are being modified
   while async copies are in flight. Focus on `block_table.copy_to_gpu()`
   and `num_computed_tokens_cpu_tensor`.

2. **Test with TP=2** — the previous agent ran with TP=2 and
   max_parallel=8 successfully (with 0% hits). Test our fix with TP=2
   to see if more memory / different block allocation avoids the race.

3. **Upstream the max_parallel=1 fix** — even without multi-parallel
   support, the fix provides 49% cache hits and 100% accuracy. This
   is a significant improvement over the baseline's 0% hits.
