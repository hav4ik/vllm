# Investigation: max_parallel >= 2 crash with PC + Eagle3

> Branch: `pc-spec-v2-nonpc-slots`
> Date: 2026-04-10

## Symptom

`CUDA error: device-side assert triggered` when 2+ requests are batched
together with prefix caching + Eagle3 spec decode enabled on NemotronH.

Works with:
- `max_parallel=1` (serial sessions) — 100% accuracy, 49% cache hits
- `CUDA_LAUNCH_BLOCKING=1` (serializes GPU ops) — also works

Does NOT work with:
- `max_parallel>=2` without CUDA_LAUNCH_BLOCKING

## Hypothesis: Request reclassification bug

When our cache hit fix enables non-zero cache hits, multi-turn requests
get reclassified from DECODE to PREFILL, which breaks spec slot indexing.

### The mechanism

1. **Classification** (`mamba_attn.py:355-365`):
   ```python
   # is_prefilling = num_computed_tokens < num_prompt_tokens
   # With cache hit: num_computed=4608, num_prompt=10000 → is_prefilling=True
   ```

2. **Batch split** (`mamba_attn.py:408-412`):
   ```python
   state_indices_tensor_d, state_indices_tensor_p = torch.split(
       state_indices_tensor, [num_decodes, num_prefills], dim=0)
   ```
   `num_decodes` is now SMALLER because the cache-hit request was moved
   to the prefill group.

3. **Spec slot indexing** (`mamba_mixer2.py:860-861`):
   ```python
   spec_ids = self._spec_slot_ids[:num_decodes]
   base_long = self._spec_base_long[:num_decodes]
   ```
   Uses the reduced `num_decodes`, which may not match the actual number
   of requests that need spec decode.

4. **Pool read** (`mamba_mixer2.py:875-877`):
   ```python
   canonical = state_indices_tensor_d.gather(
       1, block_idx_last_computed_token_d.unsqueeze(1))
   ```
   `state_indices_tensor_d` is undersized → out of bounds access → crash.

### Why this only happens with cache hits

- **Baseline (0% hits)**: New-turn requests have `num_computed_tokens=0`.
  They're classified as prefill regardless. No spec decode is active
  for them. The split is correct.

- **Our fix (non-zero hits)**: A request with a cache hit has
  `num_computed_tokens=4608` but `num_prompt_tokens=10000`. It was
  previously a running request in decode mode. Now when the scheduler
  sends it for partial prefill (with cache hit), it gets reclassified
  from decode→prefill. This shrinks `num_decodes`, but other requests
  in the batch still expect their decode slots.

### Why max_parallel=1 works

With only one request at a time, there's never a mix of decode and
prefill-with-cache-hit in the same batch. The request is either fully
in prefill or fully in decode.

## DSA debug results

With `CUDA_LAUNCH_BLOCKING=1 + TORCH_USE_CUDA_DSA=1` and `max_parallel=2`:

```
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:111:
Assertion `-sizes[i] <= index && index < sizes[i]
           && "index out of bounds"` failed.
block: [0,0,0], thread: [0,0,0]
```

**Confirmed: index out of bounds** in PyTorch's IndexKernel (used by
`gather()` / `index_select()`). Caught at `synchronize_input_prep()`
(line 3554) — asynchronous from the previous step's model forward.

This confirms the reclassification hypothesis: a `gather()` on
`state_indices_tensor_d` (or similar) is accessing an index beyond
the tensor's size because `num_decodes` was reduced by the
decode→prefill reclassification of cache-hit requests.

Note: the crash reproduces even with CUDA_LAUNCH_BLOCKING because
the index is genuinely out of bounds (not a race condition). Earlier
tests where CUDA_LAUNCH_BLOCKING prevented crashes were likely on
different code versions.

## Proposed fix

The `state_indices_tensor_d` split and `block_idx_last_computed_token_d`
slicing in `mamba_attn.py` need to account for cache-hit requests that
are reclassified as prefills. Options:

1. **Don't reclassify cache-hit requests as prefill** — override
   `is_prefilling=False` for requests with `num_computed_tokens > 0`
   AND active spec decode. They stay in the decode group.

2. **Adjust the split** — when building `state_indices_tensor_d`,
   include cache-hit requests that have spec decode active, even if
   they're in the prefill group.

3. **Separate mamba handling** — build mamba's prefill/decode split
   independently from the attention split, so cache-hit requests are
   always in the correct group for mamba state management.

Option 1 is simplest and most targeted.

## Debug round 2: Comprehensive bounds checks

Added Python-level OOB assertions before every `gather()` and
`index_select()` in `mamba_mixer2.py`:
- Prefill path (line 735): state_indices_tensor_p gather
- Spec init path (line 888): state_indices_tensor_d gather
- Commit path (line 1087): state_indices_d gather
- ssm_state indexing (line 740)

**Result: NO OOB assertions fired.** The crash is inside a CUDA kernel,
not in Python-level tensor operations.

## Crash dump analysis

The scheduler output at crash time shows:
```
2 requests: num_computed_tokens=[1940, 6085]
Both with num_scheduled_tokens=5 (spec decode: 1 target + 4 draft)
Both are DECODE (not prefill)
scheduled_spec_decode_tokens=[-1,-1,-1,-1] (all rejected from prev step)
```

**Key finding: The crash happens during DECODE, not during a cache-hit
prefill.** Both requests are in normal spec decode mode. No cache hit
is occurring at the crash step.

This means:
1. The reclassification hypothesis was WRONG for this crash
2. The crash is in the CUDA kernel itself (SSM or conv), not in PyTorch
3. Our cache hit fix changes block allocation patterns, which somehow
   causes the kernel to receive invalid indices later during decode
4. The baseline (without cache hit fix) handles 2 concurrent decodes
   fine, so the block allocation difference is the trigger

## Debug rounds 3-6: Systematic isolation

### Round 3 (debug v3): Python OOB checks on all gather/index ops
- Added bounds checks before every gather/index_select in mamba_mixer2.py
- **NONE triggered.** Crash is inside a CUDA kernel, not Python ops.

### Round 4 (debug v4): torch.cuda.synchronize() after target model forward
- Sync after _model_forward() passed → target model is NOT the source
- Crash caught at synchronize_input_prep in NEXT step
- **Conclusion: crash is in sample_tokens (commit/drafter/bookkeeping)**

### Round 5 (debug v5): 4 syncs (target, sample, commit, drafter)
- All 4 syncs passed, crash STILL at synchronize_input_prep
- **Conclusion: crash is AFTER the drafter, in post-drafter ops or
  between sample_tokens completion and next execute_model**

### Round 6 (debug v6): 5 syncs (+ end of sample_tokens)
- STILL crashes at synchronize_input_prep even with sync at END of
  sample_tokens
- This means the crash kernel is launched AFTER sample_tokens returns
- But sample_tokens is the last GPU-touching function before the next step

### Round 7: sample_complete_event
- Added CUDA event recorded at end of sample_tokens, waited at start
  of execute_model's model forward
- STILL crashes

### Key finding: Only CUDA_LAUNCH_BLOCKING=1 prevents the crash

This means it's NOT a simple ordering issue between steps. It's either:
1. A CPU-GPU buffer race (CPU overwrites a buffer being async-copied)
2. A race between different CUDA streams within the same step
3. An issue with the async batch queue where multiple steps overlap
   on the worker thread via futures

The crash at `synchronize_input_prep()` catches the error via the
`prepare_inputs_event`, which was recorded at the end of the PREVIOUS
step's execute_model(). If a kernel launched by _prepare_inputs()
(block_table copy, metadata setup) reads from a buffer that's still
being written by a concurrent operation on another stream, the
non-blocking copy could race with the kernel launch.

With CUDA_LAUNCH_BLOCKING=1, all operations become synchronous,
eliminating ALL races — but at a huge performance cost.

## Analysis: Eagle drafter block_table mismatch (ruled out)

Investigated whether the Eagle drafter uses the wrong block_table or
block_size. Finding: the drafter selects its own CommonAttentionMetadata
based on `kv_cache_gid` (line 2385 in gpu_model_runner.py), so the
block_table IS for the drafter's SlidingWindow group with correct sizes.

## Current state of understanding

The crash is a race condition confirmed by:
- Only CUDA_LAUNCH_BLOCKING=1 prevents it (full serialization)
- Event syncs between execute_model/sample_tokens don't help
- torch.cuda.synchronize() at every pipeline stage doesn't help
- The DSA error is IndexKernel.cu "index out of bounds"
- Python-level OOB checks on all mamba tensor ops don't trigger

The race is NOT between:
- execute_model(N) and execute_model(N+1)
- sample_tokens(N) and execute_model(N+1)
- target model forward and drafter forward

The race IS between something that only CUDA_LAUNCH_BLOCKING serializes.
This could be:
- An intra-kernel race (two CUDA thread blocks accessing shared memory)
- A host-device race (CPU pinned memory modified while GPU DMA reads it)
- A multi-stream race within the SAME step (e.g., async_output_copy_stream)

The key invariant: the crash ONLY happens when our cache hit fix
changes num_computed_tokens from 0 to 4608+. This is a code path
never exercised in the baseline.

## Round 8-9: Async scheduling and full sync at start

### Round 8: --no-async-scheduling
Disabled async batch queue entirely. STILL crashes at ~14K gen tokens.
Eliminates the batch queue overlap as the cause.

### Round 9: torch.cuda.synchronize() at START of execute_model
Added sync before ANY work in execute_model (syncs all streams, all
devices). STILL crashes.

This means the race is NOT between steps or between streams. The crash
happens within a SINGLE execute_model+sample_tokens call, within
operations on the DEFAULT stream.

### Current understanding (updated)

The `torch.cuda.synchronize()` at the start ensures all prior GPU
work is complete. Then execute_model runs: _prepare_inputs → build
metadata → model forward → sample → commit → drafter. ALL on the
default stream (with enforce-eager). Yet it still crashes.

The only explanation: a KERNEL-LEVEL bug where the kernel itself has
an index-out-of-bounds for certain input patterns. Our cache hit fix
changes the block table pattern (some entries point to cached blocks
instead of fresh allocations), and a kernel (Triton or CUDA) doesn't
handle this pattern correctly.

This requires CUDA compute-sanitizer (racecheck/memcheck) to debug
further. The Python-level investigation has exhausted all plausible
inter-operation races.

### Alternative: the crash is deterministic, not a race

If torch.cuda.synchronize() at every possible point doesn't help,
and CUDA_LAUNCH_BLOCKING=1 DOES help, the difference might be in
how CUDA_LAUNCH_BLOCKING affects kernel compilation or launch
parameters, not just timing. CUDA_LAUNCH_BLOCKING=1 forces
synchronous kernel launches which can change:
- Grid/block dimensions (some kernels adapt to occupancy)
- Memory allocation patterns
- Tensor layout/stride assumptions

This opens the possibility that CUDA_LAUNCH_BLOCKING=1 works not
because it serializes, but because it changes some kernel parameter
that avoids the bug.

## Round 10: Isolation test — no spec decode + max_parallel=2

Ran with our cache hit fix but WITHOUT Eagle3 speculative decoding.
Same config: enforce-eager, max_parallel=2, max_model_len=65536.

Result: **STABLE at 20K+ gen tokens, 18K cache hits, no crash.**

This DEFINITIVELY proves:
1. Our cache hit fix (kv_cache_coordinator.py) is correct
2. The mamba boundary state handling is correct  
3. The mamba/attention kernels handle shared cached blocks correctly
4. Cache hits work correctly with max_parallel=2

**The crash is 100% caused by the Eagle3 drafter** when operating on
batched requests where the block table has cached block entries from
our prefix cache fix. Without Eagle3, everything works.

The next step: instrument the Eagle3 propose() method and its kernels
(eagle_step_update_slot_mapping_and_metadata, the drafter's model
forward) to find the exact OOB index.

## Round 11: seq_lens clone fix

Found that Eagle's propose() modifies `common_attn_metadata.seq_lens`
in-place, which is a VIEW of `self.seq_lens` in gpu_model_runner.
Two modification paths:
1. `seq_lens -= num_rejected_tokens_gpu` (Python in-place op)
2. `eagle_step_update_slot_mapping_and_metadata` Triton kernel writes
   `seq_lens += 1` per drafter iteration

Fix: `common_attn_metadata.seq_lens = common_attn_metadata.seq_lens.clone()`
before any modifications.

Result: crash moved from ~8K to ~55K gen tokens with max_parallel=4.
Significant improvement but not a complete fix.

Cloning additional tensors (block_table_tensor, slot_mapping,
query_start_loc) did NOT help further — the 55K crash is a different
issue from the seq_lens corruption.

## Current status

The seq_lens clone significantly reduces the crash frequency.
The remaining crash at ~55K gen tokens requires further investigation
(possibly with CUDA compute-sanitizer or by identifying additional
shared state that the drafter corrupts).

For production use:
- max_parallel=1 + enforce-eager: FULLY STABLE, all metrics good
- max_parallel=4: survives ~55K tokens (~5 problems) with seq_lens clone
- Full 30-problem eval: needs either max_parallel=1 or TP=2 setup
