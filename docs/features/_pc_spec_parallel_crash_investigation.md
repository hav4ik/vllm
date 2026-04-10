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
