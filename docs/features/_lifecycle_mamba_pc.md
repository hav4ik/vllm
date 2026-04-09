# Mamba SSM state lifecycle in vLLM v1 (NemotronH, `mamba_cache_mode=all`)

This doc traces how Mamba-2 SSM (and conv) state flows through vLLM v1 for a
hybrid model (NemotronH / Cascade-2) with Mamba prefix caching enabled. The
focus is on SSM state blocks — not the attention KV cache blocks — though the
two share a block-id space.

## 1. Mamba pool sizing at startup

### `MambaSpec.max_memory_usage_bytes`

At `vllm/v1/kv_cache_interface.py:332-359` the `MambaSpec` dataclass reports
its per-layer memory budget with three branches:

- **`all`** (the PC mode): `cdiv(max_model_len, block_size) * page_size_bytes`.
  That is, *one SSM-state slot per mamba block of the longest possible
  sequence*. With `max_model_len=262144` and `mamba_block_size=512` that is
  512 state slots per layer — matching the full request length on the logical
  block grid.
- **`align`**: `page_size_bytes * (2 + num_speculative_blocks)` — just a tiny
  ring of slots (running state + shadow), because the manager juggles block
  ownership instead of growing a pool.
- **`none`** (no PC): `page_size_bytes * (1 + num_speculative_blocks)` — one
  slot for the running state plus speculative shadow slots.

`page_size_bytes` itself (`:342-350`) is
`sum(prod(shape) * dtype_size for (shape, dtype) in zip(shapes, dtypes))` — for
NemotronH it is conv_state + ssm_state tiled together. The page size is padded
to match the attention page size (`page_size_padded`, see
`unify_kv_cache_spec_page_size` in `vllm/v1/core/kv_cache_utils.py:914-951`).

### `get_num_blocks`

`vllm/v1/core/kv_cache_utils.py:839-854` divides available GPU memory by
`page_size * num_layers` where `num_layers` is the **group size** (max layers
per group). Because the page size has been padded to be uniform across
attention and mamba groups, a single `num_blocks` value applies to both kinds
of groups in the hybrid block pool. See also `get_kv_cache_config_from_groups`
at `:1081-1156` which builds one `KVCacheTensor` per slot in the repeating
pattern.

### Allocation of the actual GPU tensor

`GPUModelRunner._allocate_kv_cache_tensors` at
`vllm/v1/worker/gpu_model_runner.py:6548-6578` first allocates a raw
`torch.int8` buffer of size `KVCacheTensor.size` per group slot, shared between
the layers that map onto that slot.

`_reshape_kv_cache_tensors` at `:6589-6691` then reinterprets the raw buffer.
For a mamba layer (`:6661-6684`) it iterates `kv_cache_spec.shapes` (for
NemotronH: conv shape and ssm shape from `mamba2_state_shape` in
`vllm/model_executor/layers/mamba/mamba_utils.py:165-190`) and builds each
state tensor with `torch.as_strided`, using `num_element_per_page =
page_size_bytes // dtype_size` as the outer stride. The result is:

```
kv_caches[layer_name] = [conv_state, ssm_state]
conv_state.shape == (num_blocks, conv_dim, conv_kernel - 1 + num_spec)
ssm_state.shape  == (num_blocks, num_heads//tp, head_dim, ssm_state_size)
```

Inside `Mamba2Mixer.conv_ssm_forward` (`vllm/model_executor/layers/mamba/mamba_mixer2.py:582-587`)
the layer grabs `self.kv_cache[0]` as conv_state (possibly with a transpose
for layout) and **`self.kv_cache[1]` as the ssm_state**. The leading dim is
the pool slot index (block id) — not the request index.

## 2. Block table mapping for mamba layers

`vllm/v1/core/block_pool.py:129-182` defines `BlockPool`: a flat array of
`KVCacheBlock(idx)` objects, a `FreeKVCacheBlockQueue` linked list, and a
`cached_block_hash_to_block` hash map for prefix-cache lookups. Block 0 is
reserved as `null_block` (`:175-176`). **Attention KV-cache blocks and mamba
state blocks share the same BlockPool and therefore the same block-id space**
— `_reshape_kv_cache_tensors` just reinterprets each group's slice of the raw
buffer with a group-specific stride/shape.

The per-layer block_size can still differ: attention uses the model's attention
block size (e.g. 16 or 64), mamba uses `mamba_block_size` (512 in the sample
launch). When `mamba_block_size != attn_block_size`, the hash is computed at
`hash_block_size` granularity and widened for mamba via
`BlockHashListWithBlockSize` in `BlockPool.cache_full_blocks`
(`vllm/v1/core/block_pool.py:240-251`), while the shared block-id domain keeps
allocation trivial.

### `MambaManager.get_num_blocks_to_allocate` and `allocate_new_blocks`

`vllm/v1/core/single_type_kv_cache_manager.py:862-925` (`get_num_blocks_to_allocate`)
and `:927-1003` (`allocate_new_blocks`). Under `mamba_cache_mode != "align"`
(i.e. `all` or `none`), the manager inflates the requested `num_tokens` by
`block_size * num_speculative_blocks` (`:880-886`, `:931-935`). The rationale:
Eagle/MTP steps write speculative SSM states into the *next* block id, so the
pool must reserve room for them. Under `all` this simply falls through to
`SingleTypeKVCacheManager.allocate_new_blocks` (`:215-242`) which appends fresh
blocks from `block_pool.get_new_blocks`. Contrast with `align` mode (`:894-
925`, `:939-1003`), which tracks `last_state_block_idx` and rotates one
running-state slot instead of growing the block table — the null-block padding
between allocated indices is the visible symptom.

`MambaManager` also guards against cross-request races in a single step: if
the candidate prefix-cache hit was cached *this step*
(`:871-879`), it returns `num_gpu_blocks + 1` so the scheduler defers the
request — mamba PC cannot consume a state block that another request is still
writing.

## 3. Per-step decode write path

`Mamba2Mixer.conv_ssm_forward` decode branch at
`vllm/model_executor/layers/mamba/mamba_mixer2.py:820-899`.

When PC is on (`is_mamba_cache_all`), per-request **block-grid metadata** is
split out of the attention metadata (`:628-649`):

- `block_idx_last_computed_token`: index into the request's block table of the
  block that holds the SSM state at position `num_computed_tokens - 1`
  (`vllm/v1/attention/backends/mamba_attn.py:309-338`:
  `cdiv(num_computed, mamba_block_size) - 1`). This is the *source* block
  whose stored boundary state is the starting point for this step.
- `block_idx_last_scheduled_token`: index of the block that contains the last
  token being scheduled this step:
  `cdiv(seq_lens, mamba_block_size) - 1`. This is the *destination* block that
  will hold the new state after the step.
- For decode with chunk alignment, `first` and `last` scheduled indices are
  equal except at block boundaries, where `last > computed`.

At `:822-828` these indices are gathered into flat slot ids:

```python
state_indices_tensor_d_input  = state_indices_tensor_d.gather(1, last_computed.unsqueeze(1)).squeeze(1)
state_indices_tensor_d_output = state_indices_tensor_d.gather(1, last_scheduled.unsqueeze(1)).squeeze(1)
```

`state_indices_tensor_d` is `block_table_tensor` for the decode sub-batch
(`vllm/v1/attention/backends/mamba_attn.py:405-417`) — rows are requests,
columns are the per-request block-table indices mapping logical mamba blocks
→ global pool slot ids. The two gathers above therefore produce:

- `state_indices_tensor_d_input[r]` = slot id whose `ssm_state[slot]` holds the
  committed boundary state at the tail of the previous block (or the end of
  the previous step, if still inside the same block).
- `state_indices_tensor_d_output[r]` = slot id where the new state for this
  step should be written.

These are then passed into `selective_state_update`
(`vllm/model_executor/layers/mamba/mamba_mixer2.py:880-899`) with
`state_batch_indices=..._input` and `dst_state_batch_indices=..._output`.

Inside the Triton kernel at
`vllm/model_executor/layers/mamba/ops/mamba_ssm.py:140-316`:
`state_ptrs = state_ptr + state_batch_idx * stride_state_batch` and
`dst_state_ptrs = state_ptr + dst_state_batch_idx * stride_state_batch`
(`:168-202`). The kernel reads the initial SSM state from the *input* slot,
runs the recurrence `state = state * dA + dB * x[:, None]` (`:255`), writes
outputs to `out_ptrs`, and finally stores the updated state into the
*output* slot (`:287-316`). `null_block_id` masking (`:155-206`) skips padded
entries.

**At block boundaries** input and output slots are distinct: the kernel is
effectively `dst_block_state = update(src_block_state, x)`, writing a fresh
boundary state into the next block's SSM slot. When we are still inside a
block, input and output are the same slot and the update is in-place (matching
the `none`/legacy path where `dst_state_batch_indices = state_batch_indices`
at `:427` when not provided).

## 4. Per-step prefill write path (intermediate-state snapshotting)

`Mamba2Mixer.conv_ssm_forward` prefill branch
`vllm/model_executor/layers/mamba/mamba_mixer2.py:665-818`.

The causal_conv1d_fn call at `:682-697` is already prefix-cache-aware via
`block_idx_first_scheduled_token`, `block_idx_last_scheduled_token`, and
`initial_state_idx=block_idx_last_computed_token_p` — it writes boundary
*conv* states at every mamba-block boundary inside the chunked prefill. For
the SSM path the logic is explicit Python rather than kernel-internal.

First, the initial states are gathered using the per-request
`block_idx_last_computed_token_p` (`:704-715`): the kernel looks up
`ssm_state[state_indices_tensor_p.gather(1, last_computed).squeeze(1)]` to
seed the recurrence from a cached boundary state.

The scan itself (`:719-741`) is
`mamba_chunk_scan_combined_varlen(..., return_intermediate_states=is_mamba_cache_all, ...)`.
Because PC is on, `varlen_states` returns the state at the **end of every
chunk** — not just the last chunk per sequence. The shape is
`(num_total_chunks, num_heads, head_dim, ssm_state_size)`.

The boundary-snapshot loop at `:743-805` walks each prefill sequence:

```
chunk_stride = mamba_block_size // chunk_size   # e.g. 512//256 = 2
for seq in range(num_prefills):
    n_blocks_to_fill = last_scheduled - first_scheduled
    if n_blocks_to_fill == 0:
        continue
    cache_blocks_to_fill = state_indices_tensor_p[seq,
        first_scheduled:last_scheduled]           # slot ids of interior blocks
    first_chunk = 0 if seq == 0 else last_chunk_indices_p[seq-1] + 1
    first_aligned_chunk = first_chunk + chunk_stride - 1
    if num_computed_tokens[seq] % mamba_block_size > 0:
        first_aligned_chunk -= num_unaligned_computed_tokens // chunk_size
    from_where = varlen_states[first_aligned_chunk :
                               first_aligned_chunk + n_blocks_to_fill*chunk_stride
                               : chunk_stride]
    ssm_state[cache_blocks_to_fill] = from_where
```

In English: for every interior mamba-block boundary crossed during this
prefill chunk, find the intermediate state at the chunk index aligned on the
block boundary (accounting for partially-cached leading tokens), and scatter
it into the SSM slot corresponding to that interior block. Then at `:808-812`
the final (possibly partial) boundary state for each sequence is written into
the slot of its `last_scheduled` block via another gather/assign.

**This is where the boundary states that prefix caching depends on actually
get computed and written.** Under `none` or `align` mode, only the tail state
ever lands in the pool (`:818`); under `all` mode every block boundary that is
traversed during a prefill chunk gets a proper snapshot.

## 5. Block commit to the radix tree (prefix cache hashing)

`cache_blocks` base impl at
`vllm/v1/core/single_type_kv_cache_manager.py:250-274`: when
`num_tokens // block_size > num_cached_blocks`, it forwards to
`BlockPool.cache_full_blocks(request, blocks, num_cached, num_full,
block_size, group_id)`.

`BlockPool.cache_full_blocks`
(`vllm/v1/core/block_pool.py:210-318`) walks
`blocks[num_cached:num_full]`, attaches a `BlockHashWithGroupId` to each, and
inserts it into `cached_block_hash_to_block`. Null blocks (from align mode) are
skipped at `:257-262`. **Crucially this function only touches metadata — it
does not read any tensor.** The claim is: *the current contents of
`ssm_state[block_id]` are already the boundary state for this block*.

The Mamba prefill path guarantees that claim: section 4 writes each interior
boundary state into exactly the slot id whose `KVCacheBlock` object is being
committed here, because `state_indices_tensor_p` is built from the same
`req_to_blocks[request_id]` that `cache_blocks` iterates. At decode time the
claim is preserved too — at a block boundary the decode kernel writes the new
state into `block_idx_last_scheduled_token`'s slot (section 3).

`MambaManager.cache_blocks` override at `:1019-1033` calls the base and also
records the newly-cached block hashes into `cached_blocks_this_step` so that
another request in the same step will be refused the hit (`:871-879` in
`get_num_blocks_to_allocate`) — again, because the state has not yet been
committed from the worker's perspective.

`MambaManager.find_longest_cache_hit` at `:778-824` is unusual: it scans from
right to left (`for i in range(max_num_blocks - 1, -1, -1)`), takes the
*single longest* cached prefix, and pads the leading positions with null
blocks. That is because a cached mamba boundary state is sufficient on its
own — there is no chain-dependency back through the earlier blocks as there is
for KV cache. Alignment to attention block size is enforced at `:810-814`.

Eviction: a cached block returns to the free queue via
`BlockPool.free_blocks` (`vllm/v1/core/block_pool.py:409-423`) which
decrements `ref_cnt`; when another request pulls it out of the queue via
`get_new_blocks` (`:320-349`) it goes through `_maybe_evict_cached_block`
(`:352-390`) which clears its block_hash and pops it from
`cached_block_hash_to_block`. Until that moment the slot still holds its
boundary state and can be re-hit.

## 6. Per-request lifecycle

**Cold-start request (no PC hit).** The scheduler asks `MambaManager` for
enough blocks via `get_num_blocks_to_allocate`; under `all`, `num_tokens` is
padded with `block_size * num_speculative_blocks` and the base
`allocate_new_blocks` extends `req_to_blocks` from the free queue. Each new
block is a fresh slot with no state. During the first prefill chunk,
`has_initial_states_p` is False so `initial_states = None` and the chunk scan
starts from zero; the boundary-snapshot loop fills `ssm_state[...]` at every
block boundary crossed, and `cache_blocks` is called from the scheduler once
enough tokens are computed to close out each full block.

**Request with a PC hit.** `find_longest_cache_hit` returns the longest
matching block prefix for this sequence. `allocate_new_computed_blocks`
(`vllm/v1/core/single_type_kv_cache_manager.py:142-213`) touches those blocks
(`block_pool.touch`, bumping ref_cnt and possibly unlinking them from the free
queue), then appends them to `req_to_blocks[request_id]`. The matched blocks
are already in `num_cached_block` so `cache_blocks` will not re-commit them.
`num_computed_tokens` starts at `hit_len * mamba_block_size`, so the first
scheduled chunk sees `has_initial_states_p=True`,
`block_idx_last_computed_token_p` pointing at the hit block, and the SSM scan
reads `ssm_state[hit_slot]` as its starting state at `:711-715`.

`block_idx_last_computed_token` advances implicitly via
`num_computed_tokens`: after each step the scheduler updates
`request.num_computed_tokens`, and on the next step
`_compute_prefix_caching_block_indices` at
`vllm/v1/attention/backends/mamba_attn.py:309-338` recomputes
`cdiv(num_computed_tokens, mamba_block_size) - 1`. So the source-block index
ticks forward by one every time the request crosses a full mamba-block
boundary; between boundaries it stays pinned to the same slot and the decode
path reads/writes in-place (section 3).

**Spec-decode interaction.** `remove_skipped_blocks`
(`vllm/v1/core/single_type_kv_cache_manager.py:826-854`) deducts
`num_speculative_blocks` from `num_computed_tokens` before pruning, so an
unverified draft never frees state blocks that may still be needed after
rejection. The speculative shadow slots reserved at allocation time absorb
the draft writes without touching the committed boundary state.

**Finishing.** When a request finishes, the engine calls
`SingleTypeKVCacheManager.free(request_id)` (`:276-291`), which pops
`req_to_blocks[request_id]` and calls `block_pool.free_blocks` in **reverse
order** (tail blocks first) so recently-used blocks sit at the head of the
free queue and survive longer against eviction. `num_cached_block` is popped.
Cached boundary states remain valid in `ssm_state[...]` until another
`get_new_blocks` call evicts the slot via `_maybe_evict_cached_block` and
returns it to a different request. `MambaManager.free` (`:1005-1009`) just
clears `align`-mode bookkeeping and delegates.
