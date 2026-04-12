# H100 FlashAttn + PC + Eagle3 Spec Decode Fix

> Branch: `fix/h100-flashattn-spec-init-capture-safe`
> Date: 2026-04-12
> GPU: NVIDIA H100 80GB HBM3 (sm_90)
> Model: chankhavu/c2-softcpy-fp8 + chankhavu/c2.eagle3-test (Eagle3)
> Base: `upstream-pr/nemotron-h-eagle3-pc` (camera-ready, commit `aea9c146c`)

## Problem

PC + Eagle3 speculative decoding on NemotronH crashes or produces
garbage output on H100 with FlashAttention, despite working correctly
on Blackwell (RTX PRO 6000, sm_120f) with FlashInfer.

**Root cause**: On H100 with FP8 KV cache, vLLM auto-selects FlashAttention
(FA3) as the attention backend. FA3 supports `FULL_AND_PIECEWISE` cudagraph
mode with spec decode. Under FULL capture, `mamba_mixer2` (a split op in
vLLM's compilation config) is captured INSIDE the cudagraph rather than
running eagerly. On Blackwell, FlashInfer unconditionally downgrades
`FULL_AND_PIECEWISE -> PIECEWISE` when spec decode is active, so
`mamba_mixer2` always runs eagerly there — masking five bugs in the
spec-slot mamba decode path that only manifest under FULL capture.

## The five bugs

### Bug 1: `if needs_init.any()` D2H sync during stream capture

**File**: `mamba_mixer2.py:872` (camera-ready line numbers)
**Symptom**: `cudaErrorStreamCaptureUnsupported` crash at boot during
cudagraph capture.

`conv_ssm_forward` checked `if needs_init.any():` where `needs_init` is
a CUDA bool tensor. The implicit `__bool__()` calls `.item()` which is a
synchronizing GPU->CPU copy, forbidden during stream capture.

**Fix**: Hoist the one-time pool->spec-slot init into a new runner-side
method `MambaMixer2.eager_init_spec_slots()`, called from `execute_model`
inside `set_forward_context` but before `_model_forward`. The captured
forward no longer reads or writes `_spec_inited`.

### Bug 2: `torch.split` exact-sum crash under padded capture buffers

**File**: `mamba_mixer2.py:663`
**Symptom**: `RuntimeError: split_with_sizes expects split_sizes to sum
exactly to 8, but got split_sizes=[7, 0]`

`torch.split(block_idx_last_computed_token, [num_decodes, num_prefills])`
requires the sum to equal the tensor's size. Under FULL capture, the
persistent `block_idx_last_*` buffers are exposed at `[:num_decode_tokens]`
which can be larger than `num_decodes + num_prefills` for padded capture
buckets.

**Fix**: Replace `torch.split` with plain slicing `[:num_decodes]` and
`[num_decodes:num_decodes+num_prefills]`, which is robust to larger
tensor sizes.

### Bug 3: `commit_boundary_states` off-by-one at `n_done = k * block_size`

**File**: `mamba_mixer2.py:1147`
**Symptom**: Pool block state corruption, subtle accuracy drift on
multi-turn sessions.

When `n_done` (num_computed_tokens before the step) lands exactly on a
block boundary (e.g. `n_done=256, block_size=256`), the old check
`blk_before = (n_done-1)//bs; blk_after = (n_done+n_acc)//bs;
needs_commit = blk_after > blk_before` fires spuriously. It then reads
`spec_slot[base+0]` which holds the state at position `n_done` (this
step's candidate 0), not the boundary position `n_done-1`. This
overwrites the pool's correct boundary state (committed in the prior
step) with a state from the wrong block.

**Fix**: Reframe the check as "is there a block-end position inside this
step's `[n_done, n_done+n_acc)` window?" using
`blk_current = n_done // block_size` and
`boundary_pos = (blk_current + 1) * block_size - 1`.

### Bug 4: `self._spec_pending` Python side-effect invisible to cudagraph replay

**File**: `mamba_mixer2.py:914`
**Symptom**: `commit_boundary_states` silently no-ops after step 1,
pool blocks never updated.

The forward stashed commit metadata via Python attribute assignment:
`self._spec_pending = (t1.clone(), t2.clone(), t3.clone())`. Cudagraph
replay executes the recorded CUDA ops (the `.clone()` memcpys) but does
NOT re-execute Python. After the first `commit_boundary_states` sets
`self._spec_pending = None`, the replay never reassigns the tuple.

**Fix**: The runner now clones the metadata in eager mode (before
`_model_forward`) via `_build_spec_commit_stash()` and passes it to
`commit_boundary_states` as explicit arguments.

### Bug 5: Cudagraph padding writes corrupt spec slots for cycling requests (PRIMARY ACCURACY KILLER)

**File**: Runner-side, no single line — emergent from FULL capture semantics.
**Symptom**: Post-`</think>` garbage tokens on ~40% of traces. AIME-25
accuracy drops from ~97% to ~50%.

Under FULL capture, the mamba kernel's grid size is the padded bucket
size `B` (e.g. 32), not the actual `num_decodes` `A`. For positions
`[A, B)` (padding), the kernel computes from dummy `hidden_states` and
WRITES garbage to `spec_ssm[spec_ids[i, 0..K]]` — real spec slot
addresses.

When a request cycles decode -> prefill (tool-call continuation) ->
decode, its batch position temporarily falls outside `[0, num_decodes)`
during the prefill step. The captured kernel writes garbage to that
position's spec slots. When the request returns to decode,
`_spec_inited[i]` is still True (the request didn't finish, just
transitioned), so `eager_init_spec_slots` skips it. The kernel reads
the garbage state. Drift accumulates over hundreds of decode steps
until the output collapses into random tokens.

Under PIECEWISE (Blackwell), this never fires: `mamba_mixer2` runs
eagerly with exactly `num_decodes` positions, no padding.

**Fix**: After `eager_init_spec_slots`, reset
`_spec_inited[num_decodes:] = False` so any position outside the
current decode group is forced to re-initialize from pool on re-entry.

## Why Blackwell never hit these

FlashInfer unconditionally downgrades `FULL_AND_PIECEWISE -> PIECEWISE`
when `num_speculative_tokens > 0`. Under PIECEWISE:

- `mamba_mixer2` is in `splitting_ops` -> runs eagerly (Python impl)
- No stream capture -> `.any()` sync is fine (bug 1)
- No padded capture buffers -> `torch.split` sums match (bug 2)
- `self._spec_pending =` re-executes each step (bug 4)
- Kernel processes exactly `num_decodes` positions, no padding writes (bug 5)
- `commit_boundary_states` off-by-one (bug 3) fires but with less
  multi-turn reuse pressure, so corruption is rarely observed

## Verified results on H100 80GB HBM3

```
Server config:
  VLLM_USE_FLASHINFER_MOE_FP8=1
  --kv-cache-dtype fp8
  --enable-prefix-caching --mamba-cache-mode all
  --mamba-block-size 256 --max-num-seqs 32
  --max-model-len 262144
  --speculative-config eagle3 num_speculative_tokens=5
  cudagraph_mode = FULL_AND_PIECEWISE (default, NOT --enforce-eager)
```

| Metric | Pre-fix | Post-fix |
|---|---|---|
| AIME-25 per-session (n=1, T=1.0) | ~50% (garbage) | **86.7%** (26/30, clean) |
| Garbage traces | ~40% | **0** |
| Spec decode acceptance | ~40% | ~41% |
| Prefix cache hit rate | climbing to ~60% | climbing to ~60% |
| Single-request throughput | 135 tok/s | **254 tok/s** |
| vs PIECEWISE (FlashInfer) | — | **3.45x faster** |

The 86.7% vs Blackwell's 96.7% gap is sampling noise at n=1, T=1.0
(4 model misses are legitimate wrong-math, not corruption). A 2-session
majority-vote run would converge.

## Files modified

| File | Changes |
|------|---------|
| `vllm/model_executor/layers/mamba/mamba_mixer2.py` | Bugs 1-5: remove in-forward init + stash, add `eager_init_spec_slots()`, fix `commit_boundary_states` signature + off-by-one, replace `torch.split` with slicing |
| `vllm/v1/worker/gpu_model_runner.py` | Runner hooks: `eager_init_spec_slots` call, `_build_spec_commit_stash`, `_spec_inited[nd:]` reset, commit data passing |

## How to use

Same command as before — no new flags needed. The fix activates
automatically when `mamba_cache_mode=all` AND `num_speculative_tokens > 0`
on any GPU where FlashAttention supports FULL cudagraphs (H100, etc.).

On Blackwell / FlashInfer (PIECEWISE auto-downgrade), the fix is a no-op:
`eager_init_spec_slots` short-circuits, `_spec_inited` reset covers
already-False positions, `_build_spec_commit_stash` returns the same data
the forward would have stashed.
