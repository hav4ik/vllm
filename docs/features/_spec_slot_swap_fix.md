# Mamba Spec Slot Fixes for Eagle3 + Prefix Caching

Branch: `opt/masked-triton-with-fixes`

## Problem

Eagle3 speculative decoding on NemotronH (hybrid mamba-transformer)
showed degraded accuracy during multi-turn conversations with tool
calls. Per-session accuracy on AIME-25 dropped from expected ~97% to
as low as 43% with 4 sessions × 32 parallel, and premature EOS at
T=0.0 (greedy) after only 100-400 tokens.

## Root Cause

Three bugs in mamba spec-slot management caused SSM/conv state
corruption during tool-call transitions:

### Bug 1 (critical): Batch reorder doesn't swap spec slots

`_may_reorder_batch` calls `reorder_batch_to_split_decodes_and_prefills`
which swaps batch positions (via `input_batch.swap_states`) to separate
decodes from prefills. This swaps all batch metadata but NOT the
per-position mamba spec slots (`spec_ssm`, `spec_conv`, `_spec_inited`).

After reorder, decode requests end up at batch positions whose spec
slots contain a different request's mamba SSM/conv state. Since
`_spec_inited` is True at the new position (from the previous
occupant), `eager_init_spec_slots` skips re-initialization, and the
model forward uses the wrong request's mamba state.

**Trigger**: Any mixed batch where the reorder changes positions —
i.e., every tool-call transition with concurrent requests.

**Fix**: Track moves from `_may_reorder_batch` and swap spec slots
(SSM state, conv state, inited flag) to match, using the same pattern
as the existing condense-move handler.

### Bug 2 (moderate): Streaming request remove doesn't reset _spec_inited

`_update_streaming_request` (tool-call continuation) calls
`input_batch.remove_request` without resetting `_spec_inited` for the
batch position. The finished-request path correctly resets the flag.

**Fix**: Reset `_spec_inited` before remove, matching finished-request
behavior.

### Bug 3 (critical): Conv state commit from wrong spec slot

`batch_commit_kernel` used the same `src_slot = base + cand_idx` for
both SSM and conv state commits. But the conv kernel only writes the
rolling-window state to `spec_conv[base]` (slot 0) — slots 1..K are
never written and contain stale/zero data. When `cand_idx > 0` (i.e.,
the block boundary isn't at the first candidate), the pool received
garbage conv state.

**Fix**: Pass `src_slot_conv=base` (always slot 0) for conv commits.

## Verification

AIME-25, 30 problems × 2 sessions, T=1.0, top_p=0.95, fp32 SSM,
32 parallel, max_model_len=131072, Eagle3 K=4:

| Branch | Per-session | Majority-vote |
|--------|-------------|---------------|
| Before fixes (`opt/masked-triton-commit`) | 34/47 = 72% | 18/30 = 60% |
| After fixes (`opt/masked-triton-with-fixes`) | **51/51 = 100%** | **26/26 = 100%** |

(4 problems still running at time of measurement, all completed
sessions correct.)

## Files Changed

- `vllm/v1/worker/gpu_model_runner.py` — spec slot swap on reorder,
  _spec_inited reset on streaming remove, conv commit src_slot_conv
- `vllm/model_executor/layers/mamba/batch_commit_kernel.py` — separate
  src_slot_conv parameter for conv state commits

## Latency Impact

Zero in steady-state decode. The spec slot swap only fires on
tool-call transitions (when `_may_reorder_batch` actually swaps
positions). Cost: ~0.5ms one-time per tool-call turn (23 layers ×
2 small tensor swaps).
