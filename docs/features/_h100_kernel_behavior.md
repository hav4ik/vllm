# Kernel behavior: SSM vs Conv null-index handling under FULL cudagraph

## Problem

Under FULL cudagraph capture, the kernel grid is padded to the bucket
size. Positions beyond `num_decodes` (padding) get processed by the
kernel with dummy hidden_states. The kernel writes to real spec slots,
corrupting state for requests that cycle decode→prefill→decode.

## SSM kernel (`selective_state_update`)

```
state load:  mask &= state_batch_idx != null_block_id  → reads 0 if null
state write: if token_dst_idx != null_block_id → SKIP write if null
output:      tl.store(out_ptrs, out) → ALWAYS writes hidden state
```

SSM handles null output indices correctly: state is preserved (write
skipped), and output hidden states are still computed (deterministic
from zero-masked state). Setting `dst_state_batch_indices[pad] = -1`
works for SSM.

## Conv kernel (`causal_conv1d`)

```
if conv_states_input_coord == null_block_id:
    return  ← RETURNS EARLY, skips ALL processing
```

Conv does NOT separate state handling from output computation. The
`return` skips both conv state updates AND hidden state output. This
leaves `hidden_states_B_C_d` uninitialized for padded positions.

## Why this matters

1. Conv returns early → hidden state tensor has stale/garbage for padding rows
2. SSM reads stale hidden states for padding rows → computes garbage output
3. Each batch position IS independent (no cross-contamination of real positions)
4. But stale conv output → slightly wrong SSM computation → wrong residual

## Approaches tried

### 1. `_spec_inited[nd:] = False` (current)
Forces re-init from pool on tool-call transitions. Works but causes
state rewind: pool has last boundary state (up to block_size-1 tokens
behind). ~85% per-session vs 94.6% PIECEWISE.

### 2. `_spec_output_ids[nd:] = NULL_BLOCK_ID`
SSM skips writes correctly, but conv returns early leaving uninitialized
hidden states. Made accuracy worse (83.6% → 75% majority).

### 3. Save/restore spec slots (not yet tried)
Before captured forward: save spec_ssm/spec_conv for positions [nd, num_reqs).
After captured forward: restore them (undo padding writes).
Cost: only on steps with mixed decode+prefill (tool-call transitions).
No kernel changes needed.

### 4. Kernel modification (ideal but invasive)
Modify conv kernel to separate state handling from output computation:
- Skip conv state read/write for null indices (preserve spec slot)
- Still compute and write hidden state output (so downstream ops get valid data)
Requires changes to `causal_conv1d.py` Triton kernel.
