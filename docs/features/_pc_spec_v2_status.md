# PC + Eagle3 v2 status — 2026-04-10

## Where we are

The v2 approach (unified spec slots for both SSM and conv, non-APC mode,
native kernel rollback) is functionally correct on single requests but
has a **slot lifecycle management** issue with concurrent requests.

## Results so far

| Test | Per-session | Majority | Issue |
|------|-------------|----------|-------|
| No-PC Eagle3 baseline | 92.9% | 96.4% | Target |
| v2 without init flag | 0% | — | New requests read stale spec slots |
| v2 with init flag (no condensation fix) | ~31% | TBD | Condensation breaks slot mapping |
| v2 with condensation fix | TBD | TBD | Not yet implemented |

## The condensation problem

vLLM v1's `condense()` compacts the batch by shifting requests down when
earlier requests finish. Example:

```
Before: [req_A(pos=0), _, req_C(pos=2), req_D(pos=3)]
After:  [req_A(pos=0), req_C(pos=1), req_D(pos=2)]
```

Our spec slot assignment uses batch position:
`spec_slot_base = 1 + batch_position * (K+1)`

After condensation, req_C moves from pos=2 (slots 11-15) to pos=1
(slots 6-10). But its SSM/conv data is still in slots 11-15. The
spec slots don't follow the request across condensation.

## The proper fix

**Don't derive spec slots from batch position.** Instead:

### Option A: Store spec slot base per-request
Add a per-request field (like `num_computed_tokens`) that tracks the
spec slot base. Set at request admission, survives condensation.
The input_batch already has per-request arrays that get rearranged
during condense (e.g., `block_table`, `num_computed_tokens_cpu`).
Adding one more field for `spec_slot_base` is straightforward.

### Option B: Use the block table itself
Modify the metadata builder to put spec slot IDs directly into
`state_indices_tensor_d` when constructing it. The block table is
per-request and survives condensation. This is the cleanest approach
but requires changes to `mamba_attn.py`'s build path.

### Option C: Handle condensation explicitly
During condense, rearrange _spec_inited flags AND copy spec slot data
from old positions to new positions. Works but is fragile (must be
done for every layer's spec_ssm and spec_conv tensors).

**Recommendation: Option B** — it's the same thing the non-PC path
already does (per-request slot IDs in the block table). We just need
to populate the block table with spec slot IDs instead of block
table IDs when spec decode is active.

## What works correctly

1. Single-request smoke tests: correct answers, coherent output
2. Init from pool on first decode step: fixed via _spec_inited flag
3. Reset on request finish: fixed via runner hook
4. Boundary commit: detects boundary crossings, copies to pool
5. Both SSM and conv use same spec slots (no desync)
6. Native kernel rollback via init_token_idx / conv_state_token_offset

## What's broken

1. **Condensation**: spec slots don't follow requests across batch
   position changes
2. **Prefix cache hits**: still 0% (upstream #38182 — block hash
   includes speculative tokens)

## Commits on branch `pc-spec-v2-nonpc-slots`

- `ca4472ed6` docs: v2 design
- `fa70b1c27` docs: v2 design revised
- `ca4810b4d` v2 implementation (SSM scratch only)
- `608c7276d` in-scratch promotion fix
- `f7b783aef` unified spec slots (both SSM + conv)
- `b6005b3a4` one-time init via _spec_inited flag
- `3a6d8d891` fix: define spec_conv_view before use
- `0490a29c5` fix: reset _spec_inited on request finish
