# PC + Eagle3 v2: Non-PC slots with boundary commit (revised)

> Clean redesign based on v1 learnings.
> Branch: `pc-spec-v2-nonpc-slots`
> Parent: `e11f201a0` (Eagle3 NemotronH support, before any PC+spec patches)

## The goal

Make BOTH prefix caching AND speculative decoding work together on
NemotronH (hybrid Mamba2 model). Not one or the other — both.

- Spec decode alone: 92% AIME-25 (throughput benefit)
- Prefix caching alone: 95% AIME-25 (multi-turn efficiency)
- Both together: target ~90%+ (currently 50% with v1 scratch approach)

## Root cause analysis (from v1)

### Why the non-PC spec path works (92%)

In `mamba_cache_mode != "all"`, each request gets K+1 dedicated slots:
```
state_indices_tensor_d[req] = [slot_0, slot_1, ..., slot_K]
```

The SSM kernel reads from `slot[init_token_idx]` (= previous step's
accepted slot) and writes K+1 new candidate states to `slot[0..K]`.
The conv kernel reads/writes `slot[0]` with widened state for rollback.
**No copying between pool and scratch. States persist in slots.**

### Why v1 (scratch + round-trip) got 50%

The scratch approach pre-copied pool→scratch before each step and
committed scratch→pool after each step. This per-step round-trip
through 31 mamba layers introduced subtle state drift that accumulated
over thousands of decode steps, degrading accuracy.

When the conv_state round-trip was removed (Idea 1), accuracy jumped
to **84% majority vote** on a partial run (19/30 problems). This
confirms the round-trip was the culprit.

### Why the upstream PC+spec path crashes

In `mamba_cache_mode="all"`, `state_indices_tensor_d` is the block table
(shape: num_decodes × max_num_blocks). The SSM kernel indexes it with
`init_token_idx = num_accepted_prev - 1`. With `init_token_idx > 0`,
this reads `block_table[req][N-1]` = the physical slot for logical
block N-1, which is a COMPLETELY DIFFERENT block — not the accepted
candidate's state. This causes reading from wrong slots →
illegal memory access → crash.

### Why prefix cache hits are 0% with spec decode

Two confirmed upstream issues:
1. **Block hash contamination** (#38182, #31920): the block hash
   includes speculative tokens, so subsequent turns' prefixes never
   match.
2. **Mamba boundary state contamination**: speculative candidate states
   are written to block table slots during the verify step, corrupting
   the boundary states that the prefix cache would snapshot.

Both need to be fixed for prefix cache to work with spec decode.

## The v2 approach

### Core idea

Use non-PC-style slots for the mamba decode path (so the kernel's
native rollback works), and separately commit correct boundary states
to the pool (so the prefix cache snapshots correct data).

### 1. Allocate K+1 dedicated mamba slots per request

When `mamba_cache_mode=all` AND spec decode is active, allocate
`1 + num_spec_tokens` extra slots per request from a scratch tensor
(similar to v1, but the tensor is used differently).

These slots are indexed as:
```
spec_slot_ids[req] = [base, base+1, ..., base+K]
```

where `base = 1 + req_batch_idx * (K+1)` (with +1 to skip NULL_BLOCK_ID).

### 2. Route the mamba decode kernel through non-PC mode

In the layer's forward, replace `state_indices_tensor_d` (the block
table) with the spec slot IDs for the kernel call:

```python
# Instead of gathering from the block table:
state_indices_tensor_d_input = spec_slot_ids  # [base, base+1, ..., base+K]
state_indices_tensor_d_output = spec_slot_ids
ssm_state_for_kernel = spec_scratch_ssm_state

# Conv state: use the regular pool with IS_APC_ENABLED=True
# (native APC path handles conv state correctly for non-spec steps)
```

The SSM kernel uses `init_token_idx` to select which slot to read,
and writes K+1 candidate states to the K+1 slots. **Native rollback,
no round-trip.**

### 3. Initialize on first decode step

On the FIRST decode step after prefill, copy the prefill's boundary
state from the pool to `spec_scratch[base+0]`:

```python
spec_scratch.index_copy_(0, base_long,
    ssm_state.index_select(0, canonical_in_slot_long))
```

This is a ONE-TIME copy per request, not per step.

**Detection**: on the first decode step, `init_token_idx = 0` (from
prefill's `num_accepted = 1`). The kernel reads from `base+0`.
On subsequent steps, `init_token_idx = N-1 > 0`, and the kernel reads
from `base+N-1` (written by the previous step's kernel). The init
copy only matters when `init_token_idx = 0`.

**Simplification**: ALWAYS copy pool → scratch[base+0]. It's a waste
on subsequent steps when init_token_idx > 0 (the kernel doesn't read
base+0), but it's harmless and avoids first-step detection logic.

### 4. Commit boundary states to the pool

After each step, for requests that crossed a block boundary:

```python
# Identify boundary-crossing requests
boundary_mask = (block_of(n_done + num_accepted - 1)
                 != block_of(n_done - 1))

# For boundary requests: copy the accepted state to the pool
# The accepted state is at spec_scratch[base + num_accepted - 1]
ssm_state.index_copy_(0, pool_slot_for_new_block,
    spec_scratch.index_select(0, base + num_accepted - 1))
```

This ensures the prefix cache snapshots correct boundary states.
Non-boundary requests don't touch the pool.

### 5. In-scratch rollback for init_token_idx=0

After each step, copy the accepted state to base+0:
```python
spec_scratch.index_copy_(0, base_long,
    spec_scratch.index_select(0, base + num_accepted - 1))
```

This ensures that if the NEXT step has `init_token_idx = 0`
(num_accepted = 1 from rejection), base+0 has the correct state.
Combined with the always-pre-copy from step 3, whichever runs LAST
wins. The in-scratch copy should run AFTER the pre-copy.

**Wait — ordering issue**: the pre-copy overwrites base+0 with stale
pool data, then the in-scratch copy overwrites it with correct data.
But the pre-copy happens in the FORWARD (step N+1) and the in-scratch
copy happens in the COMMIT (step N). The order is:
commit(N) → forward(N+1) → commit(N+1).

So: in-scratch copy at commit(N) → pre-copy at forward(N+1) → kernel.
The pre-copy CLOBBERS the in-scratch copy! Bad.

**Fix**: do the in-scratch copy at the START of the forward (before
the pre-copy), using the PREVIOUS step's num_accepted (from metadata).
Then the pre-copy runs and overwrites base+0. But the kernel reads
from base[init_token_idx]:
- If init_token_idx > 0: reads from base+N-1 (kernel's own output). ✓
- If init_token_idx = 0: reads from base+0 (pre-copy'd pool data).

The pool data = last boundary commit (or prefill's final state). If
the pool is up to date... it's NOT up to date (we only commit at
boundaries, not every step).

**Resolution**: also commit every step (not just boundaries). This
adds one `index_copy_` per step per layer (scratch → pool) to keep
the pool current. The pre-copy then reads correct data.

This is the SAME per-step commit as v1. But the critical difference:
the INPUT indices are non-broadcast. The kernel reads from its own
output for init_token_idx > 0, which is the majority of steps. The
commit + pre-copy round-trip only matters for init_token_idx = 0
(when num_accepted_prev = 1, ~30% of steps).

### Summary of per-step operations

| Operation | When | What |
|-----------|------|------|
| Pre-copy pool → scratch[base+0] | Every step | Initialize base+0 from pool |
| SSM kernel | Every step | Process K+1 candidates, write to base+0..K |
| In-scratch copy scratch[base+N-1] → scratch[base+0] | Every step (in commit hook) | Promote accepted state to base+0 |
| Commit scratch[base+0] → pool | Every step | Keep pool current for next pre-copy |
| Boundary commit to new block | At boundaries only | Snapshot boundary state for prefix cache |

Total copies per step per layer: 3 (pre-copy + in-scratch + commit).
v1 had 2 (pre-copy + commit). The extra in-scratch copy is cheap
(same tensor, no stride mismatch).

## Open questions

1. **Will the per-step commit reintroduce v1's accuracy regression?**
   The commit in v2 is from scratch[base+0] (the accepted state) to
   pool[canonical_slot]. In v1, the commit was from scratch[base+N-1]
   to pool. The difference: v2 commits from base+0 (which was just
   written by the in-scratch copy) instead of from base+N-1 directly.
   Whether this matters depends on whether the stride mismatch between
   scratch and pool causes subtle data corruption. The v1 SSM scratch
   uses PAGE-PADDED stride (matching the pool), so copies should be
   exact.

2. **Does the init_token_idx > 0 path avoid the pool entirely?**
   For those steps, the kernel reads from scratch (not pool) and writes
   to scratch. The pre-copy is wasted (overwrites base+0 but the kernel
   reads base+N-1). The commit writes to pool. So the pool is involved
   on the WRITE side but not the READ side. If the pool write is the
   source of v1's regression, removing the pool READ might help.

3. **Is per-step commit necessary?**
   Only needed for init_token_idx = 0 cases (~30% of steps). For
   init_token_idx > 0 cases, the pre-copy is wasted. Could
   conditionally skip the commit for those steps, but this requires
   knowing the NEXT step's init_token_idx, which we don't have.

4. **Block hash contamination (#38182)**: even with correct boundary
   states, the block hash might still prevent cache hits. This is
   a SEPARATE fix needed at the scheduler/block manager level.

## Experimental results to date

| Config | Per-session | Majority | Notes |
|--------|-------------|----------|-------|
| No-PC + Eagle3 (our code) | 92.9% | 96.4% | Baseline, proves our code works |
| PC only (no spec) | 95.0% | 100% | Baseline, proves PC works alone |
| v1 SSM+conv scratch | 50.0% | 60.0% | Full round-trip killed accuracy |
| v1 SSM scratch + native conv (Idea 1) | ??% | 84.2% (partial, 19 problems) | BEST SO FAR, testing now |
| v2 non-PC slots | TBD | TBD | To be implemented |

## Files to modify

1. `vllm/v1/attention/backends/mamba_attn.py` — buffer allocation, state_indices construction
2. `vllm/model_executor/layers/mamba/mamba_mixer2.py` — decode path routing, init copy, boundary commit
3. `vllm/v1/worker/gpu_model_runner.py` — boundary detection, num_accepted handling

## Reference

- `_pc_spec_decode_ablation_results.md` — full v1 experimental record
- `_pc_spec_decode_three_approaches.md` — option comparison and off-by-one analysis
- vllm-project/vllm#38182, #31920 — prefix cache hit rate with spec decode
- vllm-project/vllm#39273 — GDN spec decode corruption (same root cause class)
- vllm-project/vllm#39146 — stale KV blocks (unfixed, may affect mamba)
