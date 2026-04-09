# PC + Eagle3 v2: Non-PC slots with boundary commit

> Clean redesign based on learnings from v1 (the scratch tensor approach).
> Branch: `pc-spec-v2-nonpc-slots`
> Parent: `e11f201a0` (Eagle3 NemotronH support, before any PC+spec patches)

## The insight

The non-PC Eagle3 path gets **92% AIME-25 accuracy**. It uses:
- K+1 direct slot IDs per request in `state_indices_tensor_d`
- The kernel's built-in `init_token_idx` for SSM rollback
- The widened conv state + `conv_state_token_offset` for conv rollback
- `IS_APC_ENABLED=False` for the conv kernel
- States persist in fixed slots across decode steps (no copying)

The v1 scratch approach got **50%** because of the per-step pool→scratch→pool
round-trip that introduced subtle state drift over thousands of steps.

## The v2 approach

**Run mamba decode EXACTLY like the non-PC path** (direct slots, native
rollback, no round-trip). Then add two surgical extensions:

### 1. Allocate non-PC-style slots even when PC is enabled

When `mamba_cache_mode=all` AND `num_speculative_tokens > 0`, the metadata
builder allocates `state_indices_tensor_d` with shape
`(num_decodes, 1 + num_spec_tokens)` instead of `(num_decodes, max_num_blocks)`.
Each entry points to a **dedicated per-request slot** (not a block table slot).

These slots are allocated from a separate scratch pool (like v1) or from
extra slots in the mamba state pool. The key difference from v1: **no
per-step copying**. States persist in these slots across all decode steps.

### 2. Boundary commit with safety margin

At mamba block boundaries (every B=512 tokens), commit the accepted state
from the non-PC slot to the block table's pool slot. This is for the prefix
cache to snapshot.

**Safety margin**: to avoid off-by-one issues at boundaries, disable spec
decode (force `num_accepted=1`) within a safety zone:
- Stop speculating at position `B - K - margin` (e.g., 512 - 5 - 2 = 505)
- Resume speculating at position `B + margin` (e.g., 512 + 2 = 514)

This gives a window of `K + 2*margin + 1 = 10` greedy steps per boundary
= ~2.0% of decode steps. Worth it for correctness.

**Force `num_accepted=1`** (not truly disable spec): the kernel still
processes K+1 candidates, but only 1 is kept. This works because:
- The kernel's native rollback handles partial acceptance correctly
- With 1 accepted token, the state advances by exactly 1 position
- At the boundary position, we commit the state to the pool
- The boundary state is guaranteed to be from a single-token advance
  (no speculative contamination)

### 3. Conv state: use the regular APC path

The conv state goes through the regular pool with `IS_APC_ENABLED=True`
(the upstream path). Since prefix cache hits are 0% with spec decode
(upstream issues #38182/#31920), contamination of cached conv boundary
states is harmless.

This eliminates the need for conv scratch entirely.

## What changes

### File: `vllm/v1/attention/backends/mamba_attn.py`

1. **Buffer allocation**: When `mamba_cache_mode=all` AND spec decode,
   allocate `state_indices_tensor_d` with shape
   `(max_bs, 1 + num_spec_tokens)` (non-PC style).

2. **Metadata build**: Populate `state_indices_tensor_d` with per-request
   slot IDs from the scratch pool (or from extra pool allocations).

3. **Add boundary-zone detection**: Compute `in_boundary_zone` per request
   based on `num_computed_tokens`, block_size, K, and margin.

### File: `vllm/model_executor/layers/mamba/mamba_mixer2.py`

1. **Decode path**: When PC + spec, use the SAME code as the non-PC spec
   path (the `else` branch at line ~1150). No scratch, no pre-copy,
   no commit. States persist in the non-PC slots.

2. **Boundary commit**: After the rejection sampler, for requests where
   `in_boundary_zone=True`, commit the accepted state from the non-PC
   slot to the block table's pool slot.

### File: `vllm/v1/worker/gpu_model_runner.py`

1. **Boundary clamp**: Force `num_accepted=1` for requests in the
   boundary zone (same as v1's approach, but with the safety margin).

2. **State initialization**: On the first decode step after prefill,
   copy the prefill's final state from the block table's pool slot
   to the non-PC slot. This is a ONE-TIME copy per request.

## Why this should work

1. **Decode path is identical to the 92% baseline**: same kernel mode,
   same slot structure, same rollback mechanism, no round-trip.

2. **Boundary handling is conservative**: the safety margin ensures
   no speculative state crosses a boundary. The greedy zone is only
   ~2% of steps.

3. **No new kernel changes**: all existing kernels are used as-is.

4. **No new state management**: non-PC slots are the same mechanism
   vLLM already uses for `mamba_cache_mode != "all"`.

## Risks

1. **Slot allocation**: need to allocate (1 + K) extra slots per running
   request from the mamba state pool. With max_num_seqs=16 and K=4,
   that's 80 extra slots. Each slot is ~2MB (SSM fp32), so ~160MB total.

2. **First-step initialization**: need to correctly detect "first decode
   step after prefill" and copy the prefill's boundary state to the
   non-PC slot.

3. **Request lifecycle**: when a request finishes, its non-PC slots must
   be freed. When a new request starts, it gets fresh slots.

4. **Prefix cache still 0% hits**: this approach doesn't fix the
   0% hit rate issue (#38182/#31920). But it doesn't make it worse.

## Comparison with v1

| Aspect | v1 (scratch) | v2 (non-PC slots) |
|--------|-------------|-------------------|
| SSM rollback | Custom broadcast + round-trip | Native kernel init_token_idx |
| Conv rollback | Custom scratch + round-trip | Native APC path (upstream) |
| Per-step copies | 2 × 31 layers (pool↔scratch) | 0 (states persist in slots) |
| Boundary handling | Force accept=1 + commit | Force accept=1 + commit (with margin) |
| Expected accuracy | ~50% (measured) | ~90% (matches non-PC baseline) |
| Memory overhead | ~5GB (SSM scratch, page-padded) | ~160MB (80 extra slots) |

## Reference: v1 ablation results

See `_pc_spec_decode_ablation_results.md` for the full experimental record
from the v1 approach, including all baselines and failure analysis.
