# PC + Eagle3 spec decode: three approaches to fixing the boundary problem

> Companion to `_pc_spec_decode_implementation_status.md`. Written 2026-04-09
> after the eager-mode AIME-25 result on the segregated-scratch implementation
> came back at 48.3% per-session vs the no-PC baseline's 90% — confirming that
> the bug is **not** the cudagraph regression we were chasing, but a deeper
> design hole in how we handle mamba block boundaries during spec verify.

## Recap of the actual root cause

vLLM's mamba PC contract: slot[b] in the regular ssm_state pool must contain
the post-state of position `(b+1)*B - 1` (the LAST position of block b) at
the moment the prefix-cache committer snapshots the block to the radix tree.
The committer snapshots when the block becomes "full", which happens whenever
the latest processed position crosses into block b+1.

In normal (non-spec) decode, every step processes 1 token. When the kernel
processes position `(b+1)*B - 1`, it writes the post-state to slot[b]
(the OLD canonical slot). When the next step processes position `(b+1)*B`,
it writes to slot[b+1] (the NEW canonical slot). The old slot stays frozen
as the boundary state. Clean.

In spec decode, each verify step processes K+1 candidate token positions
in one kernel launch. If those K+1 positions span a block boundary, the
old canonical slot must end up holding the boundary post-state, and the new
canonical slot must end up holding the latest accepted post-state past the
boundary. The kernel writes K+1 post-states to K+1 slots — but **which** K+1
slots, and how they get committed back to the canonical pool is exactly
the design surface where this gets hard.

## The three options

### Option 1: Disable spec decode at block boundaries (chosen)

**The idea.** For any verify step where the K+1 candidates would cross a
block boundary, force `num_accepted = 1` so we only accept token 0 and
commit its post-state to the right canonical slot. K candidates of work is
wasted (drafter compute thrown away) but correctness is preserved.

**Detection.** Per-request, in metadata:
```
unsafe = (block_idx_last_computed_token != block_idx_last_scheduled_token)
```
This catches both "case 1" (`num_computed_tokens` is at a block boundary so
the very first scheduled position is in a new block) and "case 2"
(some-but-not-all candidates cross into the next block).

**Per-boundary cost.** With block size B and num_speculative_tokens K, each
boundary costs K+1 consecutive forced steps (the unsafe window is
`num_computed_tokens ∈ [k*B - K, k*B]`). The fraction of decode steps with
spec disabled is approximately `(K+1) / B`. For K=4, B=512 that's about
**0.98%** — i.e., we keep ~99% of the spec-decode speedup.

**Implementation properties.**
- Pure tensor operations (no Python branches inside the captured forward),
  so it's cudagraph-compatible.
- Touches three files: metadata builder (expose `block_idx_first_scheduled_d`),
  layer (per-request `canonical_dst_slot`), runner (clamp `num_accepted=1`
  for unsafe requests after the rejection sampler).
- No new buffers, no kernel changes, no upstream coordination needed.
- Easy to reason about: one boolean per request, one branch in the commit.

**Downsides.**
- Loses ~1% of spec-decode benefit (acceptable for our use case).
- Wastes the drafter's K drafted tokens for unsafe steps (those drafts get
  thrown away — the drafter still ran them).
- Doesn't help upstream solve the "real" PC+spec problem.

### Option 2: Per-candidate slot routing (the "proper" upstream fix)

**The idea.** This is what `_pc_spec_decode_upstream_status.md` proposes. The
kernel already supports K+1 distinct destination slots per request via
`dst_state_batch_indices`. The metadata builder allocates K extra "scratch"
slots per running spec-active request. Each candidate token gets its own
slot. After the rejection sampler picks `N` accepted tokens, the worker
either copies `scratch[N-1]` -> `main_block_slot` or rotates pool indices to
swap pointers (zero-copy commit). At a block boundary, you'd have to map
candidate j to slot[block_b] for the candidates that fit in block b and to
slot[block_b+1] for those that don't.

**Why it's the "right" fix.** It preserves all spec-decode benefit (no
forced fallback at boundaries), exercises the kernel's
`IS_SPEC_DECODING` mode as designed, and gives you the GDN/SGLang
"shadow slot per draft" architecture for free. The kernel infrastructure for
this already exists in `selective_state_update` lines 156-270.

**Why nobody has shipped it (and why the existing attempts are riddled with
bugs).** Multiple parties have been chipping away at this for months. As of
2026-04-09:

- **vllm-project/vllm#33726** ([Model][Spec Decode] Nemotron-H MTP and Mamba
  Speculative Decoding Support, merged 2026-02-24 by @benchislett) — landed
  the line-114 stub that **explicitly disables spec decode in PC mode** with
  the comment "Speculative decoding not supported with prefix caching, so
  keep shape consistent with prefill buffer". The PR description says
  "choose PC for multi-turn, or spec-decode for throughput, not both". This
  was a deliberate "ship MVP, fix later" decision, not an oversight, by the
  developer with the most context.

- **vllm-project/vllm#26201** ([Tracking Issue]: Prefix Caching for Hybrid
  Models, by @bohnstingl, last activity 2026-03-06) — explicitly lists
  "Support mamba prefix caching and spec decode" as a TODO action item. No
  owner, no timeline, no PR in flight.

- **vllm-project/vllm#38898** ([Feature]: Mamba DS conv state layout / support
  spec decode with mamba_cache_mode=align, by @NickLucche, opened 2026-04-03)
  — same feature request from a different angle. No assignee, no PR.

- **vllm-project/vllm#37985** ([Mamba] Speculative Decoding Mamba) — DRAFT,
  stalled since 2026-03-30, 26 commits, doesn't touch `mamba_attn.py`. CI
  flagged a critical block_size issue for pure-Mamba draft models;
  reviewers noted "accepted draft tokens percentage was low." Currently
  abandoned.

- **vllm-project/vllm#39273** ([Bug]: NGram spec decode corrupts output on
  hybrid GDN (Qwen3.5), filed 2026-04-08) — reports the **exact same bug
  class** (state rollback on partial accept) but for GatedDeltaNet instead
  of Mamba2. The author traces the root cause to "ssm_state_indices block
  table lacking extra slots for speculative tokens" — literally the bug
  this option is supposed to fix. No fix in flight.

- Earlier SGLang attempt (referenced in `_pc_spec_decode_mamba_investigation.md`)
  — got partial-credit at 71.7% AIME-25 vs 95% baseline. Their stub handles
  the kernel call correctly but the post-verify slot promotion is broken
  for the boundary case. **Their bug is exactly the boundary case we're
  trying to handle**, and they hit it independently.

**The pattern.** Everyone who has attempted Option 2 has run into one or more
of:

1. **Cudagraph buffer shape mismatch.** The metadata builder allocates
   `state_indices_tensor_d` with `max_num_blocks` columns when PC is on,
   but the spec verify needs `K+1` columns. You have to either size for
   the max of both, allocate two buffers, or reshape per call (non-trivial
   under cudagraph).

2. **Slot ownership at the boundary.** When the K+1 candidates span a
   boundary, candidate 0 belongs to slot[block_b] but candidate K belongs
   to slot[block_b+1]. Computing this per-request, per-step, in a tensor
   op (so cudagraph captures it correctly) is fiddly. SGLang got this wrong
   and lost ~23pp accuracy.

3. **Post-verify promotion when the accept count crosses the boundary.**
   If `N` candidates are accepted and the boundary fell between candidates
   `j` and `j+1` for `j < N-1`, then BOTH slot[block_b] and slot[block_b+1]
   need to receive a state — slot[block_b] gets the post-state of
   candidate `j` (the boundary state of block b), and slot[block_b+1] gets
   the post-state of candidate `N-1` (the new running state). This is a
   two-way commit that requires per-request logic.

4. **Interaction with `init_token_idx` from the previous step's
   `num_accepted_tokens`.** The kernel reads its input state from
   `state_batch_indices[req][num_accepted - 1]` — i.e., the slot
   corresponding to the last accepted candidate of the previous step. If
   the previous step's accepted-slot landed in slot[block_b] and the
   current step starts in slot[block_b+1], the input source has to point
   at slot[block_b], not slot[block_b+1]. Off-by-one trap.

5. **Drafter advancement.** The Eagle3 drafter has its own state machine
   tracking which block it's on. Its block index can drift from the
   verifier's if the boundary handling is wrong. Several attempts have
   surfaced subtle drafter/verifier desync bugs.

6. **Mamba-cache-manager scratch slot lifecycle.** The "K extra scratch
   slots per running request" need to be allocated, freed, and recycled.
   Doing this without leaking and without contaminating the BlockPool's
   radix tree requires invasive changes to MambaCacheManager.

In short: Option 2 is **300+ LoC across 4 files plus tests** with high
cognitive load, and several months of attempts have not produced a working
PR. We could do it eventually but not on a one-day budget.

### Option 3: Roll back the whole branch and document that PC+Eagle3 doesn't work

Drop all our patches, set `enable_prefix_caching=False`, fall back to the
no-PC Eagle3 path which already works at 90% AIME-25. Add a launch-script
note documenting the limitation.

**Pros.** Zero risk of regression. Bulletproof correctness. The smallest
amount of code to maintain.

**Downsides.** No PC means tool-call workloads (like the AIME runs that
exercise multi-turn agentic loops) re-prefill the conversation prefix on
every turn, which costs many seconds per turn. The user explicitly cited
this as the reason PC matters more than spec decode for their workload —
empirically, PC > spec decode for agentic.

This is the safety net if Option 1 turns out to have a hidden bug we
didn't catch.

## Why Option 1 is the right choice for this project

Option 2 is the engineering-textbook answer. Option 1 is the answer that
ships. The constraint set we're working under:

- **One developer, narrow time budget.** Several teams have failed to ship
  Option 2 over months. We are not going to crack it in a day.
- **NemotronH is the only model that matters here.** Block size is fixed
  at 512, K is fixed at 4, so the unsafe-step rate is statically ~1%.
  For other models with different (B, K) the math may change but the
  approach generalizes.
- **Cudagraph mode is the priority.** Eager is too slow for production.
  Option 1 is implementable purely as tensor ops (boundary detection +
  masked gather + clamp), so it captures cleanly.
- **Correctness matters more than the marginal 1% throughput.** The
  user's workload is reasoning-heavy AIME-25 problems where one
  state-corruption event ruins the whole session. We give up ~1% to
  guarantee 0% corruption.
- **Self-contained.** No upstream coordination, no metadata builder
  rewrites, no MambaCacheManager surgery. Three files touched, all
  already on this branch.

If Option 1 ships and works, we still have the option of attempting
Option 2 later as an upstream PR (per the action items in
`_pc_spec_decode_upstream_status.md`). Option 1 is not in conflict with
Option 2 — it's a strict subset of the design space.

## Off-by-one analysis (the part that has to be right)

The single most common way Option 1 implementations break is by getting
the safety condition or the destination-block computation wrong. Spelling
out the math here so the implementation can be cross-checked.

### Conventions

- Positions are 0-indexed.
- `B` = mamba block size (512 in our config).
- `K` = num_speculative_tokens (4 in our config). Each verify step
  processes `K+1 = 5` candidate token positions.
- "Block b" contains positions `[b*B, b*B + B - 1]` (B positions per block).
- `block_of(p) = p // B`.
- "Post-state of position p" = the SSM state AFTER processing the token at
  position p. This is what gets stored in slot[block_of(p)].
- `n_done = num_computed_tokens` = number of tokens whose post-state has
  already been committed to the canonical pool. Equivalently: position
  `n_done` is the FIRST position that this verify step will process.

### Metadata fields (from `mamba_attn.py::_compute_prefix_caching_block_indices`)

- `block_idx_last_computed_token = max(0, cdiv(n_done, B) - 1)`
  - For `n_done = 0`: clamped to 0 (prefill case, not relevant for spec verify).
  - For `n_done = 1`: `cdiv(1, B) - 1 = 1 - 1 = 0`. ✓
  - For `n_done = B`: `cdiv(B, B) - 1 = 1 - 1 = 0`. (Last token of block 0.) ✓
  - For `n_done = B+1`: `cdiv(B+1, B) - 1 = 2 - 1 = 1`. (First token of block 1.) ✓

- `block_idx_first_scheduled_token = cdiv(n_done + 1, B) - 1`
  - For `n_done = 100`: `cdiv(101, 512) - 1 = 1 - 1 = 0`. (Position 100 in block 0.) ✓
  - For `n_done = 511`: `cdiv(512, 512) - 1 = 1 - 1 = 0`. (Position 511 in block 0.) ✓
  - For `n_done = 512`: `cdiv(513, 512) - 1 = 2 - 1 = 1`. (Position 512 in block 1.) ✓
  - For `n_done = 513`: `cdiv(514, 512) - 1 = 2 - 1 = 1`. ✓

- `block_idx_last_scheduled_token = cdiv(n_done + (K+1), B) - 1`
  - For decode with K+1 spec tokens, `seq_lens = n_done + K + 1`.
  - For `n_done = 100, K = 4`: `cdiv(105, 512) - 1 = 0`. ✓
  - For `n_done = 511, K = 4`: `cdiv(516, 512) - 1 = 1`. (Last candidate at 515 → block 1.) ✓
  - For `n_done = 513, K = 4`: `cdiv(518, 512) - 1 = 1`. ✓

### Safety condition

```
unsafe[req] = block_idx_last_computed_token[req] != block_idx_last_scheduled_token[req]
```

Equivalently (and this is how we'll check it): the OLD block (block of the
last committed position, i.e. position `n_done - 1`) is the same block as
the LAST candidate's block (position `n_done + K`). If they're equal, all
positions in `[n_done - 1, n_done + K]` fit inside one block — no
boundary crossed in or out.

Cross-check against the position table:
| n_done | last_computed | first_scheduled | last_scheduled | safe? | reason |
|--------|---------------|-----------------|----------------|-------|--------|
| 100    | 0             | 0               | 0              | yes   | all in block 0 |
| 506    | 0             | 0               | 0              | yes   | last_cand=510, still block 0 |
| 507    | 0             | 0               | 0              | yes   | last_cand=511, still block 0 |
| 508    | 0             | 0               | 1              | NO    | last_cand=512 in block 1 |
| 509    | 0             | 0               | 1              | NO    | last_cand=513 in block 1 |
| 510    | 0             | 0               | 1              | NO    | last_cand=514 in block 1 |
| 511    | 0             | 0               | 1              | NO    | last_cand=515 in block 1 |
| 512    | 0             | 1               | 1              | NO    | first_cand=512 in block 1 |
| 513    | 1             | 1               | 1              | yes   | all in block 1 |

Five consecutive unsafe steps per boundary (n_done in [508, 512]). The
unsafe rate is `(K+1) / B = 5 / 512 ≈ 0.98%`.

### Forced num_accepted = 1: where to commit

For an unsafe step, we force `num_accepted = 1`, meaning only token 0 (at
position `n_done`) is accepted. The post-state of token 0 must land in the
slot of `block_of(n_done)` = `block_idx_first_scheduled_token`.

Sanity check across the unsafe window for n_done in [508, 512]:
| n_done | token 0 position | block_of(token 0) | first_scheduled | match? |
|--------|------------------|-------------------|-----------------|--------|
| 508    | 508              | 0                 | 0               | yes    |
| 509    | 509              | 0                 | 0               | yes    |
| 510    | 510              | 0                 | 0               | yes    |
| 511    | 511 (LAST of b0) | 0                 | 0               | yes    |
| 512    | 512 (FIRST of b1)| 1                 | 1               | yes    |

So uniformly: **forced-step destination = state_indices[req][block_idx_first_scheduled[req]]**.

This is the same as the OLD canonical slot for "case 2" steps (508-511,
where the boundary crossing is on the OUTPUT side) and the NEW canonical
slot for "case 1" steps (512, where the boundary crossing is on the INPUT
side).

### Slot read for the input state

For an unsafe step, the input state to the kernel is whatever lives at
`state_indices[req][block_idx_last_computed_token[req]]` — i.e., the OLD
canonical slot. This is the post-state of position `n_done - 1`.

For case 2 (n_done in [508, 511]): position `n_done - 1` is somewhere in
block 0. slot[block 0] contains its post-state because the previous step
also ran (and was either safe or also forced to num_accepted=1 with the
destination block 0). ✓

For case 1 (n_done = 512): position `n_done - 1 = 511` is the LAST
position of block 0. slot[block 0] contains the boundary state of block 0
(written by the previous step at n_done = 511, which committed to block 0).
✓ This is exactly the boundary state the prefix-cache committer should
snapshot for block 0.

### Why every unsafe step in the window must commit

Within the K+1 unsafe steps for one boundary:
- Step 1 (n_done = 508): commit slot[block 0] = post-state of position 508.
- Step 2 (n_done = 509): commit slot[block 0] = post-state of position 509.
- Step 3 (n_done = 510): commit slot[block 0] = post-state of position 510.
- Step 4 (n_done = 511): commit slot[block 0] = post-state of position 511.
  **THIS IS THE BOUNDARY STATE.** It's now correct in slot[block 0].
- Step 5 (n_done = 512): commit slot[block 1] = post-state of position 512.
  slot[block 0] is left untouched and stays correct.

If we got the destination block wrong on step 5 (e.g., wrote to slot[block 0]
instead of slot[block 1]), we'd corrupt the boundary state. This is the
mistake the segregated-scratch implementation made — and the reason the
eager-mode AIME-25 fell from ~95% baseline to 48%.

### Edge case: very first decode after prefill

Prefill commits the post-state of `n_done - 1` to the appropriate slot.
For prefill of length L, the first decode has `n_done = L`. The check
`block_idx_last_computed != block_idx_last_scheduled` runs the same way:

- L = 200: last_comp = 0, last_sched = (200 + 4) // 512 = 0. SAFE. Use spec.
- L = 512: last_comp = 0, last_sched = (512 + 4) // 512 = 1. UNSAFE. Force =1.

For L = 512 the first decode is at the case-1 boundary; we force =1 and
commit slot[block 1] = post-state of position 512. The previous (prefill)
step left slot[block 0] holding the boundary state of block 0. ✓

### Edge case: what if num_computed_tokens = 0?

This is the prefill case. The metadata builder clamps
`block_idx_last_computed_token` to 0, but we won't enter the spec scratch
path on a prefill-only step (the gating check requires
`num_accepted_tokens is not None`, which is only set for spec-active
decode batches). So this edge case doesn't reach our code.

## Implementation skeleton (option 1)

Three changes:

### 1. Metadata: expose `block_idx_first_scheduled_token` (decode portion)

`vllm/v1/attention/backends/mamba_attn.py` already computes
`block_idx_first_scheduled_token` (the full tensor) inside
`_compute_prefix_caching_block_indices`. Today only the prefill slice
gets stored as `block_idx_first_scheduled_token_p`. We add the full tensor
to the metadata dataclass so the layer can access the decode portion via
the same `torch.split` pattern that's already used for
`block_idx_last_computed_token` and `block_idx_last_scheduled_token`.

### 2. Layer (`mamba_mixer2.py`): per-request safety mask

Inside `conv_ssm_forward` decode branch, in the spec-scratch path:

```python
# Both vectors come from metadata, shape (num_decodes,), int32
unsafe_d = block_idx_last_computed_token_d != block_idx_last_scheduled_token_d  # bool

# Canonical IN slot — same for safe & unsafe (always read from OLD slot)
canonical_in_slot_int32 = state_indices_tensor_d.gather(
    1, block_idx_last_computed_token_d.unsqueeze(1)
).squeeze(1)

# Canonical DST slot — different for safe vs unsafe
#   safe: write back to the same OLD slot (no boundary crossed, slot stays "live")
#   unsafe: write token-0's post-state to slot[block_idx_first_scheduled]
canonical_dst_slot_int32 = state_indices_tensor_d.gather(
    1,
    torch.where(
        unsafe_d, block_idx_first_scheduled_token_d, block_idx_last_computed_token_d
    ).unsqueeze(1),
).squeeze(1)

# Pre-copy: scratch[base+0] = ssm_state[canonical_in_slot]  (unchanged)

# state_indices_tensor_d_output: for safe requests, all K+1 scratch slots get
# written. For unsafe requests, only slot 0 — the rest are NULL_BLOCK_ID so
# the kernel skips them (we don't want speculative junk lying around).
output_indices = pre_state_indices_output[:num_decodes]  # (num_decodes, K+1) safe
output_indices = torch.where(
    unsafe_d.unsqueeze(1),
    pre_state_indices_unsafe_output[:num_decodes],  # (num_decodes, K+1): [base+0, NULL, ..., NULL]
    output_indices,
)
state_indices_tensor_d_output = output_indices

# Stash for commit (canonical_dst_slot, scratch_slot_base)
self._spec_scratch_pending = (canonical_dst_slot_long, scratch_slot_base_long)
```

`pre_state_indices_unsafe_output` is a NEW pre-allocated buffer with the same
shape as `pre_state_indices_output_int32` but with NULL_BLOCK_ID (= 0) in
columns 1..K. It's a constant computed at startup, allocated alongside the
existing pre-allocated index buffers.

### 3. Runner (`gpu_model_runner.py`): clamp num_accepted=1 for unsafe requests

In `_update_states_after_model_execute`, BEFORE the commit hook loop:

```python
# Clamp num_accepted_tokens to 1 for boundary-crossing requests so the
# commit hook (which uses scratch_base + num_accepted - 1) reads the
# token-0 post-state, AND so the runner emits exactly 1 output token for
# these requests.
unsafe_d = (block_idx_last_computed_token_d != block_idx_last_scheduled_token_d)
num_accepted_tokens_gpu_decode = num_accepted_tokens_gpu[:num_decodes]
torch.where(unsafe_d, torch.ones_like(num_accepted_tokens_gpu_decode),
            num_accepted_tokens_gpu_decode,
            out=num_accepted_tokens_gpu_decode)
```

Then call commit as before.

### Startup warning log

In the engine init (or in the layer init when `_spec_scratch_enabled`
becomes true), log:

```
WARNING: PC + spec decode enabled with mamba_block_size=B and
num_speculative_tokens=K. ~Y% of decode steps near block boundaries will
fall back to greedy decoding (single-token, no spec) for correctness.
This is approximate; actual rate depends on prompt+generation lengths.
```

Where `Y = 100 * (K+1) / B`. For our config, Y ≈ 0.98%.

## Risks and unknowns for option 1

Things I want to verify experimentally before declaring success:

1. **The `torch.where` mask trick on `state_indices_tensor_d_output`** — does
   the kernel correctly skip writes when the dst slot is NULL_BLOCK_ID = 0?
   YES per `mamba_ssm.py:260` (`if token_dst_idx != null_block_id:`). But
   I want to confirm the masking happens at the right granularity and
   doesn't poison the next iteration.

2. **The runner's `num_accepted_tokens` is read by other code paths** —
   the next-token sampler, the stop-token check, the output writer. Does
   clamping to 1 in the persistent buffer break any of those? Need to
   trace and verify.

3. **The drafter's view of n_done** — Eagle3's drafter advances by N each
   step. If we force N=1 for unsafe steps but the drafter thinks it
   produced K drafts, is there a desync? The drafter's drafts are
   discarded when not accepted, so the next step's drafter input is
   `n_done + 1` regardless. Should be fine but worth checking.

4. **Cudagraph interaction** — the per-request `torch.where` and
   `gather` operations all need to capture cleanly. They use only fixed
   shapes (sliced from pre-allocated buffers up to num_decodes), so this
   should work, but we'll know for sure only after a full AIME-25 run.

5. **The first decode after a prefill that ends exactly on a boundary**
   (n_done = k*B for k > 0) — need to verify that the case-1 forced step
   correctly populates slot[block k] given that slot[block k] was never
   touched before this step.

## What we'll measure to decide

Run AIME-25 in cudagraph mode with this fix and compare to
`baseline_pc` (95% per-session) and `eagle3_no_pc` (90% per-session).
Targets:
- Per-session accuracy ≥ 88%. (Within 2pp of no-PC Eagle3 baseline.)
- Spec decode acceptance rate matches no-PC Eagle3 acceptance rate (the
  ~1% boundary fallback should not move the needle on the average).
- No "stop=2" / "no_answer" failure modes from state corruption (the
  signature of the current bug).
- Throughput within 5% of cudagraph no-PC Eagle3.

If we hit those, this fix ships. If not, we fall back to Option 3
(rollback) and document.
