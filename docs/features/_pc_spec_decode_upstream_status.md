# PC + Eagle3 + NemotronH: Upstream status, the actual fix path, action items

> Companion document to `_pc_spec_decode_mamba_investigation.md`. That doc covers
> the empirical bug hunt (the 4-bug chain in mamba_attn.py) and the SGLang
> reference. **This doc covers what we now know about the upstream situation,
> the actual minimum fix, and the GitHub issues / PRs / contributors to engage.**
>
> Written 2026-04-09 after a deep code-reading session of vllm v0.19+ tip-of-main
> and a parallel sweep of vllm-project/vllm GitHub issues.

## TL;DR

1. **The kernel-level architecture for shadow slots already exists in vLLM.** The
   Triton `selective_state_update` kernel at
   `vllm/model_executor/layers/mamba/ops/mamba_ssm.py:130-270` has a fully
   implemented `IS_SPEC_DECODING` mode that takes K+1 source slot indices, K+1
   destination slot indices, and `num_accepted_tokens`, and writes one shadow
   per candidate token in a single launch. **It's exactly the GDN-style "one
   slot per draft token" architecture from SGLang's blog -- already in vLLM,
   already shipped, already wired up to mamba2_mixer.py.**

2. **The metadata layer stubs it out when prefix caching is enabled.** See
   `vllm/v1/attention/backends/mamba_attn.py:114`:
   ```python
   if self.vllm_config.cache_config.mamba_cache_mode == "all":
       max_num_blocks = cdiv(...)
       # Speculative decoding not supported with prefix caching,
       # so keep shape consistent with prefill buffer
       # TODO: reduce this size as needed for decode-only cudagraph capture
       self.state_indices_tensor_d: torch.Tensor = torch.empty(
           (self.decode_cudagraph_max_bs, max_num_blocks),  # WRONG SHAPE for spec
           dtype=torch.int32, device=device,
       )
   ```
   vs the no-PC path on line 135:
   ```python
   else:
       self.state_indices_tensor_d = torch.empty(
           (self.decode_cudagraph_max_bs, 1 + self.num_spec_tokens),  # K+1 slots
           ...
       )
   ```
   The PC path explicitly bails. The comment is load-bearing.

3. **The fix is metadata plumbing only, not a kernel rewrite.** The kernel side
   is done. We just need to (a) allocate K+1 slots when PC + spec is enabled,
   (b) populate `state_indices_tensor_d` correctly, (c) add a post-verify
   "promote shadow[N-1] to live" step. Estimated 300 LoC across 3-4 files,
   1-2 person-weeks for someone who already knows the code.

4. **Nobody has a PR in flight for this.** Multiple tracking issues acknowledge
   it as TODO; no public roadmap; no assignee; no ETA. We'd be the first to
   ship the fix.

## What the kernel actually does (the surprise)

Read `vllm/model_executor/layers/mamba/ops/mamba_ssm.py` lines 50-270. The
`_selective_state_update_kernel` Triton kernel has two key constexpr flags:
`HAS_STATE_BATCH_INDICES` and `IS_SPEC_DECODING`. When both are true:

```python
# Line 156-160:
if IS_SPEC_DECODING:
    num_accepted = tl.load(num_accepted_tokens_ptr + pid_b).to(tl.int64)
    init_token_idx = tl.maximum(num_accepted - 1, 0)
else:
    init_token_idx = 0

# Line 172-176: read source state from state_batch_indices[pid_b, init_token_idx]
state_batch_indices_ptr += (
    pid_b * stride_state_indices_batch + init_token_idx * stride_state_indices_T
)
state_batch_idx = tl.load(state_batch_indices_ptr).to(tl.int64)
```

The kernel reads the SOURCE state from the slot indexed by `(pid_b,
init_token_idx)`, where `init_token_idx = num_accepted - 1`. This is
**already the "promote previous accepted slot to current input"** logic.

Then in the loop over `seq_len` candidate tokens (line 216):
```python
for i_t in range(seq_len):
    # ... compute new state via mamba recurrence ...
    if IS_SPEC_DECODING:
        dst_idx_ptr = dst_state_batch_indices_ptr + i_t * stride_dst_state_indices_T
        token_dst_idx = tl.load(dst_idx_ptr).to(tl.int64)
        if token_dst_idx != null_block_id:
            tl.store(token_dst_ptrs, state.to(...), mask=mask)
```

For each candidate token at position `i_t`, the kernel writes the new state to
`dst_state_batch_indices[pid_b, i_t]` -- a **different slot per token**. This
is the shadow-slot dance, fully implemented.

The mixer at `vllm/model_executor/layers/mamba/mamba_mixer2.py:880` already
calls this kernel correctly:

```python
selective_state_update(
    ssm_state,
    hidden_states_d, dt_d, A_d, B_d, C_d, D_d,
    state_batch_indices=state_indices_tensor_d_input,    # K+1 source slots
    dst_state_batch_indices=state_indices_tensor_d_output,  # K+1 dest slots
    num_accepted_tokens=num_accepted_tokens,
    ...
)
```

And the metadata builder at `mamba_attn.py:135` allocates the right shape
**when PC is OFF**:
```python
self.state_indices_tensor_d = torch.empty(
    (self.decode_cudagraph_max_bs, 1 + self.num_spec_tokens),
    ...
)
```

**Everything is done except the PC + spec interaction.** When PC is on, the
metadata builder takes a different branch (line 109-124) that allocates the
buffer for the BLOCK TABLE (`max_num_blocks` columns) instead of the SPEC SLOT
TABLE (`1 + num_spec_tokens` columns). The mamba2_mixer's gather logic at line
822-828 then collapses to one slot per request, and the kernel's shadow-slot
machinery never gets exercised.

## What changes to make

This is the surgical fix path, intentionally minimal:

### Change 1: `mamba_attn.py:117` -- buffer sizing

When PC AND spec are both enabled, the cudagraph buffer must hold both kinds of
indices. The simplest correct fix is two separate buffers (one for the block
table, one for the K+1 spec slot table). Less surgical: keep one buffer and
size it to `max(max_num_blocks, 1 + num_spec_tokens)`.

```python
if self.vllm_config.cache_config.mamba_cache_mode == "all":
    max_num_blocks = cdiv(
        self.vllm_config.model_config.max_model_len,
        self.kv_cache_spec.block_size,
    )
    second_dim = max_num_blocks
    if self.use_spec_decode:
        second_dim = max(second_dim, 1 + self.num_spec_tokens)
    self.state_indices_tensor_d: torch.Tensor = torch.empty(
        (self.decode_cudagraph_max_bs, second_dim),
        dtype=torch.int32, device=device,
    )
```

This unblocks the cudagraph buffer shape mismatch (Bug #1 from the earlier
investigation log).

### Change 2: `mamba_attn.py` build() -- populate K+1 slots from a scratch pool

The current PC path computes `state_indices_tensor` as the block table (one
slot per block per request). When spec is also on, you need K+1 slot ids per
request, drawn from a **separate per-request shadow allocation** -- these are
NOT the prefix-cache block slots, they're transient shadows that live for one
verify step.

The cleanest provisioning approach: ask the existing `MambaCacheManager` for K
extra "scratch" slots per spec-active request at request admission time. These
slots are tagged as transient (not part of any block table, never snapshotted
into the radix tree).

The metadata builder then assembles:
```
state_indices_tensor_d[req] = [main_block_slot, scratch_0, scratch_1, ..., scratch_K-1]
```

Where `main_block_slot` comes from the existing prefix-cache block table (just
like today), and the K scratch slots are new.

### Change 3: `mamba_mixer2.py:822-828` -- bypass gather when spec+PC

```python
if has_decode:
    if is_mamba_cache_all:
        if not self.use_spec_decode:
            # existing path: gather single slot per request from block table
            state_indices_tensor_d_input = state_indices_tensor_d.gather(
                1, block_idx_last_computed_token_d.unsqueeze(1)
            ).squeeze(1)
            state_indices_tensor_d_output = state_indices_tensor_d.gather(
                1, block_idx_last_scheduled_token_d.unsqueeze(1)
            ).squeeze(1)
        else:
            # new path: spec verify uses K+1 slots, no gather
            state_indices_tensor_d_input = state_indices_tensor_d
            state_indices_tensor_d_output = state_indices_tensor_d
```

### Change 4: post-verify promote step (gpu_model_runner.py or eagle.py)

After the spec verifier returns N accepted tokens per request, copy
`scratch_{N-1}` -> `main_block_slot` for that request. Conceptually:

```python
mamba_state[main_block_slot] = mamba_state[scratch_slots[N-1]]
```

vLLM has a slightly cleverer approach available -- instead of copying, rotate
the pool indices: the request's `main_block_slot` pointer is updated to point
to `scratch_{N-1}` for the next forward, and the old main slot becomes
`scratch_0` for the next verify. Zero copies, just pointer swaps in the cache
manager. This logic should live in either `vllm/v1/spec_decode/eagle.py` or
`vllm/v1/worker/gpu_model_runner.py` post-forward cleanup.

### Estimated effort

| Change | Files | LoC | Risk |
|---|---|---|---|
| Cudagraph buffer fix | `mamba_attn.py` | ~15 | Low |
| K+1 scratch allocation + metadata population | `mamba_attn.py`, possibly `mamba_cache.py` | ~80 | Medium |
| Mamba2Mixer slot routing for PC+spec | `mamba_mixer2.py` | ~30 | Low |
| Post-verify promote/swap | `gpu_model_runner.py` | ~30 | Medium |
| Tests + validation | new test file | ~150 | -- |
| **Total** | **4 files** | **~300** | -- |

## Upstream GitHub state (as of 2026-04-09)

### Tracking issues that already cover this

- **vllm-project/vllm#26201** "[Tracking Issue]: Prefix Caching for Hybrid Models"
  -- author @bohnstingl, last activity 2026-03-06. **Explicitly lists "Support
  mamba prefix caching and spec decode" as a TODO action item.** No owner, no
  timeline.

- **vllm-project/vllm#17140** "[RFC]: Native support for Mamba, SSM, and hybrid
  transformer models in vLLM V1" -- author @tlrmchlsmth. Master tracker; "Spec
  decode" still unchecked.

- **vllm-project/vllm#38898** "[Feature]: Mamba DS conv state layout / support
  spec decode with mamba_cache_mode=align" -- author @NickLucche, opened
  2026-04-03, no PR in flight, no assignee. The Mamba2-side feature request.

### The PR that landed the line-114 stub

- **vllm-project/vllm#33726** "[Model][Spec Decode] Nemotron-H MTP and Mamba
  Speculative Decoding Support" -- **MERGED 2026-02-24 by @benchislett**.
  Introduced the `state_indices_tensor_p` / `state_indices_tensor_d` split and
  the line-114 comment. The PR description explicitly says:
  > "choose PC for multi-turn, or spec-decode for throughput, not both"

  This was a deliberate "ship MVP, fix later" decision, not an oversight.

### Adjacent PRs worth knowing

- **vllm-project/vllm#37985** "[Mamba] Speculative Decoding Mamba" -- **DRAFT,
  stalled since 2026-03-30**. 26 commits, doesn't touch `mamba_attn.py`.
  CI flagged a critical block_size issue for pure-Mamba draft models;
  reviewers noted "accepted draft tokens percentage was low." Currently
  abandoned. **Not the fix path we want.**

- **vllm-project/vllm#33705** "[Hybrid] Enable spec decoding in mamba cache
  align mode" -- merged 2026-02-13. Re-enables spec decode in `align` cache
  mode (the older mamba caching path). Test `test_mamba_prefix_cache.py`
  passes for align mode but not for the `all` mode where the line-114 stub
  lives.

- **vllm-project/vllm#38556** "[Bugfix][Async] Fix async spec decoding with
  hybrid models" -- merged 2026-03-31. Fixes off-by-one in backup-token
  lookup and a GPU->CPU async stale-value bug in mamba hidden-state condense.

- **vllm-project/vllm#39273** "[Bug]: NGram spec decode corrupts output on
  hybrid GDN (Qwen3.5)" -- **FILED 2026-04-08, today**. Reports the **exact
  same bug class** (state rollback on partial accept) but for GDN
  (GatedDeltaNet) instead of Mamba2. Reproduces with both
  `prefix_caching=True` and `False`, suggesting the GDN path has a more
  fundamental shadow-slot bug. The author traces the root cause to
  `ssm_state_indices` block table lacking extra slots for speculative tokens
  -- **literally the bug we documented**. No fix in flight. Watch this one.

### Key contributors to engage

Based on git log of `vllm/v1/attention/backends/mamba_attn.py` and
`vllm/model_executor/layers/mamba/mamba_mixer2.py`, the most active and
relevant maintainers:

- **@benchislett (Benjamin Chislett)** -- **TAG FIRST**. Authored #33726 (the
  PR that landed the stub) and #35447 (NemotronH MTP + chunked prefill fix).
  Also POC for spec-decode tracking issue #28947. Has the most context on
  this exact code path and intentionally documented the limitation.
- **@tdoublep (Thomas Parnell)** -- hybrid/V1 PC refactors, chunk alignment,
  piecewise CUDA graphs.
- **@NickLucche** -- filed #38898 for the Mamba2 align mode; engaged on
  Mamba2-side feature work.
- **@tlrmchlsmth (Tyler Michael Smith)** -- author of RFC #17140; the
  hybrid-models RFC owner.
- **@Ailurus1 (Asaf Joseph Gardin)** -- Mamba1 APC, metadata consolidation.
- **@bohnstingl (Thomas Ortner)** -- author of tracking issue #26201.
- Honorable mention: @cyang49 (Chih-Chieh Yang), @tomeras91.

## Action items (for hav4ik to execute later)

These are the concrete next steps to upstream the fix. **Order matters** --
opening a PR before commenting on the tracking issues will surprise people; do
the social work first.

### Step 1: Comment on existing tracking issues (do this first)

1. **Comment on vllm-project/vllm#26201** with:
   - Concrete use case: NemotronH FP8 + Eagle3 + PC, 23pp accuracy gap
     measured on AIME-25
   - Link to this fork's `nemotron3-eagle3-support` branch as evidence we're
     already deep in the code
   - Offer to PR the fix
   - Tag @benchislett directly

2. **Comment on vllm-project/vllm#38898** with:
   - The finding that the kernel-level support **already exists** in
     `selective_state_update IS_SPEC_DECODING` mode
   - The minimum fix is the metadata stub at `mamba_attn.py:114`, not a
     kernel rewrite
   - Reference to this doc for the surgical incision points

3. **Optional: comment on vllm-project/vllm#39273** noting that the same bug
   class affects Mamba2 with PC+spec, and that the fix path is metadata-side
   for both architectures (different metadata files but similar structure).

### Step 2: Open a PR (after the tracking-issue comments land)

Title: `[Mamba2][Spec Decode] Enable shadow-slot rollback for prefix caching +
speculative decoding (NemotronH)`

PR description should:
- Reference #26201 and #38898
- Cite the kernel-level IS_SPEC_DECODING infrastructure that already exists
- Point at the line-114 stub and the comment from #33726 that documented the
  limitation
- Show benchmark numbers: accuracy unchanged from baseline, throughput up by
  some measurable factor
- Note that this only fixes Mamba2 (not GDN -- GDN's #39273 is separate)
- Tag @benchislett, @NickLucche, @tdoublep as reviewers

### Step 3: Don't fragment

- **Do NOT open a third tracking issue.** #26201 and #38898 already cover
  this; opening another one will fragment discussion and irritate
  maintainers.
- **Do NOT do a giant PR** that bundles fixes for both Mamba2 and GDN. Ship
  the Mamba2 fix first; the GDN fix can be a follow-up.

## Why this approach is right

The reason this is "300 LoC" instead of "research project" is that vLLM was
already designed to support the feature. The kernel knows how to handle K+1
slots. The mixer knows how to wire them. The metadata layer knows the shape
when PC is off. **The only missing wire is the conditional that says "when PC
is on AND spec is on, allocate the K+1 scratch slots and use them instead of
collapsing to one slot per block."** Read #33726's PR description -- the
authors knew exactly what they were leaving on the table.

The empirical confirmation from our investigation:
- vLLM Eagle3 + Nemotron WITHOUT PC: 95% AIME-25 (works)
- vLLM Eagle3 + Nemotron WITH PC: blocked by 4-bug chain (this doc fixes them)
- SGLang baseline + Nemotron WITH PC: 95% AIME-25 (PC works alone)
- SGLang Eagle3 + Nemotron WITH PC: 71.7% AIME-25 (their stub also broken,
  different code path)

The conclusion: **PC works, spec works, the two together is the only thing
that's broken, and it's broken in metadata not in kernels.**

## See also

- `_pc_spec_decode_mamba_investigation.md` -- the original empirical bug hunt,
  including the 4-bug chain found in earlier investigation work
- `_partial_mamba_attn_fix.patch` -- partial patch for bugs #1 and #2 of the
  4-bug chain (state_indices_tensor_d shape mismatch and block_idx slicing)
- `_sglang_nemotron_h_eagle3_patch.patch`,
  `_sglang_server_args_extra_buffer_patch.patch` -- the SGLang reference
  patches that mostly work for the no-PC case
- `spec_decode_eagle3_nemotron_h.md` -- the user-facing docs for the working
  vLLM Eagle3 (no PC) path

---

**Document maintainer note:** if you're picking this up later, the fastest way
to re-orient is to read this doc top-to-bottom, then read `mamba_attn.py:117`
and `mamba_ssm.py:130-270` side by side. The mismatch between the two is the
entire fix.
