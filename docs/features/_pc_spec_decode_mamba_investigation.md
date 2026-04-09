# Investigation: prefix caching + speculative decoding for NemotronH (mamba) on vLLM

> **Status (2026-04-08):** Not solvable with metadata-level patches. The bug
> chain bottoms out in the mamba SSM Triton kernel, which uses an in-place
> state-update design that is **architecturally incompatible** with
> speculative decoding's "speculate, then roll back rejected suffix"
> semantics. SGLang upstream has solved this by **redesigning** the mamba
> cache as per-candidate-token **state sandboxes** (cache isolation), but
> vLLM has not yet ported that design.
>
> **For users today:** pick prefix caching, drop spec decode (see the empirical
> comparison in [`spec_decode_eagle3_nemotron_h.md`](spec_decode_eagle3_nemotron_h.md)).
>
> **For upstream contributors:** the partial patch
> [`_partial_mamba_attn_fix.patch`](_partial_mamba_attn_fix.patch) unblocks
> bugs #1 and #2 (the metadata-level mismatches) so you can reproduce bug
> #3 (the SSM kernel rollback) reliably and start porting SGLang's
> cache-isolation design. The patch alone is **not** a fix.

This document is the full investigation log from a 2026-04-08 session
trying to make `--enable-prefix-caching --mamba-cache-mode all
--speculative-config '{"model":"chankhavu/c2.eagle3-test","method":"eagle3"}'`
work for `chankhavu/c2-softcpy-fp8` (FP8 NemotronH 30B-A3B) on a single
RTX PRO 6000 Blackwell. It exists so the next person looking at this
problem doesn't have to rediscover the bug chain from scratch.

---

## Tl;dr

| What works today | What does not |
|---|---|
| Eagle3 + NemotronH **without** prefix caching | Eagle3 + NemotronH **with** prefix caching |
| MTP + NemotronH **without** prefix caching | MTP + NemotronH **with** prefix caching |
| n-gram speculator + NemotronH **without** prefix caching | n-gram speculator + NemotronH **with** prefix caching |
| Prefix caching alone (`--mamba-cache-mode all`) on NemotronH | Any combination of `mamba_cache_mode != none` + `num_speculative_tokens > 0` |
| FP8 NemotronH with prefix caching alone | FP8 NemotronH with prefix caching + spec decode (extra FlashInfer FP8 prefill kernel bug compounds) |
| BF16 NemotronH with prefix caching alone | BF16 NemotronH with prefix caching + spec decode (bypasses FlashInfer bug, exposes bugs 1-3 below) |

The "without" rows confirm the bug is **specifically** in the `spec decode +
mamba prefix caching` interaction, not in either feature alone.

---

## The bug chain (4 layers, in order of where they fire)

### Layer 1: `mamba_attn.py` cudagraph capture buffer mismatch (init time)

`vllm/v1/attention/backends/mamba_attn.py:117` allocates the
`state_indices_tensor_d` persistent buffer used during cudagraph capture
with shape:

```python
self.state_indices_tensor_d = torch.empty(
    (self.decode_cudagraph_max_bs, max_num_blocks),  # max_num_blocks = cdiv(max_model_len, block_size)
    dtype=torch.int32,
    device=device,
)
```

The code comment immediately above this allocation literally says (as of
vLLM tip-of-main on 2026-04-08):

```python
# Speculative decoding not supported with prefix caching,
# so keep shape consistent with prefill buffer
# TODO: reduce this size as needed for decode-only cudagraph capture
```

When spec decode is enabled, the runner inflates `block_table_tensor`
beyond `max_num_blocks` to reserve slots for the speculative tokens.
Specifically, the per-call shape becomes
`(num_decodes, max_num_blocks + num_spec_tokens)`. The downstream
`_update_metadata_for_cudagraph_capture` then does:

```python
self.state_indices_tensor_d[: metadata.num_decodes].copy_(
    state_indices_tensor_d, non_blocking=True
)
```

which fails with an exact size mismatch — the LHS is `max_num_blocks`
wide, the RHS is `max_num_blocks + num_spec_tokens` wide. **Crash signature:**

```
RuntimeError: The size of tensor a (26) must match the size of tensor b (29) at non-singleton dimension 1
```

(26 = `cdiv(32768, 4288/single-mamba-block-size)`, 29 = 26 + 3 spec tokens.)

This bug only fires when **both** spec decode is active **and**
`cudagraph_mode.has_full_cudagraphs() is True` (i.e. `FULL` or
`FULL_AND_PIECEWISE`). With `PIECEWISE` cudagraphs, the buggy branch is
skipped — which is why FlashInfer attention works in the "PIECEWISE
auto-fallback for spec decode" path that vLLM ships.

### Layer 2: `mamba_attn.py` block_idx slice convention mismatch (init time)

Right after layer 1, in the same `_update_metadata_for_cudagraph_capture`
function, the `block_idx_last_scheduled_token` and
`block_idx_last_computed_token` tensors are sliced to
`metadata.num_decode_tokens`:

```python
block_idx_last_scheduled_token = self.block_idx_last_scheduled_token[
    : metadata.num_decode_tokens
]
```

But the downstream consumer in
`vllm/model_executor/layers/mamba/mamba_mixer2.py:632` splits these
tensors by `[num_decodes, num_prefills]` (a request-count-based split):

```python
block_idx_last_computed_token_d, block_idx_last_computed_token_p = (
    torch.split(
        attn_metadata.block_idx_last_computed_token,
        [num_decodes, num_prefills],
        dim=0,
    )
)
```

**Without spec decode**, `num_decode_tokens == num_decodes` (one decoded
token per request) so the slice and the split agree. **With spec decode**,
`num_decode_tokens > num_decodes` because each decode request expands to
`1 + num_spec_tokens` decoded tokens. The split then fails:

```
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 16 (input tensor's size at dimension 0), but got split_sizes=[14, 0]
```

(16 = the slice width = `num_decode_tokens`; 14 = `num_decodes`; 0 =
`num_prefills` — the bug is using `num_decode_tokens` where `num_reqs`
was meant.)

### Layer 3: `_selective_scan_update_kernel` runtime illegal memory access (the actual blocker)

This is the bug layer that **cannot be fixed at the metadata level**.

`vllm/model_executor/layers/mamba/ops/mamba_ssm.py
_selective_scan_update_kernel` is the Triton kernel that performs the
mamba SSM state update during decode. It is called from
`mamba_mixer2.py:880` like:

```python
selective_state_update(
    ssm_state,
    hidden_states_d,
    dt_d, A_d, B_d, C_d, D_d,
    z=None, dt_bias=dt_bias, dt_softplus=True,
    state_batch_indices=state_indices_tensor_d_input,    # which slot to read from
    dst_state_batch_indices=state_indices_tensor_d_output, # which slot to write to
    out=preallocated_ssm_out_d.view(num_decode_tokens, -1, self.head_dim),
    num_accepted_tokens=num_accepted_tokens,
    cu_seqlens=query_start_loc_d,
    is_blackwell=self.is_blackwell,
    ...
)
```

The intent of this signature is *clearly* spec-decode-aware:
`num_accepted_tokens` and `cu_seqlens` are passed in so the kernel
*should* be able to figure out which tokens were accepted and which to
roll back. The kernel **claims** to support spec decode in its argument
list.

In practice, when invoked with a real spec-decode batch on top of mamba
prefix caching (`mamba_cache_mode=all`), it raises:

```
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
  File ".../vllm/model_executor/layers/mamba/ops/mamba_ssm.py", line 477,
  in selective_state_update
    _selective_scan_update_kernel[grid](
```

This is the actual root cause. The kernel takes spec-decode-aware
arguments but the implementation does not correctly handle the case
where:

1. Each request has *multiple* SSM cache slots (because prefix caching
   is on and we keep state at every block boundary)
2. There are `1 + num_spec_tokens` "candidate" tokens per request that
   need to be folded into the SSM forward
3. After verification, only the accepted prefix's state should be
   committed to the cache; the rejected tail's state must be discarded
4. The "discard" step is impossible with the current in-place update
   design — you cannot un-do an SSM scan once it's been folded into a
   state slot

**This is the SGLang quote from the
[PyTorch blog post](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/),
worth reading verbatim:**

> However, the in-place state updates preclude the ability to roll back
> cache entries for partial sequence matches, which complicates the
> implementation of many widely adopted features, such as prefix caching
> and speculative decoding.

vLLM's mamba SSM kernel has the same in-place-update design and the same
fundamental incompatibility.

### Layer 4 (FP8 verifier only): FlashInfer FP8 paged prefill kernel illegal memory access

When the verifier is FP8-quantized (e.g. `chankhavu/c2-softcpy-fp8`),
vLLM auto-selects the FlashInfer attention backend for the verifier's
attention layers, which uses the FP8 paged-KV prefill kernel:

```
File "/root/.cache/flashinfer/0.6.7/120f/generated/batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_head_dim_qk_128_head_dim_vo_128_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False/batch_prefill.cu", line 330, in BatchPrefillWithPagedKVCacheRun(...)::<lambda>
RuntimeError: Check failed: (status == cudaSuccess) is false: BatchPrefillWithPagedKVCache failed with error an illegal memory access was encountered
```

This kernel has a separate bug when called with a partial-prefix prefill
(prefix caching active) AND a spec-decode-inflated batch. It fires *before*
bugs 1-3 above because it's the verifier's first attention layer and runs
during the very first decode step.

This bug is **bypassed** by switching the verifier to BF16 (e.g.
`nvidia/Nemotron-Cascade-2-30B-A3B`), which makes vLLM auto-pick the
`FLASH_ATTN` backend instead. But that just exposes bugs 1-3 in the
mamba layers below it.

---

## SGLang's reference solution: cache isolation, not rollback

[Alibaba's writeup of the SGLang implementation](https://www.alibabacloud.com/blog/hybrid-model-support-%7C-sglangs-support-scheme-for-hybrid-architecture-models-like-mamba-transformer_602857)
is the cleanest explanation. The key passages:

> SGLang proposes a new cache-isolation–based architecture: it allocates
> **an independent Mamba cache slot for each candidate token**, creating
> **physically isolated state sandboxes**.

> Using the three-step candidate sequence "the → air → streets" as an
> example, the system maintains progressive state evolution in three
> separate cache slots — slot 1 stores the base state after being
> updated by "the", slot 2 injects "air" on top of that, and slot 3
> inherits the previous state and adds "streets".

The high-level design:

1. **Don't try to roll back.** It is impossible with in-place SSM state
   updates.
2. **Fork instead.** When the verifier processes `1 + num_spec_tokens`
   tokens for a request, allocate `1 + num_spec_tokens` separate SSM
   cache slots (the "sandboxes") and write each speculative step into
   its own slot.
3. **Commit on verification.** After the sampler decides which prefix
   was accepted, the corresponding sandbox slot becomes the new
   committed state for that request. The rejected sandbox slots are
   freed back to the pool.
4. **Eagle-Tree compatibility:** for tree-shaped speculation, each node
   in the speculation tree records its parent slot, and sandbox slots
   are organized as a tree mirroring the speculation tree:

> Eagle-Tree dynamically constructs attention masks to support multi-path
> parallel verification, whereas SSMs do not explicitly maintain
> attention relationships between tokens... the system explicitly
> records, for each candidate token, its parent node in the speculation
> tree.

> This design "preserves Eagle-Tree's multi-path exploration capability
> while aligning it with SSM state evolution."

### Memory cost of the SGLang approach

The naive worry is "this uses too much memory". Working it out for
`chankhavu/c2-softcpy-fp8` (NemotronH 30B-A3B, 23 mamba layers,
`mamba_num_heads=64`, `mamba_head_dim=64`, `ssm_state_size=128`):

- One SSM state slot per layer per request: `64 × 64 × 128 = 524288`
  fp32 elements ≈ **2 MiB**
- Across 23 mamba layers: **≈ 47 MiB per request per cache slot**
- With `num_spec_tokens=3`: **4 slots per active spec-decode request ≈
  188 MiB extra per request**
- For `max_num_seqs=64`: up to **64 × 188 MiB ≈ 12 GiB extra mamba state
  cache** beyond the no-spec baseline

12 GiB on a 96 GiB Blackwell card is meaningful but not prohibitive —
you'd just lower `--max-num-seqs` if you can't afford it. SGLang's
"page size 1" you may have heard about is a separate concept (the radix
cache prefix-matching granularity, controlled by FLA_CHUNK_SIZE in the
flash-linear-attention library), not the per-slot mamba state size.

**This is the right design.** vLLM has not implemented it.

---

## Investigation log: what we tried and why each failed

This is the chronological list of attempts in the 2026-04-08 session, so
the next person looking at this can avoid the dead ends.

### Attempt 1: `--mamba-cache-mode align` (lighter mamba prefix caching)

**Hypothesis:** maybe the `align` mode (cache only at scheduler-step
boundaries, not every block) avoids the rollback path.

**Result:** Engine starts. Requests succeed. **But prefix-cache hit rate
is 0%** — the alignment is too restrictive to ever match a real prefix
in practice. Prefix cache is effectively disabled. Eagle3 acceptance
rate is normal (~32%). So `align` is correctness-safe but useless for
the goal.

### Attempt 2: `--mamba-cache-mode none` (disable mamba prefix caching, keep attention prefix caching)

**Hypothesis:** maybe the bug is specifically in mamba prefix caching,
and attention prefix caching alone would still help.

**Result:** Same crash in FlashInfer FP8 paged prefill kernel. The
attention layer's prefix caching also triggers the spec-decode-on-cached-prefix
path that exposes bug #4. Plus, for hybrid models, `--mamba-cache-mode
none` just makes prefix caching skip the mamba layers entirely — the
attention layers still go through the broken path.

### Attempt 3: TRITON_ATTN attention backend with PC + Eagle3

**Hypothesis:** maybe FlashInfer is the only attention backend with the
prefill kernel bug. Triton would bypass it.

**Result:** Triton avoids bug #4 but the vLLM auto-selector keeps
`cudagraph_mode=FULL_AND_PIECEWISE` for Triton (it only auto-downgrades
to `PIECEWISE` for FlashInfer + spec decode), so bug #1 fires at engine
init. Cannot start.

### Attempt 4: Tip-of-main vLLM install

**Hypothesis:** vLLM PRs #34874, #33726, #35447 (Feb-Mar 2026) may have
fixed at least one of the bugs.

**Result:** Built tip-of-main vLLM into `customvllmenv` via
`VLLM_USE_PRECOMPILED=1 uv pip install -e .`. Required adding cublas
and nvrtc header + library symlinks from the venv's `nvidia/cu13` cache
to `/usr/local/cuda/{include,lib64}` to make the FlashInfer JIT
compilation succeed (Blackwell sm_120 needs JIT compile). Once
installed, the FP8 PC + Eagle3 path still hits the FlashInfer FP8
prefill kernel bug — it's an upstream FlashInfer kernel issue, not a
vLLM issue. Recommended setup is documented in the
[`vllm_tipmain_install` memory note](#).

### Attempt 5: BF16 verifier (`nvidia/Nemotron-Cascade-2-30B-A3B`) on tip-of-main

**Hypothesis:** the FP8 KV cache forces FlashInfer; BF16 would use
FLASH_ATTN backend which doesn't have the same kernel bug.

**Result:** **Confirmed** — FLASH_ATTN backend is auto-selected for
BF16. Bug #4 (FlashInfer FP8 prefill) is bypassed. But:
- Triton MoE is needed because the FlashInfer CUTLASS BF16 MoE kernel
  doesn't support `relu2` activation (`--kernel-config '{"moe_backend":"triton"}'`).
- Bugs #1 and #2 in `mamba_attn.py` now fire at engine init.

### Attempt 6: Patch `mamba_attn.py` for bugs #1 and #2

**Hypothesis:** the metadata-level mismatches are the only bugs; fixing
them will let the SSM kernel work.

**Result:** Patched (saved as
[`_partial_mamba_attn_fix.patch`](_partial_mamba_attn_fix.patch)):

- Buffer width: add `num_spec_tokens` margin to `max_num_blocks` in the
  `state_indices_tensor_d` allocation.
- Slice convention: replace `:metadata.num_decode_tokens` with
  `:padded_bs` for the `block_idx_last_*` tensors (which are
  request-indexed, not token-indexed).
- Use site: defensively slice the persistent buffer's dim 1 to match
  the per-call width.

Engine init now succeeds. Cudagraph capture succeeds. Then **bug #3
fires** at the first decode: `_selective_scan_update_kernel` illegal
memory access. The patch unblocked layers 1 and 2 only to expose layer 3.

### Attempt 7: PIECEWISE cudagraph mode with the patches

**Hypothesis:** maybe `FULL` cudagraph capture is the trigger for bug
#3 and PIECEWISE bypasses it.

**Result:** PIECEWISE cudagraph still crashes in the SSM kernel at
runtime. The kernel bug is independent of the cudagraph mode.

### Attempt 8: `--enforce-eager` with the patches (no cudagraphs at all)

**Hypothesis:** if all cudagraph capture/replay is bypassed, maybe it's
a cudagraph-specific issue.

**Result:** Eager mode still crashes in `_selective_scan_update_kernel`.
The bug is in the kernel call itself, not in cudagraph capture/replay.
This was the conclusive evidence that bug #3 is in the actual SSM
kernel logic, not in vLLM's surrounding metadata or graph machinery.

### Attempt 9: alternative aux-hidden-state layer ids `[5, 19, 33]` (all-attention positions)

**Hypothesis (from a thoughtful question during the investigation):**
maybe the bug is somehow related to *which* layers Eagle3 captures from,
and if Eagle3 only reads from attention layers (positions 5, 19, 33 in
the hybrid pattern `MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME`),
the bug would not fire.

**Result:** Created a snapshot of the published draft head's config
with `eagle_aux_hidden_state_layer_ids` overridden to `[5, 19, 33]`,
verified vLLM logged `Using auxiliary layers from speculative config:
(5, 19, 33)` at startup, then submitted a request. Same crash, same
backtrace. **Conclusively proves the bug is independent of which layers
the draft head reads from**, because the verifier's forward pass
executes all 52 layers regardless — the mamba layers always run, and
their state-management code path is the broken one. The aux layer
choice only controls which layer outputs the *draft* sees; it does not
change what the *verifier* computes.

### Attempt 10: n-gram prompt-lookup speculator + PC

**Hypothesis (suggested during the investigation):** maybe the bug is
specifically Eagle3 / specifically a draft model; n-gram speculator uses
no model and might bypass it.

**Result:** Same `AcceleratorError: CUDA error: an illegal memory access
was encountered`. This was the cleanest possible test of "is the bug
spec-decode-method-agnostic" because n-gram is the simplest possible
speculator (it just looks up n-grams in the prompt). It still crashes,
which **conclusively proves the bug is in the verifier-side spec-decode
path, not in any specific draft method**. MTP would hit the same crash
for the same reason — all spec methods feed draft tokens through the
same broken verifier path.

---

## Recommended upstream fix path

For someone willing to do the upstream vLLM work to make this combination
ship, the right approach is to **port SGLang's cache-isolation design**.
This is not a bug-fix; it's a feature port. Approximate scope:

1. **Mamba cache pool: support per-candidate-token slots.**
   `vllm/model_executor/layers/mamba/mamba_mixer2.py` and the surrounding
   cache management need to allocate `(1 + num_spec_tokens)` SSM slots
   per active spec-decode request, not 1.

2. **Mamba SSM kernel: write speculative steps to sandbox slots.**
   `vllm/model_executor/layers/mamba/ops/mamba_ssm.py
   _selective_scan_update_kernel` needs a new mode that writes each
   speculative step to its own sandbox slot rather than in-place
   updating the request's single state slot. The kernel signature
   already takes `dst_state_batch_indices` so the API surface is
   probably enough; the implementation just needs to actually use it
   correctly for the spec-decode + sandbox case.

3. **Verification commit: pick the accepted prefix's sandbox.**
   The verifier-side post-sample step (where vLLM's sampler decides
   which spec tokens were accepted) needs a new "commit sandbox" call
   that points the request's "current state slot" pointer at the
   correct sandbox, and frees the rest back to the pool.

4. **Tree speculation (Eagle-Tree) compatibility.**
   For Eagle-Tree, the sandboxes need to form a tree mirroring the
   speculation tree, not a flat list. This is what SGLang implemented
   for Eagle3 + NemotronH support.

5. **Metadata-level fixes (bugs 1 and 2).**
   The
   [`_partial_mamba_attn_fix.patch`](_partial_mamba_attn_fix.patch) in
   this directory patches `mamba_attn.py` to fix the buffer-sizing and
   slice-convention bugs that gate progress past engine init. These
   fixes are correct on their own; they would land cleanly as a
   stand-alone PR even before the kernel work above is done.

6. **Reference implementation:** SGLang's relevant code lives in their
   `sgl-project/sglang` repo. The PyTorch blog post linked above is the
   high-level summary; the actual implementation is in their hybrid
   model support PRs (search for "MambaRadixCache" and "cache isolation"
   on their repo).

The work is **substantial but well-scoped**: an SGLang reference exists,
the design is known to work, and the failure modes are precisely
diagnosed. The likely upstream PR shape is 3-5 PRs (one for the
metadata-level cleanup, one for the cache pool, one for the kernel, one
for the verifier-side commit, one for tree compatibility).

---

## Key files referenced

- `vllm/v1/attention/backends/mamba_attn.py` — bugs #1 and #2; metadata
  builder for the mamba state path. Has the `# Speculative decoding not
  supported with prefix caching` comment that documents the assumption
  this whole investigation falsifies.
- `vllm/model_executor/layers/mamba/mamba_mixer2.py` — the consumer of
  the mamba metadata that splits tensors by `[num_decodes, num_prefills]`
  (request count). Calls into `selective_state_update` for the SSM
  forward.
- `vllm/model_executor/layers/mamba/ops/mamba_ssm.py` — bug #3; the
  Triton kernel that needs the SGLang cache-isolation port.
- `flashinfer-python 0.6.7` `batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_*`
  kernel — bug #4, FP8-only, separate from the mamba bugs.

## Empirical confirmation table

All of these were tested on the bleeding-edge `customvllmenv` install on
2026-04-08, single RTX PRO 6000 Blackwell. Every row below crashed with
some variant of "illegal memory access" or "size mismatch".

| Verifier | KV dtype | Attn backend | Spec method | mamba_cache_mode | cudagraph mode | Result |
|---|---|---|---|---|---|---|
| FP8 | fp8_e4m3 | FlashInfer (auto) | Eagle3 | all | FULL_AND_PIECEWISE → PIECEWISE auto | crash bug #4 (FlashInfer prefill) |
| FP8 | fp8_e4m3 | FlashInfer | Eagle3 | none | PIECEWISE | crash bug #4 |
| FP8 | fp8_e4m3 | TRITON_ATTN | Eagle3 | all | FULL_AND_PIECEWISE | crash bug #1 (mamba_attn buffer) |
| FP8 | fp8_e4m3 | FlashInfer | Eagle3 | all | enforce_eager | crash bug #4 |
| FP8 | fp8_e4m3 | FlashInfer | Eagle3 (aux=[5,19,33]) | all | PIECEWISE | crash bug #4 (proves layer choice irrelevant) |
| FP8 | fp8_e4m3 | FlashInfer | Eagle3, max_num_seqs=4 | all | PIECEWISE | crash bug #4 (proves batch-size irrelevant) |
| BF16 | auto | FlashAttn (auto) | Eagle3 | all | FULL_AND_PIECEWISE | crash bug #1 |
| BF16 | auto | FlashAttn | Eagle3 + partial patch | all | FULL_AND_PIECEWISE | crash bug #3 (cudagraph replay) |
| BF16 | auto | FlashAttn | Eagle3 + partial patch | all | PIECEWISE | crash bug #3 |
| BF16 | auto | FlashAttn | Eagle3 + partial patch | all | enforce_eager | **crash bug #3** (SSM kernel) |
| BF16 | auto | FlashAttn | n-gram + partial patch | all | enforce_eager | **crash bug #3** (proves spec method irrelevant) |
| BF16 | auto | FlashAttn | Eagle3 (any spec count) | none | any | crash bug #3 (mamba mode none doesn't disable; just for the mamba state path) |

The pattern is clear and exhaustive: **any combination of (spec decode
ON) + (mamba prefix caching ON) crashes**, regardless of attention
backend, KV dtype, cudagraph mode, spec method, batch size, aux layer
choice, or `--enforce-eager`. The bug is exactly where SGLang's blog
post says it is — in-place SSM state updates being incompatible with
the speculate-then-rollback semantics of spec decode.

---

## See also

- [`spec_decode_eagle3_nemotron_h.md`](spec_decode_eagle3_nemotron_h.md) —
  the user-facing how-to docs for the working part (Eagle3 + NemotronH
  *without* prefix caching), including the empirical PC vs Eagle3
  comparison on AIME-25.
- [`_partial_mamba_attn_fix.patch`](_partial_mamba_attn_fix.patch) —
  partial patch for bugs #1 and #2; not applied to the main tree; saved
  as a starting point for the upstream fix.
- [PyTorch blog: Hybrid Models Meet SGLang](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/) —
  the SGLang reference solution.
- [Alibaba Cloud writeup](https://www.alibabacloud.com/blog/hybrid-model-support-%7C-sglangs-support-scheme-for-hybrid-architecture-models-like-mamba-transformer_602857) —
  the deeper technical explanation of SGLang's cache-isolation design.
- [vLLM PR #34874](https://github.com/vllm-project/vllm/pull/34874) —
  "Fix prefix caching for Mamba 'all' mode (Nemotron models)" — partial
  upstream work, doesn't cover spec decode.
- [vLLM PR #33726](https://github.com/vllm-project/vllm/pull/33726) —
  "Nemotron-H MTP and Mamba Speculative Decoding Support" — adds the
  spec decode metadata path that bugs 1 and 2 live in.
- [vLLM PR #35447](https://github.com/vllm-project/vllm/pull/35447) —
  "Fix NemotronH MTP + Chunked Prefill" — partial fix for some of the
  same code paths.
- [SGLang issue #21138](https://github.com/sgl-project/sglang/issues/21138) —
  related: even SGLang had/has issues with NemotronH MTP acceptance
  rates being lower than expected. So even with cache isolation, there
  are still subtleties to get right.
