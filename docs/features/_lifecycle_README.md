# vLLM v1 Lifecycle Reference Docs (NemotronH-focused)

These three documents are a developer reference for navigating vLLM v1 code
when working on the PC + Eagle3 + Mamba2 fix path. They were produced by code
sweep on 2026-04-09 against the `nemotron3-eagle3-support` branch.

The docs are intentionally **scoped to NemotronH on H100 (sm100+) FP8**:
multimodal, encoder-decoder, LoRA, PD disaggregation, pipeline parallel, and
prefill-chunking edge cases are deliberately skipped to reduce noise.

## Files

| File | What it covers |
|---|---|
| `_lifecycle_normal_decode.md` | End-to-end token lifecycle without spec decode: HTTP request → AsyncLLM → IPC → EngineCore → Scheduler → GPUModelRunner → forward → sampler → output streaming. Includes block-table maintenance for the attention KV cache. |
| `_lifecycle_mamba_pc.md` | Mamba SSM state lifecycle with `mamba_cache_mode=all`: pool sizing at startup, block-table mapping, per-step decode read/write, prefill-time intermediate-state snapshotting (the cache-fill path), block commit to the radix tree, and per-request block lifecycle. |
| `_lifecycle_eagle3_spec.md` | Eagle3 propose/verify lifecycle: drafter instantiation, propose step, verify metadata construction, rejection sampler, num_accepted_tokens plumbing, deferred state corrections, set_eagle3_layers_to_capture, and the **broken Mamba2-spec interaction** including a walk-through of `selective_state_update`'s `IS_SPEC_DECODING` mode. |

## How to read these together

For a full picture of "what happens to a single request from API to output",
read in this order:

1. `_lifecycle_normal_decode.md` (sections 1-5) — establishes the engine
   loop and the bookkeeping vocabulary.
2. `_lifecycle_mamba_pc.md` (sections 1-6) — overlays the mamba state
   machine on top, especially block-table indexing and the boundary-state
   commit semantics.
3. `_lifecycle_eagle3_spec.md` (sections 1-7) — overlays the spec decode
   propose/verify dance on top of #1 and #2; section 8 explains the failure
   mode that this branch is trying to fix.

## Companion design docs

- `_pc_spec_decode_mamba_investigation.md` — original empirical bug hunt
  (the 4-bug chain in mamba_attn.py)
- `_pc_spec_decode_upstream_status.md` — upstream GitHub state and the
  surgical incision points for the fix; references to vllm-project/vllm
  issues #26201, #38898, #39273, #33726, #17140
- `spec_decode_eagle3_nemotron_h.md` — user-facing docs for the working
  vLLM Eagle3 (no-PC) path

## Caveats

- These docs are a **snapshot** as of `nemotron3-eagle3-support` branch HEAD
  on 2026-04-09. vLLM master moves fast; line numbers will drift.
- Citations use `file:line` against the local checkout. If you're reading
  this on a different vLLM revision, use the function/class names rather
  than line numbers.
- The "broken Mamba2-spec interaction" section in `_lifecycle_eagle3_spec.md`
  is the *current state on master*; this branch's in-progress work is
  separately documented in `_pc_spec_decode_upstream_status.md`.
