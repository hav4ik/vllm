# PC + Eagle3 Spec Decode: Final Status

> Branch: `pc-spec-v2-nonpc-slots`
> Date: 2026-04-11 (superseded the 2026-04-10 "0% hits" version below)
> Model: NemotronH (chankhavu/Nemotron-Cascade-2-30B-A3B-FP8)
> Eagle3 drafter: chankhavu/c2.eagle3-test

## TL;DR — 2026-04-11 update

All three success criteria **PASS** on the 94 GB Blackwell RTX PRO 6000:

| Criterion | Result |
|---|---|
| AIME-25 per-session accuracy | **58/60 = 96.7%** (>90% target) |
| AIME-25 majority vote | **30/30 = 100%** |
| Prefix cache hit rate (cumulative) | **4.37M / 5.51M = 79.4%** |
| Eagle3 draft acceptance | **453k / 1.07M = 42.2%** |
| `max_parallel=4` + cudagraph PIECEWISE crashes | **0** |

The "0% cache hits" section below is **resolved**. Two independent
things were required:

1. **Five code fixes** (commits through `8546c2a71`). The last three —
   cudagraph buffer clone, GPU-authoritative `num_computed_d`, and
   `num_decodes` slice in `commit_boundary_states` — lived only in
   the previous agent's offline-bundle wheel for a day before being
   recovered and committed on 2026-04-11. See commit `8546c2a71`.
2. **Server config change**: `--mamba-block-size 256` instead of the
   original 512. With 512 the LCM-aligned block size inflated to
   ~4608 tokens and the Eagle block drop wiped all cache hits even
   with the coordinator-level fix. 256 keeps the LCM small enough
   that multi-turn conversations actually hit.

### Verified startup command (the one that works)

```bash
HF_HOME=/workspace/.hf_home VLLM_USE_FLASHINFER_MOE_FP8=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve chankhavu/Nemotron-Cascade-2-30B-A3B-FP8 \
  --max-model-len 131072 --trust-remote-code \
  --mamba-ssm-cache-dtype float16 --max-num-seqs 16 \
  --kv-cache-dtype fp8 --enable-prefix-caching \
  --mamba-cache-mode all --mamba-block-size 256 \
  --gpu-memory-utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --download-dir /workspace/models --host 127.0.0.1 --port 18000 \
  --speculative-config '{"model":"chankhavu/c2.eagle3-test","method":"eagle3","num_speculative_tokens":4}'
```

No `--enforce-eager`; cudagraph mode is `FULL_AND_PIECEWISE`; async
scheduling stays on by default.

### Build-env gotcha

On sm_120f (Blackwell RTX PRO 6000) with CUDA 13.0 / Python 3.12,
FlashInfer's JIT compile of `gemm` fails because `/usr/local/cuda/include`
is missing `cublasLt.h`. Install `flashinfer-jit-cache` to skip the
JIT step entirely:

```bash
uv pip install flashinfer-jit-cache==0.6.7 \
  --extra-index-url https://flashinfer.ai/whl/cu130/
```

---

## Historical record: "v2 final status" as of 2026-04-10

Everything below this line is the pre-fix snapshot. Kept for reference.

## What we built

A fix for the PC + speculative decoding interaction on Mamba2 hybrid
models. The fix makes the mamba decode path work correctly when BOTH
prefix caching (`mamba_cache_mode=all`) and speculative decoding
(Eagle3) are enabled simultaneously.

## The problem

When prefix caching AND spec decode are both enabled on NemotronH:

1. **Without any fix**: the server CRASHES with `illegal memory access`
   because the SSM kernel indexes into the block table using
   `init_token_idx` (a candidate index 0..K), but the block table has
   BLOCK entries (block 0's slot, block 1's slot, etc.). Reading
   block N-1's slot when you wanted candidate N-1's slot gives the
   wrong physical address.

2. **With v1 scratch approach** (branch `nemotron3-eagle3-support`):
   the server runs but accuracy degrades from 96% → 50-77% because
   the per-step pool↔scratch round-trip introduces subtle state drift
   over thousands of decode steps.

## The fix (v2)

**Core idea**: give both SSM and conv kernels their own K+1 dedicated
"spec slots" per request — exactly like the working non-PC path. The
kernels use native rollback (`init_token_idx` for SSM,
`conv_state_token_offset` for conv). No per-step copying to/from
the pool. States persist in spec slots across decode steps.

### Changes to `mamba_mixer2.py`

1. **Spec slot allocation** (`init_spec_slots`): allocates a separate
   SSM tensor (page-padded stride) and conv tensor (contiguous) with
   `max_seqs * (K+1) + 1` slots each. Pre-computes the slot ID table
   `_spec_slot_ids[i][j] = 1 + i * K1 + j`.

2. **Decode forward path**: when `mamba_cache_mode=all` AND spec decode
   is active AND `num_accepted_tokens` is available:
   - SSM kernel uses `spec_ssm` with `_spec_slot_ids` as both input
     and output indices (non-broadcast — native `init_token_idx`)
   - Conv kernel uses `spec_conv` with `_spec_slot_ids` in non-APC
     mode (`block_idx_last_scheduled_token=None`)
   - One-time init from pool on first decode step per request
     (detected via `_spec_inited` flag per batch position)
   - Stashes block table + old block index for boundary commit

3. **Boundary commit** (`commit_boundary_states`): after the rejection
   sampler, for each request whose accepted tokens cross a mamba block
   boundary, copies the boundary-position state from the spec slot to
   the pool's block slot. Boundary position = `(block_before + 1) *
   block_size - 1`, candidate index = `boundary_pos - n_done`.

### Changes to `gpu_model_runner.py`

1. **Eager init**: after `bind_kv_cache`, initializes spec slots for
   all MambaMixer2 layers.

2. **Request finish hook**: before `remove_request`, resets
   `_spec_inited[batch_idx]` for finished requests so new requests at
   that position get fresh initialization.

3. **Condense hook**: after `condense()`, detects moved requests
   (pre/post mapping diff) and copies their spec slot data (SSM + conv)
   to the new batch positions. Also transfers `_spec_inited` flags.

4. **Boundary commit hook**: in `_update_states_after_model_execute`,
   calls `commit_boundary_states` for each mamba layer.

## Results

| Config | Per-session | Majority | avg_prompt | avg_tools |
|--------|-------------|----------|-----------|-----------|
| No-PC Eagle3 (baseline) | 92.9% | 96.4% | 48K | 4.6 |
| PC only, no spec (baseline) | 95.0% | 100% | 78K | 6.0 |
| v1 full round-trip | 50.0% | 60.0% | 22K | 2.9 |
| v1 Idea 1 (SSM round-trip + native conv) | 63.3% | 76.7% | 49K | 4.4 |
| **v2 unified spec slots** | **91.5%** | **96.7%** | **86K** | **5.9** |

**v2 matches the no-PC Eagle3 baseline** (91.5% vs 92.9% per-session,
96.7% vs 96.4% majority). The spec decode is working correctly.

## The remaining problem: 0% prefix cache hits

Despite `--enable-prefix-caching` being active, prefix cache hits are
**exactly 0%**. This means multi-turn conversations re-prefill the
entire context every turn. Our setup is effectively "no PC + Eagle3"
— the PC flag is a no-op.

### Evidence

```
vllm:prefix_cache_queries_total: 5,106,409
vllm:prefix_cache_hits_total:    0
```

Direct test: sending the same prompt twice — the second request is
SLOWER (6.9s vs 2.3s), confirming no cache reuse.

### Root cause

This is upstream issue **#38182** / **#31920**: speculative decoding
suppresses all prefix cache hits. The exact mechanism is under
investigation, but likely involves:

1. **Block hash contamination**: the block hash computation may include
   metadata that differs between spec and non-spec execution, even
   though `request.all_token_ids` only contains accepted tokens.

2. **`num_computed_tokens` mismatch**: after draft token rejection,
   `num_computed_tokens` is reduced by the rejected count. This can
   cause the hash computation or cache lookup to use a different
   "window" of tokens, producing non-matching hashes.

3. **Hybrid coordinator alignment**: the `HybridKVCacheCoordinator`
   requires cache hits to be aligned to the LCM of all block sizes
   (attention block_size AND mamba block_size). With mamba_block_size=512,
   the minimum cacheable prefix is 512 tokens. Short prefixes never hit.

4. **Eagle-specific block dropping**: the coordinator comments mention
   "EAGLE spiral block dropping problem" (issue #32802) where Eagle
   spec decode causes extra blocks to be dropped from the cache hit.

### Impact

Without prefix cache hits, PC + spec decode provides **no speed benefit**
over no-PC + spec decode for multi-turn conversations. The model accuracy
is correct (96.7%) but the multi-turn prefill savings are absent.

### Next steps to fix

1. **Investigate block hash mismatch**: add debug logging to the hash
   computation and cache lookup to see WHY hashes don't match between
   turns.

2. **Check Eagle block dropping**: the `use_eagle` flag is passed to
   the coordinator and may cause block dropping. Test with
   `use_eagle=False` to see if hits improve.

3. **Test with a transformer-only model**: verify that spec decode +
   PC works on a pure transformer model (no mamba). If it does, the
   0% is mamba-specific. If it doesn't, it's a general spec decode
   issue.

4. **Upstream engagement**: comment on #38182 with our findings.

## Files modified

| File | Changes |
|------|---------|
| `vllm/model_executor/layers/mamba/mamba_mixer2.py` | Spec slot alloc, decode path routing, init/commit/condense |
| `vllm/v1/worker/gpu_model_runner.py` | Eager init, finish hook, condense hook, boundary commit |
| `docs/features/_pc_spec_v2_design.md` | Design doc |
| `docs/features/_pc_spec_v2_status.md` | Status doc |

## How to use

```bash
# Same command as before — no new flags needed
vllm serve chankhavu/c2-softcpy-fp8 \
  --max-model-len 65536 --trust-remote-code \
  --mamba-ssm-cache-dtype float32 --max-num-seqs 16 \
  --kv-cache-dtype fp8 --enable-prefix-caching \
  --mamba-cache-mode all --mamba-block-size 512 \
  --enable-chunked-prefill --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"model":"chankhavu/c2.eagle3-test","method":"eagle3","num_speculative_tokens":4}'
```

The fix activates automatically when `mamba_cache_mode=all` AND
`num_speculative_tokens > 0`. No additional configuration needed.
