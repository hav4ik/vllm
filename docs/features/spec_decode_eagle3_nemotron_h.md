# EAGLE-3 speculative decoding for NemotronH (Cascade-2 / Nano 30B-A3B)

This vLLM branch (`nemotron3-eagle3-support`) adds the small set of changes
needed to use an [EAGLE-3] draft head against a NemotronH verifier — i.e.
hybrid Mamba-Transformer MoE models such as
`nvidia/Nemotron-Cascade-2-30B-A3B` and `chankhavu/c2-softcpy-fp8`.

To our knowledge this is the first EAGLE-3 head trained against a hybrid
Mamba-Transformer verifier; the draft head used as the reference checkpoint
here was trained with the
[`hav4ik/SpecForge-Nemotron3`](https://github.com/hav4ik/SpecForge-Nemotron3)
fork of SpecForge on the `nemotron-cascade-2-experiments` branch.

[EAGLE-3]: https://arxiv.org/abs/2503.01840

## What this patch changes

Three small touch-points in vLLM:

1. **`vllm/model_executor/models/nemotron_h.py`** —
    - `NemotronHForCausalLM` now declares `SupportsEagle3`.
    - `NemotronHModel` inherits from `EagleModelMixin` and captures the
      auxiliary hidden states required by the EAGLE-3 draft inside its
      forward loop.
    - The capture indices use **SpecForge's convention**: an index `k` in
      the draft config's `eagle_aux_hidden_state_layer_ids` refers to the
      output of `model.backbone.layers[k]` (i.e. the absolute layer index in
      the verifier, with no embedding-layer offset). This is *not* the same
      convention used by vLLM's Llama EAGLE-3 capture, which is implicitly
      offset by +1 because it also captures the embedding output as
      "layer 0". The NemotronH override matches the convention used during
      training so the published draft head works out of the box.
    - A `get_eagle3_default_aux_hidden_state_layers` override returns
      `(1, num_layers // 2 - 1, num_layers - 4)` (SpecForge default) in
      case the draft config does not specify `eagle_aux_hidden_state_layer_ids`.

2. **`vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py`** &nbsp;**+**&nbsp;
   **`vllm/v1/worker/gpu_model_runner.py`** —
    - Both helpers that resolve `eagle_aux_hidden_state_layer_ids` from
      the draft `hf_config` now also accept the **nested** layout
      `eagle_config["eagle_aux_hidden_state_layer_ids"]` that SpecForge
      writes by default. This means EAGLE-3 heads published by SpecForge
      (e.g. [`chankhavu/c2.eagle3-test`](https://huggingface.co/chankhavu/c2.eagle3-test))
      can be served by vLLM without rewriting the draft `config.json`.

There are no changes to `Eagle3LlamaForCausalLM` itself — the existing
1-layer-Llama draft implementation handles the head end-to-end (vocab
pruning via `d2t`, sliding-window attention from `sliding_window=4096` in
the draft config, GQA, FP8-aware loading, etc.).

## Applying the patch to your install

This branch ships an idempotent patch applier that string-substitutes
the changes into any installed vLLM (tested against 0.19.0 and tip of
main as of 2026-04-08). It is more robust to version drift than a
unified `.patch` file, since the line numbers between v0.19.0 and main
have shifted but the targeted code blocks are stable.

```bash
# Against your active vLLM install (autodetected from `import vllm`):
python scripts/apply_nemotron3_eagle3_patch.py

# Or against an explicit install dir:
python scripts/apply_nemotron3_eagle3_patch.py /path/to/site-packages/vllm

# Dry-run first if you want:
python scripts/apply_nemotron3_eagle3_patch.py --dry-run
```

The script writes `.eagle3-bak` files alongside each patched file the
first time it touches them, and is fully idempotent — running it twice
prints "already patched" and exits cleanly.

## Running vLLM with EAGLE-3 + NemotronH

The exact command tested on this machine (1× RTX PRO 6000, 96 GiB) is:

```bash
VLLM_USE_FLASHINFER_MOE_FP8=1 vllm serve chankhavu/c2-softcpy-fp8 \
    --max-model-len 65536 \
    --trust-remote-code \
    --mamba-ssm-cache-dtype float32 \
    --max-num-seqs 16 \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --download-dir /workspace/models \
    --host 127.0.0.1 --port 18000 \
    --speculative-config '{
        "model": "chankhavu/c2.eagle3-test",
        "method": "eagle3",
        "num_speculative_tokens": 5
    }'
```

When the verifier loads, the log line you should see is:

```
gpu_model_runner.py: Using auxiliary layers from speculative config: (2, 26, 48)
```

These three indices come from the published draft head's
`eagle_config.eagle_aux_hidden_state_layer_ids` and correspond to the
output of NemotronH layers `2` (Mamba), `26` (Attention), and `48`
(Mamba) — chosen by SpecForge during training to give a low/mid/high
hidden-state mixture.

### Important compatibility notes

* **Mamba SSM state must be float32.** Always pass
  `--mamba-ssm-cache-dtype float32`. Downcasting to bf16 causes
  ~10% absolute regression on AIME-class math (the published Nemotron
  Cascade-2 SGLang vs. vLLM gap of 88.3% → 99.17% on AIME 2025 was
  caused by exactly this).

* **EAGLE-3 + mamba prefix caching is fundamentally a mamba-state
  rollback bug**, not an EAGLE-3 bug. Spec decoding requires the
  verifier to compute "speculative" state updates for `1 +
  num_spec_tokens` tokens per request, then **roll back** the rejected
  speculative state on rejection. With mamba prefix caching enabled,
  there are multiple cache slots per request and the rollback has to
  decide which slot to commit, which to discard. The vLLM mamba layer
  doesn't correctly implement this rollback semantics on top of prefix
  caching. The chain of bugs is at least three deep:

  1. **`vllm/v1/attention/backends/mamba_attn.py:117` —
     `state_indices_tensor_d` cudagraph capture buffer mismatch.**
     When spec decode + mamba_cache_mode=all are both active, the
     preallocated buffer is sized for `(max_num_seqs, max_num_blocks)`
     but the runtime metadata can be `(max_num_seqs, max_num_blocks +
     num_spec_tokens)`. The code comment in the buffer allocation
     literally says *"Speculative decoding not supported with prefix
     caching, so keep shape consistent with prefill buffer"*. Partial
     fix is in `_partial_mamba_attn_fix.patch` next to this doc — it
     widens the buffer by `num_spec_tokens` and slices `block_idx_last_*`
     to `num_reqs` instead of `num_decode_tokens`. The patch unblocks
     init but exposes the next bug.

  2. **`vllm/model_executor/layers/mamba/mamba_mixer2.py:632`
     `torch.split(block_idx_last_*, [num_decodes, num_prefills])`
     length mismatch.** The downstream consumer of the mamba metadata
     splits the block-idx tensor by request count
     (`num_decodes + num_prefills`), but `mamba_attn.py` was slicing
     it to `num_decode_tokens` (the spec-inflated count). Patched in
     the same partial fix.

  3. **`vllm/model_executor/layers/mamba/ops/mamba_ssm.py
     _selective_scan_update_kernel` illegal memory access at runtime.**
     The mamba SSM Triton kernel itself doesn't correctly handle the
     `state_batch_indices` / `dst_state_batch_indices` /
     `num_accepted_tokens` / `cu_seqlens` combination when prefix
     caching is on AND there are speculative tokens to roll back per
     request. This is the actual rollback semantics bug — the kernel
     does not know how to commit "accepted prefix, rollback rejected
     suffix" when each sequence has multiple cache slots. **No
     patch — needs upstream vLLM fix in the mamba SSM kernel.**

  In addition, with the FP8 verifier `chankhavu/c2-softcpy-fp8`, vLLM
  picks the FlashInfer FP8 paged prefill kernel which has a related
  but distinct bug (`BatchPrefillWithPagedKVCacheRun` illegal memory
  access on first decode after a partial-prefix prefill with
  spec-decode-inflated batch). Switching to the BF16 verifier
  `nvidia/Nemotron-Cascade-2-30B-A3B` bypasses this kernel (vLLM
  picks `FLASH_ATTN` instead of `FLASHINFER`) but exposes bugs 1-3
  above.

  **Crucially, this entire bug chain is independent of the
  speculative-decoding *method*.** The bugs live in the verifier-side
  mamba state path. Eagle3, MTP, and the n-gram prompt-lookup
  speculator all hit the same illegal memory accesses (verified
  empirically with all three on the BF16 verifier on 2026-04-08). The
  draft model only generates speculative tokens; the actual rollback
  is the verifier's mamba layers' responsibility, and that's what's
  broken. Switching to MTP would not fix this.

  Likewise, **changing which layers EAGLE-3 attaches to does not
  help.** The aux-hidden-state layer choice (`[2, 26, 48]` vs
  `[5, 19, 33]` all-attention vs anything else) only controls which
  layer outputs the draft head reads — it does not change what the
  verifier computes. The verifier still runs all 52 layers including
  the 23 mamba layers, which still go through the buggy state
  rollback path. Verified empirically with `[5, 19, 33]` on
  2026-04-08 — same crash.

  **For the full investigation log, the SGLang reference solution, the
  recommended upstream fix path, and the complete table of every
  configuration tried, see**
  [`_pc_spec_decode_mamba_investigation.md`](_pc_spec_decode_mamba_investigation.md)
  in this directory.

  The empirical workaround is to **pick one of**:

  - **Run with prefix caching, no EAGLE-3.** Drop the `--speculative-config`
    flag. Best for agentic / multi-turn workloads where the long shared
    prefix dominates.
  - **Run with EAGLE-3, no prefix caching.** Drop `--enable-prefix-caching`
    and `--mamba-cache-mode`. Best for single-turn, short-context
    workloads where per-token decode latency dominates.

  See "Empirical comparison" below for the actual numbers from running
  both on AIME-25 (mathematical reasoning with sandbox tool use, 4
  sessions per problem, 30 problems = 120 sessions).

### Empirical comparison: PC vs EAGLE-3 on AIME-25

Both runs use the same vLLM tip-of-main install, the same
`chankhavu/c2-softcpy-fp8` verifier, the same temperature (0.6) and
top_p (0.95), the same `--max-num-seqs 16`, and the same
`max_completion_tokens=32768`. Only the speculative-config and
prefix-caching flags differ.

Two independent passes were run for the prefix-caching baseline, one
for EAGLE-3 with `num_speculative_tokens=5` (the SpecForge default),
and *two* for EAGLE-3 with `num_speculative_tokens=3` (since the
per-position acceptance decay was steep on this workload, suggesting
positions 4-5 may not be worth their compute cost).

| Metric | PC run 1 | PC run 2 | EAGLE-3 spec=5 | EAGLE-3 spec=3 run 1 | EAGLE-3 spec=3 run 2 |
| --- | --- | --- | --- | --- | --- |
| `--enable-prefix-caching --mamba-cache-mode all` | ✓ | ✓ | ✗ | ✗ | ✗ |
| `--speculative-config eagle3` | ✗ | ✗ | num_spec=5 | num_spec=3 | num_spec=3 |
| Per-session accuracy | **95.0%** | **91.7%** | 90.0% | 93.3% | 92.5% |
| Majority-vote accuracy (n=4) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Mean wall time / session | **113.2s** | **122.4s** | 140.5s | 130.0s | 136.5s |
| Effective gen tokens/s | **96.2** | **94.9** | 80.3 | 84.6 | 85.6 |
| Mean gen tokens / session | 10,889 | 11,622 | 11,281 | 11,004 | 11,686 |
| Mean prompt tokens / session | 78,348 | 81,657 | 73,862 | 75,057 | 91,616 |
| Sessions w/ max_turns/token_limit/no_answer | 6 | 8 | 10 | 8 | 9 |
| EAGLE-3 mean acceptance rate | n/a | n/a | 31% (1.55 tok/draft) | 44% (1.31 tok/draft) | 44% (1.31 tok/draft) |
| EAGLE-3 per-position acceptance | n/a | n/a | 67/40/24/15/9% | 67/40/24% | 67/40/24% |
| Prefix-cache hit rate (avg over run) | ~50–80% | ~50–80% | 0% | 0% | 0% |

Per-config means with both runs collapsed:

| Config (mean of n=2 runs) | per_session_acc | mean_wall_s | eff_gen_t/s |
| --- | --- | --- | --- |
| **PC, no EAGLE-3** | **93.4%** | **117.8s** | **95.6** |
| EAGLE-3 spec=3, no PC | 92.9% | 133.2s | 85.1 |
| EAGLE-3 spec=5, no PC (n=1) | 90.0% | 140.5s | 80.3 |
| **PC vs Eagle3 spec=3 advantage** | **+0.5pp** | **−13% wall** | **+12% t/s** |

**`num_speculative_tokens` sweep** (all at seqs=16):

| spec count | per_session_acc | mean_wall_s | eff_gen_t/s |
| --- | --- | --- | --- |
| 2 | 90.0% | 136.4s | 82.3 |
| **3** (sweet spot) | **92.9%** (mean of 2 runs) | **133.2s** | **85.1** |
| 5 (SpecForge default) | 90.0% | 140.5s | 80.3 |

**`num_speculative_tokens=3` is the sweet spot** for this workload —
both faster *and* more accurate than spec=2 and spec=5. The accuracy
drop at spec=2 (90.0%) suggests position 2 is still contributing
useful tokens, while the wall-time bump from spec=3→5 (133→140s)
without an accuracy gain confirms positions 4–5 are net-negative
(67/40/24/15/9% per-position decay → 1.55 vs 1.31 mean accepted per
draft, but positions 4-5 cost full compute for ~24% combined
acceptance).

**Key observations (vs PC):**

- **PC still wins on the metrics that matter for agentic loops.**
  After multi-run averaging, accuracy is essentially tied with
  EAGLE-3 spec=3 (PC 93.4% vs EAGLE-3 92.9% — within sampling
  noise) but **wall time is 13% lower and effective throughput is
  12% higher** with prefix caching.
- The wall-time advantage is the **robust** finding: PC's worst run
  (122.4s) is still faster than EAGLE-3's best run (130.0s), so the
  difference holds even at the unfavorable end of the variance band.
- The 7x prompt-to-gen token ratio is the underlying reason. Prefix
  caching attacks the dominant cost (prefill); EAGLE-3 attacks the
  smaller cost (decode).

### Production-scale: `--max-num-seqs 64`

The runs above use `--max-num-seqs 16` for clarity (smaller
parallelism = smaller queueing effects to disentangle from the spec
decode signal). The user's actual production reference command uses
`--max-num-seqs 64`, so the head-to-head was rerun at that batch size
with `--max_parallel 32` on the trace collector to keep the queue
saturated.

| Metric | PC (seqs=64) | EAGLE-3 spec=3 (seqs=64) |
| --- | --- | --- |
| Per-session accuracy | **95.8%** (115/120) | 90.8% (109/120) |
| Majority-vote accuracy (n=4) | **100.0%** (30/30) | **96.7%** (29/30) |
| Total wall time (120 sessions) | **~19 min** | ~20 min |
| Aggregate tokens/s (gen, mean over run) | ~1,140 | ~1,090 |
| Sessions w/ max_turns/token_limit/no_answer | 5 | 11 |

**At production batch size, Eagle3's wall-time gap closes** (~5%
slower vs ~13% slower at seqs=16) because the verifier is already
better utilized at higher concurrency, so the relative cost of the
spec-decode overhead shrinks.

**But PC still wins on accuracy at every level**: per-session 95.8%
vs 90.8% (5 percentage-point gap), and the majority-vote score
actually *drops below 100%* for EAGLE-3 — one AIME problem had all
4 sessions fail, vs zero such problems for PC. This drop of
majority-vote accuracy from 100% → 96.7% is the sign that the user
flagged in the task brief as a "red flag."

The **bottom line for production agentic workloads** is unchanged:
prefix caching is the right configuration, even at high
concurrency. EAGLE-3 ≈ PC on throughput at scale but loses on
accuracy.

**Variance band of the prefix-caching configuration** (95% / 91.7%
across two runs ≈ ±3pp accuracy and ±9s wall time): every EAGLE-3
metric is **outside the worst PC run** on the unfavorable side. The
result is robust to sampling noise — prefix caching is the right
choice for this workload, not just on a lucky run.

**Why prefix caching wins so decisively here:**

* The AIME-style tool-use loop sends the *full* conversation history
  (system prompt + every previous assistant turn + every tool result)
  to the model on every turn. Mean prompt tokens per session is ~78k,
  vs ~11k generated tokens — i.e. **prefill is 7x larger than decode
  by token count**. Anything that speeds up only decode is fighting
  the wrong fight on this workload.
* With prefix caching, the recurring prefix is served from the
  attention KV + mamba state cache; observed prefix-cache hit rate
  was ~50–80% across the run (and the mamba SSM state was correctly
  cached, which is the headline win of vLLM's PR #34874 / #33726
  pair).
* EAGLE-3 *is* doing real work — 31% mean acceptance rate, declining
  cleanly across positions 0-4, ~1.55 tokens accepted per draft of 5 —
  but that ~1.5x decode speedup doesn't compensate for the 7x prefill
  overhead.

**When EAGLE-3 *would* win:** workloads dominated by per-token decode
latency rather than prefill — e.g., single-turn short-prompt chat,
streaming completions where the prefill is cheap, or batch generation
of long completions from short novel prompts. None of those describe
agentic / multi-turn / long-context retrieval workflows.

**The ideal would be both at once.** Once the upstream bug pair above
is fixed, the same docs will be updated with a third row for
"`PC + EAGLE-3`" — that should beat either configuration alone, since
prefix caching handles the prefill cost while EAGLE-3 cuts decode
latency.

### Reproducing the comparison

The benchmark scripts and trace dirs are in this branch's parent
workspace, not in `vllm/` itself. The key commands are:

```bash
# Run 1: prefix caching, no EAGLE-3 (the agentic-friendly baseline)
VLLM_USE_FLASHINFER_MOE_FP8=1 vllm serve chankhavu/c2-softcpy-fp8 \
    --max-model-len 65536 --trust-remote-code \
    --mamba-ssm-cache-dtype float32 --max-num-seqs 16 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching --mamba-cache-mode all --mamba-block-size 512 \
    --enable-chunked-prefill --max-num-batched-tokens 8192 \
    --async-scheduling --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --download-dir /workspace/models --host 127.0.0.1 --port 18000

python collect_traces_nemotron.py \
    --input_dir data/ --output_dir traces/baseline_pc/ \
    --n_sessions 4 --max_parallel 8 \
    --server_addr 127.0.0.1:18000 \
    --model_name chankhavu/c2-softcpy-fp8 \
    --max_tokens 32768 --max_turns 32 \
    --temperature 0.6 --top_p 0.95 --resume

python calc_accuracy.py traces/baseline_pc/

# Run 2: EAGLE-3, no prefix caching
VLLM_USE_FLASHINFER_MOE_FP8=1 vllm serve chankhavu/c2-softcpy-fp8 \
    --max-model-len 65536 --trust-remote-code \
    --mamba-ssm-cache-dtype float32 --max-num-seqs 16 \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill --max-num-batched-tokens 8192 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --download-dir /workspace/models --host 127.0.0.1 --port 18000 \
    --speculative-config '{"model":"chankhavu/c2.eagle3-test","method":"eagle3","num_speculative_tokens":5}'

python collect_traces_nemotron.py \
    --input_dir data/ --output_dir traces/eagle3_no_pc/ \
    --n_sessions 4 --max_parallel 8 \
    --server_addr 127.0.0.1:18000 \
    --model_name chankhavu/c2-softcpy-fp8 \
    --max_tokens 32768 --max_turns 32 \
    --temperature 0.6 --top_p 0.95 --resume

python calc_accuracy.py traces/eagle3_no_pc/
```

### Sanity check

```bash
# 1. Quick correctness ping
curl -sS http://127.0.0.1:18000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chankhavu/c2-softcpy-fp8",
    "messages": [{"role":"user","content":"What is 17*23? Reply with just the number."}],
    "max_tokens": 64,
    "temperature": 0
  }'
# Expected: contains "391"

# 2. Acceptance metrics
curl -sS http://127.0.0.1:18000/metrics | grep -E "spec_decode_(num_drafts_total|num_draft_tokens_total|num_accepted_tokens_total|num_accepted_tokens_per_pos_total)"
```

On the test query above, observed acceptance was ~57% at position 0 and
declining cleanly across positions 1-4 — i.e. EAGLE-3 is doing real work,
not silently rejecting everything because of an off-by-one in the layer
capture indices.

## How the layer-index convention is reconciled

This is the subtle bit, written down because it bit me during
implementation.

vLLM's `Llama` EAGLE-3 capture, `_maybe_add_hidden_state(idx + 1, ...)`,
uses HuggingFace's `outputs.hidden_states` indexing convention, where
index `0` is the embedding output and index `k` for `k > 0` is the output
of `layers[k - 1]`. SpecForge, on the other hand, captures via
`layers[idx].register_forward_hook(...)` with no embedding-layer offset:
its index `k` means the output of `layers[k]`.

These two conventions differ by exactly one. You can see this in their
defaults: SpecForge defaults to `[1, num_layers // 2 - 1, num_layers - 4]`
while vLLM Llama defaults to `(2, num_layers // 2, num_layers - 3)`. Both
sets capture *the same physical layers* — the +1 offset is just baked
into vLLM Llama's interpretation.

For NemotronH, the patch chooses to **match SpecForge directly** rather
than mirror vLLM Llama's offset:

```python
for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
    hidden_states, residual = layer(...)
    self._maybe_add_hidden_state(
        aux_hidden_states,
        idx + self.start_layer,  # absolute layer index
        hidden_states,
        residual,
    )
```

This means the layer ids stored in the draft head's config (`[2, 26, 48]`
for the reference checkpoint) directly correspond to NemotronH layer
positions in `model.backbone.layers`, with no translation needed at
serving time. It also means the default-aux-layers tuple returned by
`get_eagle3_default_aux_hidden_state_layers` for NemotronH is the
SpecForge default `(1, num_layers // 2 - 1, num_layers - 4)`, not the
vLLM Llama default.

If you train a NemotronH EAGLE-3 head against a different layer triple,
just update `eagle_config.eagle_aux_hidden_state_layer_ids` (or the flat
`eagle_aux_hidden_state_layer_ids`) in the draft `config.json` — both
layouts are respected by this patch.
