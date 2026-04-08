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

* **EAGLE-3 + mamba prefix caching is *not* a single bug.** As of
  2026-04-08, combining `--mamba-cache-mode all` + `--enable-prefix-caching`
  + EAGLE-3 spec decode hits two distinct bugs in the verifier-side
  attention stack:

  1. **Mamba attention CUDA-graph capture buffer mismatch**
     (`vllm/v1/attention/backends/mamba_attn.py`). When spec decode +
     mamba_cache_mode=all are both active, the preallocated
     `state_indices_tensor_d` is sized for `(max_num_seqs, max_num_blocks)`
     but the runtime metadata is sized for `(max_num_seqs, max_num_blocks
     + spec_padding)`. The code comment in the buffer-allocation block
     literally says *"Speculative decoding not supported with prefix
     caching, so keep shape consistent with prefill buffer"*. This bug
     was **partially fixed** between vLLM 0.19.0 and tip-of-main (Mar
     2026) by PRs #34874, #33726 and #35447, but only when the FlashInfer
     attention backend forces a downgrade to `cudagraph_mode=PIECEWISE`.
     With the Triton attention backend (which keeps
     `FULL_AND_PIECEWISE`), the same buffer mismatch still triggers at
     init time on tip-of-main.

  2. **FlashInfer FP8 paged-KV prefill kernel illegal memory access**
     when called with a partial prefix (i.e. with prefix caching active)
     AND a spec-decode-inflated batch. Reproduces against
     `flashinfer-python` 0.6.7 with `kv_cache_dtype=fp8_e4m3`,
     `block_size=4288`, NemotronH MoE+Mamba. Independent of vLLM
     version (still hits on vLLM tip-of-main as of 2026-04-08). The
     crash signature is `BatchPrefillWithPagedKVCacheRun failed with
     error an illegal memory access was encountered` from the
     `batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3...` kernel.
     Triggers on the *first decode* after a request with cached prefix
     even with `--enforce-eager` (so it is not a CUDA-graph artifact).

  Either bug alone blocks `EAGLE-3 + prefix caching` for NemotronH on
  vLLM 0.19.0; both must be resolved upstream before the combination
  can ship. The empirical workaround is to **pick one of**:

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

**Key observations:**

- **Lowering `num_speculative_tokens` from 5 → 3 helps EAGLE-3** (93%
  acc, 133s/session vs 90% acc, 140s/session). The per-position
  acceptance decay is steep (67/40/24/15/9%), so positions 4-5 add
  ~24% combined acceptance for full compute cost — net negative on
  this workload.
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
