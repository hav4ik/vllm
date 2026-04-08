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

* **Mamba prefix caching (`--mamba-cache-mode all`) is currently
  incompatible with EAGLE-3 spec decode** in vLLM 0.19. Combining
  `--mamba-cache-mode all` + `--enable-prefix-caching` +
  `--async-scheduling` + `--speculative-config eagle3` triggers a
  CUDA illegal memory access at first decode. Drop `--mamba-cache-mode`
  and `--enable-prefix-caching` if you want to use EAGLE-3.
  Investigation tracked separately; this is not a fundamental
  blocker — full prefix caching just needs to be plumbed through the
  hybrid drafter path. If you do not need prefix caching, the command
  above runs end-to-end with full CUDA graph capture.

* **`--async-scheduling`** also appears to be on the same incompatibility
  axis when combined with mamba prefix caching. Without mamba prefix
  caching, `--async-scheduling` is fine.

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
