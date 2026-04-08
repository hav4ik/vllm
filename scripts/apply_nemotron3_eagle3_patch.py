#!/usr/bin/env python3
"""Apply NemotronH ↔ Eagle3 speculative-decoding support to a vLLM install.

This is a string-substitution based patch script. It is intentionally written
this way (as opposed to a unified `.patch` file) so that it works across
several vLLM releases (tested against 0.19.0 and tip-of-main as of
2026-04-08), since both files have shifted line numbers across the two but
the targeted code blocks are stable.

What this patch enables
-----------------------

* Marks `NemotronHForCausalLM` as supporting EAGLE-3 speculative decoding
  (`SupportsEagle3`).
* Makes `NemotronHModel` inherit from `EagleModelMixin` and capture
  auxiliary hidden states at the absolute layer indices specified by the
  draft model's `eagle_aux_hidden_state_layer_ids`. The convention used
  matches SpecForge exactly: index `k` captures the output of
  `model.backbone.layers[k]` (NOT the off-by-one Llama convention).
* Teaches the speculative-decoding runner to read
  `eagle_aux_hidden_state_layer_ids` either from the top level of the draft
  hf_config (as upstream expects) OR from `eagle_config.eagle_aux_hidden_state_layer_ids`
  (where SpecForge actually stores it). This is the layout used by, e.g.,
  `chankhavu/c2.eagle3-test`.

Usage
-----

    # Apply against your active vLLM install:
    python scripts/apply_nemotron3_eagle3_patch.py

    # Or point at a specific install dir:
    python scripts/apply_nemotron3_eagle3_patch.py /path/to/site-packages/vllm

    # Dry-run (show what would change without writing):
    python scripts/apply_nemotron3_eagle3_patch.py --dry-run

The script is idempotent: running it twice on the same install will
report "already applied" and exit cleanly.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass


@dataclass
class Replacement:
    """A single anchored string replacement to perform on a file."""

    path: str
    needle: str
    replacement: str
    sentinel: str  # text guaranteed to exist after the patch is applied

    def is_already_applied(self, content: str) -> bool:
        return self.sentinel in content

    def apply(self, content: str) -> str:
        if self.needle not in content:
            raise RuntimeError(
                f"could not find anchor in {self.path!r} -- the file may have "
                f"drifted from the version this patch was written against.\n"
                f"Anchor begins with: {self.needle.splitlines()[0]!r}"
            )
        return content.replace(self.needle, self.replacement, 1)


# ---------------------------------------------------------------------------
# 1) NemotronH model: import EagleModelMixin / SupportsEagle3, capture aux
#    hidden states, register the SupportsEagle3 interface, override the
#    default-aux-layers method.
# ---------------------------------------------------------------------------

NEMOTRON_H_REPLACEMENTS = [
    Replacement(
        path="vllm/model_executor/models/nemotron_h.py",
        needle=(
            "from vllm.model_executor.models.interfaces import (\n"
            "    HasInnerState,\n"
            "    IsHybrid,\n"
            "    MixtureOfExperts,\n"
            "    SupportsLoRA,\n"
            "    SupportsMambaPrefixCaching,\n"
            "    SupportsPP,\n"
            "    SupportsQuant,\n"
            ")"
        ),
        replacement=(
            "from vllm.model_executor.models.interfaces import (\n"
            "    EagleModelMixin,\n"
            "    HasInnerState,\n"
            "    IsHybrid,\n"
            "    MixtureOfExperts,\n"
            "    SupportsEagle3,\n"
            "    SupportsLoRA,\n"
            "    SupportsMambaPrefixCaching,\n"
            "    SupportsPP,\n"
            "    SupportsQuant,\n"
            ")"
        ),
        sentinel="EagleModelMixin,\n",
    ),
    Replacement(
        path="vllm/model_executor/models/nemotron_h.py",
        needle="class NemotronHModel(nn.Module):",
        replacement="class NemotronHModel(nn.Module, EagleModelMixin):",
        sentinel="class NemotronHModel(nn.Module, EagleModelMixin):",
    ),
    Replacement(
        path="vllm/model_executor/models/nemotron_h.py",
        needle=(
            "        for layer in islice(self.layers, self.start_layer, self.end_layer):\n"
            "            hidden_states, residual = layer(\n"
            "                positions=positions,\n"
            "                hidden_states=hidden_states,\n"
            "                residual=residual,\n"
            "            )\n"
            "\n"
            "        if not get_pp_group().is_last_rank:\n"
            "            return IntermediateTensors(\n"
            "                {\"hidden_states\": hidden_states, \"residual\": residual}\n"
            "            )\n"
            "        hidden_states, _ = self.norm_f(hidden_states, residual)\n"
            "        return hidden_states\n"
        ),
        replacement=(
            "        # Eagle-3 auxiliary hidden state collection.\n"
            "        #\n"
            "        # SpecForge (the canonical EAGLE-3 training framework) captures the\n"
            "        # output of the verifier's transformer layers via a `register_forward_hook`\n"
            "        # on `layers[idx]`. So in SpecForge convention, the layer index `k` in\n"
            "        # `eagle_aux_hidden_state_layer_ids` refers directly to the absolute\n"
            "        # layer index in `model.backbone.layers` of the verifier (no embedding\n"
            "        # offset). For NemotronH (whose draft heads are trained with the\n"
            "        # `nemotron-cascade-2-experiments` SpecForge fork) we therefore match\n"
            "        # this convention exactly: we capture using the absolute layer index\n"
            "        # `idx + self.start_layer`, NOT the off-by-one Llama convention.\n"
            "        aux_hidden_states: list[torch.Tensor] = []\n"
            "        for idx, layer in enumerate(\n"
            "            islice(self.layers, self.start_layer, self.end_layer)\n"
            "        ):\n"
            "            hidden_states, residual = layer(\n"
            "                positions=positions,\n"
            "                hidden_states=hidden_states,\n"
            "                residual=residual,\n"
            "            )\n"
            "            self._maybe_add_hidden_state(\n"
            "                aux_hidden_states,\n"
            "                idx + self.start_layer,\n"
            "                hidden_states,\n"
            "                residual,\n"
            "            )\n"
            "\n"
            "        if not get_pp_group().is_last_rank:\n"
            "            return IntermediateTensors(\n"
            "                {\"hidden_states\": hidden_states, \"residual\": residual}\n"
            "            )\n"
            "        hidden_states, _ = self.norm_f(hidden_states, residual)\n"
            "        if len(aux_hidden_states) > 0:\n"
            "            return hidden_states, aux_hidden_states\n"
            "        return hidden_states\n"
        ),
        sentinel="aux_hidden_states: list[torch.Tensor] = []",
    ),
    Replacement(
        path="vllm/model_executor/models/nemotron_h.py",
        needle=(
            "class NemotronHForCausalLM(\n"
            "    nn.Module,\n"
            "    HasInnerState,\n"
            "    SupportsLoRA,\n"
            "    SupportsPP,\n"
            "    IsHybrid,\n"
            "    SupportsQuant,\n"
            "    MixtureOfExperts,\n"
            "    SupportsMambaPrefixCaching,\n"
            "):"
        ),
        replacement=(
            "class NemotronHForCausalLM(\n"
            "    nn.Module,\n"
            "    HasInnerState,\n"
            "    SupportsLoRA,\n"
            "    SupportsPP,\n"
            "    IsHybrid,\n"
            "    SupportsQuant,\n"
            "    MixtureOfExperts,\n"
            "    SupportsMambaPrefixCaching,\n"
            "    SupportsEagle3,\n"
            "):"
        ),
        sentinel="    SupportsMambaPrefixCaching,\n    SupportsEagle3,\n):",
    ),
    Replacement(
        path="vllm/model_executor/models/nemotron_h.py",
        needle=(
            "    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:\n"
            "        return self.model.embed_input_ids(input_ids)\n"
            "\n"
            "    def forward(\n"
            "        self,\n"
            "        input_ids: torch.Tensor | None,\n"
            "        positions: torch.Tensor,\n"
            "        intermediate_tensors: IntermediateTensors | None = None,\n"
            "        inputs_embeds: torch.Tensor | None = None,\n"
            "        **kwargs,\n"
            "    ):"
        ),
        replacement=(
            "    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:\n"
            "        return self.model.embed_input_ids(input_ids)\n"
            "\n"
            "    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:\n"
            "        \"\"\"Default Eagle3 auxiliary hidden state layer indices for NemotronH.\n"
            "\n"
            "        Uses SpecForge's convention (`layers[k]` for an absolute layer\n"
            "        index `k`) and falls back on its default\n"
            "        `[1, num_layers // 2 - 1, num_layers - 4]` if the draft config does\n"
            "        not specify `eagle_aux_hidden_state_layer_ids`.\n"
            "        \"\"\"\n"
            "        num_layers = len(self.model.layers)\n"
            "        return (1, num_layers // 2 - 1, num_layers - 4)\n"
            "\n"
            "    def forward(\n"
            "        self,\n"
            "        input_ids: torch.Tensor | None,\n"
            "        positions: torch.Tensor,\n"
            "        intermediate_tensors: IntermediateTensors | None = None,\n"
            "        inputs_embeds: torch.Tensor | None = None,\n"
            "        **kwargs,\n"
            "    ):"
        ),
        sentinel="def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:",
    ),
]

# ---------------------------------------------------------------------------
# 2) Speculative-decoding runner: support nested `eagle_config.eagle_aux_hidden_state_layer_ids`
#    layout used by SpecForge in addition to the upstream flat layout.
# ---------------------------------------------------------------------------

EAGLE3_UTILS_REPLACEMENT = Replacement(
    path="vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py",
    needle=(
        "    if not hasattr(hf_config, \"eagle_aux_hidden_state_layer_ids\"):\n"
        "        return None\n"
        "    layer_ids = hf_config.eagle_aux_hidden_state_layer_ids\n"
        "    if layer_ids and isinstance(layer_ids, (list, tuple)):\n"
        "        return tuple(layer_ids)\n"
        "    return None"
    ),
    replacement=(
        "    layer_ids = getattr(hf_config, \"eagle_aux_hidden_state_layer_ids\", None)\n"
        "    if layer_ids is None:\n"
        "        # SpecForge stores the layer ids nested under `eagle_config` (a dict\n"
        "        # field on the draft model config). Support both layouts.\n"
        "        eagle_config = getattr(hf_config, \"eagle_config\", None)\n"
        "        if isinstance(eagle_config, dict):\n"
        "            layer_ids = eagle_config.get(\"eagle_aux_hidden_state_layer_ids\")\n"
        "    if layer_ids and isinstance(layer_ids, (list, tuple)):\n"
        "        return tuple(layer_ids)\n"
        "    return None"
    ),
    sentinel="SpecForge stores the layer ids nested under `eagle_config`",
)

# `gpu_model_runner.py` has the same logic in two slightly different shapes
# across vLLM versions; we provide both anchors and apply whichever matches.

GPU_MODEL_RUNNER_REPLACEMENTS = [
    # Tip-of-main shape (handles dflash too):
    Replacement(
        path="vllm/v1/worker/gpu_model_runner.py",
        needle=(
            "        layer_ids = getattr(hf_config, \"eagle_aux_hidden_state_layer_ids\", None)\n"
            "        if not layer_ids:\n"
            "            dflash_config = getattr(hf_config, \"dflash_config\", None)\n"
            "            if dflash_config and isinstance(dflash_config, dict):\n"
            "                layer_ids = dflash_config.get(\"target_layer_ids\")\n"
        ),
        replacement=(
            "        layer_ids = getattr(hf_config, \"eagle_aux_hidden_state_layer_ids\", None)\n"
            "        if not layer_ids:\n"
            "            # SpecForge stores the layer ids nested under `eagle_config`\n"
            "            # (a dict on the draft hf_config). Support both layouts.\n"
            "            eagle_config = getattr(hf_config, \"eagle_config\", None)\n"
            "            if isinstance(eagle_config, dict):\n"
            "                layer_ids = eagle_config.get(\"eagle_aux_hidden_state_layer_ids\")\n"
            "        if not layer_ids:\n"
            "            dflash_config = getattr(hf_config, \"dflash_config\", None)\n"
            "            if dflash_config and isinstance(dflash_config, dict):\n"
            "                layer_ids = dflash_config.get(\"target_layer_ids\")\n"
        ),
        sentinel="SpecForge stores the layer ids nested under `eagle_config`",
    ),
    # 0.19.0 shape (no dflash branching):
    Replacement(
        path="vllm/v1/worker/gpu_model_runner.py",
        needle=(
            "        hf_config = self.speculative_config.draft_model_config.hf_config\n"
            "        if not hasattr(hf_config, \"eagle_aux_hidden_state_layer_ids\"):\n"
            "            return None\n"
            "\n"
            "        layer_ids = hf_config.eagle_aux_hidden_state_layer_ids\n"
            "        if layer_ids and isinstance(layer_ids, (list, tuple)):\n"
            "            return tuple(layer_ids)\n"
            "\n"
            "        return None"
        ),
        replacement=(
            "        hf_config = self.speculative_config.draft_model_config.hf_config\n"
            "        layer_ids = getattr(hf_config, \"eagle_aux_hidden_state_layer_ids\", None)\n"
            "        if not layer_ids:\n"
            "            # SpecForge stores the layer ids nested under `eagle_config`\n"
            "            # (a dict on the draft hf_config). Support both layouts.\n"
            "            eagle_config = getattr(hf_config, \"eagle_config\", None)\n"
            "            if isinstance(eagle_config, dict):\n"
            "                layer_ids = eagle_config.get(\"eagle_aux_hidden_state_layer_ids\")\n"
            "\n"
            "        if layer_ids and isinstance(layer_ids, (list, tuple)):\n"
            "            return tuple(layer_ids)\n"
            "\n"
            "        return None"
        ),
        sentinel="SpecForge stores the layer ids nested under `eagle_config`",
    ),
]


def find_vllm_root(arg_path: str | None) -> str:
    """Resolve the directory containing the vLLM Python package."""
    if arg_path:
        if os.path.basename(arg_path.rstrip(os.sep)) != "vllm":
            arg_path = os.path.join(arg_path, "vllm")
        if not os.path.isdir(arg_path):
            raise SystemExit(f"vLLM directory not found at {arg_path}")
        return arg_path

    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise SystemExit(
            "vllm is not importable in the current Python environment. "
            "Either install vllm first or pass the path explicitly."
        )
    return os.path.dirname(spec.origin)


def patch_file(vllm_root: str, replacements: list[Replacement], dry_run: bool) -> str:
    """Apply a list of replacements to a single file. Returns a status string."""
    rel = replacements[0].path
    assert all(r.path == rel for r in replacements)
    full = os.path.join(vllm_root, rel[len("vllm/"):])
    if not os.path.isfile(full):
        return f"  SKIP {rel} (not present)"

    with open(full, encoding="utf-8") as f:
        content = f.read()
    original = content

    if all(r.is_already_applied(content) for r in replacements):
        return f"  OK   {rel} (already patched)"

    for r in replacements:
        if r.is_already_applied(content):
            continue
        content = r.apply(content)

    if content == original:
        return f"  OK   {rel} (no change)"

    if dry_run:
        return f"  DRY  {rel} (would be modified, {len(content) - len(original):+} bytes)"

    backup = full + ".eagle3-bak"
    if not os.path.exists(backup):
        with open(backup, "w", encoding="utf-8") as f:
            f.write(original)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    return f"  OK   {rel} (patched, backup at {os.path.basename(backup)})"


def patch_gpu_model_runner(vllm_root: str, dry_run: bool) -> str:
    """gpu_model_runner has two possible anchors; try the tip-of-main one first."""
    rel = GPU_MODEL_RUNNER_REPLACEMENTS[0].path
    full = os.path.join(vllm_root, rel[len("vllm/"):])
    if not os.path.isfile(full):
        return f"  SKIP {rel} (not present)"
    with open(full, encoding="utf-8") as f:
        content = f.read()
    if GPU_MODEL_RUNNER_REPLACEMENTS[0].sentinel in content:
        return f"  OK   {rel} (already patched)"
    for r in GPU_MODEL_RUNNER_REPLACEMENTS:
        if r.needle in content:
            new_content = r.apply(content)
            if dry_run:
                return f"  DRY  {rel} (would be modified)"
            backup = full + ".eagle3-bak"
            if not os.path.exists(backup):
                with open(backup, "w", encoding="utf-8") as f:
                    f.write(content)
            with open(full, "w", encoding="utf-8") as f:
                f.write(new_content)
            return f"  OK   {rel} (patched, backup at {os.path.basename(backup)})"
    raise RuntimeError(
        f"could not find any known anchor in {rel}; this vLLM version is "
        f"likely too old or too new for this patch."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "vllm_path",
        nargs="?",
        default=None,
        help="path to the `vllm` directory inside site-packages "
        "(autodetected from `import vllm` if omitted)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show what would be changed without writing"
    )
    args = parser.parse_args()

    try:
        vllm_root = find_vllm_root(args.vllm_path)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 1

    print(f"Patching vLLM at: {vllm_root}")
    if args.dry_run:
        print("(dry-run mode -- no files will be modified)")

    statuses = []
    statuses.append(patch_file(vllm_root, NEMOTRON_H_REPLACEMENTS, args.dry_run))
    statuses.append(patch_file(vllm_root, [EAGLE3_UTILS_REPLACEMENT], args.dry_run))
    statuses.append(patch_gpu_model_runner(vllm_root, args.dry_run))

    print()
    print("Results:")
    for s in statuses:
        print(s)
    print()
    print("Done. Restart vLLM and use --speculative-config '{\"model\": ..., "
          "\"method\": \"eagle3\", ...}' to enable Eagle3 with NemotronH.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
