#!/usr/bin/env python3
"""Apply NemotronH ↔ EAGLE-3 speculative-decoding support to any vLLM install.

This is a single-file, stdlib-only, string-substitution based patch script.
It is intentionally written this way (as opposed to a unified `.patch`
file) so that it works across vLLM releases — line numbers between
v0.19.0 and tip-of-main have shifted, but the targeted code blocks are
stable. Each modification has multiple alternative anchors to handle
small drift across versions.

Currently known to apply cleanly against:

* vLLM **0.19.0** (the user's pinned production version, also typical for
  Kaggle / Colab pip installs)
* vLLM tip-of-main as of **2026-04-08**

If your install is from a different version and an anchor fails to
match, the script prints the first line of the anchor it was looking
for, the closest matching context in the file, and the path so you
can either patch by hand or open an issue at
https://github.com/hav4ik/vllm/issues with that snippet attached.

What this patch enables
-----------------------

* Marks `NemotronHForCausalLM` as supporting EAGLE-3 speculative decoding
  (`SupportsEagle3`).
* Makes `NemotronHModel` inherit from `EagleModelMixin` and capture
  auxiliary hidden states at the absolute layer indices specified by the
  draft model's `eagle_aux_hidden_state_layer_ids`. The convention
  matches SpecForge exactly: index `k` captures the output of
  `model.backbone.layers[k]` (NOT the off-by-one Llama convention).
* Teaches the speculative-decoding runner to read
  `eagle_aux_hidden_state_layer_ids` either from the top level of the draft
  `hf_config` (as upstream expects) OR from
  `eagle_config.eagle_aux_hidden_state_layer_ids` (where SpecForge
  actually stores it). This is the layout used by, e.g.,
  `chankhavu/c2.eagle3-test`.

Quick usage
-----------

Local checkout::

    python scripts/apply_nemotron3_eagle3_patch.py            # apply
    python scripts/apply_nemotron3_eagle3_patch.py --verify   # check status only
    python scripts/apply_nemotron3_eagle3_patch.py --dry-run  # show what would change
    python scripts/apply_nemotron3_eagle3_patch.py --revert   # restore from backups

One-shot on Kaggle / Colab / arbitrary cloud GPU box::

    # Download the script and run against the active vLLM install:
    curl -sSLO https://raw.githubusercontent.com/hav4ik/vllm/nemotron3-eagle3-support/scripts/apply_nemotron3_eagle3_patch.py
    python apply_nemotron3_eagle3_patch.py

Or as a true one-liner via process substitution (bash/zsh)::

    python <(curl -sSL https://raw.githubusercontent.com/hav4ik/vllm/nemotron3-eagle3-support/scripts/apply_nemotron3_eagle3_patch.py)

Or pointing at a specific install dir (useful when multiple environments)::

    python apply_nemotron3_eagle3_patch.py /path/to/site-packages/vllm

The script is idempotent: running it twice on the same install reports
"already patched" and exits cleanly. The first time it touches a file,
it writes a `<file>.eagle3-bak` next to it; `--revert` restores from
those backups.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass, field

__version__ = "1.1"

# Stable raw URL for the curl one-liner above. Update if the branch is renamed.
SCRIPT_URL = (
    "https://raw.githubusercontent.com/hav4ik/vllm/"
    "nemotron3-eagle3-support/scripts/apply_nemotron3_eagle3_patch.py"
)

# Known-good vLLM versions for this patch revision. Pure documentation.
SUPPORTED_VLLM_VERSIONS = ["0.19.0", "tip-of-main as of 2026-04-08"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Modification:
    """One semantic change to a file.

    A modification can have several `needles` (alternative anchor texts)
    in case different vLLM versions express the same code block slightly
    differently — they will be tried in order until one matches.
    """

    description: str
    needles: list[str]
    replacement: str
    sentinel: str  # text guaranteed to be present after the modification is applied

    def is_applied(self, content: str) -> bool:
        return self.sentinel in content

    def find_anchor(self, content: str) -> str | None:
        """Return the first anchor that exists in `content`, or None."""
        for n in self.needles:
            if n in content:
                return n
        return None

    def apply(self, content: str) -> str:
        anchor = self.find_anchor(content)
        if anchor is None:
            raise AnchorNotFoundError(self)
        return content.replace(anchor, self.replacement, 1)


@dataclass
class FilePatch:
    path: str  # relative to the vllm/ package root (always starts with "vllm/")
    description: str
    modifications: list[Modification] = field(default_factory=list)


class AnchorNotFoundError(Exception):
    """Raised when none of a Modification's anchors match the file."""

    def __init__(self, modification: Modification):
        super().__init__(modification.description)
        self.modification = modification


# ---------------------------------------------------------------------------
# 1) NemotronH model: import EagleModelMixin / SupportsEagle3, capture aux
#    hidden states, declare SupportsEagle3, override default-aux-layers.
# ---------------------------------------------------------------------------

NEMOTRON_H_PATCH = FilePatch(
    path="vllm/model_executor/models/nemotron_h.py",
    description="NemotronH EAGLE-3 support (model class)",
    modifications=[
        Modification(
            description="import EagleModelMixin and SupportsEagle3 from interfaces",
            needles=[
                "from vllm.model_executor.models.interfaces import (\n"
                "    HasInnerState,\n"
                "    IsHybrid,\n"
                "    MixtureOfExperts,\n"
                "    SupportsLoRA,\n"
                "    SupportsMambaPrefixCaching,\n"
                "    SupportsPP,\n"
                "    SupportsQuant,\n"
                ")",
            ],
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
            sentinel="    EagleModelMixin,\n",
        ),
        Modification(
            description="NemotronHModel inherits from EagleModelMixin",
            needles=["class NemotronHModel(nn.Module):"],
            replacement="class NemotronHModel(nn.Module, EagleModelMixin):",
            sentinel="class NemotronHModel(nn.Module, EagleModelMixin):",
        ),
        Modification(
            description="NemotronHModel.forward captures aux hidden states",
            needles=[
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
                "        return hidden_states\n",
            ],
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
        Modification(
            description="NemotronHForCausalLM declares SupportsEagle3",
            needles=[
                "class NemotronHForCausalLM(\n"
                "    nn.Module,\n"
                "    HasInnerState,\n"
                "    SupportsLoRA,\n"
                "    SupportsPP,\n"
                "    IsHybrid,\n"
                "    SupportsQuant,\n"
                "    MixtureOfExperts,\n"
                "    SupportsMambaPrefixCaching,\n"
                "):",
            ],
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
        Modification(
            description="NemotronHForCausalLM.get_eagle3_default_aux_hidden_state_layers",
            needles=[
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
                "    ):",
            ],
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
    ],
)


# ---------------------------------------------------------------------------
# 2) eagle3_utils: also accept nested eagle_config layout
# ---------------------------------------------------------------------------

EAGLE3_UTILS_PATCH = FilePatch(
    path="vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py",
    description="Eagle3 aux-layer config helper supports nested SpecForge layout",
    modifications=[
        Modification(
            description="get_eagle3_aux_layers_from_config falls back to eagle_config",
            needles=[
                "    if not hasattr(hf_config, \"eagle_aux_hidden_state_layer_ids\"):\n"
                "        return None\n"
                "    layer_ids = hf_config.eagle_aux_hidden_state_layer_ids\n"
                "    if layer_ids and isinstance(layer_ids, (list, tuple)):\n"
                "        return tuple(layer_ids)\n"
                "    return None",
            ],
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
        ),
    ],
)


# ---------------------------------------------------------------------------
# 3) gpu_model_runner: same fix in the duplicate copy of the helper.
#    The shape differs between v0.19.0 (no dflash branch) and tip-of-main
#    (with dflash branch). We support both via alternative anchors.
# ---------------------------------------------------------------------------

GPU_MODEL_RUNNER_PATCH = FilePatch(
    path="vllm/v1/worker/gpu_model_runner.py",
    description="Eagle3 aux-layer config helper supports nested SpecForge layout (runner copy)",
    modifications=[
        Modification(
            description="_get_eagle3_aux_layers_from_config falls back to eagle_config",
            needles=[
                # Tip-of-main shape (handles dflash too):
                "        layer_ids = getattr(hf_config, \"eagle_aux_hidden_state_layer_ids\", None)\n"
                "        if not layer_ids:\n"
                "            dflash_config = getattr(hf_config, \"dflash_config\", None)\n"
                "            if dflash_config and isinstance(dflash_config, dict):\n"
                "                layer_ids = dflash_config.get(\"target_layer_ids\")\n",
                # vLLM 0.19.0 shape (no dflash branching):
                "        hf_config = self.speculative_config.draft_model_config.hf_config\n"
                "        if not hasattr(hf_config, \"eagle_aux_hidden_state_layer_ids\"):\n"
                "            return None\n"
                "\n"
                "        layer_ids = hf_config.eagle_aux_hidden_state_layer_ids\n"
                "        if layer_ids and isinstance(layer_ids, (list, tuple)):\n"
                "            return tuple(layer_ids)\n"
                "\n"
                "        return None",
            ],
            replacement=(
                # We need version-specific replacements per matched anchor. The
                # cleanest way to handle that is per-anchor in __post_init__:
                # see _GPU_MODEL_RUNNER_REPLACEMENTS below.
                "<<MULTI_ANCHOR_REPLACEMENT_PLACEHOLDER>>"
            ),
            sentinel="SpecForge stores the layer ids nested under `eagle_config`",
        ),
    ],
)

# Per-anchor replacement table for the gpu_model_runner modification: each
# row maps a possible anchor text to its targeted replacement, since the two
# anchor shapes need different replacements.
_GPU_MODEL_RUNNER_REPLACEMENTS: list[tuple[str, str]] = [
    # Tip-of-main: insert eagle_config fallback BEFORE the dflash fallback.
    (
        "        layer_ids = getattr(hf_config, \"eagle_aux_hidden_state_layer_ids\", None)\n"
        "        if not layer_ids:\n"
        "            dflash_config = getattr(hf_config, \"dflash_config\", None)\n"
        "            if dflash_config and isinstance(dflash_config, dict):\n"
        "                layer_ids = dflash_config.get(\"target_layer_ids\")\n",
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
        "                layer_ids = dflash_config.get(\"target_layer_ids\")\n",
    ),
    # 0.19.0: replace the entire small block.
    (
        "        hf_config = self.speculative_config.draft_model_config.hf_config\n"
        "        if not hasattr(hf_config, \"eagle_aux_hidden_state_layer_ids\"):\n"
        "            return None\n"
        "\n"
        "        layer_ids = hf_config.eagle_aux_hidden_state_layer_ids\n"
        "        if layer_ids and isinstance(layer_ids, (list, tuple)):\n"
        "            return tuple(layer_ids)\n"
        "\n"
        "        return None",
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
        "        return None",
    ),
]

ALL_PATCHES: list[FilePatch] = [NEMOTRON_H_PATCH, EAGLE3_UTILS_PATCH, GPU_MODEL_RUNNER_PATCH]


# ---------------------------------------------------------------------------
# Apply / verify / revert
# ---------------------------------------------------------------------------


def find_vllm_root(arg_path: str | None) -> str:
    """Resolve the directory containing the vLLM Python package."""
    if arg_path:
        candidate = arg_path.rstrip(os.sep)
        if os.path.basename(candidate) != "vllm":
            candidate = os.path.join(candidate, "vllm")
        if not os.path.isdir(candidate):
            raise SystemExit(f"vLLM directory not found at {arg_path}")
        return candidate

    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise SystemExit(
            "vllm is not importable in the current Python environment. "
            "Either `pip install vllm` first or pass the path explicitly."
        )
    return os.path.dirname(spec.origin)


def _abs_path(vllm_root: str, rel_path: str) -> str:
    """Convert a 'vllm/x/y.py' relative path to an absolute path."""
    return os.path.join(vllm_root, rel_path[len("vllm/"):])


def _read(p: str) -> str:
    with open(p, encoding="utf-8") as f:
        return f.read()


def _write(p: str, content: str) -> None:
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)


def _gpu_model_runner_apply(content: str) -> tuple[str, bool]:
    """Special-case the gpu_model_runner modification due to per-anchor replacements.

    Returns (new_content, made_change).
    """
    for needle, repl in _GPU_MODEL_RUNNER_REPLACEMENTS:
        if needle in content:
            return content.replace(needle, repl, 1), True
    return content, False


def apply_patch(patch: FilePatch, vllm_root: str, dry_run: bool) -> str:
    full = _abs_path(vllm_root, patch.path)
    if not os.path.isfile(full):
        return f"  SKIP {patch.path} (file not present)"

    content = _read(full)
    original = content

    # Per-modification application
    pending: list[Modification] = []
    for mod in patch.modifications:
        if mod.is_applied(content):
            continue
        pending.append(mod)

    if not pending:
        return f"  OK   {patch.path} (already patched)"

    for mod in pending:
        # Special case: gpu_model_runner needs per-anchor replacements
        if patch is GPU_MODEL_RUNNER_PATCH:
            new_content, ok = _gpu_model_runner_apply(content)
            if not ok:
                _print_anchor_failure(patch, mod, content)
                raise AnchorNotFoundError(mod)
            content = new_content
            continue
        try:
            content = mod.apply(content)
        except AnchorNotFoundError:
            _print_anchor_failure(patch, mod, content)
            raise

    if content == original:
        return f"  OK   {patch.path} (no change)"

    if dry_run:
        delta = len(content) - len(original)
        return f"  DRY  {patch.path} (would be modified, {delta:+} bytes)"

    backup = full + ".eagle3-bak"
    if not os.path.exists(backup):
        _write(backup, original)
    _write(full, content)
    return (
        f"  OK   {patch.path} (patched, backup at {os.path.basename(backup)})"
    )


def verify_patch(patch: FilePatch, vllm_root: str) -> str:
    full = _abs_path(vllm_root, patch.path)
    if not os.path.isfile(full):
        return f"  SKIP {patch.path} (file not present)"

    content = _read(full)

    applied = sum(1 for mod in patch.modifications if mod.is_applied(content))
    total = len(patch.modifications)

    if applied == total:
        return f"  OK   {patch.path} ({applied}/{total} modifications applied)"
    if applied == 0:
        return f"  ---  {patch.path} (0/{total} modifications applied)"
    return f"  ⚠    {patch.path} ({applied}/{total} modifications applied -- partial)"


def revert_patch(patch: FilePatch, vllm_root: str, dry_run: bool) -> str:
    full = _abs_path(vllm_root, patch.path)
    backup = full + ".eagle3-bak"
    if not os.path.isfile(backup):
        return f"  SKIP {patch.path} (no backup file -- file may not have been patched by this script)"
    if dry_run:
        return f"  DRY  {patch.path} (would restore from {os.path.basename(backup)})"
    backup_content = _read(backup)
    _write(full, backup_content)
    os.remove(backup)
    return f"  OK   {patch.path} (restored from backup; backup removed)"


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _print_anchor_failure(patch: FilePatch, mod: Modification, content: str) -> None:
    print()
    print(f"!! Could not find any anchor for modification:")
    print(f"   file:        {patch.path}")
    print(f"   description: {mod.description}")
    print()
    print("   Tried these anchors (only the first line of each is shown):")
    for i, n in enumerate(mod.needles, 1):
        first_line = n.splitlines()[0]
        print(f"     [{i}] {first_line}")
    print()
    print(
        "   This usually means your vLLM version differs from the versions\n"
        "   this patch was written against. Please open an issue at\n"
        "     https://github.com/hav4ik/vllm/issues\n"
        "   with the following info attached:\n"
        f"     - the output of `python -c 'import vllm; print(vllm.__version__)'`\n"
        f"     - the relevant portion of {patch.path} (around the function above)"
    )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _detect_vllm_version() -> str:
    try:
        import vllm  # type: ignore

        return getattr(vllm, "__version__", "unknown")
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply NemotronH ↔ EAGLE-3 speculative-decoding support "
            f"(patch revision {__version__})"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Known to apply against:\n"
            + "\n".join(f"  - vLLM {v}" for v in SUPPORTED_VLLM_VERSIONS)
        ),
    )
    parser.add_argument(
        "vllm_path",
        nargs="?",
        default=None,
        help="path to the `vllm` directory inside site-packages "
        "(autodetected from `import vllm` if omitted)",
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--verify",
        action="store_true",
        help="report current patch status without modifying anything",
    )
    actions.add_argument(
        "--revert",
        action="store_true",
        help="restore original files from .eagle3-bak backups",
    )
    actions.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would change without writing",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=(
            f"%(prog)s patch revision {__version__} "
            f"(supports vLLM: {', '.join(SUPPORTED_VLLM_VERSIONS)})"
        ),
    )
    args = parser.parse_args()

    try:
        vllm_root = find_vllm_root(args.vllm_path)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 1

    detected_version = _detect_vllm_version()
    print(f"Patch revision: {__version__}")
    print(f"Detected vLLM:  {detected_version}")
    print(f"vLLM location:  {vllm_root}")
    if args.dry_run:
        print("(dry-run mode -- no files will be modified)")
    if args.verify:
        print("(verify mode -- no files will be modified)")
    if args.revert:
        print("(revert mode -- restoring from .eagle3-bak backups)")
    print()

    statuses: list[str] = []
    failed = False
    try:
        if args.verify:
            for patch in ALL_PATCHES:
                statuses.append(verify_patch(patch, vllm_root))
        elif args.revert:
            for patch in ALL_PATCHES:
                statuses.append(revert_patch(patch, vllm_root, dry_run=False))
        else:
            for patch in ALL_PATCHES:
                statuses.append(apply_patch(patch, vllm_root, dry_run=args.dry_run))
    except AnchorNotFoundError:
        failed = True

    print("Results:")
    for s in statuses:
        print(s)
    print()

    if failed:
        print(
            "Patch FAILED. See the diagnostic above. To open an issue, please "
            "include the diagnostic and the output of `python "
            "apply_nemotron3_eagle3_patch.py --verify`."
        )
        return 2

    if args.verify:
        print(
            "Verify complete. To apply the patch run the same command without "
            "--verify; to revert, use --revert."
        )
    elif args.revert:
        print("Revert complete.")
    elif args.dry_run:
        print(
            "Dry run complete. Re-run without --dry-run to apply the changes."
        )
    else:
        print(
            "Patch applied. Restart vLLM and use --speculative-config "
            "'{\"model\": <draft>, \"method\": \"eagle3\", "
            "\"num_speculative_tokens\": 5}' to enable EAGLE-3 with NemotronH."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
