@AGENTS.md

# Project: NemotronH Eagle3 + Prefix Caching

See `/workspace/vllm_nemotron3_eagle3/CLAUDE.md` for the full project context.

## Quick context for this repo

This is `hav4ik/vllm` fork. Current branch: `pc-spec-v2-nonpc-slots`.

**What we fixed**: mamba state corruption when prefix caching + speculative decoding are both enabled on NemotronH hybrid model. The fix uses K+1 dedicated spec slots per request for both SSM and conv kernels, with native `init_token_idx` rollback. AIME-25 accuracy: 96.7% majority (matches 96.4% baseline).

**What's broken**: prefix cache hits are 0% when spec decode is active (upstream #38182/#31920). The fix is correct but PC provides no speedup.

**Key files**:
- `vllm/model_executor/layers/mamba/mamba_mixer2.py` — spec slot allocation, decode path routing
- `vllm/v1/worker/gpu_model_runner.py` — init/finish/condense hooks, boundary commit
- `docs/features/_pc_spec_v2_final_status.md` — full status doc (START HERE)
- `docs/features/_pc_spec_decode_ablation_results.md` — all experimental results
