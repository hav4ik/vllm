# Verified Results — Eagle3 + Prefix Caching on NemotronH (H100)

Branch: `stable/eagle3-pc-verified`
Commit: `6285b3cebb5`
Wheel: `vllm-0.1.dev19+g6285b3ceb.cu129-cp312-cp312-linux_x86_64.whl`

## AIME-25 Accuracy

| Config | Per-session | Majority-vote | Notes |
|--------|-------------|---------------|-------|
| T=1.0, top_p=0.95, fp32 SSM, n=2 | **58/59 = 98.3%** | **30/30 = 100%** | Recommended |
| T=0.0, fp32 SSM, n=2 | **56/58 = 96.6%** | **29/30 = 96.7%** | 1 problem hit context limit |

Eval script: `/tmp/vllm_pc_v2_dev/eval/collect_traces_nemotron.py`
Accuracy script: `/tmp/vllm_pc_v2_dev/eval/calc_accuracy.py`

## Throughput (p=8, AIME-25 hard, vLLM engine metrics, last 2 min of 3)

| Config | Mean tok/s | Median tok/s | vs PC-only |
|--------|-----------|-------------|------------|
| **PC-only (no Eagle3)** | **1293** | **1294** | baseline |
| **Upstream vLLM 0.19.0 PC-only** | **1279** | **1277** | -1% |
| **K=4 T=0.0** | **1682** | **1683** | **+30%** |
| **K=4 T=0.7** | **1555** | **1560** | **+20%** |
| **K=7 T=0.7** | **1497** | **1499** | **+16%** |

All throughput numbers from fresh server (no prefix cache).

## Server Launch

```bash
python -m vllm.entrypoints.openai.api_server \
    --model chankhavu/c2-softcpy-fp8 --max-model-len 131072 --trust-remote-code \
    --mamba-ssm-cache-dtype float32 --max-num-seqs 32 --kv-cache-dtype fp8 \
    --enable-prefix-caching --mamba-cache-mode all --mamba-block-size 256 \
    --gpu-memory-utilization 0.85 --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder --download-dir /workspace/models \
    --host 127.0.0.1 --port 18000 \
    --speculative-config '{"model":"chankhavu/c2.eagle3-test","method":"eagle3","num_speculative_tokens":5}'
```

Use `--mamba-ssm-cache-dtype float32` for accuracy, `float16` for throughput.

## What's in this branch

Built on `fix/h100-flashattn-spec-init-capture-safe` (all upstream correctness
fixes for FULL cudagraph + hybrid mamba spec decode) plus:

### Performance optimizations
- Masked Triton commit kernel — eliminates `.nonzero()` D2H sync in deferred
  mamba state commit
- Fast decode attn_meta — skips 26/27 `update_block_table()` calls on pure
  decode steps

### Correctness fixes (new, not in fix/ branch)
- **Spec slot swap on batch reorder** — `_may_reorder_batch` now swaps
  spec_ssm/spec_conv/_spec_inited when batch positions are swapped during
  tool-call transitions
- **Conv state commit from base slot** — `batch_commit_kernel` uses
  `src_slot_conv=base` (slot 0) for conv state, not `base + cand_idx`
  (conv kernel only writes rolling-window state to slot 0)
- **Streaming request _spec_inited reset** — tool-call continuation path
  resets the flag before removing the request, matching finished-request
  behavior

### Experiments tried (not included)
- FULL CUDA graph for drafter decode loop: -9% to -24% regression at p=8
  due to GPU pipeline stall from graph replay blocking `commit_block_table`
- Removing `sample_complete_event.synchronize()`: partial improvement for
  T=0.0 but worse for T=0.7
