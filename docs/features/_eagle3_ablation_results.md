# Eagle3 Spec Decode Ablation Results on NemotronH (H100)

## Setup

- **Model**: chankhavu/c2-softcpy-fp8 (NemotronH 30B, 3B active MoE, FP8)
- **Drafter**: chankhavu/c2.eagle3-test (Eagle3 Llama, 1 layer, BF16)
- **Hardware**: Single H100 80GB
- **KV cache**: FP8, prefix caching enabled, mamba_cache_mode=all
- **SSM state**: float16, block_size=256
- **max_model_len**: 131072
- **Benchmark**: AIME-25 hardest 8 problems, 4 sessions per problem, with tool calls (sandbox code execution), enable_thinking=true
- **Methodology**: 3 minutes per config, stats from last 2 minutes (1 min warmup)
- **Branch**: `opt/masked-triton-commit` @ `a82d3041ea`

## Aggregate Throughput (tok/s)

| Config | p=1 | p=8 | p=12 |
|--------|-----|-----|------|
| **PC-only (no Eagle3)** | **303** | **1186** | **1383** |
| Eagle3 K=4 T=1.0 | 298 | 1368 | 1843 |
| Eagle3 K=4 T=0.9 | — | 1424 | — |
| Eagle3 K=4 T=0.8 | — | 1496 | — |
| Eagle3 K=4 T=0.7 | 349 | 1552 | 1837 |
| Eagle3 K=7 T=1.0 | 325 | 1349 | 1815 |
| Eagle3 K=7 T=0.7 | 355 | 1458 | 1885 |

## Speedup vs PC-only

| Config | p=1 | p=8 | p=12 |
|--------|-----|-----|------|
| Eagle3 K=4 T=1.0 | -2% | +15% | +33% |
| Eagle3 K=4 T=0.9 | — | +20% | — |
| Eagle3 K=4 T=0.8 | — | +26% | — |
| Eagle3 K=4 T=0.7 | +15% | +31% | +33% |
| Eagle3 K=7 T=1.0 | +7% | +14% | +31% |
| Eagle3 K=7 T=0.7 | +17% | +23% | +36% |

## Temperature Sweep (K=4, p=8)

| T | tok/s | vs PC-only | Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 | Avg accept |
|---|-------|------------|-------|-------|-------|-------|-------|------------|
| 0.7 | **1552** | **+31%** | 73.3% | 49.8% | 34.5% | 24.7% | 17.3% | 39.9% |
| 0.8 | 1496 | +26% | 73.8% | 51.8% | 35.8% | 25.1% | 18.1% | 40.9% |
| 0.9 | 1424 | +20% | 69.3% | 47.1% | 31.9% | 22.9% | 16.4% | 37.5% |
| 1.0 | 1368 | +15% | 69.2% | 47.5% | 33.5% | 23.6% | 17.1% | 38.2% |

T=0.8 is a strong middle ground: only 4% less throughput than T=0.7
(1496 vs 1552) while preserving more output diversity — important for
best-of-N sampling on hard competition problems. T=0.7-0.8 is the
recommended range for Kaggle deployment.

## Per-Position Acceptance Rate

### K=4 (5 draft positions)

| Config | Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 | Avg |
|--------|-------|-------|-------|-------|-------|-----|
| K=4 T=1.0 p=8 | 69.2% | 47.5% | 33.5% | 23.6% | 17.1% | 38.2% |
| K=4 T=1.0 p=12 | 68.7% | 47.4% | 32.5% | 23.4% | 17.8% | 38.0% |
| K=4 T=0.7 p=8 | 73.3% | 49.8% | 34.5% | 24.7% | 17.3% | 39.9% |
| K=4 T=0.7 p=12 | 75.0% | 54.7% | 40.4% | 29.4% | 22.2% | 44.3% |

### K=7 (8 draft positions)

| Config | Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 | Pos 6 | Pos 7 | Pos 8 | Avg |
|--------|-------|-------|-------|-------|-------|-------|-------|-------|-----|
| K=7 T=1.0 p=8 | 67.2% | 42.9% | 28.8% | 20.0% | 13.0% | 9.1% | 6.4% | 4.3% | 24.0% |
| K=7 T=1.0 p=12 | 68.9% | 46.1% | 32.4% | 22.7% | 16.3% | 10.9% | 8.0% | 5.9% | 26.4% |
| K=7 T=0.7 p=8 | 72.1% | 50.2% | 35.3% | 24.8% | 17.4% | 12.4% | 9.1% | 6.5% | 28.4% |
| K=7 T=0.7 p=12 | 66.3% | 46.3% | 33.9% | 24.9% | 18.9% | 14.6% | 11.2% | 8.3% | 28.1% |

## Analysis

### Temperature effect
- T=0.7→0.8 is a small throughput gap (~4%) with better output diversity
- T=0.8→0.9 is a larger gap (~5%) — the sweet spot boundary
- T=0.9→1.0 shows little difference in acceptance but throughput drops
- **Recommended: T=0.8 for competitions** (good throughput + diverse answers)

### K=4 vs K=7
- K=4 wins at p=8 (1368-1552 vs 1349-1458 tok/s) — drafter overhead dominates
- K=7 competitive at p=12 (1815-1885 vs 1837-1843 tok/s) — more tokens/step amortizes overhead
- K=7 positions 6-8 have <15% acceptance — diminishing returns beyond K=5

### Parallelism scaling
- Eagle3 scales better than PC-only with parallelism:
  - PC-only p=8→p=12: +17% (1186→1383)
  - Eagle3 K=4 T=1.0 p=8→p=12: +35% (1368→1843)
- At p=12, all Eagle3 configs beat PC-only by 31-36%

### Recommendation for Kaggle (8 parallel sessions)
- **Best throughput**: K=4, T=0.7 — **1552 tok/s** (+31% vs PC-only)
- **Best diversity/throughput tradeoff**: K=4, T=0.8 — **1496 tok/s** (+26% vs PC-only)
- K=4 preferred over K=7 at p=8 due to lower drafter overhead

## Per-step Profiling (K=4, p=1)

| Component | No-Eagle | Eagle3 (optimized) | Overhead |
|-----------|----------|-------------------|----------|
| update | 0.10ms | 0.47ms | +0.37ms |
| prep | 0.62ms | 1.72ms | +1.10ms |
| attn_meta | 0.67ms | 1.40ms | +0.73ms |
| other_pre | 1.18ms | 0.79ms | -0.39ms |
| sample | 0.15ms | 0.50ms | +0.35ms |
| draft | 0.00ms | 3.15ms | +3.15ms |
| **Total step** | **3.3ms** | **8.9ms** | +5.6ms |
| **tok/step** | 1.0 | ~2.7 | +1.7 |
| **tok/s** | 303 | 298 | -2% |

## Optimizations Applied

1. **Masked Triton commit kernel**: Eliminates `.nonzero()` D2H sync in deferred mamba state commit. Uses per-request boolean mask; kernel early-exits for non-boundary requests.

2. **Remove deferred commit sync**: The `sample_complete_event.synchronize()` was vestigial — CUDA stream ordering guarantees GPU work completes in order. Saved ~1.0ms/step.

3. **Fast decode attn_meta**: Skip 26 of 27 `update_block_table()` calls on pure decode steps. Only `state_indices_tensor_d` differs per mamba layer; all other fields are shared persistent buffers. Saved ~0.3ms/step.

## SGLang v0.5.10 Comparison

| Engine | Config | p=1 tok/s |
|--------|--------|-----------|
| **vLLM (this branch)** | PC-only | **303** |
| **vLLM (this branch)** | Eagle3 K=4 T=0.7 | **349** |
| SGLang v0.5.10.post1 | PC-only (bf16 kv) | 136 |
| SGLang v0.5.10.post1 | PC-only (fp8 kv) | 131 |

vLLM is **2.2× faster** than SGLang on NemotronH.
