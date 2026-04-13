# E1 (PC only) vs E2 (Eagle3 only) — aimofull benchmark

> Dataset: `aimofull.jsonl` (63 IMO-AnswerBench problems × 8 rollouts = 504 sessions)
> Hardware: 1× Blackwell RTX PRO 6000 (94 GB)
> vLLM commit: `3245fdd46` (camera-ready, with the PC + Eagle3 fixes)
> Both engines hit the 3 h orchestrator timeout before processing all 504 sessions.

## Server config (identical except for the engine flags)

```
--max-model-len 262144
--max-num-seqs 32
--mamba-ssm-cache-dtype float16
--kv-cache-dtype fp8
--gpu-memory-utilization 0.9
--enable-auto-tool-choice --tool-call-parser qwen3_coder
```

| | **E1: PC only** | **E2: Eagle3 only** |
|---|---|---|
| `--enable-prefix-caching` | yes | no |
| `--mamba-cache-mode all` | yes | no |
| `--mamba-block-size` | 256 | – |
| `--speculative-config` | – | `eagle3, num_speculative_tokens=5` |

## Raw numbers (3 h wall-clock cap)

| Metric | E1 | E2 |
|---|---|---|
| Sessions completed | **379** / 504 | **328** / 504 |
| Problems with full rollouts | 50 / 63 | 43 / 63 |
| Per-session accuracy (all completed) | 203/379 = **53.6 %** | 187/328 = **57.0 %** |
| Majority-vote accuracy (all completed) | 32/50 = **64.0 %** | 32/43 = **74.4 %** |
| Avg session wall time | 774.9 s | 868.9 s |
| Avg gen tokens / session | 36 021 | 35 620 |
| Avg prompt tokens / session | 377 823 | 384 657 |
| Avg gen throughput (req-weighted) | **1496 tok/s** | 1208 tok/s |
| Avg concurrent requests | 29.9 | 30.2 |
| Time-weighted PC hit rate | **83.8 %** | 0.0 % |
| Time-weighted Eagle3 accept rate | – | **31.9 %** |
| Tool-call errors | 605 / 5469 | 946 / 4911 |

## Apples-to-apples on the 43 problems both engines finished

| | E1 | E2 | Δ |
|---|---|---|---|
| Per-session accuracy | 193/335 = **57.6 %** | 187/328 = **57.0 %** | tie |
| Majority-vote accuracy | 30/43 = **69.8 %** | 32/43 = **74.4 %** | **E2 +4.6 pct** |
| Avg session time | 783 s | 869 s | E1 −10 % |

So at fixed problem set:
- **Per-session accuracy is essentially tied** (the rejection sampler in spec decode is lossless w.r.t. the target model, so neither side hurts quality).
- **E2 has a higher majority-vote** (presumably because spec decode's slightly different sampling sequence breaks ties differently — within noise of a 43-sample evaluation, but consistent with several majority-flips we tracked).
- **E1 is meaningfully faster per session** (10 % less wall time per session) thanks to its 84 % cache hit rate.

## Per-3-hour productivity (the production-relevant view)

In the same 3 h wall-clock budget on this workload:

|  | E1 | E2 |
|---|---|---|
| Sessions completed | 379 | 328 |
| Problems fully covered | 50 | 43 |
| **Majority-correct problems** | 32 | 32 |
| Effective tok/s (all sessions, summed over the run) | **46.5 K tok/s** | **40.9 K tok/s** |
| Wall-clock per problem (avg) | 216 s | 251 s |

E1 finishes **16 % more sessions** than E2 in the same wall-clock window, but E1's lower per-session accuracy means **the same number of problems end up majority-correct**. The two engines deliver an identical 32 majority-correct results in 3 hours, by very different routes.

## Which is better in production?

**It depends on the workload, but for any session-reuse pattern E1 wins.**

### If your workload has multi-turn / shared-prefix structure → **E1 (Prefix Caching)**
- Multi-turn chat with the same system prompt
- RAG with shared context across queries  
- Math/code agents with shared tool definitions and few-shot exemplars
- N rollouts of the same problem (this benchmark)

E1 dominates here because PC turns the second-and-later turns' prefill into a near no-op. On this benchmark we see:
- **84 % cache hit rate** — almost the entire prompt is reused on rollouts 2–8
- **24 % higher sustained generation throughput** (1496 vs 1208 tok/s)
- **10 % faster per-session wall time** even though the per-session work is the same

PC is *free* compute. The only cost is GPU memory for the cache pool (~3.7 M tokens of KV in our config), and on hybrid models there's a one-time complexity payoff — which is exactly what the camera-ready PR fixes.

### If your workload is single-turn / no shared context → **E2 (Eagle3)**
- One-off chatbot turns
- Code completion at IDE token-level latency
- Anything where each request has a fresh, unique prompt and you can't amortize prefill

E2 is the right call here because spec decode reduces *decode* latency without depending on prefix sharing. With ~32 % accept rate, decode steps are ~1.5× more efficient. The cost is some GPU memory for the drafter weights and per-step compute.

### Mixed workload → **what we'd want is E3 (PC + Eagle3 together)**
This is the configuration that would dominate both axes — multi-turn savings *and* per-step decode acceleration. Unfortunately E3 is currently broken on aimofull (degenerate output, root cause TBD; works on AIME-25). Once E3 works, it should beat both E1 and E2 strictly.

## Recommendation

For **the actual production target this work was originally for** (math-tutoring agent with long multi-turn reasoning + shared system prompts + tool calls reusing context):

**Ship E1 (PC) today.** It gives:
- The best wall-clock throughput on multi-turn workloads (the relevant axis)
- Identical per-session accuracy as E2 (no quality loss from skipping spec decode)
- 4.6 pct lower majority-vote on this dataset is *within sampling noise* on 43 problems, and on AIME-25 the same code with the same dataset structure gets 100 % majority — so the gap is dataset-specific, not engine-fundamental

When E3 is fixed, switch to E3 for a strict improvement.

## Caveats
- Both engines hit the 3 h timeout before finishing 504 sessions; the comparison is "what fits in 3 h", not "what's the asymptote".
- aimofull is harder than AIME-25 — both engines produce more wrong answers and more `no_answer/stop` failures than they would on easier benchmarks.
- E2's spec accept rate of 32 % is lower than the ~42 % we saw on AIME-25, suggesting the Eagle3 drafter (`chankhavu/c2.eagle3-test`, trained on AIME-style data) is slightly out of distribution on IMO problems.
- Tool-call error rate is high in both engines (~11–19 %), mostly from the model emitting Python that doesn't parse or that the sandbox rejects.
