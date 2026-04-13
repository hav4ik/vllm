# Eagle3 throughput analysis on NemotronH (H100)

## Architecture

NemotronH has 52 layers:
- 23 mamba layers (recurrent: O(1) per token, no seq_len scaling)
- ~6 real attention layers (O(K × seq_len) per token)
- Remaining layers are MoE MLP-only
- Active parameters per token much less than 30B due to MoE

## Benchmark: pure generation, no tools, no PC reuse

AIME-25 problems, temperature 1.0, max-model-len 131072,
single H100, `--no-enable-chunked-prefill`.

### Per-request throughput (tok/s)

| Concurrency | Eagle3 (K=4) | No-Eagle | Ratio |
|-------------|-------------|----------|-------|
| p=1         | 158         | 294      | 0.54x |
| p=2         | 143         | 266      | 0.54x |

### Aggregate throughput (tok/s)

| Concurrency | Eagle3 (K=4) | No-Eagle | Ratio |
|-------------|-------------|----------|-------|
| p=1         | 151         | 295      | 0.51x |
| p=2         | 229         | 462      | 0.50x |

Eagle3 is ~2x SLOWER than no-Eagle at all tested concurrency levels.

## Spec decode acceptance

Mean acceptance length: 3.11 tokens/step (out of K+1=5).
Per-position acceptance: 74.2%, 52.6%, 37.7%, 27.2%, 19.6%.
Overall acceptance rate: ~42%.

## Why Eagle3 doesn't help

The speculative decoding premise is: verifying K tokens costs roughly
the same as generating 1, because the forward pass is weight-reading
dominated (memory-bandwidth bound). This should hold well for
NemotronH because:

- Mamba layers are recurrent (O(1) per token)
- Only ~6 attention layers scale with seq_len
- MoE means small active parameter set per token

Yet Eagle3 is 2x slower. Possible explanations:

1. **Drafter overhead**: Eagle3 drafter runs K=4 sequential forward
   passes per step to propose tokens. Even at ~400MB, 4 sequential
   calls add up relative to the fast MoE target forward.

2. **Spec decode pipeline overhead**: Rejection sampling, token
   management, metadata construction for K+1 tokens per request.

3. **Low acceptance rate**: 42% at T=1.0 means 58% of drafted
   tokens are wasted compute. At T=0.6 or greedy, acceptance would
   be higher.

4. **CUDA graph overhead**: The captured graph processes K+1=5
   tokens per request. Even if the compute is similar, the graph
   replay overhead for larger batches may not scale linearly.

## Recommendation

For Kaggle AIME (8 parallel sessions, long-context tool use):
- **Use PC-only** (no Eagle3): 1170 tok/s aggregate, 99.2% accuracy
- Eagle3 is a net loss at all concurrency levels tested

Eagle3 might help with:
- Lower temperature (higher acceptance)
- Short-context generation (KV cache small → attention cheap)
- Different drafter with higher acceptance rate
- K=1 or K=2 (lower overhead, higher per-position acceptance)

## TODO

- Test K=1 and K=2 to find breakeven point
- Test T=0.6 for acceptance rate improvement
- Profile drafter forward time vs target forward time
- Test with `--max-num-seqs 32` (higher concurrency favors spec decode)
