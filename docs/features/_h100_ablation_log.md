# H100 PC + Eagle3 Ablation Log

All runs: AIME-25, 32 parallel, chankhavu/c2-softcpy-fp8 + Eagle3 drafter,
mamba-block-size 256, max-model-len 262144, kv-cache-dtype fp8.

## Accuracy comparison

| Config | SSM dtype | n | Per-session | Majority | Notes |
|---|---|---|---|---|---|
| Ref vLLM 0.19.0+cu130, PC only | fp32 | 4 | 60/60=100% (ongoing) | 19/19=100% | No Eagle3, no spec-slot path |
| Dev branch (Blackwell, PIECEWISE) | fp16 | 2 | 53/56=94.6% | 30/30=100% | Eagle3 active, mamba runs eagerly |
| Fix branch (H100, FULL capture) | fp16 | 4 | ~96/112=85.7% | 27/30=90% | Garbage eliminated but per-session regressed |
| Fix branch (H100, FULL capture) | fp32 | 4 | 98/120=81.7% | 27/30=90% | fp32 SSM didn't help — same regression |
| Fix branch (H100, enforce-eager) | fp16 | ~1 | 14/14=100% | — | Spec-slot code correct in eager mode |
| Fix branch (H100, FULL, pre-padding-fix) | fp16 | 1 | ~50% | — | Garbage tokens, state corruption |

## Key finding: _spec_inited reset causes state rewind

The `_spec_inited[num_decodes:] = False` reset (bug #5 fix) prevents
padding-write corruption but introduces a **state rewind** on every
decode → prefill → decode transition (tool-call continuations):

1. Request at position i is actively decoding (_spec_inited[i]=True)
2. Tool call triggers a new turn → request enters prefill → position i
   falls outside [0, num_decodes) for that step
3. FULL cudagraph kernel writes garbage to spec_ssm[spec_ids[i, 0..K]]
   (padding writes from captured kernel grid)
4. Reset sets _spec_inited[i] = False
5. Next decode step: eager_init_spec_slots re-initializes from POOL
   (last committed boundary state, up to block_size-1 tokens behind)
6. Kernel resumes from stale state → ~100-200 tokens of mamba progress lost

Under PIECEWISE (Blackwell): step 3 never happens (eager forward
processes exact num_decodes, no padding writes), so _spec_inited stays
True and spec slots retain correct state across tool-call transitions.

## Proper fix direction

Suppress padding writes at the kernel level: set output indices to
NULL_BLOCK_ID (-1) for padded positions so the kernel's existing
`if token_dst_idx != null_block_id` check skips the write. Then
remove the blunt _spec_inited reset entirely.

## Throughput

| Config | Single-request tok/s |
|---|---|
| PIECEWISE (FlashInfer fallback) | 73.7 |
| FULL_AND_PIECEWISE (fix branch) | 254 |
| Speedup | 3.45x |

## Failure classification (fix branch, fp16, n=4, 112 sessions)

- 11 hit max_tokens (32768) — hard problems
- 1 tool-call parser failure
- 10 wrong math (coherent, incorrect answer)
- 0 garbage/corruption
