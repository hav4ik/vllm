# Benchmark failure patterns — running notes

Live notes from the overnight `aimofull` benchmark. Updated on each self-check cycle while Engine E1/E2/E3 are running. IDs reference the problem id field from `aimofull.jsonl`.

## Legend
- **no_answer**: session ended without a `\boxed{…}` in any assistant turn
- **wrong_answer**: boxed answer was produced but didn't match expected
- **max_turns**: hit `--max_turns` without converging
- **answer_found**: finish reason when a boxed answer was present

## Observations

### E1 (PC only, no spec decode)
- **424e18:4** — `pred=62140 exp=21818 reason=answer_found` — model computed `v_5(2^{20}!) = 262140 mod 10^5 = 62140` but expected is 21818. Reasoning tail shows it wrote "v_5((2^{20}!)) because v_2 > v_5" — an internal logic slip (chose the wrong valuation: v_5 when it should have been v_2), then arithmetic was consistent with the wrong choice. Not a tool/exec error; pure reasoning slip.
- **92ba6a:7** — `pred=2 exp=50 reason=answer_found` — "Alice/Bob sweets" word problem. 15 tool calls, 27K gen tokens. The reasoning tail is visibly contorted ("under the interpretation that the two numbers become equal"), concludes ages 2 and 1, product 2. Failure mode: over-tortured interpretation of an ambiguous statement. The model found *an* interpretation that worked rather than the intended one.
- **86e8e5** — **0/8 correct** (consistent failure across rollouts). Problem involves a function `g(c)` summed over 6 specific large values, then `p+q mod 99991`. Different rollouts arrive at *different* wrong answers (23, 7, 97424, …) — the model computes a *plausible* sum of 6 terms each time but disagrees with itself across rollouts on what `g(c)` actually evaluates to. Not a single wrong branch; this problem sits in a region where the model can't reliably derive `g(c)` regardless of seed. ~50K tokens/session, ~20 tools/session — expensive failure every time. Majority vote will not save this one.
- **a295e9** — 3/8 correct, majority-wrong. Not yet inspected in detail.
- **641659** — 3/8 correct, just became majority-wrong. Not yet inspected.
- **Recurring failure mode: no_answer / rambling without committing** (`9debd6:3`, pred=None, finish_reason=stop). Tail shows model describing a pattern ("the string '10' repeats many times, with occasional '001' at positions maybe indicating a defect near boundaries") but it never produces a `\boxed{}`. The model runs out of tokens or gives up mid-analysis. Distinct from the earlier "commits to wrong answer" cases — here it just never commits.

- **⚠ Degenerate token output — possible state corruption** (`9debd6:0`, pred=None, finish_reason=stop, 19771 tokens). Tail transitions mid-sentence from coherent English ("for rows i in range(n), if i%2==0 (even rows), then for columns") into garbage tokens mixing multiple languages and fragments: "dist g for for. for . يم?? Mit?).). th). Fäh?: instance мг Arcade Cym. asks kings". This is at temperature=0 so it's deterministic. Probably either: (a) the model entering an attention-repetition-loop state, or (b) something about this specific problem's long context triggering numerical instability. Worth watching — if it only happens on 9debd6 and a few similar problems, it's a model-level pathology on tough problems. If it starts happening broadly, it's a server-side issue.

## E3 (PC + Eagle3): boot OOM at max_num_seqs=32 + num_spec=5

E3 server failed to boot at 16:21:16 with `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 52.29 GiB`. Root cause: the spec-slot allocation in `MambaMixer2.init_spec_slots` (the wheel/PR-C fix) does not fit at `max_num_seqs=32 × num_spec_tokens=5` × 23 mamba layers × `mamba_page_size_padded` (~50 MB after padding to attention LCM at `max_model_len=262144`). Total spec-slot footprint ~40-50 GB, exceeding the available headroom after model weights + Eagle drafter.

E1 and E2 both worked at `max_num_seqs=32` because neither allocates spec slots: E1 has no spec decode at all, E2 has spec decode but not `--mamba-cache-mode all` (no spec slots).

**Workaround for E3**: drop to `--max-num-seqs 16` so the spec-slot tensor halves. Comparison vs E1/E2 isn't strictly apples-to-apples on parallelism but is the only way to fit at this `max_model_len + num_spec` combo on a 94 GB GPU. A proper fix would refactor `init_spec_slots` to use a sparser layout that doesn't scale linearly with `max_num_seqs`.

## E3: degenerate output at num_speculative_tokens=5 + max_model_len=262144

After fixing the boot OOM (max_num_seqs=16, gpu_mem_util=0.85), E3 booted cleanly but the **first turn of every session produces unbalanced tool-call closing tags without an opening tag**. The model emits coherent prose reasoning, then suddenly drops `</parameter>\n</function>\n</tool_call>` at the end. The qwen3_coder tool-call parser sees this as malformed and classifies the whole thing as plain content (`num_tool_calls=0`), so the eval can't proceed past turn 2 and gives up with `no_answer/stop`.

Sample (`bench_e3_pc_eagle/0e644e:0`, 890 tokens, time 61s, ends with):
```
... Now D is very large maybe, but compute directly.
</parameter>
</function>
</tool_call>
```

This is a regression from the camera-ready verified config (`max_model_len=131072`, `num_speculative_tokens=4`). At the new config (`max_model_len=262144`, `num_speculative_tokens=5`), the PC + Eagle path produces this degenerate sequence. Either the longer context, the higher spec K, or both interact badly with the spec-slot decode path.

**Workaround**: drop E3 to `num_speculative_tokens=4` (matching the previously verified setting). E1 and E2 keep `num_speculative_tokens=5` (E2 only — E1 has no spec). Documentation note: **E2 vs E3 is no longer apples-to-apples on K** either.

## E3: degenerate output persists even at the verified config

After also dropping E3 to `--max-model-len 131072` (matching the exact config from the earlier 30/30-on-AIME-25 verified run), E3 is **still producing degenerate output on aimofull**: 1 correct / 2 wrong / 42 no_answer out of 45 sessions (93% no_answer rate). Same `</parameter></function></tool_call>` closing-tag artifact in the assistant content; same `num_tool_calls=0`.

This is a real puzzle. The same code (camera-ready commit `3245fdd46`) was verified end-to-end on AIME-25 with 58/59 per-session and 30/30 majority. The only thing different is the dataset:
- **AIME-25** (30 problems, US olympiad) → **works**
- **aimofull** (63 problems, IMO-AnswerBench subset) → **fails 93% of sessions**

Hypotheses, ranked:
1. **Long prompt + harder reasoning depth**: aimofull prompts are longer and the model spends more tokens reasoning before reaching its first tool call. The PC+Eagle path may have a corner case that only triggers after N tokens of deep reasoning (maybe state corruption that compounds, maybe a kernel that misbehaves on long generations). AIME-25's shorter problems never reach that point.
2. **Tool-call payload structure**: aimofull problems often have geometric/combinatorial setups where the model writes very long Python code blocks in `<parameter>` fields. If the spec slot path mishandles tokens from the parameter region, it could break the structured output.
3. **Drafter distribution mismatch**: the Eagle3 drafter was trained on SpecForge data; harder math problems may push the drafter into low-acceptance, high-divergence regions where the rejected token sampling exposes a bug.

**Conclusion for this benchmark run**: E3 cannot be meaningfully compared to E1/E2 on aimofull. The benchmark will report E3 as "broken — degenerate output, 93% no_answer", with the actual root-cause investigation deferred.

## E3: ROOT CAUSE — client/server parallelism oversubscription

**Found it.** Dropping the eval client from `--max_parallel 32` (against `--max-num-seqs 16` server) to `--max_parallel 16` (1:1 ratio) **completely fixes** E3. First 10 sessions on aimofull went from 0/10 correct + 93% no_answer + 0% cache hit + degenerate output → 10/10 correct + 0% no_answer + 87% cache hit + 41% spec accept.

`num_preemptions_total = 0` in both the broken and working runs, so it's NOT preemption (the hypothesis I was leaning toward).

The pattern at `p=32 against max_num_seqs=16`:
- 16 sessions running, 16 sessions queued in the API server
- Cache hit rate stays at exactly **0.0%** the entire run — the cache never engages
- Sessions emit a few hundred tokens of plausible reasoning then degenerate into unbalanced `</tool_call>` artifacts

The pattern at `p=16 against max_num_seqs=16` (1:1):
- 16 sessions running, 0 queued
- Cache hit rate **87%** (the prefix cache works as designed)
- Spec accept **41%**, gen throughput **2.3x** higher

**Probable mechanism**: with 32 parallel clients all sending sub-second-spaced requests for the same problem (8 rollouts × 4 problems = 32 nearly-identical prompts), they all hit the API server, all queue or get scheduled simultaneously, and a thundering-herd race in the scheduler prevents the cache from engaging. The first prefill races against the second, neither completes in time for the third to find a hit, etc. **Plus** something about the rapid batch composition changes (requests joining/leaving every few hundred ms) triggers the spec slot init path to read from blocks that have been freshly assigned but not yet contain valid SSM state — producing the degenerate output.

This is consistent with the "cache eviction hypothesis" but at a finer grain than preemption: it's not whole-request preemption, it's the prefix-cache LRU churning fast enough that block contents can change between when our forward path reads them and when the kernel uses them.

**Workaround for the benchmark**: drop E3's client `--max_parallel` to 16 to match server `--max-num-seqs 16`. **Production guidance**: never oversubscribe the client side beyond `max-num-seqs` when using PC + spec decode on hybrid models, until the underlying race is properly fixed in `init_spec_from_pool` / commit_boundary_states.

**Real long-term fix**: instrument the spec slot init path to either (a) wait for the source pool block to be marked "valid" before reading, or (b) reload from kernel forward instead of from pool, so eviction churn can't corrupt the spec slot.

- **Recurring failure mode: problem misreading**. Example:
  - **9e0e88:2** (pred=3 exp=498). Model ran BFS over a state space (2^16 states), found n=4 "impossible", concluded "n=3 is the only small n that works". The tail even shows the model hesitating ("I want to be sure that n=3 is indeed the largest") before committing. The actual answer is 498 — the model misread the problem entirely, answering a much simpler question than what was asked.

- **Recurring failure mode: unverified hand-wave extrapolation**. Examples:
  - **1793b2:7** (pred=12 exp=13, off-by-one). "A construction with n=12 is known (e.g., … such a family exists by combinatorial design arguments)." Asserts existence without verifying.
  - **a295e9:4** (pred=501 exp=520). "Starting at n=5, it appears that k=n+1. This pattern is achieved by constructions that use a series of 1×h rectangles … For n=500, this generalization gives k=501." Model extrapolates a small-case pattern to n=500 without proving the formula holds at that scale.
  - Common: the model observes a few small-case values, guesses a formula, commits to it. The actual answer deviates from the guessed formula in the large-case regime. Tool use (sympy, python) typically isn't invoked to *verify* the extrapolation at the target size.
