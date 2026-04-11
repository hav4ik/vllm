# Evaluation scripts for NemotronH Eagle3 + Prefix Caching

## Overview

These scripts evaluate NemotronH (`chankhavu/Nemotron-Cascade-2-30B-A3B-FP8`)
with Eagle3 speculative decoding on the AIME-25 math benchmark. The benchmark
tests multi-turn tool-integrated reasoning (the model can call Python code via
a sandbox).

## Prerequisites

1. **vLLM server** running with NemotronH + Eagle3 — use the
   *verified* config (mamba-block-size 256, not 512; see
   `docs/features/_pc_spec_v2_final_status.md` for the historical
   reasoning):
   ```bash
   HF_HOME=/workspace/.hf_home VLLM_USE_FLASHINFER_MOE_FP8=1 \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   vllm serve chankhavu/Nemotron-Cascade-2-30B-A3B-FP8 \
     --max-model-len 131072 --trust-remote-code \
     --mamba-ssm-cache-dtype float16 --max-num-seqs 16 \
     --kv-cache-dtype fp8 --enable-prefix-caching \
     --mamba-cache-mode all --mamba-block-size 256 \
     --gpu-memory-utilization 0.9 \
     --enable-auto-tool-choice --tool-call-parser qwen3_coder \
     --download-dir /workspace/models --host 127.0.0.1 --port 18000 \
     --speculative-config '{"model":"chankhavu/c2.eagle3-test","method":"eagle3","num_speculative_tokens":4}'
   ```

   On sm_120f (Blackwell RTX PRO 6000) with CUDA 13.0 you also need
   `flashinfer-jit-cache` installed — see the final-status doc.

2. **Sandbox server** for Python code execution (needed for tool calls):
   ```bash
   python eval/sandbox_server.py
   ```
   This starts on port 6000 by default.

## Running the evaluation

### Full AIME-25 with tool calls (recommended)
```bash
python eval/collect_traces_nemotron.py \
  --input_dir eval/data \
  --output_dir traces/my_run \
  --n_sessions 2 \
  --max_parallel 8 \
  --model_name chankhavu/Nemotron-Cascade-2-30B-A3B-FP8 \
  --max_tokens 32768 \
  --server_addr http://127.0.0.1:18000 \
  --metrics_url http://127.0.0.1:18000/metrics \
  --watchdog_interval 30 \
  --resume
```

**Parameters:**
- `--n_sessions N`: number of attempts per problem (default 2 for majority vote)
- `--max_parallel N`: concurrent requests to the server
- `--max_tokens N`: max completion tokens per turn
- `--resume`: skip already-completed sessions (safe to restart)

### Compute accuracy
```bash
python eval/calc_accuracy.py traces/my_run/
```

This prints per-problem results, per-session accuracy, and majority-vote accuracy.

### Quick no-tool test (single-turn, no sandbox needed)
```bash
python -c "
import asyncio, json, aiohttp, re
SYSTEM = 'You are a helpful assistant. You are not allowed to use any tools.'
SERVER = 'http://127.0.0.1:18000/v1/chat/completions'
MODEL = 'chankhavu/Nemotron-Cascade-2-30B-A3B-FP8'
problems = [json.loads(l) for l in open('eval/data/aime25.jsonl')]

async def solve(session, prob):
    async with session.post(SERVER, json={
        'model': MODEL, 'messages': [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': prob['problem']},
        ], 'max_tokens': 16384, 'temperature': 0,
    }, timeout=aiohttp.ClientTimeout(total=600)) as resp:
        data = await resp.json()
        content = data['choices'][0]['message']['content']
        matches = re.findall(r'\\\\boxed\{([^}]+)\}', content)
        pred = int(matches[-1].strip()) if matches else None
        ok = str(pred) == str(prob.get('expected_answer'))
        return f'{prob[\"id\"]:12s}: {\"OK\" if ok else \"FAIL\"} pred={pred} exp={prob.get(\"expected_answer\")}'

async def main():
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=8)) as s:
        results = await asyncio.gather(*[solve(s, p) for p in problems])
    for r in sorted(results): print(r)
    correct = sum(1 for r in results if 'OK' in r)
    print(f'{correct}/{len(results)} correct')

asyncio.run(main())
"
```

## Expected results

Verified 2026-04-11 with the `mamba-block-size 256` config and
commit `8546c2a71`, on a single 94 GB Blackwell RTX PRO 6000,
`max_parallel=4`, cudagraph PIECEWISE (no `--enforce-eager`):

| Config | Per-session | Majority vote | PC hit | Eagle3 accept |
|--------|-------------|---------------|--------|---------------|
| PC + Eagle3 (verified) | **96.7%** (58/60) | **100%** (30/30) | **79.4%** | **42.2%** |

Earlier runs with `mamba-block-size 512` showed 0% PC hits even
with the `kv_cache_coordinator` fix — the LCM-aligned block became
~4608 tokens and the Eagle block drop wiped every hit. The 256
setting is what makes multi-turn prefix caching actually work. See
`docs/features/_pc_spec_v2_final_status.md` and commit
`8546c2a71` for the rationale.

## Data format

`eval/data/aime25.jsonl` contains 30 AIME-25 problems, one per line:
```json
{
  "id": "aime25-0",
  "problem": "Find the sum of all integer bases b>9 for which ...",
  "expected_answer": "70",
  "answer_type": "nonneg_int",
  "source": "aime25"
}
```

## Trace format

Output traces are JSONL files (one per problem, multiple sessions per file):
```json
{
  "id": "aime25-0",
  "expected_answer": "70",
  "predicted_answer": 70,
  "num_completion_tokens": 1261,
  "num_prompt_tokens": 1652,
  "num_tool_calls": 1,
  "finish_reason": "answer_found",
  "conversation": [...],
  "generation": "..."
}
```
