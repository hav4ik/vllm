# Evaluation scripts for NemotronH Eagle3 + Prefix Caching

## Overview

These scripts evaluate NemotronH (chankhavu/c2-softcpy-fp8) with Eagle3
speculative decoding on the AIME-25 math benchmark. The benchmark tests
multi-turn tool-integrated reasoning (the model can call Python code via
a sandbox).

## Prerequisites

1. **vLLM server** running with NemotronH + Eagle3:
   ```bash
   cd /workspace/vllm_nemotron3_eagle3/vllm
   vllm serve chankhavu/c2-softcpy-fp8 \
     --max-model-len 65536 --trust-remote-code \
     --mamba-ssm-cache-dtype float32 --max-num-seqs 16 \
     --kv-cache-dtype fp8 --enable-prefix-caching \
     --mamba-cache-mode all --mamba-block-size 512 \
     --enable-chunked-prefill --max-num-batched-tokens 8192 \
     --enable-auto-tool-choice --tool-call-parser qwen3_coder \
     --download-dir /workspace/models --host 127.0.0.1 --port 18000 \
     --speculative-config '{"model":"chankhavu/c2.eagle3-test","method":"eagle3","num_speculative_tokens":4}'
   ```

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
  --model_name chankhavu/c2-softcpy-fp8 \
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
MODEL = 'chankhavu/c2-softcpy-fp8'
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

| Config | Per-session | Majority vote |
|--------|-------------|---------------|
| No PC + Eagle3 (baseline) | 92.9% | 96.4% |
| PC + Eagle3 (v2 fix) | 91.5% | 96.7% |
| PC only (no spec) | 95.0% | 100% |

**Note**: prefix cache hits are currently 0% when spec decode is active
(upstream issue #38182). This means the PC+Eagle3 config runs correctly
but without PC speedup for multi-turn conversations. See
`docs/features/_pc_spec_v2_final_status.md` for details.

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
