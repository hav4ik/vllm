#!/usr/bin/env python3
"""
Collect reasoning traces from Nemotron-Cascade-2-30B.

Reads a folder of JSONL datasets (keys: id, problem, expected_answer, source,
source_id), generates N reasoning sessions per problem using the OpenAI v1 Chat
Completions API with native function calling, and saves full conversation traces
as individual JSONL files named <id>-<attempt>.jsonl.

The user prompt template is selected based on the expected answer type:
  - Non-negative integer 0-99999: "The answer is expected to be an integer."
  - Any integer: "The answer is expected to be an integer."
  - Symbolic/other: just "Put your final answer in \\boxed{}"

Requires:
  - Model server running at SERVER_ADDR (OpenAI-compatible /v1/chat/completions)
  - Sandbox server (sandbox_server.py) running for Python code execution

Usage:
  python big_inference_run/scripts/collect_traces_nemotron.py \
      --input_dir big_inference_run/data/ \
      --output_dir big_inference_run/traces/ \
      --n_sessions 8 \
      --max_parallel 16
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import requests
from openai import OpenAI

# Import sandbox client from local scripts directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sandbox_client import LocalSandbox


LOG = logging.getLogger(__name__)

SANDBOX_PREIMPORT = (
    "import math\n"
    "import numpy\n"
    "import sympy\n"
    "import itertools\n"
    "import collections\n"
    "import mpmath\n"
    "mpmath.mp.dps = 64\n"
)

SYSTEM_PROMPT = "You are a helpful assistant and a IMO-level math problem solver."

# --- User prompt templates ---
# Symbolic / general answer (default)
USER_PROMPT_SYMBOLIC = "{problem}\n\nPut your final answer in \\boxed{{}}"
# Integer answer (any integer, including negative)
USER_PROMPT_INTEGER = "{problem}\n\nThe answer is expected to be an integer. Put your final answer in \\boxed{{}}"
# Non-negative integer 0-99999
USER_PROMPT_NONNEG_INT = "{problem}\n\nThe answer is expected to be an integer from 0 to 99999, inclusive. Put your final answer in \\boxed{{}}"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "stateful_python_code_exec",
        "description": (
            "Call this function to execute Python code in a stateful Jupyter notebook environment. "
            "Python will respond with the output of the execution or time out after 120.0 seconds."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Code to execute",
                }
            },
            "required": ["code"],
        },
    },
}]


# ---------------------------------------------------------------------------
# Answer type classification
# ---------------------------------------------------------------------------

def classify_answer_type(expected_answer: str) -> str:
    """Classify the expected answer as 'nonneg_int', 'integer', or 'symbolic'.

    Returns:
        'nonneg_int' — non-negative integer in [0, 99999]
        'integer'    — any integer (including negative or > 99999)
        'symbolic'   — everything else (fractions, expressions, etc.)
    """
    s = expected_answer.strip()
    # Check if it's an integer (possibly negative)
    try:
        val = int(s)
        if 0 <= val <= 99999:
            return "nonneg_int"
        return "integer"
    except ValueError:
        pass
    return "symbolic"


def get_user_prompt(problem: str, answer_type: str) -> str:
    """Return the formatted user prompt for the given answer type."""
    if answer_type == "nonneg_int":
        return USER_PROMPT_NONNEG_INT.format(problem=problem)
    elif answer_type == "integer":
        return USER_PROMPT_INTEGER.format(problem=problem)
    else:
        return USER_PROMPT_SYMBOLIC.format(problem=problem)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _extract_boxed(text: str) -> list[str]:
    """Extract all \\boxed{...} contents, handling nested braces."""
    results = []
    for m in re.finditer(r'\\boxed\s*\{', text):
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[start:i-1].strip())
    return results


def _scan_for_answer_kaggle(text: str) -> int | None:
    """Extract integer answer (0-99999) from text. Kaggle competition format."""
    # Try \boxed{DIGITS} first
    pattern = r'\\boxed\s*\{\s*([0-9,]+)\s*\}'
    matches = re.findall(pattern, text)
    if matches:
        try:
            value = int(matches[-1].replace(',', ''))
            if 0 <= value <= 99999:
                return value
        except ValueError:
            pass

    # Fallback: \boxed{...expr = DIGITS}
    boxed_contents = re.findall(r'\\boxed\s*\{([^}]+)\}', text)
    if boxed_contents:
        nums = re.findall(r'(\d[\d,]*)', boxed_contents[-1])
        if nums:
            try:
                value = int(nums[-1].replace(',', ''))
                if 0 <= value <= 99999:
                    return value
            except ValueError:
                pass

    pattern = r'final\s+answer\s+is\s*([0-9,]+)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches:
        try:
            value = int(matches[-1].replace(',', ''))
            if 0 <= value <= 99999:
                return value
        except ValueError:
            pass

    return None


def _scan_for_answer_generic(text: str) -> str | None:
    """Extract the last \\boxed{...} content from text. Returns raw string."""
    matches = _extract_boxed(text)
    if matches:
        return matches[-1]
    # Fallback: "final answer is ..."
    m = re.search(r'final\s+answer\s+is\s*[:\s]*(.+?)(?:\.|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Single session runner
# ---------------------------------------------------------------------------

async def run_session(
    *,
    problem_id: str,
    problem: str,
    expected_answer: str,
    source: str,
    source_id: str,
    answer_type: str,
    session_index: int,
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    tools: list[dict],
    sandbox_host: str,
    sandbox_port: int,
    sampling_params: dict,
    max_tokens: int,
    max_turns: int,
    python_timeout: int,
    max_output_characters: int,
    enable_thinking: bool,
    verbose: bool,
) -> dict:
    """Run a single reasoning session. Returns a trace dict."""
    start_time = time.time()
    finish_reason = "max_turns"
    final_answer = None
    total_completion_tokens = 0
    total_prompt_tokens = 0
    python_calls = 0
    python_errors = 0

    ipython_session_id = uuid4().hex
    sandbox = LocalSandbox(host=sandbox_host, port=sandbox_port)

    user_content = get_user_prompt(problem, answer_type)

    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    turn_timestamps = []

    try:
        # Pre-import packages in sandbox
        try:
            await sandbox.execute_code(
                generated_code=SANDBOX_PREIMPORT,
                session_id=ipython_session_id,
                timeout=60,
            )
        except Exception:
            LOG.debug(f"[{problem_id}:{session_index}] sandbox pre-import failed")

        extra_body = {"skip_special_tokens": False}
        if enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}

        # Split sampling params into OpenAI-standard and vLLM-extension
        _OPENAI_PARAMS = {"temperature", "top_p"}
        openai_sampling = {k: v for k, v in sampling_params.items() if k in _OPENAI_PARAMS}
        vllm_sampling = {k: v for k, v in sampling_params.items() if k not in _OPENAI_PARAMS}
        extra_body.update(vllm_sampling)

        for turn in range(max_turns):
            # Cap completion tokens to remaining session budget.
            # If max_tokens <= 0, don't pass max_completion_tokens at all
            # and let the server use max_model_len - prompt_len, which
            # avoids 400 errors when prompt + max_tokens > max_model_len.
            remaining_tokens = max_tokens - total_completion_tokens
            if remaining_tokens <= 0:
                finish_reason = "token_limit"
                break

            # Call Chat Completions API (non-streaming)
            turn_start = time.time()
            turn_completion_tokens = 0
            turn_prompt_tokens = 0
            try:
                completion_kwargs = dict(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    **openai_sampling,
                    **({"extra_body": extra_body} if extra_body else {}),
                )
                if max_tokens > 0:
                    completion_kwargs["max_completion_tokens"] = remaining_tokens
                response = client.chat.completions.create(**completion_kwargs)
            except Exception as e:
                LOG.error(f"[{problem_id}:{session_index}] API error: {e}")
                finish_reason = "error"
                break

            # Extract response
            choice = response.choices[0] if response.choices else None
            if not choice:
                finish_reason = "error"
                break

            assistant_content = choice.message.content or ""
            reasoning_content = getattr(choice.message, 'reasoning_content', None) or ""
            finish_reason_chunk = choice.finish_reason

            # Extract tool calls
            tool_calls = []
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tool_calls.append({
                        "id": tc.id or f"call_{uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    })

            # Token usage
            if response.usage:
                turn_completion_tokens = response.usage.completion_tokens or 0
                turn_prompt_tokens = response.usage.prompt_tokens or 0

            total_completion_tokens += turn_completion_tokens
            total_prompt_tokens += turn_prompt_tokens

            turn_end = time.time()
            turn_timestamps.append({
                "turn": turn,
                "role": "assistant",
                "start_time": round(turn_start, 3),
                "end_time": round(turn_end, 3),
                "duration": round(turn_end - turn_start, 3),
                "completion_tokens": turn_completion_tokens,
                "prompt_tokens": turn_prompt_tokens,
            })

            # Build assistant message for conversation history
            assistant_msg = {"role": "assistant", "content": assistant_content or None}
            if reasoning_content:
                assistant_msg["reasoning_content"] = reasoning_content
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            # Handle tool calls
            if tool_calls:
                tool_start = time.time()
                for tc in tool_calls:
                    python_calls += 1
                    func_args_str = tc["function"]["arguments"]

                    code = ""
                    try:
                        func_args = json.loads(func_args_str)
                        code = func_args.get("code", "")
                    except json.JSONDecodeError:
                        code = func_args_str

                    if code:
                        try:
                            tool_output, _ = await sandbox.execute_code(
                                generated_code=code,
                                session_id=ipython_session_id,
                                timeout=python_timeout,
                                max_output_characters=max_output_characters,
                            )
                            stdout = tool_output.get("stdout", "")
                            stderr = tool_output.get("stderr", "")
                            if stderr:
                                result = f'{stdout.rstrip()}\n{stderr}' if stdout else stderr
                            else:
                                result = stdout if stdout.strip() else '[WARN] No output. Use print() to see results.'
                        except Exception as e:
                            result = f"[ERROR] {type(e).__name__}: {e}"
                    else:
                        result = "[ERROR] No code provided."

                    if result.startswith('[ERROR]') or 'Traceback' in result or 'Error' in result:
                        python_errors += 1

                    messages.append({
                        "role": "tool",
                        "name": tc["function"]["name"],
                        "tool_call_id": tc["id"],
                        "content": result,
                    })

                tool_end = time.time()
                turn_timestamps.append({
                    "turn": turn,
                    "role": "tool",
                    "start_time": round(tool_start, 3),
                    "end_time": round(tool_end, 3),
                    "duration": round(tool_end - tool_start, 3),
                    "num_calls": len(tool_calls),
                })
                continue

            # No tool calls — check for answer
            _scan = _scan_for_answer_kaggle if answer_type == "nonneg_int" else _scan_for_answer_generic
            text_to_check = _strip_thinking(assistant_content) if assistant_content else ""
            answer = _scan(text_to_check)
            if answer is None and assistant_content:
                answer = _scan(assistant_content)
            if answer is not None:
                final_answer = answer
                finish_reason = "answer_found"
            elif finish_reason_chunk == "stop":
                finish_reason = "stop"
            else:
                finish_reason = "no_answer"
            break

    except Exception as exc:
        LOG.error(f"[{problem_id}:{session_index}] Session error: {exc}")
        finish_reason = "error"
    finally:
        try:
            await sandbox.delete_session(ipython_session_id)
        except Exception:
            pass
        try:
            await sandbox.close()
        except Exception:
            pass

    generation_end_time = time.time()
    generation_time = generation_end_time - start_time

    generation_text = ""
    for m in messages:
        if m.get("role") == "assistant" and m.get("content"):
            generation_text += m["content"]

    return {
        "id": problem_id,
        "source": source,
        "source_id": source_id,
        "problem": problem,
        "expected_answer": expected_answer,
        "answer_type": answer_type,
        "session_index": session_index,
        "predicted_answer": final_answer,
        "num_completion_tokens": total_completion_tokens,
        "num_prompt_tokens": total_prompt_tokens,
        "finish_reason": finish_reason,
        "num_tool_calls": python_calls,
        "num_tool_errors": python_errors,
        "conversation": messages,
        "tools": tools,
        "generation_start_time": start_time,
        "generation_end_time": generation_end_time,
        "generation_time": round(generation_time, 2),
        "generation": generation_text,
        "turn_timestamps": turn_timestamps,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_WRITE_LOCK = threading.Lock()


def _append_result(output_dir: Path, result: dict):
    """Append a single result to the per-problem JSONL file (thread-safe)."""
    outfile = output_dir / f"{result['id']}.jsonl"
    line = json.dumps(result, ensure_ascii=False) + "\n"
    with _WRITE_LOCK:
        with open(outfile, "a") as f:
            f.write(line)


def _existing_session_indices(output_dir: Path, problem_id: str) -> set[int]:
    """Return set of completed session indices for a problem."""
    outfile = output_dir / f"{problem_id}.jsonl"
    if not outfile.exists():
        return set()
    indices = set()
    with open(outfile) as f:
        for line in f:
            line = line.strip()
            if line:
                indices.add(json.loads(line)["session_index"])
    return indices


def _worker(kwargs):
    """Thread worker: runs async session in its own event loop."""
    return asyncio.run(run_session(**kwargs))


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

def check_vllm_health(server_url: str, api_key: str = "", timeout: float = 10) -> str | None:
    """Ping vLLM /v1/models. Returns error string or None if healthy."""
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = requests.get(f"{server_url}/models", headers=headers, timeout=timeout)
        if resp.status_code != 200:
            return f"vLLM returned status {resp.status_code}"
        return None
    except Exception as e:
        return f"vLLM unreachable: {e}"


def check_sandbox_health(host: str, port: int, timeout: float = 120) -> str | None:
    """Execute trivial code in sandbox. Returns error string or None if healthy."""
    async def _check():
        sandbox = LocalSandbox(host=host, port=port)
        try:
            session_id = f"healthcheck_{uuid4().hex[:8]}"
            output, _ = await sandbox.execute_code(
                generated_code="print('ok')",
                session_id=session_id,
                timeout=timeout,
            )
            try:
                await sandbox.delete_session(session_id)
            except Exception:
                pass
            stdout = output.get("stdout", "").strip()
            if stdout != "ok":
                return f"sandbox returned unexpected output: {stdout!r}"
            return None
        finally:
            await sandbox.close()

    try:
        return asyncio.run(_check())
    except Exception as e:
        return f"sandbox unreachable: {e}"


def run_health_checks(server_urls: list[str], api_key: str, sandbox_host: str, sandbox_port: int) -> str | None:
    """Run all health checks. Returns first error or None if all healthy."""
    for url in server_urls:
        err = check_vllm_health(url, api_key=api_key)
        if err:
            return f"{url}: {err}"
    err = check_sandbox_health(sandbox_host, sandbox_port)
    if err:
        return err
    return None


def _route_to_server(problem_id: str, num_servers: int) -> int:
    """Deterministically route a problem to a server index."""
    if num_servers == 1:
        return 0
    # Try to interpret first 16 chars as hex (canonical UUID-like ids); fall
    # back to a stable Python hash for human-readable ids like "aime25-0".
    try:
        return int(problem_id[:16], 16) % num_servers
    except ValueError:
        import zlib
        return zlib.crc32(problem_id.encode()) % num_servers


_ABORT_EVENT = threading.Event()

# Shared progress counters (updated by main thread, read by watchdog)
_PROGRESS = {"completed": 0, "total": 0, "errors": 0, "sandbox_calls": 0, "sandbox_errors": 0,
             "correct": 0, "problems_done": 0, "problems_correct": 0}
_PROGRESS_LOCK = threading.Lock()


def _fetch_vllm_metrics(url: str) -> dict[str, float]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            text = resp.read().decode()
    except Exception:
        return {}
    metrics = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r'^(\S+?)(?:\{([^}]*)\})?\s+(\S+)', line)
        if m:
            key, labels, val = m.group(1), m.group(2) or "", m.group(3)
            try:
                v = float(val)
            except ValueError:
                continue
            full = f"{key}{{{labels}}}" if labels else key
            metrics[full] = v
            if key not in metrics:
                metrics[key] = v
    return metrics


def _get_metric(m: dict, name: str, labels: str = "") -> float:
    # Try the original name, then with ':' replaced by '_', then with vllm: -> sglang:
    candidates = [name, name.replace(":", "_")]
    if name.startswith("vllm:"):
        sgl_name = name.replace("vllm:", "sglang:", 1)
        candidates += [sgl_name, sgl_name.replace(":", "_")]
    for n in candidates:
        if labels:
            for k, v in m.items():
                if k.startswith(n + "{") and labels in k:
                    return v
        else:
            # plain match (no labels)
            if n in m:
                return m[n]
            # also accept any labelled variant
            for k, v in m.items():
                if k.startswith(n + "{"):
                    return v
    return 0.0


def _human(n: float) -> str:
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return f"{n:.0f}"


def _watchdog(server_urls: list[str], api_key: str, sandbox_host: str, sandbox_port: int,
              metrics_urls: list[str] = None, interval: float = 30,
              health_check_interval: float = 300):
    """Background thread: progress + vLLM metrics table + periodic health checks."""
    metrics_urls = metrics_urls or []

    HEADER = (
        f"{'Time':<9} "
        f"{'Done':>12} "
        f"{'Acc':>10} "
        f"{'Sandbox':>12} "
        f"{'Gen t/s':>8} "
        f"{'Pfx t/s':>8} "
        f"{'Reqs':>6} "
        f"{'Run':>4} "
        f"{'Wait':>4} "
        f"{'Cache%':>7} "
        f"{'Cached':>9} "
        f"{'Spec%':>6} "
        f"{'KV%':>6} "
        f"{'KV used/total':>16}"
    )
    SEP = "─" * len(HEADER)

    # CSV setup
    csv_path = "vllm-watchdog-live.csv"
    CSV_HEADER = "timestamp,done,total,sb_errors,sb_calls,gen_tps,pfx_tps,reqs,running,waiting,cache_hit_pct,cached_tokens,spec_accept_pct,kv_usage_pct,kv_used_tokens,kv_total_tokens\n"
    csv_file = open(csv_path, "a")
    if csv_file.tell() == 0:
        csv_file.write(CSV_HEADER)
        csv_file.flush()

    prev_prompt = prev_gen = prev_reqs = prev_cache_q = prev_cache_h = None
    prev_spec_draft = prev_spec_accept = None
    prev_sb_calls = prev_sb_errors = 0
    prev_time = None
    row_count = 0
    total_blocks = None
    block_size = None
    last_health_check = time.time()

    while not _ABORT_EVENT.is_set():
        _ABORT_EVENT.wait(interval)
        if _ABORT_EVENT.is_set():
            break

        now = time.time()

        # Periodic health check
        if now - last_health_check >= health_check_interval:
            err = run_health_checks(server_urls, api_key, sandbox_host, sandbox_port)
            if err:
                LOG.error(f"HEALTH CHECK FAILED: {err}")
                LOG.error("Aborting — no new tasks will start. In-flight tasks will finish.")
                _ABORT_EVENT.set()
                break
            last_health_check = now

        # Progress counters
        with _PROGRESS_LOCK:
            done = _PROGRESS["completed"]
            total_tasks = _PROGRESS["total"]
            cur_sb_calls = _PROGRESS["sandbox_calls"]
            cur_sb_errors = _PROGRESS["sandbox_errors"]
            correct = _PROGRESS["correct"]
        done_str = f"{done}/{total_tasks}" if total_tasks else "—"
        acc_str = f"{correct}/{done}" if done > 0 else "—"
        d_sb_calls = cur_sb_calls - prev_sb_calls
        d_sb_errors = cur_sb_errors - prev_sb_errors
        sb_str = f"{d_sb_errors}/{d_sb_calls}" if d_sb_calls else "0/0"
        prev_sb_calls = cur_sb_calls
        prev_sb_errors = cur_sb_errors

        # Fetch and aggregate vLLM metrics across all servers
        cur_prompt = cur_gen = cur_reqs = cur_cache_q = cur_cache_h = 0.0
        cur_spec_draft = cur_spec_accept = 0.0
        running = waiting = 0.0
        agg_kv_used = agg_kv_total = 0

        any_metrics = False
        for murl in metrics_urls:
            m = _fetch_vllm_metrics(murl)
            if not m:
                continue
            any_metrics = True

            # Grab KV block info per server (vLLM uses num_gpu_blocks/block_size, SGLang uses num_pages/page_size)
            srv_blocks = None
            srv_bsz = None
            for k, v in m.items():
                if "cache_config_info" in k:
                    bg = re.search(r'num_gpu_blocks="(\d+)"', k) or re.search(r'num_pages="(\d+)"', k)
                    bs = re.search(r'(?<![a-z_])block_size="(\d+)"', k) or re.search(r'page_size="(\d+)"', k)
                    if bg:
                        srv_blocks = int(bg.group(1))
                    if bs:
                        srv_bsz = int(bs.group(1))
                    break

            cur_prompt  += _get_metric(m, "vllm:prompt_tokens_total")
            cur_gen     += _get_metric(m, "vllm:generation_tokens_total")
            # vLLM uses request_success_total{finished_reason}; SGLang uses num_requests_total
            vllm_reqs = (_get_metric(m, "vllm:request_success_total", labels='finished_reason="stop"') +
                         _get_metric(m, "vllm:request_success_total", labels='finished_reason="length"'))
            if vllm_reqs > 0:
                cur_reqs += vllm_reqs
            else:
                cur_reqs += _get_metric(m, "sglang:num_requests_total")
            # vLLM uses prefix_cache_queries/hits; SGLang exposes cached_tokens_total (cumulative cached prompt tokens)
            # so we use prompt_tokens_total as denominator and cached_tokens_total as numerator
            vllm_q = _get_metric(m, "vllm:prefix_cache_queries_total")
            if vllm_q > 0:
                cur_cache_q += vllm_q
                cur_cache_h += _get_metric(m, "vllm:prefix_cache_hits_total")
            else:
                cur_cache_q += _get_metric(m, "sglang:prompt_tokens_total")
                cur_cache_h += _get_metric(m, "sglang:cached_tokens_total")
            # Spec decode acceptance
            cur_spec_draft  += _get_metric(m, "vllm:spec_decode_num_draft_tokens_total")
            cur_spec_accept += _get_metric(m, "vllm:spec_decode_num_accepted_tokens_total")

            v_run = _get_metric(m, "vllm:num_requests_running")
            v_wait = _get_metric(m, "vllm:num_requests_waiting")
            if v_run == 0.0 and v_wait == 0.0:
                v_run = _get_metric(m, "sglang:num_running_reqs")
                v_wait = _get_metric(m, "sglang:num_queue_reqs")
            running += v_run
            waiting += v_wait

            srv_kv_pct = _get_metric(m, "vllm:kv_cache_usage_perc")
            if srv_kv_pct == 0.0:
                srv_kv_pct = _get_metric(m, "sglang:token_usage")
            bsz = srv_bsz or 1
            srv_total = srv_blocks * bsz if srv_blocks else 0
            agg_kv_total += srv_total
            agg_kv_used  += int(srv_kv_pct * srv_total)

        kv_pct = (agg_kv_used / agg_kv_total * 100) if agg_kv_total else 0.0
        used_tok = agg_kv_used
        total_tok = agg_kv_total

        if prev_time is not None:
            dt = now - prev_time
            if dt > 0:
                if row_count % 20 == 0:
                    print(SEP, flush=True)
                    print(HEADER, flush=True)
                    print(SEP, flush=True)

                gen_tps    = (cur_gen - prev_gen) / dt
                prompt_tps = (cur_prompt - prev_prompt) / dt
                d_reqs     = cur_reqs - prev_reqs
                d_cache_q  = cur_cache_q - prev_cache_q
                d_cache_h  = cur_cache_h - prev_cache_h
                cache_pct  = (d_cache_h / d_cache_q * 100) if d_cache_q > 0 else 0.0
                d_spec_d   = cur_spec_draft - prev_spec_draft if prev_spec_draft is not None else 0
                d_spec_a   = cur_spec_accept - prev_spec_accept if prev_spec_accept is not None else 0
                spec_pct   = (d_spec_a / d_spec_d * 100) if d_spec_d > 0 else 0.0

                ts = datetime.now().strftime("%H:%M:%S")
                kv_str = f"{_human(used_tok)}/{_human(total_tok)}" if total_tok else "?"

                print(
                    f"{ts:<9} "
                    f"{done_str:>12} "
                    f"{acc_str:>10} "
                    f"{sb_str:>12} "
                    f"{gen_tps:>8.1f} "
                    f"{prompt_tps:>8.1f} "
                    f"{d_reqs:>6.0f} "
                    f"{running:>4.0f} "
                    f"{waiting:>4.0f} "
                    f"{cache_pct:>6.1f}% "
                    f"{_human(cur_cache_h):>9} "
                    f"{spec_pct:>5.1f}% "
                    f"{kv_pct:>5.1f}% "
                    f"{kv_str:>16}",
                    flush=True,
                )
                row_count += 1

                # Write CSV
                iso_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                csv_file.write(
                    f"{iso_ts},{done},{total_tasks},{d_sb_errors},{d_sb_calls},"
                    f"{gen_tps:.1f},{prompt_tps:.1f},{d_reqs:.0f},"
                    f"{running:.0f},{waiting:.0f},{cache_pct:.1f},"
                    f"{cur_cache_h:.0f},{spec_pct:.1f},{kv_pct:.2f},{used_tok},{total_tok}\n"
                )
                csv_file.flush()

        prev_prompt      = cur_prompt
        prev_gen         = cur_gen
        prev_reqs        = cur_reqs
        prev_cache_q     = cur_cache_q
        prev_spec_draft  = cur_spec_draft
        prev_spec_accept = cur_spec_accept
        prev_cache_h = cur_cache_h
        prev_time    = now

    csv_file.close()


def load_problems_from_dir(input_dir: str) -> list[dict]:
    """Load all JSONL files from a directory, deduplicating by problem id."""
    problems = []
    seen_ids = set()
    input_path = Path(input_dir)

    jsonl_files = sorted(input_path.glob("*.jsonl"))
    if not jsonl_files:
        LOG.error(f"No .jsonl files found in {input_dir}")
        sys.exit(1)

    for fpath in jsonl_files:
        count = 0
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                pid = record["id"]
                if pid in seen_ids:
                    LOG.warning(f"Duplicate problem id {pid} in {fpath.name}, skipping")
                    continue
                seen_ids.add(pid)
                problems.append(record)
                count += 1
        LOG.info(f"  {fpath.name}: {count} problems")

    return problems


def main():
    parser = argparse.ArgumentParser(description="Collect reasoning traces from Nemotron-Cascade-2")
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing JSONL dataset files")
    parser.add_argument("--output_dir", required=True, help="Output directory for traces")
    parser.add_argument("--n_sessions", type=int, default=8, help="Sessions per problem")
    parser.add_argument("--max_parallel", type=int, default=128, help="Max concurrent sessions per server")
    parser.add_argument("--server_addr", nargs="+", default=None,
                        help="Model server URL(s) (default: env SERVER_ADDR). Multiple servers for load distribution.")
    parser.add_argument("--api_key", default=None, help="API key (default: env OPENAI_API_KEY)")
    parser.add_argument("--model_name", default="chankhavu/Nemotron-Cascade-2-30B-A3B-FP8",
                        help="Model name for API calls")
    parser.add_argument("--sandbox_host", default="127.0.0.1", help="Sandbox server host")
    parser.add_argument("--sandbox_port", type=int, default=6000, help="Sandbox server port")
    parser.add_argument("--max_turns", type=int, default=128, help="Max tool-use turns per session")
    parser.add_argument("--python_timeout", type=int, default=60, help="Sandbox code execution timeout (seconds)")
    parser.add_argument("--max_output_characters", type=int, default=3000, help="Max sandbox output chars")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=0, help="Max completion tokens per session. 0 = no limit (server uses max_model_len - prompt_len)")
    parser.add_argument("--system_prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--enable_thinking", action="store_true", default=True,
                        help="Enable <think> reasoning mode")
    parser.add_argument("--no_thinking", action="store_true", help="Disable <think> reasoning mode")
    parser.add_argument("--health_check_interval", type=int, default=300,
                        help="Seconds between health checks (0 to disable)")
    parser.add_argument("--metrics_url", default=None,
                        help="vLLM /metrics URL for watchdog (default: derived from server_addr)")
    parser.add_argument("--watchdog_interval", type=int, default=30,
                        help="Seconds between watchdog table rows")
    parser.add_argument("--rescan_interval", type=int, default=60,
                        help="Seconds to wait before re-scanning data dir for new files")
    parser.add_argument("--resume", action="store_true", help="Skip completed traces")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.no_thinking:
        args.enable_thinking = False

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # Silence sandbox client logs unless verbose — only show errors (failed executions)
    if not args.verbose:
        logging.getLogger("sandbox_client.py").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Server addresses
    raw_addrs = args.server_addr or [os.environ.get("SERVER_ADDR", "")]
    if not any(raw_addrs):
        print("ERROR: --server_addr or SERVER_ADDR env var required", file=sys.stderr)
        sys.exit(1)
    server_addrs = []
    for addr in raw_addrs:
        if not addr.startswith("http"):
            addr = f"http://{addr}"
        if not addr.endswith("/v1"):
            addr = f"{addr}/v1"
        server_addrs.append(addr)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "sk-local")

    # Initialize one OpenAI client per server
    clients = [OpenAI(base_url=addr, api_key=api_key, timeout=None) for addr in server_addrs]
    LOG.info(f"Servers: {len(clients)} — {server_addrs}")

    # Initial health check — fail fast if any service is down
    LOG.info("Running initial health check...")
    err = run_health_checks(server_addrs, api_key, args.sandbox_host, args.sandbox_port)
    if err:
        LOG.error(f"Initial health check failed: {err}")
        sys.exit(1)
    LOG.info("Health check passed")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive metrics URLs from server addresses (strip /v1, add /metrics)
    if args.metrics_url is not None:
        metrics_urls = [args.metrics_url] if args.metrics_url else []
    else:
        metrics_urls = [re.sub(r'/v1$', '', addr) + "/metrics" for addr in server_addrs]

    # Start background watchdog (metrics table + health checks)
    if args.health_check_interval > 0 or metrics_urls:
        watchdog_thread = threading.Thread(
            target=_watchdog,
            args=(server_addrs, api_key, args.sandbox_host, args.sandbox_port),
            kwargs={
                "metrics_urls": metrics_urls,
                "interval": args.watchdog_interval,
                "health_check_interval": args.health_check_interval,
            },
            daemon=True,
        )
        watchdog_thread.start()

    # Build sampling params — only include non-None values
    sampling_params = {}
    if args.temperature is not None:
        sampling_params["temperature"] = args.temperature
    if args.top_p is not None:
        sampling_params["top_p"] = args.top_p
    if args.top_k is not None:
        sampling_params["top_k"] = args.top_k
    if args.min_p is not None:
        sampling_params["min_p"] = args.min_p
    LOG.info(f"Sampling params: {sampling_params}")

    total_completed = 0
    total_errors = 0
    round_num = 0

    # Main loop: re-scan data dir each round to pick up new files
    while not _ABORT_EVENT.is_set():
        round_num += 1

        # (Re)load problems from all JSONL files in input_dir
        LOG.info(f"[Round {round_num}] Scanning {args.input_dir} for datasets...")
        problems = load_problems_from_dir(args.input_dir)
        LOG.info(f"[Round {round_num}] {len(problems)} unique problems")

        # Classify answer types
        type_counts = {"nonneg_int": 0, "integer": 0, "symbolic": 0}
        for prob in problems:
            prob["_answer_type"] = classify_answer_type(prob["expected_answer"])
            type_counts[prob["_answer_type"]] += 1
        LOG.info(f"[Round {round_num}] Answer types: {type_counts}")

        # Build task list (always check existing traces for resume)
        tasks = []
        skipped = 0
        for prob in problems:
            pid = str(prob["id"])
            done = _existing_session_indices(output_dir, pid)
            for si in range(args.n_sessions):
                if si in done:
                    skipped += 1
                    continue
                tasks.append({
                    "problem_id": pid,
                    "problem": prob["problem"],
                    "expected_answer": prob["expected_answer"],
                    "source": prob.get("source", ""),
                    "source_id": prob.get("source_id", ""),
                    "answer_type": prob["_answer_type"],
                    "session_index": si,
                    "client": clients[_route_to_server(pid, len(clients))],
                    "model_name": args.model_name,
                    "system_prompt": args.system_prompt,
                    "tools": TOOLS,
                    "sandbox_host": args.sandbox_host,
                    "sandbox_port": args.sandbox_port,
                    "sampling_params": sampling_params,
                    "max_tokens": args.max_tokens,
                    "max_turns": args.max_turns,
                    "python_timeout": args.python_timeout,
                    "max_output_characters": args.max_output_characters,
                    "enable_thinking": args.enable_thinking,
                    "verbose": args.verbose,
                })

        if not tasks:
            if round_num == 1:
                LOG.info("No tasks to run (all complete or empty input)")
            else:
                LOG.info(f"[Round {round_num}] No new tasks. Waiting {args.rescan_interval}s before next scan...")
            # Wait before re-scanning, but respect abort
            if _ABORT_EVENT.wait(timeout=args.rescan_interval):
                break
            continue

        total_parallel = args.max_parallel * len(clients)
        if skipped:
            LOG.info(f"[Round {round_num}] Skipped {skipped} already-completed sessions")
        LOG.info(f"[Round {round_num}] Running {len(tasks)} sessions "
                 f"({args.max_parallel}/server x {len(clients)} servers = {total_parallel} parallel)")

        completed = 0
        errors = 0
        start_round_time = time.time()
        last_progress_time = start_round_time
        aborted = False

        with _PROGRESS_LOCK:
            _PROGRESS["completed"] = 0
            _PROGRESS["total"] = len(tasks)
            _PROGRESS["errors"] = 0

        # Partition tasks by server
        tasks_by_server = [[] for _ in clients]
        for t in tasks:
            srv_idx = _route_to_server(t["problem_id"], len(clients))
            tasks_by_server[srv_idx].append(t)

        for i, st in enumerate(tasks_by_server):
            LOG.info(f"  Server {i} ({server_addrs[i]}): {len(st)} tasks")

        try:
            # One executor per server, each capped at max_parallel
            executors = [ThreadPoolExecutor(max_workers=args.max_parallel) for _ in clients]
            futures = {}
            for srv_idx, srv_tasks in enumerate(tasks_by_server):
                for t in srv_tasks:
                    f = executors[srv_idx].submit(_worker, t)
                    futures[f] = t
            LOG.info(f"[Round {round_num}] Submitted {len(futures)} tasks across {len(clients)} executors")

            for future in as_completed(futures):
                if _ABORT_EVENT.is_set() and not aborted:
                    aborted = True
                    LOG.error("Abort signal received. In-flight tasks will finish.")

                task_info = futures[future]
                pid = task_info["problem_id"]
                si = task_info["session_index"]

                try:
                    result = future.result()
                    _append_result(output_dir, result)
                    completed += 1
                    last_progress_time = time.time()
                    # Check accuracy
                    is_correct = (str(result.get("predicted_answer", ""))
                                  == str(result.get("expected_answer", "")))
                    with _PROGRESS_LOCK:
                        _PROGRESS["completed"] = completed
                        _PROGRESS["sandbox_calls"] += result.get("num_tool_calls", 0)
                        _PROGRESS["sandbox_errors"] += result.get("num_tool_errors", 0)
                        if is_correct:
                            _PROGRESS["correct"] += 1

                    if args.verbose:
                        ans = result["predicted_answer"]
                        reason = result["finish_reason"]
                        tokens = result["num_completion_tokens"]
                        tools_used = result["num_tool_calls"]
                        t = result["generation_time"]
                        print(f"  [{completed}/{len(tasks)}] {pid[:16]}..:{si} "
                              f"answer={ans} reason={reason} tokens={tokens} "
                              f"tools={tools_used} time={t}s")

                except Exception as exc:
                    errors += 1
                    with _PROGRESS_LOCK:
                        _PROGRESS["errors"] = errors
                    LOG.error(f"  [{pid[:16]}..:{si}] Failed: {exc}")

                # Progress every 100 completions, or every 5 min
                now = time.time()
                if completed % 100 == 0 or now - last_progress_time >= 300:
                    elapsed = round(now - start_round_time)
                    LOG.info(f"  [{completed}/{len(tasks)}] sessions done, "
                             f"{errors} errors, {elapsed}s elapsed")
                    last_progress_time = now

        except Exception as exc:
            LOG.error(f"[Round {round_num}] Executor error: {exc}")
            aborted = True
        finally:
            for ex in executors:
                ex.shutdown(wait=False)

        total_completed += completed
        total_errors += errors
        LOG.info(f"[Round {round_num}] {completed} completed, {errors} errors")

        if aborted or _ABORT_EVENT.is_set():
            break

        # Brief pause before re-scanning for new data
        LOG.info(f"[Round {round_num}] Round complete. Re-scanning in {args.rescan_interval}s...")
        if _ABORT_EVENT.wait(timeout=args.rescan_interval):
            break

    _ABORT_EVENT.set()  # signal watchdog to stop

    LOG.info(f"Total: {total_completed} completed, {total_errors} errors. Output: {output_dir}")
    if _ABORT_EVENT.is_set() and total_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
