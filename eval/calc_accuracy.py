#!/usr/bin/env python3
"""
Calculate overall and per-task accuracy from trace JSONL files.

Usage:
  python scripts/calc_accuracy.py traces/gptoss-aime25/
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _get_gen_tokens(session: dict) -> int:
    """Extract generated token count from a session, with fallback estimation."""
    for key in ("num_completion_tokens", "num_generated_tokens"):
        val = session.get(key, 0)
        if val and val > 0:
            return val
    # Fallback: estimate from conversation text (~4 chars per token)
    conv = session.get("conversation", [])
    if not conv:
        trace = session.get("trace", "")
        return len(trace) // 4 if trace else 0
    chars = 0
    for msg in conv:
        if msg.get("role") == "assistant":
            chars += len(msg.get("content", "") or "")
            for tc in msg.get("tool_calls", []):
                chars += len(tc.get("function", {}).get("arguments", ""))
    return chars // 4


def load_traces(trace_dir: Path) -> dict[str, list[dict]]:
    """Load all traces grouped by problem ID."""
    problems = defaultdict(list)
    for f in sorted(trace_dir.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    problems[r["id"]].append(r)
    return dict(problems)


def majority_vote(sessions: list[dict]) -> int | None:
    """Return the most common predicted answer (None if all None)."""
    answers = [s["predicted_answer"] for s in sessions if s["predicted_answer"] is not None]
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]


def main():
    parser = argparse.ArgumentParser(description="Calculate accuracy from trace files")
    parser.add_argument("trace_dir", help="Directory containing per-problem JSONL trace files")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    if not trace_dir.is_dir():
        print(f"ERROR: {trace_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    problems = load_traces(trace_dir)
    if not problems:
        print("No traces found.", file=sys.stderr)
        sys.exit(1)

    rows = []
    total_correct = 0
    total_sessions = 0
    total_maj_correct = 0
    total_gen_tokens = 0
    total_prompt_tokens = 0
    total_tools = 0
    total_errors = 0
    total_time = 0.0

    for pid, sessions in problems.items():
        expected = sessions[0]["expected_answer"]
        n = len(sessions)

        correct = sum(
            1 for s in sessions
            if s["predicted_answer"] is not None and str(s["predicted_answer"]) == str(expected)
        )
        acc = correct / n if n else 0

        maj_pred = majority_vote(sessions)
        maj_correct = int(maj_pred is not None and str(maj_pred) == str(expected))

        gen_tokens = sum(_get_gen_tokens(s) for s in sessions)
        prompt_tokens = sum(
            s.get("num_prompt_tokens",
                  (len(s["token_ids"]) - s.get("num_completion_tokens", 0))
                  if "token_ids" in s and s.get("num_completion_tokens", 0) > 0 else 0)
            for s in sessions
        )
        tools = sum(s.get("num_tool_calls", 0) for s in sessions)
        errors = sum(s.get("num_tool_errors", 0) for s in sessions)
        time_s = sum(s.get("generation_time_seconds", s.get("generation_time", 0)) for s in sessions)
        reasons = Counter(s["finish_reason"] for s in sessions)

        total_correct += correct
        total_sessions += n
        total_maj_correct += maj_correct
        total_gen_tokens += gen_tokens
        total_prompt_tokens += prompt_tokens
        total_tools += tools
        total_errors += errors
        total_time += time_s

        rows.append({
            "id": pid,
            "expected": expected,
            "sessions": n,
            "correct": correct,
            "acc": acc,
            "maj_pred": maj_pred,
            "maj_correct": maj_correct,
            "gen_tokens": gen_tokens,
            "prompt_tokens": prompt_tokens,
            "tools": tools,
            "errors": errors,
            "time": time_s,
            "reasons": reasons,
        })

    rows.sort(key=lambda r: r["id"])

    # Per-task table
    print(f"{'ID':<16} {'Exp':>6} {'N':>3} {'Cor':>4} {'Acc':>6} {'Maj':>6} {'MajOK':>5} "
          f"{'GenTok':>8} {'PrmTok':>8} {'Tools':>6} {'Errs':>5} {'Time':>7}  Reasons")
    print("-" * 130)

    for r in rows:
        reasons_str = ", ".join(f"{k}={v}" for k, v in r["reasons"].most_common())
        maj_str = str(r["maj_pred"]) if r["maj_pred"] is not None else "-"
        maj_ok = "Y" if r["maj_correct"] else "N"
        print(f"{r['id']:<16} {str(r['expected']):>6} {r['sessions']:>3} {r['correct']:>4} "
              f"{r['acc']:>5.0%} {maj_str:>6} {maj_ok:>5} "
              f"{r['gen_tokens']:>8} {r['prompt_tokens']:>8} {r['tools']:>6} {r['errors']:>5} {r['time']:>6.0f}s  {reasons_str}")

    print("-" * 130)

    # Overall summary
    n_problems = len(problems)
    per_session_acc = total_correct / total_sessions if total_sessions else 0
    maj_acc = total_maj_correct / n_problems if n_problems else 0
    total_all_tokens = total_gen_tokens + total_prompt_tokens

    print(f"\n{'Per-session accuracy:':<24} {total_correct}/{total_sessions} = {per_session_acc:.1%}")
    print(f"{'Majority-vote accuracy:':<24} {total_maj_correct}/{n_problems} = {maj_acc:.1%}")
    print(f"{'Problems:':<24} {n_problems}  ({total_sessions} sessions total)")
    print(f"{'Avg gen tokens/session:':<24} {total_gen_tokens / total_sessions:.0f}")
    print(f"{'Avg prompt tok/session:':<24} {total_prompt_tokens / total_sessions:.0f}")
    print(f"{'Avg total tok/session:':<24} {(total_all_tokens) / total_sessions:.0f}")
    print(f"{'Avg time/session:':<24} {total_time / total_sessions:.1f}s")
    print(f"{'Total tool calls:':<24} {total_tools}  ({total_errors} errors)")
    print(f"{'Total time:':<24} {total_time:.0f}s ({total_time / 60:.1f}m)")


if __name__ == "__main__":
    main()
