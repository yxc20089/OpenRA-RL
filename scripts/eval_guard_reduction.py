#!/usr/bin/env python3
"""Evaluate repeated tool-call reduction from before/after agent logs.

Zero-intrusion metric script:
- Does not modify runtime code.
- Parses existing verbose agent logs produced by examples/llm_agent.py.
- Compares repeated-call rate between two sets of runs.

Expected log pattern (verbose mode):
  [Tool] build_structure({"building_type":"powr"})
  [Result] {"guard_status":"defer","guard_reason":"already_in_queue",...}
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TOOL_LINE_RE = re.compile(r"\[Tool\]\s+([A-Za-z0-9_]+)\((.*)\)\s*$")
RESULT_LINE_RE = re.compile(r"\[Result\]\s+(\{.*\})\s*$")

# Reasons related to repeated/duplicate/loop-like calls.
REPEAT_REASONS = {
    "agent_repeat_guard",
    "duplicate_control_same_tick",
    "already_pending",
    "already_in_queue",
    "cancel_repeat_backoff",
    "cancel_build_loop",
}


@dataclass
class Metrics:
    files: int = 0
    total_tool_calls: int = 0
    immediate_repeats: int = 0
    window_repeats: int = 0
    unique_signatures: int = 0
    truncated_tool_lines: int = 0
    guard_block_or_defer: int = 0
    guard_repeat_block_or_defer: int = 0
    parseable_result_json: int = 0

    def immediate_repeat_rate(self) -> float:
        return (self.immediate_repeats / self.total_tool_calls) if self.total_tool_calls else 0.0

    def window_repeat_rate(self) -> float:
        return (self.window_repeats / self.total_tool_calls) if self.total_tool_calls else 0.0


def _iter_log_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    candidates = [p for p in path.rglob("*") if p.is_file()]
    # Keep typical text-like logs.
    allowed_suffix = {".log", ".txt", ".jsonl"}
    return sorted([p for p in candidates if p.suffix.lower() in allowed_suffix])


def _safe_load_json(s: str) -> dict | None:
    try:
        obj = json.loads(s)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def parse_logs(path: Path, repeat_window: int) -> tuple[Metrics, Counter]:
    files = _iter_log_files(path)
    if not files:
        raise ValueError(f"No log files found under: {path}")

    m = Metrics(files=len(files))
    reason_counts: Counter = Counter()
    global_seen: set[str] = set()

    for fp in files:
        last_sig: str | None = None
        last_seen_index: dict[str, int] = {}
        idx = 0

        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                mt = TOOL_LINE_RE.search(line)
                if mt:
                    fn_name = mt.group(1)
                    args_blob = mt.group(2).strip()
                    sig = f"{fn_name}:{args_blob}"
                    m.total_tool_calls += 1
                    idx += 1

                    if args_blob.endswith("..."):
                        m.truncated_tool_lines += 1

                    if last_sig == sig:
                        m.immediate_repeats += 1

                    prev_idx = last_seen_index.get(sig)
                    if prev_idx is not None and (idx - prev_idx) <= repeat_window:
                        m.window_repeats += 1
                    last_seen_index[sig] = idx
                    last_sig = sig
                    global_seen.add(sig)
                    continue

                mr = RESULT_LINE_RE.search(line)
                if not mr:
                    continue
                obj = _safe_load_json(mr.group(1))
                if obj is None:
                    continue
                m.parseable_result_json += 1
                status = str(obj.get("guard_status", ""))
                reason = str(obj.get("guard_reason", ""))
                if status in {"block", "defer"}:
                    m.guard_block_or_defer += 1
                    reason_counts[reason] += 1
                    if reason in REPEAT_REASONS:
                        m.guard_repeat_block_or_defer += 1

    m.unique_signatures = len(global_seen)
    return m, reason_counts


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _reduction(before_rate: float, after_rate: float) -> float | None:
    if before_rate <= 0:
        return None
    return (before_rate - after_rate) / before_rate


def _print_block(name: str, m: Metrics) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    print(f"files: {m.files}")
    print(f"total_tool_calls: {m.total_tool_calls}")
    print(f"unique_signatures: {m.unique_signatures}")
    print(f"immediate_repeats: {m.immediate_repeats} ({_pct(m.immediate_repeat_rate())})")
    print(f"window_repeats: {m.window_repeats} ({_pct(m.window_repeat_rate())})")
    print(f"guard_block_or_defer: {m.guard_block_or_defer}")
    print(f"guard_repeat_block_or_defer: {m.guard_repeat_block_or_defer}")
    print(f"truncated_tool_lines: {m.truncated_tool_lines}")
    print(f"parseable_result_json: {m.parseable_result_json}")


def _write_csv(
    out_csv: Path,
    before: Metrics,
    after: Metrics,
    immediate_reduction: float | None,
    window_reduction: float | None,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "before", "after", "reduction_pct"])
        rows = [
            ("total_tool_calls", before.total_tool_calls, after.total_tool_calls, ""),
            ("immediate_repeat_rate", before.immediate_repeat_rate(), after.immediate_repeat_rate(), ""),
            ("window_repeat_rate", before.window_repeat_rate(), after.window_repeat_rate(), ""),
            (
                "guard_repeat_block_or_defer_rate",
                (before.guard_repeat_block_or_defer / before.total_tool_calls) if before.total_tool_calls else 0.0,
                (after.guard_repeat_block_or_defer / after.total_tool_calls) if after.total_tool_calls else 0.0,
                "",
            ),
            (
                "immediate_repeat_reduction_pct",
                "",
                "",
                "" if immediate_reduction is None else round(immediate_reduction * 100, 4),
            ),
            (
                "window_repeat_reduction_pct",
                "",
                "",
                "" if window_reduction is None else round(window_reduction * 100, 4),
            ),
        ]
        for r in rows:
            w.writerow(r)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare repeated-call metrics between before/after logs."
    )
    parser.add_argument("--before", required=True, help="Path to baseline log file or directory.")
    parser.add_argument("--after", required=True, help="Path to improved log file or directory.")
    parser.add_argument(
        "--repeat-window",
        type=int,
        default=3,
        help="A call is counted as repeated if same signature appears within N tool calls (default: 3).",
    )
    parser.add_argument("--csv-out", default="", help="Optional path to save summary CSV.")
    args = parser.parse_args()

    before_metrics, before_reasons = parse_logs(Path(args.before), args.repeat_window)
    after_metrics, after_reasons = parse_logs(Path(args.after), args.repeat_window)

    _print_block("BEFORE", before_metrics)
    _print_block("AFTER", after_metrics)

    b_immediate = before_metrics.immediate_repeat_rate()
    a_immediate = after_metrics.immediate_repeat_rate()
    b_window = before_metrics.window_repeat_rate()
    a_window = after_metrics.window_repeat_rate()
    immediate_reduction = _reduction(b_immediate, a_immediate)
    window_reduction = _reduction(b_window, a_window)

    print("\nREDUCTION")
    print("---------")
    print(f"immediate_repeat_rate: {_pct(b_immediate)} -> {_pct(a_immediate)}")
    print(
        "immediate_repeat_reduction: "
        + ("N/A (baseline is 0)" if immediate_reduction is None else _pct(immediate_reduction))
    )
    print(f"window_repeat_rate: {_pct(b_window)} -> {_pct(a_window)}")
    print(
        "window_repeat_reduction: "
        + ("N/A (baseline is 0)" if window_reduction is None else _pct(window_reduction))
    )

    if before_reasons or after_reasons:
        print("\nTop guard reasons (before):")
        for reason, cnt in before_reasons.most_common(8):
            print(f"  - {reason or '<empty>'}: {cnt}")
        print("Top guard reasons (after):")
        for reason, cnt in after_reasons.most_common(8):
            print(f"  - {reason or '<empty>'}: {cnt}")

    if args.csv_out:
        out = Path(args.csv_out)
        _write_csv(out, before_metrics, after_metrics, immediate_reduction, window_reduction)
        print(f"\nCSV written: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
