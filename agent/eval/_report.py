# pyright: strict
"""Render a ScoreReport as a table, persist the full report as JSON, and append
a one-line summary to a committed trend file so week-over-week pass-rate is
answerable without a database (same zero-infra pattern as benefits.jsonl).

`timestamp` is passed in (not read from the clock) so the function is
deterministic and unit-testable; callers stamp `time.strftime(...)` at the edge.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.eval._types import ScoreReport


def render_table(report: ScoreReport) -> str:
    lines = [
        f"=== {report.layer} ===",
        f"  total={report.total}  pass={report.passed}  fail={report.failed}  error={report.errored}",
        f"  pass_rate={report.pass_rate:.2%}",
    ]
    if report.failure_modes:
        lines.append("  failure modes:")
        for mode, count in sorted(report.failure_modes.items(), key=lambda kv: kv[0].value):
            lines.append(f"    {mode.value}: {count}")
    return "\n".join(lines)


def write_report(
    report: ScoreReport,
    *,
    results_dir: Path,
    history_path: Path,
    timestamp: str,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    safe_ts = timestamp.replace(":", "").replace("-", "")
    out_path = results_dir / f"{report.layer}-{safe_ts}.json"
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    summary = {
        "timestamp": timestamp,
        "layer": report.layer,
        "total": report.total,
        "passed": report.passed,
        "failed": report.failed,
        "errored": report.errored,
        "pass_rate": report.pass_rate,
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    return out_path
