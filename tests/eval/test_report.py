# pyright: strict
"""Unit tests for report rendering + persistence."""

from __future__ import annotations

import json
from pathlib import Path

from agent.eval._report import render_table, write_report
from agent.eval._types import CaseResult, EvalOutcome, FailureMode, ScoreReport


def _report() -> ScoreReport:
    return ScoreReport.from_results(
        layer="ivr_tool_choice",
        results=[
            CaseResult(case_id="a", outcome=EvalOutcome.PASS),
            CaseResult(case_id="b", outcome=EvalOutcome.FAIL, failure_mode=FailureMode.WRONG_TOOL),
        ],
    )


def test_render_table_includes_layer_and_counts() -> None:
    text = render_table(_report())
    assert "ivr_tool_choice" in text
    assert "1" in text and "2" in text  # passed / total
    assert "wrong_tool" in text


def test_write_report_writes_json_and_appends_history(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    history = tmp_path / "eval_history.jsonl"
    write_report(
        _report(),
        results_dir=results_dir,
        history_path=history,
        timestamp="2026-05-29T12:00:00Z",
    )

    written = list(results_dir.glob("*.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text())
    assert payload["layer"] == "ivr_tool_choice"

    hist_lines = history.read_text().strip().splitlines()
    assert len(hist_lines) == 1
    hist = json.loads(hist_lines[0])
    assert hist["layer"] == "ivr_tool_choice"
    assert hist["pass_rate"] == 0.5
    assert hist["timestamp"] == "2026-05-29T12:00:00Z"


def test_write_report_appends_second_run_to_history(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    history = tmp_path / "eval_history.jsonl"
    write_report(
        _report(),
        results_dir=results_dir,
        history_path=history,
        timestamp="2026-05-29T12:00:00Z",
    )
    write_report(
        _report(),
        results_dir=results_dir,
        history_path=history,
        timestamp="2026-05-29T13:00:00Z",
    )
    assert len(history.read_text().strip().splitlines()) == 2
