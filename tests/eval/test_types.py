# pyright: strict
"""Unit tests for eval-harness type models."""

from __future__ import annotations

from agent.eval._types import (
    CaseResult,
    EvalOutcome,
    FailureMode,
    IVREvalCase,
    RepEvalCase,
    ScoreReport,
)
from agent.schemas import Benefits, Turn


def test_ivr_case_round_trips_through_json() -> None:
    case = IVREvalCase(
        id="aetna-main-press-2",
        payer="aetna",
        history=[Turn(role="user", content="For billing press 2")],
        expected_tool="send_dtmf",
        expected_args={"digits": "2"},
        rationale="billing is option 2",
    )
    reloaded = IVREvalCase.model_validate_json(case.model_dump_json())
    assert reloaded == case
    assert reloaded.expected_args == {"digits": "2"}


def test_ivr_case_expected_args_defaults_empty() -> None:
    case = IVREvalCase(
        id="x",
        payer="p",
        history=[],
        expected_tool="wait",
        rationale="r",
    )
    assert case.expected_args == {}


def test_rep_case_round_trips() -> None:
    case = RepEvalCase(
        id="copay-30",
        history=[Turn(role="user", content="Her copay is $30")],
        expected_extracted=Benefits(copay=30.0),
        expected_phase="extracting",
        rationale="copay stated",
    )
    reloaded = RepEvalCase.model_validate_json(case.model_dump_json())
    assert reloaded == case
    assert reloaded.expect_nonempty_reply is True  # default


def test_score_report_aggregates_counts() -> None:
    results = [
        CaseResult(case_id="a", outcome=EvalOutcome.PASS),
        CaseResult(case_id="b", outcome=EvalOutcome.FAIL, failure_mode=FailureMode.WRONG_TOOL),
        CaseResult(case_id="c", outcome=EvalOutcome.ERROR, error="boom"),
        CaseResult(case_id="d", outcome=EvalOutcome.FAIL, failure_mode=FailureMode.WRONG_TOOL),
    ]
    report = ScoreReport.from_results(layer="ivr_tool_choice", results=results)
    assert report.total == 4
    assert report.passed == 1
    assert report.failed == 2
    assert report.errored == 1
    assert report.pass_rate == 0.25
    assert report.failure_modes[FailureMode.WRONG_TOOL] == 2


def test_score_report_pass_rate_zero_cases_is_zero() -> None:
    report = ScoreReport.from_results(layer="x", results=[])
    assert report.total == 0
    assert report.pass_rate == 0.0
