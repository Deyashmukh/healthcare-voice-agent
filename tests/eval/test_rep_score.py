# pyright: strict
"""Unit tests for the rep-extraction scorer (pure, offline)."""

from __future__ import annotations

from agent.eval._types import EvalOutcome, FailureMode, RepEvalCase
from agent.eval.rep_extraction._score import score_rep
from agent.schemas import Benefits, RepTurnOutput, Turn


def _case(
    extracted: Benefits,
    phase: str = "extracting",
    expect_reply: bool = True,
) -> RepEvalCase:
    return RepEvalCase(
        id="c1",
        history=[Turn(role="user", content="Her copay is forty dollars")],
        expected_extracted=extracted,
        expected_phase=phase,  # type: ignore[arg-type]
        expect_nonempty_reply=expect_reply,
        rationale="r",
    )


def _out(reply: str, extracted: Benefits, phase: str = "extracting") -> RepTurnOutput:
    return RepTurnOutput(reply=reply, extracted=extracted, phase=phase)  # type: ignore[arg-type]


def test_pass() -> None:
    case = _case(Benefits(copay=40.0))
    assert (
        score_rep(case, _out("Got it, thanks.", Benefits(copay=40.0))).outcome is EvalOutcome.PASS
    )


def test_missed_extraction() -> None:
    case = _case(Benefits(copay=40.0))
    result = score_rep(case, _out("Okay.", Benefits()))
    assert result.failure_mode is FailureMode.MISSED_EXTRACTION


def test_hallucinated_field() -> None:
    case = _case(Benefits())  # expect nothing extracted
    result = score_rep(case, _out("Sure.", Benefits(copay=40.0)))
    assert result.failure_mode is FailureMode.HALLUCINATED_FIELD


def test_bad_arg_value() -> None:
    case = _case(Benefits(copay=40.0))
    result = score_rep(case, _out("Got it.", Benefits(copay=30.0)))
    assert result.failure_mode is FailureMode.BAD_ARG


def test_both_missing_and_extra_labels_missed_extraction() -> None:
    # Expected one field, got a totally different one (missing AND extra). The
    # scorer checks `missing` first; MISSED_EXTRACTION (failed to capture the
    # expected field) is the load-bearing label, and `detail` carries both dicts.
    case = _case(Benefits(copay=40.0))
    result = score_rep(case, _out("Got it.", Benefits(deductible_remaining=300.0)))
    assert result.failure_mode is FailureMode.MISSED_EXTRACTION


def test_premature_complete() -> None:
    case = _case(Benefits(copay=40.0), phase="extracting")
    result = score_rep(case, _out("All set, bye.", Benefits(copay=40.0), phase="complete"))
    assert result.failure_mode is FailureMode.PREMATURE_COMPLETE


def test_wrong_phase_non_complete() -> None:
    case = _case(Benefits(copay=40.0), phase="extracting")
    result = score_rep(case, _out("Sorry, bye.", Benefits(copay=40.0), phase="stuck"))
    assert result.failure_mode is FailureMode.WRONG_PHASE


def test_empty_reply_when_required() -> None:
    case = _case(Benefits(), expect_reply=True)
    result = score_rep(case, _out("   ", Benefits()))
    assert result.failure_mode is FailureMode.EMPTY_REPLY
