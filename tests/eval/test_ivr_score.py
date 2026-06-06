# pyright: strict
"""Unit tests for the IVR tool-choice scorer (pure, offline)."""

from __future__ import annotations

from agent.eval._types import EvalOutcome, FailureMode, IVREvalCase
from agent.eval.ivr_tool_choice._score import score_ivr
from agent.schemas import IVRTurnResponse, ToolCall, Turn


def _case(tool: str = "send_dtmf", args: dict[str, object] | None = None) -> IVREvalCase:
    return IVREvalCase(
        id="c1",
        payer="p",
        history=[Turn(role="user", content="For billing press 2")],
        expected_tool=tool,  # type: ignore[arg-type]
        expected_args=args or {"digits": "2"},
        rationale="r",
    )


def _resp(name: str, args: dict[str, object]) -> IVRTurnResponse:
    return IVRTurnResponse(tool_calls=[ToolCall(name=name, args=args)])  # type: ignore[arg-type]


def test_pass_on_matching_tool_and_args() -> None:
    result = score_ivr(_case(), _resp("send_dtmf", {"digits": "2", "purpose": "menu"}))
    assert result.outcome is EvalOutcome.PASS


def test_wrong_tool() -> None:
    result = score_ivr(_case(), _resp("wait", {}))
    assert result.outcome is EvalOutcome.FAIL
    assert result.failure_mode is FailureMode.WRONG_TOOL


def test_no_tool_call_is_wrong_tool() -> None:
    result = score_ivr(_case(), IVRTurnResponse(tool_calls=[]))
    assert result.outcome is EvalOutcome.FAIL
    assert result.failure_mode is FailureMode.WRONG_TOOL
    assert "no tool call" in result.detail


def test_bad_arg_when_digit_differs() -> None:
    result = score_ivr(_case(args={"digits": "2"}), _resp("send_dtmf", {"digits": "9"}))
    assert result.outcome is EvalOutcome.FAIL
    assert result.failure_mode is FailureMode.BAD_ARG


def test_only_expected_args_are_checked() -> None:
    # expected_args lists only `digits`; the model's extra `purpose` is ignored.
    result = score_ivr(
        _case(args={"digits": "0"}), _resp("send_dtmf", {"digits": "0", "purpose": "rep"})
    )
    assert result.outcome is EvalOutcome.PASS


def test_rep_purpose_asserted_when_listed() -> None:
    case = _case(args={"digits": "0", "purpose": "rep"})
    assert (
        score_ivr(case, _resp("send_dtmf", {"digits": "0", "purpose": "rep"})).outcome
        is EvalOutcome.PASS
    )
    bad = score_ivr(case, _resp("send_dtmf", {"digits": "0", "purpose": "menu"}))
    assert bad.outcome is EvalOutcome.FAIL
    assert bad.failure_mode is FailureMode.BAD_ARG
