# pyright: strict
"""Unit tests for the IVR tool-choice scorer (pure, offline)."""

from __future__ import annotations

import pytest

from agent.eval._runner import run_eval
from agent.eval._types import CaseResult, EvalOutcome, FailureMode, IVREvalCase
from agent.eval.ivr_tool_choice._score import NoToolCallError, score_ivr
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


def test_no_tool_call_raises_not_fails() -> None:
    # Under tool_choice=required, an empty response is a transient provider glitch,
    # not a tool choice — it raises so the runner retries / records ERROR rather
    # than a false WRONG_TOOL FAIL.
    with pytest.raises(NoToolCallError):
        score_ivr(_case(), IVRTurnResponse(tool_calls=[]))


async def test_no_tool_call_becomes_retried_error_in_runner() -> None:
    # End-to-end through the runner: a persistently-empty response is retried
    # then recorded as ERROR (excluded from the wrong_tool FAIL count), with the
    # transient-glitch detail preserved.
    calls: list[str] = []

    async def scorer(case: IVREvalCase) -> CaseResult:
        calls.append(case.id)
        return score_ivr(case, IVRTurnResponse(tool_calls=[]))

    report = await run_eval([_case()], scorer, layer="ivr", per_case_retries=1)
    assert report.errored == 1
    assert report.failed == 0  # NOT a wrong_tool FAIL
    assert len(calls) == 2  # initial + 1 retry (transient blips get a second shot)
    assert "no tool call" in (report.results[0].error or "")


def test_bad_arg_when_digit_differs() -> None:
    result = score_ivr(_case(args={"digits": "2"}), _resp("send_dtmf", {"digits": "9"}))
    assert result.outcome is EvalOutcome.FAIL
    assert result.failure_mode is FailureMode.BAD_ARG


def test_json_number_digit_is_not_a_false_bad_arg() -> None:
    # Groq may emit the digit as a JSON number (2) rather than a string ("2").
    # The scorer compares as strings so a correct press isn't mislabeled BAD_ARG.
    result = score_ivr(_case(args={"digits": "2"}), _resp("send_dtmf", {"digits": 2}))
    assert result.outcome is EvalOutcome.PASS


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
