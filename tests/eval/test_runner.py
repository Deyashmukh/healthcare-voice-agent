# pyright: strict
"""Unit tests for the eval runner (offline; fake scorers)."""

from __future__ import annotations

import structlog.testing

from agent.eval._runner import run_eval
from agent.eval._types import CaseResult, EvalOutcome, FailureMode, IVREvalCase


def _case(case_id: str) -> IVREvalCase:
    return IVREvalCase(id=case_id, payer="p", history=[], expected_tool="wait", rationale="r")


async def test_run_eval_aggregates_pass_and_fail() -> None:
    cases = [_case("a"), _case("b")]

    async def scorer(case: IVREvalCase) -> CaseResult:
        if case.id == "a":
            return CaseResult(case_id=case.id, outcome=EvalOutcome.PASS)
        return CaseResult(
            case_id=case.id, outcome=EvalOutcome.FAIL, failure_mode=FailureMode.WRONG_TOOL
        )

    report = await run_eval(cases, scorer, layer="ivr")
    assert report.passed == 1
    assert report.failed == 1
    assert report.failure_modes[FailureMode.WRONG_TOOL] == 1


async def test_run_eval_retries_then_errors_on_persistent_raise() -> None:
    calls: list[str] = []

    async def scorer(case: IVREvalCase) -> CaseResult:
        calls.append(case.id)
        raise RuntimeError("api down")

    report = await run_eval([_case("a")], scorer, layer="ivr", per_case_retries=1)
    assert report.errored == 1
    assert report.results[0].error is not None
    assert len(calls) == 2  # initial attempt + 1 retry


async def test_run_eval_retry_succeeds_on_second_attempt() -> None:
    attempts: list[str] = []

    async def scorer(case: IVREvalCase) -> CaseResult:
        attempts.append(case.id)
        if len(attempts) == 1:
            raise RuntimeError("transient")
        return CaseResult(case_id=case.id, outcome=EvalOutcome.PASS)

    report = await run_eval([_case("a")], scorer, layer="ivr", per_case_retries=1)
    assert report.passed == 1
    assert len(attempts) == 2


async def test_run_eval_aggregate_retry_budget_caps_total_retries() -> None:
    calls: list[str] = []

    async def scorer(case: IVREvalCase) -> CaseResult:
        calls.append(case.id)
        raise RuntimeError("down")

    # 3 cases, each would retry once = 3 retries wanted, but budget is 1.
    report = await run_eval(
        [_case("a"), _case("b"), _case("c")],
        scorer,
        layer="ivr",
        per_case_retries=1,
        total_retry_budget=1,
    )
    assert report.errored == 3  # all error out
    # Directly pin the cap: 3 initial attempts + exactly 1 budgeted retry = 4
    # calls. Without the budget the count would be 6 (each case retrying once),
    # so this assertion is what actually distinguishes a working cap from none.
    assert len(calls) == 4


async def test_run_eval_returned_error_is_not_retried() -> None:
    """Only RAISES are retried. A scorer that RETURNS an ERROR-outcome
    CaseResult has made a final scoring decision — the runner records it as-is
    and calls the scorer exactly once."""
    calls: list[str] = []

    async def scorer(case: IVREvalCase) -> CaseResult:
        calls.append(case.id)
        return CaseResult(case_id=case.id, outcome=EvalOutcome.ERROR, error="unscoreable")

    report = await run_eval([_case("a")], scorer, layer="ivr", per_case_retries=3)
    assert report.errored == 1
    assert len(calls) == 1  # not retried despite per_case_retries=3


async def test_run_eval_max_cases_truncates_and_logs() -> None:
    async def scorer(case: IVREvalCase) -> CaseResult:
        return CaseResult(case_id=case.id, outcome=EvalOutcome.PASS)

    with structlog.testing.capture_logs() as captured:
        report = await run_eval(
            [_case("a"), _case("b"), _case("c")], scorer, layer="ivr", max_cases=2
        )
    assert report.total == 2
    assert any(e.get("event") == "eval_corpus_truncated" for e in captured)
