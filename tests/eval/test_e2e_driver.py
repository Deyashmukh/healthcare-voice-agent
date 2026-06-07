# pyright: strict
"""Hermetic tests for the E2E driver, scripted payer, and terminal scorer.
Uses fake LLM clients producing scripted tool calls / rep outputs — zero network.
"""

from __future__ import annotations

import pytest

from agent.call_session import CallSessionRunner
from agent.eval._types import EvalOutcome, FailureMode
from agent.eval.e2e_trajectory import _driver
from agent.eval.e2e_trajectory._driver import RecordingActuator, run_scenario, score_scenario
from agent.eval.e2e_trajectory._scripted_payer import Scenario
from agent.schemas import (
    Benefits,
    CallSession,
    IVRTurnResponse,
    PatientInfo,
    RepTurnOutput,
    ToolCall,
)
from agent.tools import dispatch
from tests.unit.conftest import FakeAnthropicRepClient, FakeIVRLLMClient


def _runner(
    ivr: FakeIVRLLMClient, rep: FakeAnthropicRepClient, actuator: RecordingActuator
) -> CallSessionRunner:
    session = CallSession(
        call_sid="E2E-test",
        patient=PatientInfo(member_id="M1", first_name="A", last_name="B", dob="1990-01-01"),
    )
    return CallSessionRunner(
        session=session,
        ivr_llm=ivr,
        rep_llm=rep,
        tool_dispatcher=dispatch,
        ivr_system_prompt="ivr",
        rep_system_prompt="rep",
        tools=[],
        actuator=actuator,
    )


async def test_wait_for_turn_gives_up_cleanly_when_no_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Poll-cap exhaustion: if a turn never advances, the driver gives up cleanly
    # (returns session.done == False) instead of hanging forever. Shrink the cap
    # so the test is fast.
    monkeypatch.setattr(_driver, "_POLL_ITERATIONS", 2)
    runner = _runner(
        FakeIVRLLMClient(responses=[]), FakeAnthropicRepClient(responses=[]), RecordingActuator()
    )
    # Runner not started → turn_count stays 0 and the call is never done.
    ended = await _driver._wait_for_turn(  # pyright: ignore[reportPrivateUsage]
        runner, runner.session.turn_count
    )
    assert ended is False


async def test_happy_path_reaches_rep_complete() -> None:
    # IVR: press rep digit, then transfer_to_rep. Rep: extract then complete.
    ivr = FakeIVRLLMClient(
        responses=[
            IVRTurnResponse(
                tool_calls=[ToolCall(name="send_dtmf", args={"digits": "0", "purpose": "rep"})]
            ),
            IVRTurnResponse(tool_calls=[ToolCall(name="transfer_to_rep", args={})]),
        ]
    )
    rep = FakeAnthropicRepClient(
        responses=[
            RepTurnOutput(
                reply="Hi, this is Morgan.", extracted=Benefits(copay=40.0), phase="extracting"
            ),
            RepTurnOutput(reply="Thanks so much!", extracted=Benefits(), phase="complete"),
        ]
    )
    actuator = RecordingActuator()
    runner = _runner(ivr, rep, actuator)
    scenario = Scenario(
        id="t-happy",
        script=("press 0 for a rep", "connecting you", "hi this is Jamie", "copay is 40", "bye"),
        expected_completion_reason="rep_complete",
        expected_benefits=Benefits(copay=40.0),
    )
    await runner.start()
    try:
        await run_scenario(runner, scenario)
    finally:
        await runner.stop()
    assert runner.session.completion_reason == "rep_complete"
    result = score_scenario(scenario, runner)
    assert result.outcome is EvalOutcome.PASS


async def test_scorer_flags_wrong_completion_reason() -> None:
    scenario = Scenario(id="t", script=("x",), expected_completion_reason="rep_complete")
    rep = FakeAnthropicRepClient(responses=[])
    runner = _runner(FakeIVRLLMClient(responses=[]), rep, RecordingActuator())
    runner.session.completion_reason = "rep_stuck"
    result = score_scenario(scenario, runner)
    assert result.outcome is EvalOutcome.FAIL
    assert result.failure_mode is FailureMode.WRONG_COMPLETION_REASON


async def test_scorer_flags_benefits_mismatch() -> None:
    scenario = Scenario(
        id="t",
        script=("x",),
        expected_completion_reason="rep_complete",
        expected_benefits=Benefits(copay=40.0),
    )
    runner = _runner(
        FakeIVRLLMClient(responses=[]), FakeAnthropicRepClient(responses=[]), RecordingActuator()
    )
    runner.session.completion_reason = "rep_complete"
    runner.session.benefits = Benefits(copay=30.0)
    result = score_scenario(scenario, runner)
    assert result.outcome is EvalOutcome.FAIL
    assert result.failure_mode is FailureMode.MISSED_EXTRACTION
