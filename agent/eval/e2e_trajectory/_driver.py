# pyright: strict
"""E2E trajectory driver: run the real runner against a scripted payer, then
score the terminal state.

`RecordingActuator` captures the agent's intents instead of doing TTS/DTMF.
`run_scenario` owns the turn loop: submit the opening line, then feed one
scripted line per completed turn (repeating the last line if the agent needs
extra turns to wrap up), until `session.done` or a turn cap. `score_scenario`
checks `completion_reason` and (if the scenario specifies them) the final
benefits. No barge-in / VAD here — that path is out of scope by design.
"""

from __future__ import annotations

import asyncio

from agent.call_session import CallSessionRunner
from agent.eval._types import CaseResult, EvalOutcome, FailureMode
from agent.eval.e2e_trajectory._scripted_payer import Scenario
from agent.schemas import SideEffectIntent

_POLL_INTERVAL_S = 0.01
_POLL_ITERATIONS = 3000  # ~30s ceiling per turn; live rep turns are a few seconds


class RecordingActuator:
    """Satisfies the runner's `Actuator` protocol; records intents instead of
    performing TTS/DTMF so the scripted payer side does no real I/O."""

    def __init__(self) -> None:
        self.intents: list[SideEffectIntent] = []

    async def execute(self, intent: SideEffectIntent) -> None:
        self.intents.append(intent)


async def _wait_for_turn(runner: CallSessionRunner, prev_turns: int) -> bool:
    """Poll until the turn count advances or the call ends. Returns True if the
    call ended (`session.done`). Iteration-capped, not wall-clock, so a slow
    scheduler doesn't trip a spurious timeout."""
    for _ in range(_POLL_ITERATIONS):
        if runner.session.done:
            return True
        if runner.session.turn_count > prev_turns:
            return False
        await asyncio.sleep(_POLL_INTERVAL_S)
    return runner.session.done


async def run_scenario(runner: CallSessionRunner, scenario: Scenario) -> None:
    """Drive `runner` through `scenario`. Caller owns runner.start()/stop()."""
    runner.submit_transcript(scenario.script[0])
    prev_turns = runner.session.turn_count  # captured ONCE, advanced per turn
    line = 1
    for _ in range(scenario.max_turns):
        ended = await _wait_for_turn(runner, prev_turns)
        if runner.session.done or ended:
            return
        prev_turns = runner.session.turn_count
        next_line = scenario.script[line] if line < len(scenario.script) else scenario.script[-1]
        runner.submit_transcript(next_line)
        line += 1


def score_scenario(scenario: Scenario, runner: CallSessionRunner) -> CaseResult:
    session = runner.session
    if session.completion_reason != scenario.expected_completion_reason:
        return CaseResult(
            case_id=scenario.id,
            outcome=EvalOutcome.FAIL,
            failure_mode=FailureMode.WRONG_COMPLETION_REASON,
            detail=f"completion_reason={session.completion_reason}, "
            f"expected {scenario.expected_completion_reason}",
        )
    if scenario.expected_benefits is not None:
        expected = scenario.expected_benefits.model_dump(exclude_none=True)
        actual = session.benefits.model_dump(exclude_none=True)
        if actual != expected:
            return CaseResult(
                case_id=scenario.id,
                outcome=EvalOutcome.FAIL,
                failure_mode=FailureMode.MISSED_EXTRACTION,
                detail=f"benefits={actual}, expected {expected}",
            )
    return CaseResult(case_id=scenario.id, outcome=EvalOutcome.PASS)
