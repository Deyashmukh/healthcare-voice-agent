# pyright: strict
"""Pure scorer for the IVR tool-choice eval.

Scores the LLM's first tool call against the case's expected tool + deterministic
args. No network — the live call lives in eval.py. Tool name is exact-matched;
only the args listed in `expected_args` are checked (freeform args like
`speak.text` are intentionally not asserted here).

Caveat: a missing tool call is scored FAIL/WRONG_TOOL. `GroqToolCallingClient`
swallows transient API errors into an empty `IVRTurnResponse`, so a provider blip
also surfaces here as "no tool call" rather than as a runner ERROR. This is a
known, documented limitation (see the eval-harness spec §6); the dominant signal
— the model picking the wrong tool or arg — scores correctly.
"""

from __future__ import annotations

from pathlib import Path

from agent.eval._types import CaseResult, EvalOutcome, FailureMode, IVREvalCase
from agent.schemas import IVRTurnResponse

CORPUS = Path(__file__).parent / "corpus" / "cases.jsonl"


def score_ivr(case: IVREvalCase, response: IVRTurnResponse) -> CaseResult:
    if not response.tool_calls:
        return CaseResult(
            case_id=case.id,
            outcome=EvalOutcome.FAIL,
            failure_mode=FailureMode.WRONG_TOOL,
            detail=f"expected {case.expected_tool}, got no tool call",
        )
    call = response.tool_calls[0]
    if call.name != case.expected_tool:
        return CaseResult(
            case_id=case.id,
            outcome=EvalOutcome.FAIL,
            failure_mode=FailureMode.WRONG_TOOL,
            detail=f"expected {case.expected_tool}, got {call.name}",
        )
    mismatched = {
        key: {"expected": value, "actual": call.args.get(key)}
        for key, value in case.expected_args.items()
        if call.args.get(key) != value
    }
    if mismatched:
        return CaseResult(
            case_id=case.id,
            outcome=EvalOutcome.FAIL,
            failure_mode=FailureMode.BAD_ARG,
            detail=f"arg mismatch: {mismatched}",
        )
    return CaseResult(case_id=case.id, outcome=EvalOutcome.PASS)
