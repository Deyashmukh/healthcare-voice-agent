# pyright: strict
"""Pure scorer for the IVR tool-choice eval.

Scores the LLM's first tool call against the case's expected tool + deterministic
args. No network — the live call lives in eval.py. Tool name is exact-matched;
only the args listed in `expected_args` are checked (freeform args like
`speak.text` are intentionally not asserted here).

Arg values are compared as strings. `ToolCall.args` is the raw `json.loads` of
the model's output (no per-key Pydantic coercion), so a digit may come back as a
JSON number (`2`) or string (`"2"`); a string compare keeps that provider quirk
from mislabeling a correct press as `BAD_ARG` — which would silently understate
accuracy in the instrument itself. The deterministic args we assert (digits,
purpose, reason) are all cleanly stringifiable.

No tool call is NOT a behavioral FAIL — it RAISES `NoToolCallError`. The IVR LLM
runs with `tool_choice="required"`, so the model should always emit exactly one
tool call; an empty response means a transient provider glitch (a 429/5xx, or a
`tool_use_failed` token-degeneration that `GroqToolCallingClient` swallows into
an empty `IVRTurnResponse`), not a tool choice. Raising routes it through the
runner's retry-then-ERROR path: a transient blip is absorbed by the retry, and a
persistent one is recorded as ERROR (infra, not the model) rather than a false
`wrong_tool` FAIL that would silently understate the model. (Resolves spec §6 /
SF-3.)
"""

from __future__ import annotations

from pathlib import Path

from agent.errors import AgentError
from agent.eval._types import CaseResult, EvalOutcome, FailureMode, IVREvalCase
from agent.schemas import IVRTurnResponse

CORPUS = Path(__file__).parent / "corpus" / "cases.jsonl"


class NoToolCallError(AgentError):
    """The IVR LLM returned no tool call under tool_choice=required — a transient
    provider glitch, not a tool choice. Raised so the runner retries then records
    ERROR instead of a false WRONG_TOOL FAIL."""


def score_ivr(case: IVREvalCase, response: IVRTurnResponse) -> CaseResult:
    if not response.tool_calls:
        raise NoToolCallError(
            f"{case.id}: expected {case.expected_tool}, got no tool call "
            "(transient Groq tool_use_failed / API blip under tool_choice=required)"
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
        if str(call.args.get(key)) != str(value)
    }
    if mismatched:
        return CaseResult(
            case_id=case.id,
            outcome=EvalOutcome.FAIL,
            failure_mode=FailureMode.BAD_ARG,
            detail=f"arg mismatch: {mismatched}",
        )
    return CaseResult(case_id=case.id, outcome=EvalOutcome.PASS)
