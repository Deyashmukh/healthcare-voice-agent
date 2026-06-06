# pyright: strict
"""Pure scorer for the rep-extraction eval.

Scores the rep LLM's single-turn output against the case. Checks in priority
order: the extracted Benefits delta (the business outcome), then the phase, then
reply-presence. Extraction is compared on non-None fields only — the rep model
emits just the fields learned this turn. No network; the live call is in eval.py.
"""

from __future__ import annotations

from pathlib import Path

from agent.eval._types import CaseResult, EvalOutcome, FailureMode, RepEvalCase
from agent.schemas import RepTurnOutput

CORPUS = Path(__file__).parent / "corpus" / "cases.jsonl"


def score_rep(case: RepEvalCase, output: RepTurnOutput) -> CaseResult:
    expected = case.expected_extracted.model_dump(exclude_none=True)
    actual = output.extracted.model_dump(exclude_none=True)
    if actual != expected:
        missing = set(expected) - set(actual)
        extra = set(actual) - set(expected)
        if missing:
            mode = FailureMode.MISSED_EXTRACTION
        elif extra:
            mode = FailureMode.HALLUCINATED_FIELD
        else:
            mode = FailureMode.BAD_ARG  # same fields, different values
        return CaseResult(
            case_id=case.id,
            outcome=EvalOutcome.FAIL,
            failure_mode=mode,
            detail=f"extracted {actual}, expected {expected}",
        )
    if output.phase != case.expected_phase:
        premature = case.expected_phase == "extracting" and output.phase == "complete"
        return CaseResult(
            case_id=case.id,
            outcome=EvalOutcome.FAIL,
            failure_mode=FailureMode.PREMATURE_COMPLETE if premature else FailureMode.WRONG_PHASE,
            detail=f"phase {output.phase}, expected {case.expected_phase}",
        )
    if case.expect_nonempty_reply and not output.reply.strip():
        return CaseResult(
            case_id=case.id,
            outcome=EvalOutcome.FAIL,
            failure_mode=FailureMode.EMPTY_REPLY,
            detail="empty reply where a non-empty one was required",
        )
    return CaseResult(case_id=case.id, outcome=EvalOutcome.PASS)
