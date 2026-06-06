# pyright: strict
"""Type models for the eval harness.

Cases reuse the production `Turn` / `Benefits` / `ToolName` types so a schema
change can't silently desync the corpus — a `ToolName` literal change breaks
corpus loading at parse time instead of mis-scoring at runtime.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from agent.schemas import Benefits, ToolName, Turn


class EvalOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"  # the case could not be scored (API error, malformed response)


class FailureMode(StrEnum):
    WRONG_TOOL = "wrong_tool"
    BAD_ARG = "bad_arg"
    MISSED_EXTRACTION = "missed_extraction"
    HALLUCINATED_FIELD = "hallucinated_field"
    WRONG_PHASE = "wrong_phase"
    EMPTY_REPLY = "empty_reply"
    PREMATURE_COMPLETE = "premature_complete"
    WRONG_COMPLETION_REASON = "wrong_completion_reason"


class EvalCase(BaseModel):
    """Shared base for every eval case: a stable `id` used in results and logs.
    Bounding the loader/runner typevars to this (not bare `BaseModel`) makes
    `case.id` a typed attribute, so a future case model that forgets `id` is a
    type error rather than a silent `"<unknown>"` at runtime."""

    # Reject unknown fields so a misspelled or stale corpus key ('expcted_args',
    # a field renamed in a refactor) fails loudly at load time as a CorpusError,
    # instead of being silently dropped and scoring the case against defaults.
    model_config = ConfigDict(extra="forbid")

    id: str


class IVREvalCase(EvalCase):
    """One IVR tool-choice case. `history` MUST carry the menu as transcript
    text — that is the only channel the LLM sees menu options through."""

    payer: str
    history: list[Turn]
    expected_tool: ToolName
    expected_args: dict[str, object] = Field(default_factory=dict)  # deterministic args only
    rationale: str


class RepEvalCase(EvalCase):
    """One rep-extraction case. `history` is POST-FLIP (rep-phase) turns only,
    matching what `_rep_turn` sends after slicing at `rep_mode_index`."""

    history: list[Turn]
    expected_extracted: Benefits
    expected_phase: Literal["extracting", "complete", "stuck"]
    expect_nonempty_reply: bool = True
    rationale: str


class CaseResult(BaseModel):
    case_id: str
    outcome: EvalOutcome
    failure_mode: FailureMode | None = None
    detail: str = ""  # human-readable note (expected vs actual)
    error: str | None = None  # populated only when outcome is ERROR


class ScoreReport(BaseModel):
    layer: str
    total: int
    passed: int
    failed: int
    errored: int
    pass_rate: float
    failure_modes: dict[FailureMode, int]
    results: list[CaseResult]

    @classmethod
    def from_results(cls, *, layer: str, results: list[CaseResult]) -> Self:
        passed = sum(1 for r in results if r.outcome is EvalOutcome.PASS)
        failed = sum(1 for r in results if r.outcome is EvalOutcome.FAIL)
        errored = sum(1 for r in results if r.outcome is EvalOutcome.ERROR)
        modes = Counter(r.failure_mode for r in results if r.failure_mode is not None)
        total = len(results)
        return cls(
            layer=layer,
            total=total,
            passed=passed,
            failed=failed,
            errored=errored,
            # pass_rate is over ALL cases including ERROR; B/C scorers exclude
            # ERROR from accuracy by reporting it separately when they need to.
            pass_rate=(passed / total) if total else 0.0,
            failure_modes=dict(modes),
            results=results,
        )
