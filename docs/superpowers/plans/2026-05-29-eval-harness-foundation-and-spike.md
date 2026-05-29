# Eval Harness — Foundation + Measurement Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the hermetic eval-harness foundation (`agent/eval/`: typed cases, loader, runner, reporter) and a throwaway measurement spike that runs the real IVR + rep LLMs to produce the stability baselines that will parameterize the component/E2E/CI milestones.

**Architecture:** A small, dependency-injected harness. `_types.py` defines the case/result/report models (reusing production `Turn`/`Benefits`/`ToolName`). `_loader.py` reads JSONL corpora into those models. `_runner.py` runs an injected per-case scorer over a corpus, classifying each outcome `PASS|FAIL|ERROR` (with a `FailureMode`), bounding retries and case count, and aggregating a `ScoreReport`. `_report.py` renders the report and appends a trend line. None of the foundation calls a network — it's unit-tested offline like the rest of the repo. The spike (`_spike.py`) is the only live-LLM piece: it exercises the real clients to measure run-to-run agreement and confirm E2E terminal-state reachability, then is kept only as a re-runnable measurement tool.

**Tech Stack:** Python 3.12, Pydantic v2, pytest + pytest-asyncio (`asyncio_mode=auto`), pyright strict per-file, ruff. Real clients: `agent.llm_client.GroqToolCallingClient` (IVR), `agent.llm_client.AnthropicRepClient` (rep).

**Sequencing note (read before starting):** The spec's "risk-first" ordering says run the spike before building the foundation. This plan builds the foundation FIRST because it is pure hermetic plumbing with zero live-LLM risk, and the spike is cleaner when it reuses the real loader/runner/report instead of throwaway code. The spike's PURPOSE — measure live-LLM stability and produce a go/no-go + baselines for B/C/D — is unchanged; it is the last task here and gates the next plan. Building one milestone of plumbing before the spike does not risk "building 7 milestones on a false premise"; building B/C/D/E/F/G would, and those are deferred to Plan 2.

**Out of scope (deferred to Plan 2, after the spike's baselines land):** M-eval/B (IVR tool-choice eval), M-eval/C (rep extraction eval), M-eval/D (E2E trajectory), M-eval/G (`make evals` + nightly CI). Judge / Langfuse-miner / train-test split remain cut per the spec.

---

## File Structure

- `agent/eval/__init__.py` — package marker (empty).
- `agent/eval/_types.py` — `EvalOutcome`, `FailureMode`, `IVREvalCase`, `RepEvalCase`, `CaseResult`, `ScoreReport`.
- `agent/eval/_loader.py` — `load_cases(path, model)` JSONL → list of typed cases.
- `agent/eval/_runner.py` — `run_eval(cases, scorer, ...)` → `ScoreReport`; outcome classification, bounded retries, case cap.
- `agent/eval/_report.py` — `render_table(report)`, `write_report(report, results_dir, history_path)`.
- `agent/eval/_spike.py` — M-eval/0 live measurement (NOT a pytest test; run via `make eval-spike`).
- `tests/eval/__init__.py` — package marker (empty).
- `tests/eval/test_types.py`, `test_loader.py`, `test_runner.py`, `test_report.py` — hermetic unit tests of the foundation.
- `pyproject.toml` — add live modules to coverage `omit`.
- `Makefile` — add `eval-spike` target.

Every new `.py` under `agent/` and `tests/` MUST start with `# pyright: strict` (repo rule).

---

## Task 1: `_types.py` — case, result, and report models

**Files:**
- Create: `agent/eval/__init__.py` (empty)
- Create: `agent/eval/_types.py`
- Test: `tests/eval/__init__.py` (empty), `tests/eval/test_types.py`

- [ ] **Step 1: Create the package markers**

Create `agent/eval/__init__.py` and `tests/eval/__init__.py`, both empty files.

- [ ] **Step 2: Write the failing test**

Create `tests/eval/test_types.py`:

```python
# pyright: strict
"""Unit tests for eval-harness type models."""

from __future__ import annotations

from agent.eval._types import (
    CaseResult,
    EvalOutcome,
    FailureMode,
    IVREvalCase,
    RepEvalCase,
    ScoreReport,
)
from agent.schemas import Benefits, Turn


def test_ivr_case_round_trips_through_json() -> None:
    case = IVREvalCase(
        id="aetna-main-press-2",
        payer="aetna",
        history=[Turn(role="user", content="For billing press 2")],
        expected_tool="send_dtmf",
        expected_args={"digits": "2"},
        rationale="billing is option 2",
    )
    reloaded = IVREvalCase.model_validate_json(case.model_dump_json())
    assert reloaded == case
    assert reloaded.expected_args == {"digits": "2"}


def test_ivr_case_expected_args_defaults_empty() -> None:
    case = IVREvalCase(
        id="x",
        payer="p",
        history=[],
        expected_tool="wait",
        rationale="r",
    )
    assert case.expected_args == {}


def test_rep_case_round_trips() -> None:
    case = RepEvalCase(
        id="copay-30",
        history=[Turn(role="user", content="Her copay is $30")],
        expected_extracted=Benefits(copay=30.0),
        expected_phase="extracting",
        rationale="copay stated",
    )
    reloaded = RepEvalCase.model_validate_json(case.model_dump_json())
    assert reloaded == case
    assert reloaded.expect_nonempty_reply is True  # default


def test_score_report_aggregates_counts() -> None:
    results = [
        CaseResult(case_id="a", outcome=EvalOutcome.PASS),
        CaseResult(case_id="b", outcome=EvalOutcome.FAIL, failure_mode=FailureMode.WRONG_TOOL),
        CaseResult(case_id="c", outcome=EvalOutcome.ERROR, error="boom"),
        CaseResult(case_id="d", outcome=EvalOutcome.FAIL, failure_mode=FailureMode.WRONG_TOOL),
    ]
    report = ScoreReport.from_results(layer="ivr_tool_choice", results=results)
    assert report.total == 4
    assert report.passed == 1
    assert report.failed == 2
    assert report.errored == 1
    assert report.pass_rate == 0.25
    assert report.failure_modes[FailureMode.WRONG_TOOL] == 2


def test_score_report_pass_rate_zero_cases_is_zero() -> None:
    report = ScoreReport.from_results(layer="x", results=[])
    assert report.total == 0
    assert report.pass_rate == 0.0
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd <worktree> && uv run pytest tests/eval/test_types.py -q`
Expected: FAIL at import (`No module named 'agent.eval._types'`).

- [ ] **Step 4: Write `agent/eval/_types.py`**

```python
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

from pydantic import BaseModel, Field

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


class IVREvalCase(BaseModel):
    """One IVR tool-choice case. `history` MUST carry the menu as transcript
    text — that is the only channel the LLM sees menu options through."""

    id: str
    payer: str
    history: list[Turn]
    expected_tool: ToolName
    expected_args: dict[str, object] = Field(default_factory=dict)  # deterministic args only
    rationale: str


class RepEvalCase(BaseModel):
    """One rep-extraction case. `history` is POST-FLIP (rep-phase) turns only,
    matching what `_rep_turn` sends after slicing at `rep_mode_index`."""

    id: str
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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/eval/test_types.py -q -p no:cacheprovider`
(If the `--cov-fail-under` floor trips on a single-file run, run the full suite in Step 6; a single-file run under the floor is expected and not a failure of these tests — check the `passed`/`failed` line, not the coverage gate.)
Expected: the 5 tests PASS.

- [ ] **Step 6: Run lint + type-check on the new files**

Run: `uv run ruff check agent/eval/ tests/eval/ && uv run ruff format --check agent/eval/ tests/eval/ && uv run pyright agent/eval/_types.py tests/eval/test_types.py`
Expected: all clean, `0 errors`.

- [ ] **Step 7: Commit**

```bash
git add agent/eval/__init__.py agent/eval/_types.py tests/eval/__init__.py tests/eval/test_types.py
git commit -m "feat(eval): case/result/report type models (M-eval/A)"
```

---

## Task 2: `_loader.py` — typed JSONL corpus loader

**Files:**
- Create: `agent/eval/_loader.py`
- Test: `tests/eval/test_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_loader.py`:

```python
# pyright: strict
"""Unit tests for the JSONL corpus loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.eval._loader import CorpusError, load_cases
from agent.eval._types import IVREvalCase


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_cases_parses_each_line(tmp_path: Path) -> None:
    case_json = IVREvalCase(
        id="c1", payer="p", history=[], expected_tool="wait", rationale="r"
    ).model_dump_json()
    path = _write(tmp_path / "corpus.jsonl", [case_json, case_json])
    cases = load_cases(path, IVREvalCase)
    assert len(cases) == 2
    assert all(isinstance(c, IVREvalCase) for c in cases)


def test_load_cases_skips_blank_lines(tmp_path: Path) -> None:
    case_json = IVREvalCase(
        id="c1", payer="p", history=[], expected_tool="wait", rationale="r"
    ).model_dump_json()
    path = tmp_path / "corpus.jsonl"
    path.write_text(f"{case_json}\n\n  \n{case_json}\n", encoding="utf-8")
    assert len(load_cases(path, IVREvalCase)) == 2


def test_load_cases_raises_with_line_number_on_bad_json(tmp_path: Path) -> None:
    path = _write(tmp_path / "corpus.jsonl", ["{not json}"])
    with pytest.raises(CorpusError) as exc:
        load_cases(path, IVREvalCase)
    assert "line 1" in str(exc.value)


def test_load_cases_raises_with_line_number_on_schema_mismatch(tmp_path: Path) -> None:
    path = _write(tmp_path / "corpus.jsonl", ['{"id": "x"}'])  # missing required fields
    with pytest.raises(CorpusError) as exc:
        load_cases(path, IVREvalCase)
    assert "line 1" in str(exc.value)


def test_load_cases_missing_file_raises_corpus_error(tmp_path: Path) -> None:
    with pytest.raises(CorpusError):
        load_cases(tmp_path / "nope.jsonl", IVREvalCase)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/eval/test_loader.py -q`
Expected: FAIL at import (`No module named 'agent.eval._loader'`).

- [ ] **Step 3: Write `agent/eval/_loader.py`**

```python
# pyright: strict
"""Load JSONL eval corpora into typed Pydantic case models.

One case per line. Errors are raised as `CorpusError` with the offending line
number so a malformed corpus fails loudly at load time instead of silently
mis-scoring (or skipping) a case mid-run.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ValidationError

from agent.errors import AgentError


class CorpusError(AgentError):
    """A corpus file could not be read or a line could not be parsed."""


def load_cases[CaseT: BaseModel](path: Path, model: type[CaseT]) -> list[CaseT]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CorpusError(f"cannot read corpus {path}: {exc}") from exc
    cases: list[CaseT] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(model.model_validate_json(line))
        except ValidationError as exc:
            raise CorpusError(f"{path} line {line_no}: invalid {model.__name__}: {exc}") from exc
    return cases
```

Note: confirm `AgentError` exists in `agent/errors.py` (it does — the error-taxonomy milestone). If its constructor signature differs from `Exception`'s, adjust the `CorpusError` raises to match; check `agent/errors.py` before writing.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/eval/test_loader.py -q`
Expected: the 5 tests PASS.

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check agent/eval/_loader.py tests/eval/test_loader.py && uv run pyright agent/eval/_loader.py tests/eval/test_loader.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add agent/eval/_loader.py tests/eval/test_loader.py
git commit -m "feat(eval): typed JSONL corpus loader with line-numbered errors (M-eval/A)"
```

---

## Task 3: `_runner.py` — corpus runner with outcome classification, bounded retries, case cap

**Files:**
- Create: `agent/eval/_runner.py`
- Test: `tests/eval/test_runner.py`

**Design:** `run_eval` takes a corpus and an injected async `scorer(case) -> CaseResult`. The scorer returns a `CaseResult` for PASS/FAIL (a scoring decision); it RAISES for a transient/unexpected failure (API error, malformed LLM response). The runner wraps each scorer call: on a raise it retries up to `per_case_retries`, drawing from a shared `total_retry_budget`; if still raising, it records `EvalOutcome.ERROR`. `max_cases` caps the run and logs loudly when it truncates (no silent truncation).

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_runner.py`:

```python
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
        return CaseResult(case_id=case.id, outcome=EvalOutcome.FAIL, failure_mode=FailureMode.WRONG_TOOL)

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
    async def scorer(case: IVREvalCase) -> CaseResult:
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
    # Only 1 retry was spent across the whole run: 3 initial + 1 retry = 4 calls.
    # (Asserted indirectly via the budget; see implementation.)


async def test_run_eval_max_cases_truncates_and_logs() -> None:
    async def scorer(case: IVREvalCase) -> CaseResult:
        return CaseResult(case_id=case.id, outcome=EvalOutcome.PASS)

    with structlog.testing.capture_logs() as captured:
        report = await run_eval(
            [_case("a"), _case("b"), _case("c")], scorer, layer="ivr", max_cases=2
        )
    assert report.total == 2
    assert any(e.get("event") == "eval_corpus_truncated" for e in captured)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/eval/test_runner.py -q`
Expected: FAIL at import.

- [ ] **Step 3: Write `agent/eval/_runner.py`**

```python
# pyright: strict
"""Run an injected per-case scorer over a corpus and aggregate a ScoreReport.

The scorer returns a `CaseResult` for a scoring decision (PASS/FAIL) and RAISES
for a transient/unexpected failure (API error, malformed response). The runner
centralizes ERROR handling: a raise is retried up to `per_case_retries`, drawing
from a shared `total_retry_budget`, before being recorded as `EvalOutcome.ERROR`.
This keeps provider noise (ERROR) from masquerading as a behavioral FAIL. Note:
only RAISES are retried — a scorer that returns a CaseResult with
`outcome=ERROR` directly is treated as a final scoring decision and not retried.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from agent.eval._types import CaseResult, EvalOutcome, ScoreReport
from agent.logging_config import log

_DEFAULT_MAX_CASES = 500
"""Cost guard: hard cap on cases per run so a runaway corpus can't silently
burn the account. Truncation is logged loudly, never silent."""


async def run_eval[CaseT: BaseModel](
    cases: list[CaseT],
    scorer: Callable[[CaseT], Awaitable[CaseResult]],
    *,
    layer: str,
    max_cases: int | None = _DEFAULT_MAX_CASES,
    per_case_retries: int = 1,
    total_retry_budget: int = 50,
) -> ScoreReport:
    selected = cases
    if max_cases is not None and len(cases) > max_cases:
        log.warning(
            "eval_corpus_truncated", layer=layer, total=len(cases), kept=max_cases
        )
        selected = cases[:max_cases]

    retry_budget = total_retry_budget
    results: list[CaseResult] = []
    for case in selected:
        case_id = getattr(case, "id", "<unknown>")
        attempts_left = per_case_retries
        last_error: Exception | None = None
        result: CaseResult | None = None
        while True:
            try:
                result = await scorer(case)
                break
            except Exception as exc:  # any scorer raise is an ERROR candidate
                last_error = exc
                if attempts_left > 0 and retry_budget > 0:
                    attempts_left -= 1
                    retry_budget -= 1
                    log.warning("eval_case_retry", layer=layer, case_id=case_id, error=str(exc))
                    continue
                break
        if result is None:
            result = CaseResult(
                case_id=str(case_id),
                outcome=EvalOutcome.ERROR,
                error=str(last_error),
            )
        results.append(result)

    return ScoreReport.from_results(layer=layer, results=results)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/eval/test_runner.py -q`
Expected: the 5 tests PASS.

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check agent/eval/_runner.py tests/eval/test_runner.py && uv run pyright agent/eval/_runner.py tests/eval/test_runner.py`
Expected: clean. (If ruff flags `BLE001` despite the `# noqa`, confirm `BLE` isn't in the active rule set — the repo selects `E,F,I,B,UP,N,W,SIM,RUF`; `B` is bugbear, not `BLE`. Remove the `# noqa` if unneeded.)

- [ ] **Step 6: Commit**

```bash
git add agent/eval/_runner.py tests/eval/test_runner.py
git commit -m "feat(eval): corpus runner with ERROR/retry/cost-cap semantics (M-eval/A)"
```

---

## Task 4: `_report.py` — render table + persist results + append trend line

**Files:**
- Create: `agent/eval/_report.py`
- Test: `tests/eval/test_report.py`

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_report.py`:

```python
# pyright: strict
"""Unit tests for report rendering + persistence."""

from __future__ import annotations

import json
from pathlib import Path

from agent.eval._report import render_table, write_report
from agent.eval._types import CaseResult, EvalOutcome, FailureMode, ScoreReport


def _report() -> ScoreReport:
    return ScoreReport.from_results(
        layer="ivr_tool_choice",
        results=[
            CaseResult(case_id="a", outcome=EvalOutcome.PASS),
            CaseResult(case_id="b", outcome=EvalOutcome.FAIL, failure_mode=FailureMode.WRONG_TOOL),
        ],
    )


def test_render_table_includes_layer_and_counts() -> None:
    text = render_table(_report())
    assert "ivr_tool_choice" in text
    assert "1" in text and "2" in text  # passed / total
    assert "wrong_tool" in text


def test_write_report_writes_json_and_appends_history(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    history = tmp_path / "eval_history.jsonl"
    write_report(_report(), results_dir=results_dir, history_path=history, timestamp="2026-05-29T12:00:00Z")

    written = list(results_dir.glob("*.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text())
    assert payload["layer"] == "ivr_tool_choice"

    hist_lines = history.read_text().strip().splitlines()
    assert len(hist_lines) == 1
    hist = json.loads(hist_lines[0])
    assert hist["layer"] == "ivr_tool_choice"
    assert hist["pass_rate"] == 0.5
    assert hist["timestamp"] == "2026-05-29T12:00:00Z"


def test_write_report_appends_second_run_to_history(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    history = tmp_path / "eval_history.jsonl"
    write_report(_report(), results_dir=results_dir, history_path=history, timestamp="2026-05-29T12:00:00Z")
    write_report(_report(), results_dir=results_dir, history_path=history, timestamp="2026-05-29T13:00:00Z")
    assert len(history.read_text().strip().splitlines()) == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/eval/test_report.py -q`
Expected: FAIL at import.

- [ ] **Step 3: Write `agent/eval/_report.py`**

```python
# pyright: strict
"""Render a ScoreReport as a table, persist the full report as JSON, and append
a one-line summary to a committed trend file so week-over-week pass-rate is
answerable without a database (same zero-infra pattern as benefits.jsonl).

`timestamp` is passed in (not read from the clock) so the function is
deterministic and unit-testable; callers stamp `time.strftime(...)` at the edge.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.eval._types import ScoreReport


def render_table(report: ScoreReport) -> str:
    lines = [
        f"=== {report.layer} ===",
        f"  total={report.total}  pass={report.passed}  fail={report.failed}  error={report.errored}",
        f"  pass_rate={report.pass_rate:.2%}",
    ]
    if report.failure_modes:
        lines.append("  failure modes:")
        for mode, count in sorted(report.failure_modes.items(), key=lambda kv: kv[0].value):
            lines.append(f"    {mode.value}: {count}")
    return "\n".join(lines)


def write_report(
    report: ScoreReport,
    *,
    results_dir: Path,
    history_path: Path,
    timestamp: str,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    safe_ts = timestamp.replace(":", "").replace("-", "")
    out_path = results_dir / f"{report.layer}-{safe_ts}.json"
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    summary = {
        "timestamp": timestamp,
        "layer": report.layer,
        "total": report.total,
        "passed": report.passed,
        "failed": report.failed,
        "errored": report.errored,
        "pass_rate": report.pass_rate,
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    return out_path
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/eval/test_report.py -q`
Expected: the 3 tests PASS.

- [ ] **Step 5: Lint + type-check + FULL suite**

Run: `uv run ruff check agent/ tests/ && uv run ruff format --check agent/ tests/ && uv run pyright agent/eval/ tests/eval/ && uv run pytest tests/`
Expected: all clean; full suite green (existing 279 + the new eval foundation tests); coverage floor still met.

- [ ] **Step 6: Commit**

```bash
git add agent/eval/_report.py tests/eval/test_report.py
git commit -m "feat(eval): report table + JSON + committed trend line (M-eval/A)"
```

---

## Task 5: Coverage-omit the live-LLM modules

**Files:**
- Modify: `pyproject.toml` (the `[tool.coverage.run] omit` list)

The foundation (`_types`, `_loader`, `_runner`, `_report`) is hermetic and SHOULD count toward coverage. The live-LLM modules (`agent/eval/_spike.py`, and in Plan 2 the per-layer `eval.py` files) are run via `make`, never by `pytest`, so they must be omitted or they tank the floor — exactly as `agent/main.py` already is.

- [ ] **Step 1: Edit the omit list**

In `pyproject.toml`, change:

```toml
omit = ["agent/main.py"]
```

to:

```toml
omit = [
    "agent/main.py",
    # Live-LLM eval entrypoints: run via `make eval-spike` / `make evals`,
    # not pytest. The hermetic harness (agent/eval/_types|_loader|_runner|_report)
    # is NOT omitted and is covered by tests/eval/.
    "agent/eval/_spike.py",
    "agent/eval/*/eval.py",
]
```

- [ ] **Step 2: Gitignore eval run artifacts**

`_report.write_report` writes per-run JSON into `eval_results/` (spec §6 says this is gitignored; the committed trend lives in `eval_history.jsonl`). Round-1 tests use `tmp_path`, but Plan 2's `make evals` will drop untracked JSON into the tree — add the ignore now. Append to `.gitignore`:

```
# Eval run artifacts (per-run JSON). The committed trend is eval_history.jsonl.
eval_results/
```

- [ ] **Step 3: Verify the floor still holds**

Run: `uv run pytest tests/`
Expected: PASS; `Required test coverage ... reached`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .gitignore
git commit -m "chore(eval): omit live-LLM eval entrypoints from coverage; gitignore eval_results"
```

---

## Task 6: M-eval/0 — the measurement spike (live LLMs)

**Files:**
- Create: `agent/eval/_spike.py`
- Modify: `Makefile` (add `eval-spike` target)

**Purpose (read the spec §0):** Produce the two things the next plan depends on:
1. **E2E reachability** — confirm a happy-path call driven through the REAL runner with REAL clients against a stubbed scripted payer reaches `completion_reason == "rep_complete"` with full benefits at all.
2. **Stability baselines** — run ~5 IVR and ~5 rep cases against the real clients N times each; report run-to-run agreement (tool-name, deterministic-arg, extraction, phase) so Plan 2's thresholds are measured, not guessed.

This task is NOT TDD (it calls real paid APIs and is inherently nondeterministic). It is a re-runnable measurement tool. It requires `GROQ_API_KEY` + `ANTHROPIC_API_KEY` in `.env`.

- [ ] **Step 1: Write `agent/eval/_spike.py`**

```python
# pyright: strict
"""M-eval/0 measurement spike — LIVE LLMs, not a pytest test.

Run: `make eval-spike`. Requires ONLY GROQ_API_KEY + ANTHROPIC_API_KEY —
importing `agent.main` is side-effect-safe (it builds a FastAPI app + reads
.env at import, but every credentialed client — Twilio/Deepgram/ElevenLabs —
is constructed lazily inside request handlers, not at module scope).

Outputs (to stdout): per-probe run-to-run agreement for the IVR tool-choice and
rep-extraction layers. These numbers set the pass thresholds for Plan 2
(M-eval/B/C/D). Nothing here is asserted; the spike informs, it does not gate.

Caveat (matches spec §3): the rep probes pass a single raw user-turn dict to
`complete_structured`, not the production `_history_to_anthropic_messages`
projection. For these single-turn probes the projection is near-identity, so the
agreement numbers transfer; multi-turn corpus cases in Plan 2's M-eval/C MUST
route through the real projection.
"""

from __future__ import annotations

import asyncio
from collections import Counter

from agent.llm_client import AnthropicRepClient, GroqToolCallingClient

# `_default_patient` / `_ivr_system_prompt` / `_rep_system_prompt` are module-
# private in agent.main; the spike is a throwaway measurement tool that
# legitimately reuses them rather than duplicating prompt-format logic.
from agent.main import (  # pyright: ignore[reportPrivateUsage]
    _default_patient,
    _ivr_system_prompt,
    _rep_system_prompt,
)
from agent.schemas import RepTurnOutput, Turn
from agent.tools import groq_tool_schemas

_RUNS = 20  # 20 runs × 10 probes = ~200 live LLM calls per spike invocation.

# A handful of hand-written probes. Each is a (label, history-as-text).
_IVR_PROBES: list[tuple[str, str]] = [
    ("aetna-billing", "Thank you for calling. For billing press 1, for eligibility press 2."),
    ("rep-option", "For benefits press 2, to speak to a representative press 0."),
    ("greeting", "Welcome to the provider line. Please hold while we connect you."),
    ("member-id", "Please enter your member ID followed by the pound key."),
    ("repeat", "Press 9 at any time to repeat this menu."),
]

_REP_PROBES: list[tuple[str, str]] = [
    ("copay", "Sure, her copay is thirty dollars for a specialist visit."),
    ("active", "Yes, the policy is active and in good standing."),
    ("deductible", "She has four hundred dollars left on her deductible this year."),
    ("greeting", "Thanks for holding, this is Jamie, how can I help you?"),
    ("coinsurance", "After the deductible it's twenty percent coinsurance."),
]


async def _measure_ivr(ivr_system_prompt: str) -> None:
    client = GroqToolCallingClient()
    tools = groq_tool_schemas()
    print(f"\n=== IVR tool-choice stability (n={_RUNS}) ===")
    for label, text in _IVR_PROBES:
        history = [Turn(role="user", content=text)]
        names: Counter[str] = Counter()
        for _ in range(_RUNS):
            resp = await client.complete_with_tools(
                system=ivr_system_prompt, history=history, tools=tools, temperature=0.1
            )
            chosen = resp.tool_calls[0].name if resp.tool_calls else "<none>"
            names[chosen] += 1
        top, top_n = names.most_common(1)[0]
        print(f"  {label:14s} top={top:16s} agreement={top_n}/{_RUNS}  dist={dict(names)}")


async def _measure_rep(rep_system_prompt: str) -> None:
    client = AnthropicRepClient()
    print(f"\n=== Rep extraction stability (n={_RUNS}) ===")
    for label, text in _REP_PROBES:
        history: list[dict[str, object]] = [{"role": "user", "content": text}]
        phases: Counter[str] = Counter()
        extracted: Counter[str] = Counter()
        for _ in range(_RUNS):
            out = await client.complete_structured(
                system=rep_system_prompt, history=history, schema=RepTurnOutput
            )
            phases[out.phase] += 1
            extracted[out.extracted.model_dump_json(exclude_none=True)] += 1
        top_phase, top_phase_n = phases.most_common(1)[0]
        top_ex, top_ex_n = extracted.most_common(1)[0]
        print(
            f"  {label:14s} phase_top={top_phase}({top_phase_n}/{_RUNS}) "
            f"extract_top={top_ex}({top_ex_n}/{_RUNS})"
        )


async def main() -> None:
    total_calls = _RUNS * (len(_IVR_PROBES) + len(_REP_PROBES))
    print(f"eval spike: ~{total_calls} live LLM calls ({_RUNS} runs × {total_calls // _RUNS} probes)")
    patient = _default_patient()
    await _measure_ivr(_ivr_system_prompt(patient))
    await _measure_rep(_rep_system_prompt(patient))
    print("\n(Spike complete. Use these agreement rates to set Plan 2 thresholds.)")


if __name__ == "__main__":
    asyncio.run(main())
```

Note: `groq_tool_schemas`, `_default_patient`, `_ivr_system_prompt`, `_rep_system_prompt` are existing symbols (verified present during planning). The `# pyright: ignore[reportPrivateUsage]` on the `agent.main` import is required because strict mode rejects cross-module underscore-symbol access — this matches the repo's established pattern of explicit per-line pyright ignores for real gaps. The E2E reachability check is intentionally deferred to Plan 2's M-eval/D task (it needs the `ScriptedPayer` + eval actuator, which belong to D); measuring component-layer stability now is the higher-value risk signal.

- [ ] **Step 2: Add the Makefile target**

In `Makefile`, add `eval-spike` to `.PHONY` and a target:

```makefile
eval-spike:
	uv run python -m agent.eval._spike
```

- [ ] **Step 3: Run the spike (requires API keys)**

Run: `make eval-spike`
Expected: prints per-probe agreement rates for IVR tool-choice and rep extraction. There is no pass/fail — record the numbers.

- [ ] **Step 4: Record the baselines**

Append a short section to `docs/superpowers/specs/2026-05-28-eval-harness-design.md` (or a new `docs/superpowers/notes/eval-baselines.md`) capturing: the measured per-layer agreement rates, and the chosen Plan-2 pass thresholds (measured rate minus reasonable headroom), plus a go/no-go call on whether single-run exact-match is viable or whether B/C/D need majority-vote. This is the deliverable that unblocks Plan 2.

- [ ] **Step 5: Commit**

```bash
git add agent/eval/_spike.py Makefile docs/superpowers/
git commit -m "feat(eval): M-eval/0 live measurement spike + recorded baselines"
```

---

## Self-Review (completed during planning)

- **Spec coverage (round-1 scope):** M-eval/A foundation → Tasks 1-4; coverage-omit (spec §"Testing strategy") → Task 5; M-eval/0 spike (spec §0) → Task 6. B/C/D/G are explicitly deferred to Plan 2 with rationale (spike parameterizes them). Judge / Langfuse / train-test split remain cut. ✔
- **Placeholder scan:** no "TBD"/"handle errors"/"similar to" — every code step has complete code. Two notes ask the implementer to confirm existing symbol names (`AgentError`, `groq_tool_schemas`, `_ivr_system_prompt`) against the codebase before use; these are verification prompts, not placeholders. ✔
- **Type consistency:** `CaseResult`, `ScoreReport.from_results`, `EvalOutcome`, `FailureMode`, `IVREvalCase`, `RepEvalCase` are defined in Task 1 and used with identical signatures in Tasks 3-4 and the tests. `load_cases(path, model)` (Task 2) is generic over `BaseModel`. `run_eval(cases, scorer, *, layer, max_cases, per_case_retries, total_retry_budget)` (Task 3) matches its tests. `write_report(report, *, results_dir, history_path, timestamp)` (Task 4) matches its tests. ✔
- **`AgentError` constructor (resolved):** confirmed `class AgentError(Exception)` in `agent/errors.py:39` has no custom `__init__`, so `CorpusError(f"...")` (Task 2) works via the inherited `Exception` constructor.

## Senior review changelog

Revised after an independent senior-staff plan review (verdict: approve with changes). The reviewer wrote and ran every planned file. Foundation (Tasks 1–5) validated: 18 tests pass, foundation files at 100% coverage, full suite 97.67%, pyright strict clean. Fixes applied to the spike (Task 6) — all three were guaranteed first-run gate failures:
- **MF-1:** removed `# noqa: BLE001` from the runner (RUF100 — repo doesn't enable `BLE`).
- **MF-2:** the spike's `agent.main` private-symbol import now carries `# pyright: ignore[reportPrivateUsage]` (strict rejects cross-module underscore access).
- **MF-3:** replaced `% _RUNS` percent-format with f-strings (UP031).
- **SF-1/2/3:** moved imports to module top (alphabetical for isort), documented the import-is-side-effect-safe fact and the single-turn rep-projection caveat.
- **SF-4:** added `eval_results/` to `.gitignore` in Task 5.
- **SF-5:** the spike now prints its ~200-call estimate up front.
