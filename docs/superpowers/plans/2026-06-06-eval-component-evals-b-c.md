# Eval Harness — Component Evals (M-eval/B + M-eval/C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the two component evals on top of the merged foundation — IVR tool-choice (B) and rep extraction (C) — each as a pure, hermetically-tested scorer plus a thin live-LLM wrapper, with hand-authored seed corpora and a CLI to run them.

**Architecture:** For each layer, split the work in two. A pure `_score.py` (`score_ivr` / `score_rep`) takes a case plus the model's already-produced output and returns a classified `CaseResult` — no network, fully TDD-tested, counts toward coverage. A thin `eval.py` wrapper calls the real client, hands the output to the scorer, and runs it through the foundation's `run_eval`; it's the only live-LLM piece and is coverage-omitted. Seed corpora are hand-authored JSONL anchored to the real prompt-iteration bugs (#37/#38). A small `cli.py` + `make evals` runs a layer (or all) and writes a report via the foundation's reporter.

**Tech Stack:** Python 3.12, Pydantic v2, pytest + pytest-asyncio, pyright strict, ruff. Reuses the merged `agent/eval/{_types,_loader,_runner,_report}.py`. Live clients: `GroqToolCallingClient` (IVR), `AnthropicRepClient` (rep).

**Baseline from M-eval/0 (`docs/superpowers/notes/eval-baselines.md`):** GREEN — single-run exact-match is viable for both layers (no majority-vote). This plan uses one model call per case and an exact assertion.

**Scope note — private imports:** the live `eval.py` wrappers reuse module-private helpers from `agent.main` (`_default_patient`, `_ivr_system_prompt`, `_rep_system_prompt`) and `agent.call_session` (`_history_to_anthropic_messages`), the same ones the spike already imports. We keep the established per-line `# pyright: ignore[reportPrivateUsage]` pattern rather than promoting them to public, to stay scoped to B/C and avoid churning production code. If a 4th+ consumer appears, revisit promoting them.

**Out of scope (Plan 3):** M-eval/D (E2E trajectory) and M-eval/G (nightly CI). This plan ships a `make evals` that runs B/C on demand; the nightly workflow + cost-estimate logging land in G.

---

## File Structure

- `agent/eval/ivr_tool_choice/__init__.py` — empty marker.
- `agent/eval/ivr_tool_choice/_score.py` — `CORPUS` path + `score_ivr(case, response) -> CaseResult` (pure).
- `agent/eval/ivr_tool_choice/eval.py` — live wrapper `run() -> ScoreReport` (coverage-omitted).
- `agent/eval/ivr_tool_choice/corpus/cases.jsonl` — IVR seed corpus.
- `agent/eval/rep_extraction/__init__.py` — empty marker.
- `agent/eval/rep_extraction/_score.py` — `CORPUS` path + `score_rep(case, output) -> CaseResult` (pure).
- `agent/eval/rep_extraction/eval.py` — live wrapper (coverage-omitted).
- `agent/eval/rep_extraction/corpus/cases.jsonl` — rep seed corpus.
- `agent/eval/cli.py` — `python -m agent.eval.cli {ivr,rep,all}` (coverage-omitted).
- `tests/eval/test_ivr_score.py`, `tests/eval/test_rep_score.py`, `tests/eval/test_corpora.py` — hermetic.
- `pyproject.toml` — add `agent/eval/cli.py` to coverage omit.
- `Makefile` — add `evals` target.

Every new `.py` under `agent/` and `tests/` starts with `# pyright: strict`.

---

## Task 1: IVR scorer (pure) + tests

**Files:**
- Create: `agent/eval/ivr_tool_choice/__init__.py` (empty)
- Create: `agent/eval/ivr_tool_choice/_score.py`
- Test: `tests/eval/test_ivr_score.py`

- [ ] **Step 1: Create the empty package marker** `agent/eval/ivr_tool_choice/__init__.py`.

- [ ] **Step 2: Write the failing test** `tests/eval/test_ivr_score.py`:

```python
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
    result = score_ivr(_case(args={"digits": "0"}), _resp("send_dtmf", {"digits": "0", "purpose": "rep"}))
    assert result.outcome is EvalOutcome.PASS


def test_rep_purpose_asserted_when_listed() -> None:
    case = _case(args={"digits": "0", "purpose": "rep"})
    assert score_ivr(case, _resp("send_dtmf", {"digits": "0", "purpose": "rep"})).outcome is EvalOutcome.PASS
    bad = score_ivr(case, _resp("send_dtmf", {"digits": "0", "purpose": "menu"}))
    assert bad.outcome is EvalOutcome.FAIL
    assert bad.failure_mode is FailureMode.BAD_ARG
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/eval/test_ivr_score.py -q` → FAIL at import.

- [ ] **Step 4: Write `agent/eval/ivr_tool_choice/_score.py`**

```python
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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/eval/test_ivr_score.py -q` → 6 PASS.

- [ ] **Step 6: Lint + type-check**

Run: `uv run ruff check agent/eval/ivr_tool_choice/ tests/eval/test_ivr_score.py && uv run pyright agent/eval/ivr_tool_choice/_score.py tests/eval/test_ivr_score.py` → clean.

- [ ] **Step 7: Commit**

```bash
git add agent/eval/ivr_tool_choice/__init__.py agent/eval/ivr_tool_choice/_score.py tests/eval/test_ivr_score.py
git commit -m "feat(eval): IVR tool-choice scorer (M-eval/B)"
```

---

## Task 2: Rep scorer (pure) + tests

**Files:**
- Create: `agent/eval/rep_extraction/__init__.py` (empty)
- Create: `agent/eval/rep_extraction/_score.py`
- Test: `tests/eval/test_rep_score.py`

- [ ] **Step 1: Create the empty package marker** `agent/eval/rep_extraction/__init__.py`.

- [ ] **Step 2: Write the failing test** `tests/eval/test_rep_score.py`:

```python
# pyright: strict
"""Unit tests for the rep-extraction scorer (pure, offline)."""

from __future__ import annotations

from agent.eval._types import EvalOutcome, FailureMode, RepEvalCase
from agent.eval.rep_extraction._score import score_rep
from agent.schemas import Benefits, RepTurnOutput, Turn


def _case(
    extracted: Benefits,
    phase: str = "extracting",
    expect_reply: bool = True,
) -> RepEvalCase:
    return RepEvalCase(
        id="c1",
        history=[Turn(role="user", content="Her copay is forty dollars")],
        expected_extracted=extracted,
        expected_phase=phase,  # type: ignore[arg-type]
        expect_nonempty_reply=expect_reply,
        rationale="r",
    )


def _out(reply: str, extracted: Benefits, phase: str = "extracting") -> RepTurnOutput:
    return RepTurnOutput(reply=reply, extracted=extracted, phase=phase)  # type: ignore[arg-type]


def test_pass() -> None:
    case = _case(Benefits(copay=40.0))
    assert score_rep(case, _out("Got it, thanks.", Benefits(copay=40.0))).outcome is EvalOutcome.PASS


def test_missed_extraction() -> None:
    case = _case(Benefits(copay=40.0))
    result = score_rep(case, _out("Okay.", Benefits()))
    assert result.failure_mode is FailureMode.MISSED_EXTRACTION


def test_hallucinated_field() -> None:
    case = _case(Benefits())  # expect nothing extracted
    result = score_rep(case, _out("Sure.", Benefits(copay=40.0)))
    assert result.failure_mode is FailureMode.HALLUCINATED_FIELD


def test_bad_arg_value() -> None:
    case = _case(Benefits(copay=40.0))
    result = score_rep(case, _out("Got it.", Benefits(copay=30.0)))
    assert result.failure_mode is FailureMode.BAD_ARG


def test_premature_complete() -> None:
    case = _case(Benefits(copay=40.0), phase="extracting")
    result = score_rep(case, _out("All set, bye.", Benefits(copay=40.0), phase="complete"))
    assert result.failure_mode is FailureMode.PREMATURE_COMPLETE


def test_wrong_phase_non_complete() -> None:
    case = _case(Benefits(copay=40.0), phase="extracting")
    result = score_rep(case, _out("Sorry, bye.", Benefits(copay=40.0), phase="stuck"))
    assert result.failure_mode is FailureMode.WRONG_PHASE


def test_empty_reply_when_required() -> None:
    case = _case(Benefits(), expect_reply=True)
    result = score_rep(case, _out("   ", Benefits()))
    assert result.failure_mode is FailureMode.EMPTY_REPLY
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/eval/test_rep_score.py -q` → FAIL at import.

- [ ] **Step 4: Write `agent/eval/rep_extraction/_score.py`**

```python
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
```

- [ ] **Step 5: Run the test** → 7 PASS.

- [ ] **Step 6: Lint + type-check**

Run: `uv run ruff check agent/eval/rep_extraction/ tests/eval/test_rep_score.py && uv run pyright agent/eval/rep_extraction/_score.py tests/eval/test_rep_score.py` → clean.

- [ ] **Step 7: Commit**

```bash
git add agent/eval/rep_extraction/__init__.py agent/eval/rep_extraction/_score.py tests/eval/test_rep_score.py
git commit -m "feat(eval): rep-extraction scorer (M-eval/C)"
```

---

## Task 3: Seed corpora + integrity test

**Files:**
- Create: `agent/eval/ivr_tool_choice/corpus/cases.jsonl`
- Create: `agent/eval/rep_extraction/corpus/cases.jsonl`
- Test: `tests/eval/test_corpora.py`

The cases are hand-authored and anchored to real failures (#38 rep-priority; #37/#38 never-go-silent), per the spec's anti-circularity note. One case per line.

- [ ] **Step 1: Write `agent/eval/ivr_tool_choice/corpus/cases.jsonl`** (exactly these 6 lines):

```jsonl
{"id": "ivr-eligibility-press-2", "payer": "generic", "history": [{"role": "user", "content": "Thank you for calling provider services. For claim status press 1, for eligibility and benefits press 2."}], "expected_tool": "send_dtmf", "expected_args": {"digits": "2"}, "rationale": "Eligibility and benefits is the goal; option 2."}
{"id": "ivr-rep-priority-press-0", "payer": "generic", "history": [{"role": "user", "content": "For claim status press 1, for benefits press 2, to speak with a representative press 0."}], "expected_tool": "send_dtmf", "expected_args": {"digits": "0", "purpose": "rep"}, "rationale": "A live rep beats DTMF even when benefits (2) sounds relevant; press 0 with purpose=rep (#38)."}
{"id": "ivr-greeting-wait", "payer": "generic", "history": [{"role": "user", "content": "Welcome to the health plan provider services line."}], "expected_tool": "wait", "rationale": "Opening greeting, no menu yet; acknowledge with wait."}
{"id": "ivr-member-id-speak", "payer": "generic", "history": [{"role": "user", "content": "Please say or enter the member ID number to continue."}], "expected_tool": "speak", "rationale": "Identifier request; speak the member ID. speak.text is freeform, not asserted."}
{"id": "ivr-hold-wait", "payer": "generic", "history": [{"role": "user", "content": "Please continue to hold while we connect you to the next available representative."}], "expected_tool": "wait", "rationale": "Hold announcement; wait."}
{"id": "ivr-closing-complete", "payer": "generic", "history": [{"role": "user", "content": "That completes this call. Thank you for calling, goodbye."}], "expected_tool": "complete_call", "rationale": "Explicit IVR closing; complete_call. reason is ambiguous among the enum, so not asserted."}
```

- [ ] **Step 2: Write `agent/eval/rep_extraction/corpus/cases.jsonl`** (exactly these 6 lines):

```jsonl
{"id": "rep-copay", "history": [{"role": "user", "content": "Sure. Her specialist copay is forty dollars per visit."}], "expected_extracted": {"copay": 40.0}, "expected_phase": "extracting", "rationale": "Copay stated explicitly."}
{"id": "rep-active", "history": [{"role": "user", "content": "Yes, the policy is active and effective as of January first."}], "expected_extracted": {"active": true}, "expected_phase": "extracting", "rationale": "Coverage active."}
{"id": "rep-deductible", "history": [{"role": "user", "content": "She has three hundred dollars remaining on her individual deductible."}], "expected_extracted": {"deductible_remaining": 300.0}, "expected_phase": "extracting", "rationale": "Deductible remaining stated."}
{"id": "rep-coinsurance", "history": [{"role": "user", "content": "After the deductible is met it is twenty percent coinsurance."}], "expected_extracted": {"coinsurance": 20.0}, "expected_phase": "extracting", "rationale": "Coinsurance 20 percent; model emits 20.0 (observed convention, see baselines)."}
{"id": "rep-out-of-network", "history": [{"role": "user", "content": "Out of network services are not covered under this plan."}], "expected_extracted": {"out_of_network_coverage": false}, "expected_phase": "extracting", "rationale": "Out-of-network not covered."}
{"id": "rep-backchannel-no-silence", "history": [{"role": "user", "content": "Okay, give me just one moment to pull that up."}], "expected_extracted": {}, "expected_phase": "extracting", "rationale": "Backchannel; nothing to extract, but the agent must NOT go silent (#38) so a non-empty reply is required."}
```

- [ ] **Step 3: Write the integrity test** `tests/eval/test_corpora.py`:

```python
# pyright: strict
"""Hermetic integrity checks for the seed corpora — they parse, ids are unique,
and the deterministic-arg promise holds. Catches a malformed seed case at test
time instead of mid live-run."""

from __future__ import annotations

from agent.eval._loader import load_cases
from agent.eval._types import IVREvalCase, RepEvalCase
from agent.eval.ivr_tool_choice._score import CORPUS as IVR_CORPUS
from agent.eval.rep_extraction._score import CORPUS as REP_CORPUS


def test_ivr_corpus_loads_and_ids_unique() -> None:
    cases = load_cases(IVR_CORPUS, IVREvalCase)
    assert len(cases) >= 6
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


def test_rep_corpus_loads_and_ids_unique() -> None:
    cases = load_cases(REP_CORPUS, RepEvalCase)
    assert len(cases) >= 6
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


def test_ivr_expected_args_only_for_send_dtmf() -> None:
    # The deterministic-arg assertion only makes sense where we list args; today
    # that is send_dtmf. A case with expected_args on another tool is a mistake.
    for case in load_cases(IVR_CORPUS, IVREvalCase):
        if case.expected_args:
            assert case.expected_tool == "send_dtmf", f"{case.id}: args on {case.expected_tool}"
```

- [ ] **Step 4: Run** `uv run pytest tests/eval/test_corpora.py -q` → 3 PASS.

- [ ] **Step 5: Lint** `uv run ruff check tests/eval/test_corpora.py && uv run pyright tests/eval/test_corpora.py` → clean. (JSONL files aren't linted.)

- [ ] **Step 6: Commit**

```bash
git add agent/eval/ivr_tool_choice/corpus/cases.jsonl agent/eval/rep_extraction/corpus/cases.jsonl tests/eval/test_corpora.py
git commit -m "feat(eval): seed corpora for IVR + rep evals, anchored to #37/#38"
```

---

## Task 4: Live `eval.py` wrappers (coverage-omitted)

**Files:**
- Create: `agent/eval/ivr_tool_choice/eval.py`
- Create: `agent/eval/rep_extraction/eval.py`

These call the real clients. They are NOT pytest tests and are already covered by the `agent/eval/*/eval.py` coverage-omit glob. Verify by lint + pyright + an import smoke (no API call on import).

- [ ] **Step 1: Write `agent/eval/ivr_tool_choice/eval.py`**

```python
# pyright: strict
"""Live IVR tool-choice eval — runs the real Groq client over the seed corpus.
Not a pytest test; run via `make evals`. Coverage-omitted.
"""

from __future__ import annotations

from agent.eval._loader import load_cases
from agent.eval._runner import run_eval
from agent.eval._types import CaseResult, IVREvalCase, ScoreReport
from agent.eval.ivr_tool_choice._score import CORPUS, score_ivr
from agent.llm_client import GroqToolCallingClient
from agent.main import _default_patient, _ivr_system_prompt  # pyright: ignore[reportPrivateUsage]
from agent.tools import groq_tool_schemas

LAYER = "ivr_tool_choice"


async def run() -> ScoreReport:
    client = GroqToolCallingClient()
    tools = groq_tool_schemas()
    system = _ivr_system_prompt(_default_patient())
    cases = load_cases(CORPUS, IVREvalCase)

    async def scorer(case: IVREvalCase) -> CaseResult:
        response = await client.complete_with_tools(
            system=system, history=case.history, tools=tools, temperature=0.1
        )
        return score_ivr(case, response)

    return await run_eval(cases, scorer, layer=LAYER)
```

(`CaseResult` is imported because the inner `scorer` is annotated `-> CaseResult`; the import block is alphabetized for ruff isort.)

- [ ] **Step 2: Write `agent/eval/rep_extraction/eval.py`**

```python
# pyright: strict
"""Live rep-extraction eval — runs the real Anthropic client over the seed
corpus, routing history through the production projection so the eval sees what
`_rep_turn` sends. Not a pytest test; run via `make evals`. Coverage-omitted.
"""

from __future__ import annotations

from agent.call_session import _history_to_anthropic_messages  # pyright: ignore[reportPrivateUsage]
from agent.eval._loader import load_cases
from agent.eval._runner import run_eval
from agent.eval._types import CaseResult, RepEvalCase, ScoreReport
from agent.eval.rep_extraction._score import CORPUS, score_rep
from agent.llm_client import AnthropicRepClient
from agent.main import _default_patient, _rep_system_prompt  # pyright: ignore[reportPrivateUsage]
from agent.schemas import RepTurnOutput

LAYER = "rep_extraction"


async def run() -> ScoreReport:
    client = AnthropicRepClient()
    system = _rep_system_prompt(_default_patient())
    cases = load_cases(CORPUS, RepEvalCase)

    async def scorer(case: RepEvalCase) -> CaseResult:
        output = await client.complete_structured(
            system=system,
            history=_history_to_anthropic_messages(case.history),
            schema=RepTurnOutput,
        )
        return score_rep(case, output)

    return await run_eval(cases, scorer, layer=LAYER)
```

- [ ] **Step 3: Lint + type-check + import smoke (no API calls)**

Run:
```
uv run ruff check agent/eval/ && uv run ruff format --check agent/eval/
uv run pyright agent/eval/ivr_tool_choice/eval.py agent/eval/rep_extraction/eval.py
uv run python -c "import agent.eval.ivr_tool_choice.eval, agent.eval.rep_extraction.eval; print('import OK')"
```
Expected: ruff clean, pyright `0 errors`, import prints OK (constructing clients happens inside `run()`, not at import).

- [ ] **Step 4: Confirm coverage still passes** (the new eval.py files are omitted; scorers + tests are not):

Run: `uv run pytest tests/` → green, floor reached.

- [ ] **Step 5: Commit**

```bash
git add agent/eval/ivr_tool_choice/eval.py agent/eval/rep_extraction/eval.py
git commit -m "feat(eval): live IVR + rep eval wrappers (M-eval/B + C)"
```

---

## Task 5: CLI + `make evals`

**Files:**
- Create: `agent/eval/cli.py`
- Modify: `pyproject.toml` (coverage omit), `Makefile`

- [ ] **Step 1: Write `agent/eval/cli.py`**

```python
# pyright: strict
"""Run the component evals and write a report. Live LLMs; coverage-omitted.

Usage: `python -m agent.eval.cli {ivr,rep,all}` (default: all), or `make evals`.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from agent.eval._report import render_table, write_report
from agent.eval._types import ScoreReport
from agent.eval.ivr_tool_choice import eval as ivr_eval
from agent.eval.rep_extraction import eval as rep_eval

_RESULTS_DIR = Path("eval_results")
_HISTORY_PATH = Path("eval_history.jsonl")


async def _run_selected(layers: list[str]) -> list[ScoreReport]:
    reports: list[ScoreReport] = []
    if "ivr" in layers:
        reports.append(await ivr_eval.run())
    if "rep" in layers:
        reports.append(await rep_eval.run())
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Run component evals.")
    parser.add_argument("layer", nargs="?", default="all", choices=["ivr", "rep", "all"])
    args = parser.parse_args()
    layers = ["ivr", "rep"] if args.layer == "all" else [args.layer]

    reports = asyncio.run(_run_selected(layers))
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for report in reports:
        print(render_table(report))
        out = write_report(
            report, results_dir=_RESULTS_DIR, history_path=_HISTORY_PATH, timestamp=timestamp
        )
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Coverage-omit the CLI.** In `pyproject.toml`, add `"agent/eval/cli.py"` to the `omit` list (next to `agent/eval/_spike.py`).

- [ ] **Step 2b: Gitignore the run-appended trend file.** `write_report` appends to `eval_history.jsonl` on every `make evals`. For round-1 (manual runs) that's local churn, so ignore it — whether to COMMIT a trend (per the reporter docstring's "committed trend") is deferred to M-eval/G, where nightly CI is the real trend source. Append to `.gitignore`:

```
# Eval trend file appended by `make evals`. Whether to commit a trend is an
# M-eval/G (CI) decision; for now it's local-run scratch like eval_results/.
eval_history.jsonl
```

- [ ] **Step 3: Add the Makefile target.** Add `evals` to `.PHONY` and:

```makefile
# Component evals (IVR + rep) over the seed corpora. Live LLMs (~12 calls).
# `make evals` runs both; `python -m agent.eval.cli ivr` runs one.
evals:
	uv run python -m agent.eval.cli all
```

- [ ] **Step 4: Lint + type-check + import smoke + full suite**

Run:
```
uv run ruff check agent/ tests/ && uv run ruff format --check agent/ tests/
uv run pyright agent/eval/cli.py
uv run python -c "import agent.eval.cli; print('import OK')"
uv run pytest tests/
```
Expected: all clean; full suite green; coverage floor reached.

- [ ] **Step 5: Commit**

```bash
git add agent/eval/cli.py pyproject.toml Makefile .gitignore
git commit -m "feat(eval): make evals CLI runner (M-eval/B + C)"
```

---

## Task 6: Smoke-run `make evals` (live) + record first scores

Live, ~12 calls. Cheap (the spike was 200). Either the implementer runs it, or it pauses for the user — match the same convention used for the spike.

- [ ] **Step 1: Run** `make evals` (needs `GROQ_API_KEY` + `ANTHROPIC_API_KEY`).

- [ ] **Step 2: Record** the first real per-layer pass rates + any FailureMode breakdown into `docs/superpowers/notes/eval-baselines.md` (append a "First corpus run" section). If any case fails, decide whether it's a real prompt gap (file it) or a corpus error (fix the case), per the spec's corpus-review discipline.

  **Cases to watch (flagged by plan review — unverified by the spike, which only probed tool choice, not these arg/field specifics):**
  - `ivr-rep-priority-press-0` expects `purpose:"rep"` as a scored arg. `purpose` defaults to `"menu"` in the schema; if the model relies on the default instead of emitting `purpose` explicitly, `args.get("purpose")` is `None` → spurious `BAD_ARG`. If it fails this way, it's a corpus-vs-prompt question (the prompt does instruct `purpose='rep'`), not a scorer bug — decide whether to keep asserting `purpose` or relax to digit-only.
  - `ivr-member-id-speak` expects `speak`, but the input says "say **or enter**" — the model could legitimately pick `send_dtmf`. Spike probed this 20/20 `speak`, so likely fine.
  - `rep-out-of-network` expects `{"out_of_network_coverage": false}` — no spike probe covered this field; low risk but first-seen.

- [ ] **Step 3: Commit** the recorded scores (docs only).

---

## Self-Review (completed during planning)

- **Spec coverage:** M-eval/B → Tasks 1, 3, 4 (scorer + corpus + live wrapper); M-eval/C → Tasks 2, 3, 4; runner/reporter reused from the foundation; `make evals` → Task 5; first-run baseline → Task 6. Single-run exact-match per the GREEN spike; no majority-vote. ✔
- **Placeholder scan:** every code + corpus step has complete content; corpora are concrete JSONL lines, not "add cases here." ✔
- **Type consistency:** `score_ivr(IVREvalCase, IVRTurnResponse) -> CaseResult` and `score_rep(RepEvalCase, RepTurnOutput) -> CaseResult` match their tests and their eval.py callers. `CORPUS` is defined in each `_score.py` and imported by both the eval.py wrapper and the corpus test. `run()` returns `ScoreReport`; the CLI consumes `render_table`/`write_report` with the foundation's exact signatures (`write_report(report, *, results_dir, history_path, timestamp)`). ✔
- **Known limitation documented:** Groq's swallow-to-empty means a provider blip scores as FAIL/WRONG_TOOL not ERROR (spec §6 SF-3); recorded in `_score.py`'s docstring. Acceptable for round-1; revisit if it muddies real scores.
- **Open risk for the implementer:** the rep corpus is all `phase="extracting"`. `complete`/`stuck` are stateful (multi-turn) and can't be elicited from a single-turn case; those are deferred to M-eval/D's trajectory eval. Noted, not a gap in B/C.

## Senior review changelog

Revised after an independent senior-staff plan review (verdict: approve with changes). The reviewer built and ran every scorer/wrapper/corpus/test: all 13 scorer assertions pass, both scorers hit 100% branch coverage, corpora load under `extra="forbid"`, all files pyright-strict + ruff clean. Validated: single-line `# pyright: ignore[reportPrivateUsage]` suppresses both names on the import line (unlike the spike's multi-line case); the `scorer` closures type-check against `run_eval`; `import eval as ivr_eval` doesn't shadow the builtin; coverage globs keep scorers in and omit eval.py/cli.py. Changes applied:
- **MF1:** fixed the IVR `eval.py` Step 1 import block to include `CaseResult` (was self-corrected in a later step; now correct in the artifact) and removed the redundant fix-step.
- **SF3:** gitignore `eval_history.jsonl` (round-1 local-run churn; committed-trend decision deferred to M-eval/G).
- **SF4/5:** added explicit "cases to watch" notes to Task 6 (the `purpose:"rep"` scored arg, `member-id`→`speak`, `out-of-network`) — unverified by the spike, most likely to surface at the first live run; with guidance that a failure there is corpus-vs-prompt, not a scorer bug.
- Accepted as-is: the test helpers' `# type: ignore[arg-type]` (works; the repo doesn't enable `reportUnnecessaryTypeIgnoreComment`), and the Groq-swallow→FAIL caveat (documented; an honest round-1 limitation per spec §6).
