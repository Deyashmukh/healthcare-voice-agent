# Eval Harness — E2E Trajectory (M-eval/D) + Nightly CI (M-eval/G) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an end-to-end trajectory eval that drives the REAL `CallSessionRunner` (real Groq + Anthropic) against a deterministic scripted payer and asserts on terminal state, plus a nightly GitHub Actions workflow that runs the full eval suite.

**Architecture (the inverted-fakes pattern, per the spec):** Unit tests use fake LLMs against real glue; this is the inverse — real LLMs against a fake payer. A `ScriptedPayer` plays the other end of the call as an ordered script of utterances. A `RecordingActuator` (satisfies the runner's `Actuator` protocol) captures the agent's emitted intents instead of doing TTS/DTMF. An `E2EDriver` owns the loop: `start()` the runner, submit the opening payer line, then after each turn feed the next scripted line, until `session.done` or a turn cap. We assert on terminal state (`completion_reason` + final `benefits`). The spike (M-eval/0) showed the models are near-deterministic, so a single-run terminal-state assertion is the baseline (a stability margin can be added later if a scenario proves flaky).

**Scope (per spec V-3):** D covers the decision loop + LLMs + dispatcher. It does NOT cover barge-in / coalescing / VAD (those live in `StateMachineProcessor`, which this layer bypasses — already covered by `test_state_processor.py` + `test_barge_in_latency.py`).

**Pure/live split (same as B/C):** the `ScriptedPayer`, `RecordingActuator`, `E2EDriver`, and terminal-state scorer are hermetically unit-tested using the existing `FakeIVRLLMClient` / `FakeAnthropicRepClient` fakes (zero network, counts toward coverage). Only `e2e_trajectory/eval.py` (real clients) is coverage-omitted.

**Tech Stack:** Python 3.12, Pydantic v2, pytest + pytest-asyncio, pyright strict, ruff. Reuses the merged `agent/eval/` foundation + the `make evals` CLI.

**Out of scope:** judge / Langfuse-miner / train-test split (still cut). Per-PR `ruff+pyright+pytest` CI (CLAUDE.md M8'/B) is a SEPARATE track from this eval-nightly workflow; this plan adds only the nightly eval workflow.

---

## File Structure

- `agent/eval/e2e_trajectory/__init__.py` — empty.
- `agent/eval/e2e_trajectory/_scripted_payer.py` — `Scenario` dataclass + the seed scenarios.
- `agent/eval/e2e_trajectory/_driver.py` — `RecordingActuator` + `run_scenario(...)` driver + `score_scenario(...)` terminal-state scorer.
- `agent/eval/e2e_trajectory/eval.py` — live wrapper `run() -> ScoreReport` (coverage-omitted).
- `agent/eval/e2e_trajectory/_spike.py` — Task 1 reachability spike (coverage-omitted; can be deleted after).
- `tests/eval/test_e2e_driver.py` — hermetic tests of the driver/payer/scorer using fakes.
- `agent/eval/cli.py` — add `e2e` to the layer choices.
- `pyproject.toml` — omit `agent/eval/e2e_trajectory/eval.py` (matches existing `*/eval.py` glob — verify) + `_spike.py`.
- `.github/workflows/evals.yml` — nightly cron workflow (M-eval/G).

Every new `.py` starts with `# pyright: strict`.

---

## Task 1: Reachability spike (validate the wiring before building the framework)

The senior review of the harness spec flagged E2E as the riskiest layer and wanted reachability proven first. This is a throwaway-but-rerunnable spike (live LLMs, coverage-omitted) that proves a single happy-path call can be driven through the real runner to `rep_complete` with full benefits. If the wiring doesn't work, we learn it here before building the framework.

**Files:** Create `agent/eval/e2e_trajectory/__init__.py` (empty) and `agent/eval/e2e_trajectory/_spike.py`.

- [ ] **Step 1: Write `_spike.py`**

```python
# pyright: strict
"""M-eval/D reachability spike — LIVE LLMs, not a pytest test.

Run: `uv run python -m agent.eval.e2e_trajectory._spike` (needs GROQ_API_KEY +
ANTHROPIC_API_KEY). Proves one happy-path call drives the real CallSessionRunner
through IVR nav -> rep handoff -> extraction -> completion. Prints the terminal
state. Nothing asserted; this de-risks the driver wiring for Tasks 2-6.
"""

from __future__ import annotations

import asyncio

from agent.actuator import Actuator
from agent.call_session import CallSessionRunner
from agent.llm_client import AnthropicRepClient, GroqToolCallingClient
from agent.main import (  # pyright: ignore[reportPrivateUsage]
    _default_patient,
    _ivr_system_prompt,
    _rep_system_prompt,
)
from agent.schemas import CallSession, SideEffectIntent
from agent.tools import dispatch, groq_tool_schemas

_SCRIPT = [
    "Thank you for calling provider services. For claims press 1, to speak with a representative press 0.",
    "One moment, connecting you to a representative.",
    "Hi, this is Jamie in benefits, who am I speaking with?",
    "Sure, I can help. Her plan is active. The specialist copay is forty dollars.",
    "She has three hundred dollars left on her deductible, and it's twenty percent coinsurance.",
    "Out of network is not covered. Is there anything else?",
    "You're welcome, have a great day. Goodbye.",
]


class _RecordingActuator:  # satisfies Actuator
    def __init__(self) -> None:
        self.intents: list[SideEffectIntent] = []

    async def execute(self, intent: SideEffectIntent) -> None:
        self.intents.append(intent)


async def main() -> None:
    session = CallSession(call_sid="E2E-spike", patient=_default_patient())
    actuator: Actuator = _RecordingActuator()
    runner = CallSessionRunner(
        session=session,
        ivr_llm=GroqToolCallingClient(),
        rep_llm=AnthropicRepClient(),
        tool_dispatcher=dispatch,
        ivr_system_prompt=_ivr_system_prompt(session.patient),
        rep_system_prompt=_rep_system_prompt(session.patient),
        tools=groq_tool_schemas(),
        actuator=actuator,
    )
    await runner.start()
    try:
        runner.submit_transcript(_SCRIPT[0])
        line = 1
        for _ in range(30):  # turn cap
            await _wait_turn(runner)
            if session.done:
                break
            runner.submit_transcript(_SCRIPT[line] if line < len(_SCRIPT) else _SCRIPT[-1])
            line += 1
    finally:
        await runner.stop()
    print(f"completion_reason={session.completion_reason}")
    print(f"mode_at_end={session.mode}  turns={session.turn_count}")
    print(f"benefits={session.benefits.model_dump()}")


async def _wait_turn(runner: CallSessionRunner) -> None:
    start = runner.session.turn_count
    for _ in range(2000):  # ~ up to 20s at 10ms; live LLM turns are seconds
        if runner.session.turn_count > start or runner.session.done:
            return
        await asyncio.sleep(0.01)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Confirm it imports (no API call) + lint/type-check**

Run: `uv run ruff check agent/eval/e2e_trajectory/ && uv run pyright agent/eval/e2e_trajectory/_spike.py && uv run python -c "import agent.eval.e2e_trajectory._spike"`
Expected: clean; import OK.

- [ ] **Step 3: Run the spike live (needs API keys)**

Run: `uv run python -m agent.eval.e2e_trajectory._spike`
Expected: `completion_reason=rep_complete`, `mode_at_end=rep`, benefits populated (copay 40.0, active true, deductible 300.0, coinsurance 20.0, out_of_network false). **If it does NOT reach `rep_complete`, STOP and report** — the script lines or the driver cadence need adjusting before building the framework. Record the observed terminal state.

- [ ] **Step 4: Broaden the spike coverage-omit BEFORE committing.** The existing `pyproject.toml` omit has `agent/eval/_spike.py`, which does NOT match the nested `agent/eval/e2e_trajectory/_spike.py` (verified). Change that entry to the recursive glob `agent/eval/**/_spike.py` so both the M-eval/0 spike and this one are omitted. Without this, any `pytest tests/` run sees a 0%-covered `_spike.py` and the 90% floor breaks.

- [ ] **Step 5: Commit the spike**

```bash
git add agent/eval/e2e_trajectory/__init__.py agent/eval/e2e_trajectory/_spike.py pyproject.toml
git commit -m "feat(eval): M-eval/D reachability spike (E2E happy path)"
```

---

## Task 2: `RecordingActuator` + driver + terminal-state scorer

**Files:**
- Create: `agent/eval/e2e_trajectory/_driver.py`
- (tests come in Task 4 after scenarios exist; this task is the mechanism.)

The driver generalizes the spike. It is pure orchestration (no LLM of its own), so it's hermetically testable with fake clients.

- [ ] **Step 1: Write `agent/eval/e2e_trajectory/_driver.py`**

```python
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
```

Cadence correctness is load-bearing: `prev_turns` is captured ONCE before the loop and re-read after each turn, so `_wait_for_turn` actually blocks until the turn advances. (A buggy recompute inside the loop — e.g. `turn_count - 1` — would be always-true, never wait, and fire every script line into the drop-oldest `in_queue` so the call never completes.) Task 4's hermetic test (fakes drain exactly) and Task 1's spike (live → `rep_complete`) both validate this end to end.

- [ ] **Step 2: Lint + type-check**

Run: `uv run ruff check agent/eval/e2e_trajectory/_driver.py && uv run pyright agent/eval/e2e_trajectory/_driver.py`
(Will error until `_scripted_payer.Scenario` exists — Task 3 defines it; if doing strictly in order, write Task 3 Step 1 first. The subagent executor should implement Task 3's `Scenario` before type-checking this file.)

No commit yet — Tasks 2-4 form one logical unit (driver + scenarios + tests); commit at the end of Task 4.

---

## Task 3: `Scenario` model + seed scenarios

**Files:**
- Create: `agent/eval/e2e_trajectory/_scripted_payer.py`

- [ ] **Step 1: Write `agent/eval/e2e_trajectory/_scripted_payer.py`**

```python
# pyright: strict
"""Scripted-payer scenarios for the E2E trajectory eval.

A Scenario is an ordered list of payer utterances (`script`) plus the terminal
state we expect the agent to reach. The driver feeds one line per agent turn;
write each line so the agent takes a single action in response, and make the
last line a clear closing that triggers `complete_call`.
"""

from __future__ import annotations

from agent.eval._types import EvalCase
from agent.schemas import Benefits, CompletionReason


class Scenario(EvalCase):
    # Subclass EvalCase (provides `id` + extra="forbid") so it satisfies the
    # `run_eval[CaseT: EvalCase]` bound — passing a bare BaseModel fails pyright.
    script: tuple[str, ...]  # payer utterances, one fed per agent turn
    expected_completion_reason: CompletionReason
    expected_benefits: Benefits | None = None
    max_turns: int = 25


_HAPPY_PATH = Scenario(
    id="happy-path-full-extraction",
    script=(
        "Thank you for calling provider services. For claims press 1, "
        "to speak with a representative press 0.",
        "One moment, connecting you to a representative.",
        "Hi, this is Jamie in benefits. Who am I speaking with today?",
        "Sure, I can help with that. Her plan is active and effective. "
        "The specialist copay is forty dollars.",
        "She has three hundred dollars remaining on her deductible, "
        "and after that it's twenty percent coinsurance.",
        "Out of network services are not covered under this plan. "
        "Is there anything else I can help with?",
        "You're very welcome. Have a great day. Goodbye.",
    ),
    expected_completion_reason="rep_complete",
    expected_benefits=Benefits(
        active=True,
        copay=40.0,
        deductible_remaining=300.0,
        coinsurance=20.0,
        out_of_network_coverage=False,
    ),
)

_REP_STUCK = Scenario(
    id="rep-stuck-no-info",
    script=(
        "Thank you for calling. To speak with a representative press 0.",
        "Connecting you now.",
        "Hi, this is Sam. I'm sorry, our benefits system is down right now, "
        "I can't pull up any of that information.",
        "I really can't access it, I'm sorry. There's nothing I can tell you today.",
        "No, still nothing, I apologize.",
    ),
    expected_completion_reason="rep_stuck",
)

SCENARIOS: tuple[Scenario, ...] = (_HAPPY_PATH, _REP_STUCK)
```

Note: `expected_benefits` is asserted only for the happy path. `_REP_STUCK` asserts just the terminal reason. Per the spec, complete/stuck are stateful, so they live in E2E (not the single-turn rep component eval). If the spike or first live run shows `rep-stuck` is flaky (the rep model declaring `stuck` twice is a model judgment), keep it but mark it advisory in Task 6's recorded notes.

- [ ] **Step 2: Lint + type-check** `uv run ruff check agent/eval/e2e_trajectory/_scripted_payer.py && uv run pyright agent/eval/e2e_trajectory/_scripted_payer.py` → clean.

---

## Task 4: Hermetic tests of the driver (fakes, zero network) + commit

**Files:**
- Create: `tests/eval/test_e2e_driver.py`

The driver/payer/scorer are pure orchestration — test them with the existing fakes producing scripted intents, no live calls.

- [ ] **Step 1: Write `tests/eval/test_e2e_driver.py`**

```python
# pyright: strict
"""Hermetic tests for the E2E driver, scripted payer, and terminal scorer.
Uses fake LLM clients producing scripted tool calls / rep outputs — zero network.
"""

from __future__ import annotations

from agent.call_session import CallSessionRunner
from agent.eval._types import EvalOutcome, FailureMode
from agent.eval.e2e_trajectory._driver import RecordingActuator, run_scenario, score_scenario
from agent.eval.e2e_trajectory._scripted_payer import Scenario
from agent.schemas import Benefits, CallSession, IVRTurnResponse, PatientInfo, RepTurnOutput, ToolCall
from agent.tools import dispatch

from tests.unit.conftest import FakeAnthropicRepClient, FakeIVRLLMClient


def _runner(ivr: FakeIVRLLMClient, rep: FakeAnthropicRepClient, actuator: RecordingActuator) -> CallSessionRunner:
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


async def test_happy_path_reaches_rep_complete() -> None:
    # IVR: press rep digit, then transfer_to_rep. Rep: extract then complete.
    ivr = FakeIVRLLMClient(
        responses=[
            IVRTurnResponse(tool_calls=[ToolCall(name="send_dtmf", args={"digits": "0", "purpose": "rep"})]),
            IVRTurnResponse(tool_calls=[ToolCall(name="transfer_to_rep", args={})]),
        ]
    )
    rep = FakeAnthropicRepClient(
        responses=[
            RepTurnOutput(reply="Hi, this is Morgan.", extracted=Benefits(copay=40.0), phase="extracting"),
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
    runner = _runner(FakeIVRLLMClient(responses=[]), FakeAnthropicRepClient(responses=[]), RecordingActuator())
    runner.session.completion_reason = "rep_complete"
    runner.session.benefits = Benefits(copay=30.0)
    result = score_scenario(scenario, runner)
    assert result.outcome is EvalOutcome.FAIL
    assert result.failure_mode is FailureMode.MISSED_EXTRACTION
```

- [ ] **Step 2: Run + verify** `uv run pytest tests/eval/test_e2e_driver.py -q` → 3 PASS. Then full suite `uv run pytest tests/` → green, floor reached.

Note: `test_happy_path_reaches_rep_complete` exercises the real `dispatch` + runner + driver with fake LLMs; if the fake-response sequencing doesn't line up with the driver's one-line-per-turn cadence (e.g. the transfer turn consumes a script line without an actuator intent), adjust the fake response count / script length so the rep `complete` response is reached. This is the hermetic proof the driver loop is correct independent of live models.

- [ ] **Step 3: Lint + type-check + commit**

Run: `uv run ruff check agent/ tests/ && uv run ruff format --check agent/ tests/ && uv run pyright agent/eval/ tests/eval/`
```bash
git add agent/eval/e2e_trajectory/_driver.py agent/eval/e2e_trajectory/_scripted_payer.py tests/eval/test_e2e_driver.py
git commit -m "feat(eval): E2E driver, scripted payer, terminal scorer + hermetic tests (M-eval/D)"
```

---

## Task 5: Live `eval.py` wrapper + CLI wiring

**Files:**
- Create: `agent/eval/e2e_trajectory/eval.py`
- Modify: `agent/eval/cli.py`, `pyproject.toml`

- [ ] **Step 1: Write `agent/eval/e2e_trajectory/eval.py`**

```python
# pyright: strict
"""Live E2E trajectory eval — drives the real runner against each scripted payer
scenario. Not a pytest test; run via `make evals`. Coverage-omitted.
"""

from __future__ import annotations

from agent.actuator import Actuator
from agent.call_session import CallSessionRunner
from agent.eval._runner import run_eval
from agent.eval._types import CaseResult, ScoreReport
from agent.eval.e2e_trajectory._driver import RecordingActuator, run_scenario, score_scenario
from agent.eval.e2e_trajectory._scripted_payer import SCENARIOS, Scenario
from agent.llm_client import AnthropicRepClient, GroqToolCallingClient
from agent.main import (  # pyright: ignore[reportPrivateUsage]
    _default_patient,
    _ivr_system_prompt,
    _rep_system_prompt,
)
from agent.schemas import CallSession
from agent.tools import dispatch, groq_tool_schemas

LAYER = "e2e_trajectory"


async def run() -> ScoreReport:
    patient = _default_patient()
    tools = groq_tool_schemas()
    ivr_system = _ivr_system_prompt(patient)
    rep_system = _rep_system_prompt(patient)

    async def scorer(scenario: Scenario) -> CaseResult:
        session = CallSession(call_sid=f"E2E-{scenario.id}", patient=patient)
        actuator: Actuator = RecordingActuator()
        runner = CallSessionRunner(
            session=session,
            ivr_llm=GroqToolCallingClient(),
            rep_llm=AnthropicRepClient(),
            tool_dispatcher=dispatch,
            ivr_system_prompt=ivr_system,
            rep_system_prompt=rep_system,
            tools=tools,
            actuator=actuator,
        )
        await runner.start()
        try:
            await run_scenario(runner, scenario)
        finally:
            await runner.stop()
        return score_scenario(scenario, runner)

    return await run_eval(list(SCENARIOS), scorer, layer=LAYER)
```

- [ ] **Step 2: Wire `e2e` into the CLI.** In `agent/eval/cli.py`, add `from agent.eval.e2e_trajectory import eval as e2e_eval`, add `"e2e": e2e_eval.run` to the `_LAYERS` dict, extend the argparse choices to `["ivr", "rep", "e2e", "all"]`, and include `"e2e"` in the `all` layer list.

- [ ] **Step 3: Coverage-omit.** Confirm `agent/eval/*/eval.py` already covers `agent/eval/e2e_trajectory/eval.py` (it does — same glob). The nested `_spike.py` was already handled in Task 1 Step 4 (broadened to `agent/eval/**/_spike.py`). Verify with `uv run pytest tests/` that the floor still holds.

- [ ] **Step 4: Lint + type-check + import smoke + full suite**

Run:
```
uv run ruff check agent/ tests/ && uv run ruff format --check agent/ tests/
uv run pyright agent/eval/e2e_trajectory/eval.py agent/eval/cli.py
uv run python -c "import agent.eval.cli; print('OK')"
uv run pytest tests/
```
Expected: clean; full suite green; floor reached.

- [ ] **Step 5: Commit**

```bash
git add agent/eval/e2e_trajectory/eval.py agent/eval/cli.py pyproject.toml
git commit -m "feat(eval): live E2E eval wrapper + CLI wiring (M-eval/D)"
```

---

## Task 6: Live smoke run of E2E (PAUSE for the user) + record

Live, real calls (the happy-path scenario alone is ~7 turns × 2 models). Pause for the user, matching the spike convention; do NOT run it autonomously.

- [ ] **Step 1:** Hand the user `make evals e2e` (or `python -m agent.eval.cli e2e`). It needs `GROQ_API_KEY` + `ANTHROPIC_API_KEY`.
- [ ] **Step 2:** Record the per-scenario outcomes in `docs/superpowers/notes/eval-baselines.md` (append an "E2E first run" section). Triage any failure: a real trajectory bug (file it) vs a scenario-script issue (fix the script) vs a flaky stateful scenario (mark advisory). Commit the recorded notes (docs only).

---

## Task 7 (M-eval/G): Nightly CI workflow

**Files:**
- Create: `.github/workflows/evals.yml`

There is no `.github/workflows/` dir yet. This adds the nightly eval workflow ONLY (the per-PR ruff+pyright+pytest CI is a separate M8'/B track). The eval suite makes live LLM calls, so it runs on a nightly cron + manual dispatch, never on every push.

- [ ] **Step 1: Write `.github/workflows/evals.yml`**

```yaml
name: nightly-evals

on:
  schedule:
    - cron: "0 8 * * *"  # 08:00 UTC nightly
  workflow_dispatch: {}    # manual trigger from the Actions tab

permissions:
  contents: read

jobs:
  evals:
    runs-on: ubuntu-latest
    timeout-minutes: 30  # ~14-20 sequential live round-trips; headroom for slow-provider nights
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v5
      - name: Sync deps
        run: uv sync --frozen
      - name: Run component + E2E evals
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: uv run python -m agent.eval.cli all
      - name: Upload eval results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: |
            eval_results/
            eval_history.jsonl
          if-no-files-found: ignore
```

- [ ] **Step 2: Validate the YAML.** Run `uv run python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/evals.yml')); print('yaml OK')"` (PyYAML ships with the dev deps; if not, skip and rely on GitHub's parser). Confirm indentation/keys.

- [ ] **Step 3: Note for the user.** The workflow needs repo secrets `GROQ_API_KEY` + `ANTHROPIC_API_KEY` set in GitHub (Settings → Secrets and variables → Actions). The user sets these — I can't. Until then, `workflow_dispatch` runs will fail on missing secrets. Document this in the PR body. Whether to gate (fail the job on pass-rate threshold) vs advisory (always green, results in artifact) is deferred — for now it's advisory: the job runs the evals and uploads results; it does not fail on a low pass rate (live-LLM flake shouldn't redden the repo). A future iteration can add a threshold gate once trend data exists.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/evals.yml
git commit -m "ci(eval): nightly eval workflow (M-eval/G)"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** M-eval/D → Tasks 1-6 (reachability spike, driver/payer/scorer, hermetic tests, live wrapper, CLI, live smoke); M-eval/G → Task 7 (nightly workflow). Terminal-state assertion single-run per the GREEN baselines; pure/live split mirrors B/C. ✔
- **Placeholder scan:** complete code for the driver, scorer, scenarios, tests, eval wrapper, CLI wiring, and YAML. The reachability spike (Task 1) and the two "Note" adjustments (driver cadence off-by-one, fake-response sequencing) are explicitly flagged as things the spike/tests will confirm — not placeholders, but validation gates. ✔
- **Type consistency:** `Scenario` (id, script: tuple[str,...], expected_completion_reason: CompletionReason, expected_benefits: Benefits|None, max_turns) is defined in `_scripted_payer.py` and used identically in `_driver.py`, `eval.py`, and the tests. `run_scenario(runner, scenario) -> None` and `score_scenario(scenario, runner) -> CaseResult` match all callers. `RecordingActuator` satisfies the `Actuator` protocol (`async execute(intent)->None`). The live `scorer` closure returns `CaseResult` and is passed to `run_eval(list(SCENARIOS), scorer, layer=...)`. ✔
- **Risk flagged for the implementer:** the driver's one-line-per-turn cadence + the transfer-to-rep turn (which produces NO actuator intent, only a mode flip) is the load-bearing assumption. Task 1's spike validates it live and Task 4's hermetic test validates it with fakes BEFORE the live wrapper. If either shows the cadence is off (e.g. the agent needs a wait turn on the greeting, consuming a script line early), fix `run_scenario` / the scenario scripts and note it. This is the spec-flagged "E2E is the riskiest layer" — the spike-first ordering is deliberate.
- **Scope honesty:** only 2 seed scenarios (happy path + rep-stuck). The spec wanted ~5-10; this is a first cut. IVR-dead-end and hold-timeout scenarios are deferred (harder to make deterministic) — noted, not hidden.

## Senior review changelog

Revised after an independent senior-staff plan review (verdict: needs rework → fixed). The reviewer compiled/ran the code. Three blocking defects fixed (two proven by execution):
- **MF1 (driver cadence, proven):** `run_scenario` recomputed the wait baseline as `turn_count - 1` inside the loop → always-true → never waited → fired every script line into the drop-oldest queue → `completion_reason=None`/FAIL. The spike couldn't catch it (its loop is written correctly). Fixed: capture `prev_turns` once, advance per iteration.
- **MF2 (typevar bound, proven via pyright):** `run_eval[CaseT: EvalCase]` rejected `Scenario(BaseModel)`. Fixed: `Scenario(EvalCase)` (inherits `id` + `extra="forbid"`).
- **MF3 (test import, proven):** `from .conftest import` fails — fakes live in `tests/unit/conftest.py` and `tests/eval/` has no conftest. Fixed: `from tests.unit.conftest import ...`.
- **SF:** nested `_spike.py` omit broadened to `agent/eval/**/_spike.py` and moved into Task 1 (before its commit) so the floor never breaks; CI `timeout-minutes` 20→30 with a note on `REP_LLM_TIMEOUT_S` (8s) flakiness under CI jitter.

Validated by the reviewer (no change needed): `transfer_to_rep` produces no actuator intent (driver cadence handles it), `RecordingActuator` structurally satisfies the `Actuator` protocol and avoids `out_queue` backpressure, fake-response counts drain exactly under the corrected cadence, all `FailureMode`/`CompletionReason` literals exist, `uv.lock` committed so `uv sync --frozen` works, and the corrected driver/scenario files pass pyright strict.
