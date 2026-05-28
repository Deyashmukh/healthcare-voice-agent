# Eval Harness Design

Date: 2026-05-28
Status: Approved (revised after senior staff review — see "Senior review changelog")

## Problem

The agent has two LLM-driven decision surfaces — IVR navigation (Groq/Llama-4-Scout
tool-calling) and rep conversation (Anthropic/Haiku-4.5 structured output) — plus a
glue layer (`CallSessionRunner`) that sequences them. Today we have unit tests with
offline fakes, but **no measurement of the agent's actual behavior under real LLMs.**
Every prompt change is a shot in the dark: commits #35–#38 were all prompt-iteration
bug fixes (forbid-conversation rule, timeout filler, empty-reply guard, rep-priority
IVR rule) made with no regression signal beyond manual phone calls.

A solid eval harness is the difference between "prompts that worked once" and "we
know the agent's behavior under perturbation."

## Scope decision (post-review)

This is a **pre-traffic learning project**: the user roleplays the payer by dialing
their own cell, so there is currently **zero real call traffic and zero Langfuse
traces from real payers.** CLAUDE.md demands hard YAGNI discipline. Accordingly, this
round builds a *focused* harness, not the full 7-layer platform:

**In scope (round 1):**
- M-eval/0 — measurement spike (de-risk before building)
- M-eval/A — foundation (slim)
- M-eval/B — IVR tool-choice component eval
- M-eval/C — rep-extraction component eval
- M-eval/D — E2E trajectory (promoted from the spike; advisory at first)
- M-eval/G — CI (nightly only)

**Deferred until real/roleplay trace volume exists (explicitly out of scope now):**
- LLM-as-judge (warmth/naturalness) — nothing to detect drift *from* yet, and it
  would grade prose we wrote ourselves (circular). Defer.
- `from_langfuse.py` corpus miner — no traces to mine. Defer.
- train/test split — a hand-written corpus of ~10–20 cases/layer is too small for a
  held-out set to carry signal (one case = a 5–10% swing). Author one corpus now;
  add the split (pure-additive `_loader.py` change) when corpora exceed ~50/layer.

## Hard dependency: production bug must be fixed first

The senior review surfaced a latent production gap that blocks a valid IVR eval:
`CallSession.recent_menu_options` is **never populated in production** — it is set
only in test fixtures. Consequences (verified):
- `agent/tools.py:195` `if session.recent_menu_options:` is always false on a real
  call, so the `send_dtmf` digit-allowlist validator is **dead code** — the agent can
  press any digit and the validator never objects.
- The IVR LLM never receives menu options as structured state; `agent/main.py:183`
  formats only patient fields into the prompt. The LLM knows the menu only from raw
  transcript text.

**This is fixed in its own worktree/PR before any eval milestone.** Constraint: the
fix must NOT re-introduce the retired regex IVR *classifier* (CLAUDE.md YAGNI cut).
The allowlist population is a narrow digit-extraction concern, not menu-type
classification — design TBD in that PR's own brainstorm.

## Goals

A production-grade component + trajectory eval harness that:
1. Scores each LLM call in isolation (IVR tool-choice, rep extraction) against a
   curated corpus, classified by failure mode.
2. Scores the full decision loop (`CallSessionRunner` turn loop — NOT the audio /
   barge-in / VAD layer, which `test_state_processor.py` + `test_barge_in_latency.py`
   already cover) end-to-end against scripted payers.
3. Runs nightly in CI, three-way PASS/FAIL/ERROR, with thresholds derived from a
   measured baseline (not guessed).

## Non-goals

- Per-PR eval gating (cost + flake — nightly only).
- A persisted scores database / dashboard.
- Replacing the offline unit suite — evals are a separate, additive layer.
- Judge-as-CI-gate (judge is deferred entirely this round).
- Coverage of barge-in / coalescing / VAD (out of the E2E seam by design).

## Architecture

A new top-level `agent/eval/` package — parallel to `agent/` and `tests/`, not under
`tests/`, because evals are first-class behavioral measurement that call **real**
LLMs, not hermetic fakeable units.

```
agent/eval/
├── __init__.py
├── _types.py             # EvalCase variants, EvalOutcome, FailureMode, ScoreReport
├── _loader.py            # typed JSONL → Pydantic loaders
├── _runner.py            # base runner: load corpus, run cases, classify, aggregate
├── _report.py            # ScoreReport → printed table + eval_results/<ts>.json
│                         #              + append summary line to eval_history.jsonl
├── ivr_tool_choice/
│   ├── corpus/cases.jsonl
│   └── eval.py
├── rep_extraction/
│   ├── corpus/cases.jsonl
│   └── eval.py
└── e2e_trajectory/
    ├── scenarios/        # one .py per scripted-payer call graph
    ├── _scripted_payer.py
    ├── _eval_actuator.py
    └── eval.py

tests/eval/                   # hermetic offline unit tests OF the harness
scripts/eval.py               # CLI entrypoint behind `make evals`
.github/workflows/evals.yml   # nightly cron only
```

Coverage: `agent/eval/**/eval.py` and `scripts/eval.py` (the live-LLM modules `make
test` never executes) are added to the coverage-omit list, the same way `main.py`
already is — so the 90% `agent/` floor measures only hermetic code.

## §0 Measurement spike (M-eval/0) — DO THIS FIRST

A throwaway script (no package, no tests, deleted or folded into A afterward) that
de-risks the two assumptions the rest of the design rests on. CLAUDE.md requires this
("measure first, then assert against the measured baseline plus reasonable headroom").

1. **E2E reachability:** drive the existing start-and-poll runner pattern
   (`runner.start()` + poll `_current_turn.done()`, as `conftest.py:77-108` /
   `test_call_session.py:518-524` already do) with the REAL Groq+Anthropic clients
   against one hand-stubbed scripted payer for one happy-path call. Confirm terminal
   state (`completion_reason == "rep_complete"`, full benefits) is reachable at all.
2. **Stability baseline:** run ~5 IVR cases and ~5 rep cases against the real clients
   20× each. Measure tool-name agreement, deterministic-arg agreement, extraction
   agreement, phase agreement, and E2E terminal-state agreement run-to-run.

**Output:** the measured baselines that set every threshold below. If the layers are
too noisy to gate, we learn it in a day instead of after building five milestones.

## §2 Corpus schema (`_types.py`)

One Pydantic `EvalCase` variant per component layer, JSONL (git-diffable, reviewable).

```python
class IVREvalCase(BaseModel):
    id: str                              # stable slug, e.g. "aetna-main-menu-press-2"
    payer: str                           # for score slicing
    history: list[Turn]                  # conversation state fed to the IVR LLM —
                                         #   the menu MUST appear as transcript text
                                         #   here; this is the ONLY channel the LLM
                                         #   sees menu options through.
    expected_tool: ToolName
    expected_args: dict[str, object] = {}  # only DETERMINISTIC args asserted
    rationale: str                       # doc only, not asserted

class RepEvalCase(BaseModel):
    id: str
    history: list[Turn]                  # POST-FLIP (rep-phase) turns only — see §3
    expected_extracted: Benefits         # the field delta THIS utterance should yield
    expected_phase: Literal["extracting", "complete", "stuck"]
    expect_nonempty_reply: bool = True
    rationale: str
```

**`recent_menu_options` is intentionally absent from `IVREvalCase`** (senior review
MF-1). It does not reach the LLM — it only feeds the dispatcher's digit validator,
which is already covered offline by `test_tools.py` + `test_tools_hypothesis.py`. The
live IVR eval tests *tool choice from the transcript*, full stop. The menu must be
written into the `history` transcript text for the case to exercise anything.

**Deterministic-vs-freeform arg split:** tool *name* is always exact-matched. Among
args, only deterministic keys are asserted — `send_dtmf.digits`,
`record_benefit.field`, `record_benefit.value`, `complete_call.reason`. `speak.text`
is freeform and is NOT asserted (it would be the judge layer's job, deferred).

Reuse the production `Turn` / `Benefits` / `ToolName` types — a `ToolName` literal
change then breaks corpus loading at parse time instead of desyncing silently.

## §3 Scoring (per layer)

Each case yields an `EvalOutcome`: `PASS | FAIL | ERROR`, with a `FailureMode` on FAIL.

```python
class FailureMode(StrEnum):
    WRONG_TOOL = "wrong_tool"
    BAD_ARG = "bad_arg"
    MISSED_EXTRACTION = "missed_extraction"
    HALLUCINATED_FIELD = "hallucinated_field"
    WRONG_PHASE = "wrong_phase"
    EMPTY_REPLY = "empty_reply"
    PREMATURE_COMPLETE = "premature_complete"
    WRONG_COMPLETION_REASON = "wrong_completion_reason"
```

- **IVR:** PASS iff `tool_name == expected_tool` AND every key in `expected_args`
  matches. Tool-name accuracy (primary) and arg-correctness among name-correct cases
  (secondary) reported separately, so "right tool, wrong digit" is a visible,
  separable regression.
- **Rep:** PASS iff `extracted == expected_extracted` AND `phase == expected_phase`
  AND (`reply` non-empty when `expect_nonempty_reply`). Float fields compared with
  exact `==` (dollar amounts; `30.0 == 30` holds, model emits canonical floats); this
  is stated, not assumed. The rep eval drives the bare client
  (`complete_structured`) — it guards the prompt-level rules (extraction, phase, the
  "never go silent" rule that #38 touched), **not** the timeout-filler path in
  `_rep_turn` (that lives behind `asyncio.wait_for` and is covered by unit tests +
  the E2E layer). The corpus `history` must be POST-FLIP turns only and is run through
  the production `_history_to_anthropic_messages` projection (reused, not
  reimplemented) so it faithfully matches what `_rep_turn` sends (`call_session.py:390`).
- **Thresholds** for all three come from M-eval/0's measured baseline + headroom, and
  may be majority-vote or tolerance-based if the spike shows exact-match is too noisy.

## §4 E2E trajectory — the inverted-fakes pattern

The unit suite uses **fake LLMs** against real glue. This layer is the inverse: the
**real** `CallSessionRunner` with real Groq + Anthropic clients, against a
**deterministic scripted fake payer**.

```python
class ScriptedPayer:
    """State machine playing the other end. Keyed on the agent's emitted intent:
    a correct DTMF digit (or rep-appropriate utterance) advances to the next node;
    a wrong/absent one replays the current prompt. Reactive AND deterministic —
    no LLM on the payer side."""
```

Wiring uses the runner's existing `actuator: Actuator | None` injection point
(`call_session.py:181`). The eval actuator routes the agent's `SpeakIntent` /
`DTMFIntent` into the `ScriptedPayer`; the payer's response is fed back via
`runner.submit_transcript()`.

**Driver model (senior review V-2): start-and-poll, not synchronous step.**
`submit_transcript` only enqueues (`call_session.py:207-227`); turns run on the
`_consume` consumer task. The eval driver must `await runner.start()`, then loop:
feed payer line → poll until the turn completes or `session.done` → let the eval
actuator feed the next payer line. This mirrors the existing test harness pattern.

**Scope (V-3):** this covers the decision loop + LLM + dispatcher. It does NOT cover
barge-in / coalescing / VAD (those live in `StateMachineProcessor`, which this layer
bypasses) — already covered offline. The spec does not overclaim "full pipeline."

**Assertions:** terminal `completion_reason` + final `benefits`, expressed as a
**stability rate** (≥X of N runs reach the expected terminal state), with N and X set
from M-eval/0 — not a single-run exact assertion, because watchdog two-strikes limits
(`REP_STUCK_LIMIT`/`IVR_NO_PROGRESS_LIMIT = 2`) make terminal state the cumulative
product of every per-turn model decision. Seed: ~5–10 scenarios (happy path, IVR
dead-end, rep-stuck, transfer-to-rep flip, hold-timeout). Advisory at first; promote
to a gate only once the stability rate is shown to be reliably high.

## §6 CI, reporting, error handling

- **Run surface:** `make evals` (all) / `make evals ARGS=ivr` (one layer), via
  `scripts/eval.py`. Results → printed table + `eval_results/<ts>.json` (gitignored)
  + one appended summary line in committed `eval_history.jsonl` (same zero-infra
  pattern as `benefits.jsonl`, `call_session.py:58-108`) so week-over-week trend is
  answerable without a DB.
- **CI:** `.github/workflows/evals.yml` — **nightly cron only** (not on-push-to-main;
  infrequent main pushes don't justify doubling live-LLM spend). API keys as GH
  Actions secrets.
- **Three-way outcome:** `PASS / FAIL / ERROR`. API error → `ERROR`, excluded from
  accuracy. **Caveat (senior review SF-3):** `GroqToolCallingClient` already swallows
  transient `APIError` into an empty `IVRTurnResponse` (`llm_client.py:184-202`), so
  an IVR API blip surfaces to the eval as a 0-tool-call *behavioral* FAIL, not an
  ERROR. The IVR eval must detect the empty-response-after-swallow case (e.g. treat
  "zero tool calls when a tool was expected" as ERROR-suspect, or thread the client's
  failure signal out) so provider noise doesn't masquerade as a WRONG_TOOL regression.
- **Retry budget:** one retry per case on a transient API error AND an aggregate
  per-run retry ceiling, so a provider brown-out can't silently 2× the spend up to the
  cost cap.
- **Cost guard:** total-case cap per run + a logged spend estimate. If the cap
  truncates the run, log it loudly (no silent truncation).
- **Corpus authorship (SF-4):** the answer key is hand-authored by the same people who
  wrote the prompts — acknowledged circularity. Mitigations: (a) corpus cases get a
  code-reviewer pass like all other code in the milestone chain; (b) seed cases are
  anchored to the *actual* failures of #35–#38, not imagination, so the corpus is
  grounded in observed reality.

## Decisions locked

1. `speak.text` unchecked at component layer (judge deferred).
2. E2E uses real LLMs + scripted payer (not record/replay); start-and-poll driver.
3. Judge, `from_langfuse.py`, and train/test split all DEFERRED (YAGNI, no traffic).
4. Thresholds set from M-eval/0 measurement, never guessed.
5. Production `recent_menu_options` bug fixed in its own PR before eval milestones.

## Testing strategy

`tests/eval/` holds offline, hermetic unit tests of the harness: scorer logic
(PASS/FAIL/ERROR + FailureMode mapping), JSONL loader (malformed-line handling),
`ScriptedPayer` state machine, eval actuator routing, report rendering, the
empty-response-after-Groq-swallow ERROR detection. These run in `make test` with zero
network — pyright strict, same bar as the rest of the repo. The evals themselves are
exercised by a 1–2-case smoke corpus in the nightly job; full corpus runs nightly.

## Milestone decomposition + sequencing (risk-first)

1. **PROD-FIX** — populate `recent_menu_options` (own brainstorm + PR). Blocks B's
   ability to also exercise the validator path, but B can start against `history`-only
   cases in parallel if needed.
2. **M-eval/0** — measurement spike. Sets all thresholds. Do before A.
3. **M-eval/A** — foundation (slim): `_types`, `_loader`, `_runner`, `_report` +
   hermetic harness tests. Thresholds seeded from M-eval/0.
4. **M-eval/B + M-eval/C** — parallel worktrees (component evals; the direct ROI on
   the #35–#38 bug class).
5. **M-eval/D** — E2E trajectory (promote the spike), advisory.
6. **M-eval/G** — CI nightly + `make evals` + cost guard + reporting/trend wiring.

Each milestone: implement → simplify → code-reviewer → verify → commit → PR →
self-review.

## Senior review changelog

Revised from v1 after an independent senior-staff design review (verdict: needs
rework). Changes:
- **Descoped** judge, `from_langfuse.py`, train/test split (YAGNI — zero traffic).
- **Added M-eval/0 measurement spike** and made all thresholds baseline-derived
  (CLAUDE.md measure-first rule was being violated).
- **Removed `recent_menu_options` from `IVREvalCase`** (MF-1: it never reaches the
  LLM) and recorded the underlying production bug as a blocking prerequisite fix.
- **Specified the E2E driver as start-and-poll** (V-2) and **scoped E2E to the
  decision loop, not the audio layer** (V-3, was overclaiming).
- **Specified rep corpus as post-flip history through `_history_to_anthropic_messages`**
  and clarified the component rep eval does not cover the timeout-filler path (SF-5).
- **Flagged the Groq-swallow ERROR-detection gap** (SF-3), pinned float comparison
  (SF-1), added an aggregate retry ceiling (SF-3), a committed trend line (SF-7), the
  coverage-omit requirement (SF-6), and corpus-authorship mitigations (SF-4).
- **Switched CI to nightly-only** (dropped on-push-to-main).
