# Eval Harness Design

Date: 2026-05-28
Status: Approved (pending senior review + user review)

## Problem

The agent has two LLM-driven decision surfaces — IVR navigation (Groq/Llama-4-Scout
tool-calling) and rep conversation (Anthropic/Haiku-4.5 structured output) — plus a
glue layer (`CallSessionRunner`) that sequences them. Today we have unit tests with
offline fakes, but **no measurement of the agent's actual behavior under real LLMs.**
Every prompt change is a shot in the dark: commits #35–#38 were all prompt-iteration
bug fixes (forbid-conversation rule, timeout filler, empty-reply guard, rep-priority
IVR rule) made with no regression signal beyond manual phone calls.

A solid eval harness is the difference between "prompts that worked once" and "we
know the agent's behavior under perturbation." This is standard practice for
production LLM agents and the CLAUDE.md bar ("dev-quality bar is production-grade")
calls for it.

## Goals

Build a production-grade eval harness covering the seven capabilities that define a
mature agent eval setup:

1. Component-level evals per LLM call in isolation (IVR tool-choice, rep extraction)
2. Trajectory / end-to-end evals over the full `CallSessionRunner` loop
3. LLM-as-judge for subjective dimensions exact-match can't score (conversational warmth)
4. Trace-sourced corpus tooling (Langfuse → corpus suggestions)
5. Held-out test set (train/test split, anti-Goodhart)
6. Failure-mode classification (typed taxonomy, not just pass/fail)
7. CI / regression gating (nightly + on-main, never per-PR)

## Non-goals

- Per-PR eval gating (cost + flake — nightly + on-main only).
- A persisted scores database / dashboard (no real traffic to benchmark against yet).
- Replacing the offline unit suite — evals are a separate, additive layer.
- Judge-as-CI-gate — the judge is advisory only (see §5).

## Architecture

A new top-level `agent/eval/` package — parallel to `agent/` and `tests/`, not under
`tests/`, because evals are first-class behavioral measurement that call **real**
LLMs, not hermetic fakeable units.

```
agent/eval/
├── __init__.py
├── _types.py             # EvalCase variants, EvalOutcome, FailureMode, ScoreReport
├── _loader.py            # typed JSONL → Pydantic loaders (train/test split aware)
├── _runner.py            # base runner: load corpus, run cases, classify, aggregate
├── _report.py            # ScoreReport → printed table + eval_results/<ts>.json
├── ivr_tool_choice/
│   ├── corpus/{train,test}.jsonl
│   └── eval.py
├── rep_extraction/
│   ├── corpus/{train,test}.jsonl
│   └── eval.py
├── e2e_trajectory/
│   ├── scenarios/        # one .py per scripted-payer call graph
│   ├── _scripted_payer.py
│   ├── _eval_actuator.py
│   └── eval.py
├── judge/
│   ├── rubric.py
│   └── eval.py
└── corpus_tools/
    ├── from_langfuse.py      # traces → corpus suggestions (manual-review gate)
    └── classify_failures.py  # tag observed failures by taxonomy

tests/eval/                   # unit tests OF the eval harness (hermetic, offline)
scripts/eval.py               # CLI entrypoint behind `make evals`
.github/workflows/evals.yml   # nightly cron + on-push-to-main
```

Rationale:
- Per-layer subpackages → each milestone is independently implementable in its own
  worktree (parallel execution viable).
- `corpus_tools/from_langfuse.py` lives in `eval/` (not `scripts/`) because corpus
  governance is part of the eval contract.
- `tests/eval/` holds offline unit tests of the *harness itself* (scorers, loaders,
  scripted payer) — these run in the normal `make test` suite. The evals proper do
  not.

## §2 Corpus schema (`_types.py`)

One Pydantic `EvalCase` variant per component layer, stored as JSONL (one case per
line — git-diffable, append-friendly, reviewable).

```python
class IVREvalCase(BaseModel):
    id: str                              # stable slug, e.g. "aetna-main-menu-press-2"
    payer: str                           # for score slicing
    history: list[Turn]                  # conversation state fed to the IVR LLM
    recent_menu_options: list[str]       # menu digits in play (drives arg validation)
    expected_tool: ToolName              # the tool we expect this turn
    expected_args: dict[str, object] = {}  # only DETERMINISTIC args asserted
    rationale: str                       # doc only, not asserted

class RepEvalCase(BaseModel):
    id: str
    history: list[Turn]                  # rep-mode conversation up to this utterance
    expected_extracted: Benefits         # the field delta THIS utterance should yield
    expected_phase: Literal["extracting", "complete", "stuck"]
    expect_nonempty_reply: bool = True
    rationale: str
```

**Deterministic-vs-freeform arg split (decision #1, default):** tool *name* is always
exact-matched. Among args, only deterministic ones are asserted — `send_dtmf.digits`,
`record_benefit.field`, `record_benefit.value`, `complete_call.reason`. `speak.text`
is freeform and is **not** exact-matched at the component layer; its quality is the
judge layer's job. So `expected_args` lists only the deterministic keys for a case.

Reusing the production `Turn` / `Benefits` / `ToolName` types (not parallel eval-only
types) keeps the corpus faithful to what the LLM actually receives and means a schema
change can't silently desync the corpus.

## §3 Scoring (per layer)

Each case yields an `EvalOutcome`: `PASS | FAIL | ERROR`, plus a `FailureMode` on
FAIL and an optional error string on ERROR.

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
  matches. Reported metrics: tool-name accuracy (primary) and arg-correctness among
  name-correct cases (secondary), so "right tool, wrong digit" is a visible,
  separable regression.
- **Rep:** PASS iff `extracted == expected_extracted` (field-exact, None-aware) AND
  `phase == expected_phase` AND (`reply` non-empty when `expect_nonempty_reply`).
  Three sub-metrics tracked independently — extraction is the business outcome, phase
  drives the watchdog, reply-presence is what #37/#38 regressed on.
- **Aggregate:** `ScoreReport` = pass-rate per layer + a `FailureMode` histogram. Not
  a bare number — a breakdown of *how* it failed.

## §4 E2E trajectory — the inverted-fakes pattern

The unit suite uses **fake LLMs** returning canned responses against real glue. The
trajectory eval is the inverse: the **real** `CallSessionRunner` with real Groq +
Anthropic clients, against a **scripted fake payer**.

```python
class ScriptedPayer:
    """State machine playing the other end of the call. Keyed on the agent's
    emitted intent: a correct DTMF digit (or rep-appropriate utterance) advances
    to the next node; a wrong/absent one replays the current prompt. Reactive
    AND deterministic — no LLM on the payer side."""
```

Wiring uses the runner's existing `actuator: Actuator | None` injection point
(`call_session.py:181`). The eval actuator (`_eval_actuator.py`) routes the agent's
`SpeakIntent` / `DTMFIntent` into the `ScriptedPayer` instead of a TTS queue, and the
payer's response is fed back via `runner.submit_transcript()`. No Pipecat, no audio.

Each scenario is one `.py` file declaring a payer call-graph. Assertions are on
**terminal state**: `session.completion_reason` and final `session.benefits`. Seed
set ~5–10 scenarios: happy-path full extraction, IVR dead-end (`ivr_no_progress`),
rep-stuck (`rep_stuck`), transfer-to-rep flip, hold-timeout.

**Decision #2 (default): live LLMs, not record/replay.** Live-scripted is more
valuable (it tests the actual model the agent ships with) at the cost of per-run
tokens and tolerance for LLM nondeterminism. Temperature is already pinned (IVR 0.1);
assertions are on coarse terminal state, not exact wording, which absorbs most
nondeterminism. Record/replay was considered and rejected as stale-prone.

## §5 LLM-as-judge — the only non-deterministic layer

The judge (Anthropic Opus, latest — stronger than both agent models, a prerequisite
for a valid judge) scores the subjective dimensions exact-match cannot, lifted
straight from `rep_turn.v1.txt`'s rules:

- **warmth** — greets, acknowledges, sounds like a person
- **naturalness** — not robotic, not a checklist deposition
- **no-id-restating** — states patient context once, doesn't re-dump the identity packet
- **no-stage-directions** — no "[pause]", no "I need to ask…"
- **brevity** — one to two sentences, not curt

Judges rep replies (from the rep-extraction corpus) and full E2E transcripts. Returns
per-dimension 1–5 + rationale via structured output.

**Decision #3 (default): advisory only, never a hard gate.** The judge runs N=3 per
item; report median + variance. A noisy ±1 swing must never fail a build. The judge's
job is to surface "the agent started sounding robotic" drift to a human reading the
nightly report — not to block.

## §6 CI, reporting, error handling

- **Run surface:** `make evals` (all layers) / `make evals ARGS=ivr` (one layer),
  driven by `scripts/eval.py`. Results → printed table + `eval_results/<ts>.json`
  (gitignored).
- **CI:** `.github/workflows/evals.yml` — nightly cron + on-push-to-`main`. **Never
  per-PR** (token cost + LLM flake). API keys as GH Actions secrets.
- **Three-way outcome:** `PASS / FAIL / ERROR`. An API 500 is `ERROR`, excluded from
  accuracy — it does not masquerade as a behavioral failure. One retry on transient
  API error per case (consistent with CLAUDE.md's "≤2 retry surfaces, hand-rolled
  per-call budget"; this is one such surface).
- **Cost guard:** total-case cap per run + a logged spend estimate, so a runaway loop
  cannot silently burn the account. If the cap truncates the run, log it loudly (no
  silent truncation).
- **train/test split (decision #4 corpus governance):** iterate prompts against
  `train.jsonl`; `test.jsonl` is scored only at gate time. Anti-Goodhart.

## Decisions locked (all defaults)

1. `speak.text` unchecked at component layer; quality deferred to judge.
2. E2E uses real LLMs + scripted payer (not record/replay).
3. Judge advisory-only, never gates a build.
4. Corpus hand-authored first; `from_langfuse.py` ships but seeds come from hand
   authoring (no real traffic yet).

## Testing strategy

- `tests/eval/` holds **offline, hermetic** unit tests of the harness: scorer logic
  (PASS/FAIL/ERROR classification, FailureMode mapping), JSONL loader (train/test
  split, malformed-line handling), `ScriptedPayer` state machine, eval actuator
  routing, report rendering. These run in `make test` with zero network — pyright
  strict, same bar as the rest of the repo.
- The evals themselves are exercised by a tiny smoke corpus (1–2 cases per layer) in
  CI's nightly job; the full corpus runs nightly.

## Milestone decomposition (for the plan)

Independently worktree-able:
- **M-eval/A** — `_types.py`, `_loader.py`, `_runner.py`, `_report.py` + harness unit
  tests. (Foundation; others depend on it.)
- **M-eval/B** — IVR tool-choice layer + seed corpus.
- **M-eval/C** — rep-extraction layer + seed corpus.
- **M-eval/D** — E2E trajectory (scripted payer, eval actuator, scenarios).
- **M-eval/E** — LLM-as-judge (rubric, judge eval).
- **M-eval/F** — corpus_tools (`from_langfuse.py`, `classify_failures.py`).
- **M-eval/G** — CI workflow + `make evals` + cost guard + reporting wiring.

A is the dependency root; B/C/E/F can parallelize after A; D depends on A; G depends
on everything (it wires them together). Each milestone: implement → simplify →
code-reviewer → verify → commit → PR → self-review.
