# Voice-Layer Eval Track — Design Spec

Date: 2026-06-10
Status: Revised after senior review (APPROVE-WITH-FIXES, 2026-06-12 — all
must-fix and should-fix findings incorporated)

## Goal

Close the realism gap in the eval harness. Today every eval feeds the decision
layer clean, perfectly-transcribed text; production feeds it Deepgram output
over an 8kHz phone line. Two new surfaces fix this:

1. **Noisy-transcript variants (M-voice/A)** — the existing component eval
   cases re-spoken through TTS, degraded to phone quality, and transcribed by
   the production STT; whatever Deepgram actually emits becomes a frozen,
   audited corpus. Measures decision-layer robustness on realistic input.
2. **Speaking scripted payer (M-voice/B)** — the existing E2E scenarios run as
   real audio over the agent's actual Media Streams websocket endpoint,
   exercising VAD, STT, TTS, turn-taking, and timing end to end. The agent's
   decision and audio paths run unmodified; the only agent-side changes are a
   runner-registry hook and an auto-hangup env flag (both below).

A third deliverable rides along: **the RLVR track is demoted to paused**, with
a measurable gate (below) deciding whether it is archived or re-scoped.

## Why

The harness has proven itself on the decision layer (three real bugs caught),
but its 100% pass rates are measured on idealized input. The layers that make
voice agents hard — transcription noise, timing, barge-in, turn-taking — have
no automated coverage. Separately, the RLVR plan as written would train on the
already-easy clean-text task; whether any trainable headroom exists is exactly
what the noisy surface measures. One track answers both.

## Decisions locked during brainstorm

- **Audio path for the speaking payer:** websocket loopback. A fake Twilio
  client speaks the Media Streams protocol against the agent's real `/ws`
  endpoint. No phone network, no second number, no per-minute cost. The
  existing manual phone-call runs remain the sampled real-network check.
- **Noise source:** harvested, not modeled. Real TTS → telephony degradation →
  real Deepgram output, frozen as a corpus. We do not hand-model ASR errors.
- **Assertions on day one (audio E2E):** outcome only (completion reason +
  benefits, same as text E2E). Timing is recorded, not asserted; budgets get
  pinned in a follow-up after baselines exist. Same measure-first convention
  as barge-in latency (M8'/F).
- **RLVR gate (pre-committed, decidable):** measured on the pooled 12 cases
  (6 IVR + 6 rep), K=3 runs of the clean surfaces and K=3 runs of the noisy
  surfaces. Any ERROR-outcome case is re-run until it resolves to PASS/FAIL
  (ERRORs are provider glitches, not signal, and must not pollute the
  denominator). The metric is the mean net case-level delta: average number
  of cases that fail on noisy minus average failing on clean. Bands, fixed
  now, before measurement:
  - delta ≤ 1 case → RLVR archived for this project;
  - delta ≥ 3 cases → RLVR re-scoped around noise robustness, with the noisy
    corpus as training signal;
  - delta of exactly 2 → explicit user judgment call, with the decision and
    reasoning recorded in the RLVR spec status header.
  Multiple runs because the clean surfaces themselves have swung 83–100%
  across runs in `eval_history.jsonl`; a single-run delta on n=12 is inside
  the harness's own variance. Decision rules are pre-committed precisely so
  the threshold can't be picked after seeing the number (the measure-first
  convention is for performance budgets, not decision rules).

## Architecture

### Shared audio toolkit — `agent/eval/audio/`

Three functions behind a clean seam, mirroring the harness's pure-scorer /
live-wrapper split:

- `synthesize(text: str) -> bytes` — ElevenLabs TTS, returns raw PCM. Live
  wrapper (coverage-omitted, same convention as `eval.py` live files).
- `degrade(pcm: bytes) -> bytes` — pure function: resample to 8kHz, encode
  mu-law, decode back. Codec loss IS the degradation; additive background
  noise is deferred until first measurement shows the codec alone is too
  gentle. Offline, `# pyright: strict`, fully unit-tested.
- `transcribe(audio: bytes) -> str` — Deepgram. Production currently uses
  pipecat's default model implicitly (`nova-3-general`, streaming live API);
  there is no config constant to import. Deliverable: hoist a
  `DEEPGRAM_MODEL` constant into agent config, consumed by BOTH `main.py`
  and the toolkit, so they cannot drift. The toolkit uses Deepgram's
  prerecorded API; the streaming-vs-batch divergence (no endpointing or
  interim results in batch) is accepted and documented in the module
  docstring. Live wrapper, coverage-omitted.

### M-voice/A — noisy corpus harvest + gate

**Harvest script** (`agent/eval/audio/harvest.py`, live, coverage-omitted):
for each of the 12 component cases (6 IVR + 6 rep), take each `user`-role
turn's text, run synthesize → degrade → transcribe, and write
`corpus/cases_noisy.jsonl` next to the clean corpus. Ids are suffixed
`-noisy`; a `source_id: str | None = None` field links back to the clean
case. `EvalCase` is `extra="forbid"`, so adding `source_id` to the base
model is an explicit schema-change deliverable of M-voice/A (with a loader
test asserting the linkage), not an incidental detail. Multi-turn
histories: only `user` turns are re-transcribed (they are the "heard"
side); assistant/tool turns pass through unchanged — currently a one-rule
provision since all 12 cases have single-user-turn histories; no machinery.
The harvest also writes the degraded WAVs next to the staging file
(gitignored): the audit sometimes needs ears, not just text, to judge
whether TTS or STT mangled a digit.

**Re-harvest trigger (pre-committed):** if the median word-level WER of
noisy vs clean user-turn text is below 5%, the codec-only degradation is
declared too gentle; the deferred background-noise overlay is added to
`degrade` and the harvest re-run. The RLVR gate may not be declared decided
on a sub-5%-WER corpus.

**Human audit (mandatory, blocking):** the harvest output is presented to the
user as clean/noisy pairs. Where the corruption destroys the label (e.g. the
menu digit itself got mangled so the expected tool call is no longer
inferable), the case is relabeled or dropped, with a one-line note in the
case's `rationale`. Only the audited file is committed. Evals run from the
frozen file and never re-harvest, so the surface is deterministic and incurs
no audio-API cost per run (the usual per-case LLM calls still apply, same as
the clean surfaces). The harvest script refuses to overwrite an existing corpus without
`--force`.

**CLI registration:** `ivr-noisy` and `rep-noisy` layers in
`agent/eval/cli.py`, reusing `score_ivr` / `score_rep` unchanged — only the
corpus path differs. Included in `all`. Registration touches both the
`_LAYERS` dict and the hard-coded argparse `choices` list — both, or the
layers are unreachable.

**The measurement:** first run records clean-vs-noisy pass-rate deltas in the
eval report. That number decides the RLVR gate.

**RLVR demotion (same PR):** the RLVR spec
(`docs/superpowers/specs/2026-06-07-rlvr-phone-agent-design.md`) and slice
plan (`docs/superpowers/plans/2026-06-08-rlvr-vertical-slice.md`) get a
status header: `Paused — pending noisy-eval headroom evidence`, with the gate
wording above and a pointer to this spec.

### M-voice/B — speaking scripted payer

**`FakeTwilioCaller`** (`agent/eval/e2e_audio/_caller.py`): a websocket
client that impersonates Twilio Media Streams against the agent's real `/ws`
endpoint:

- Sends `connected` and `start` messages with fabricated `streamSid` /
  `callSid` matching Twilio's JSON shapes (the `/ws` handler ignores
  `connected` and waits for `start` — verified).
- Streams payer lines as base64 mu-law `media` messages in 20ms frames at
  real-time pace (the pipeline's VAD assumes real-time audio).
- **Streams continuous mu-law silence (0xFF frames) at 20ms cadence whenever
  it is not speaking, for the lifetime of the connection.** This is
  load-bearing, not cosmetic: the transcript flush is VAD-driven
  (`state_processor.py` fires on `VADUserStoppedSpeakingFrame`, which Silero
  emits only by analyzing silence *audio*, never from the absence of
  frames), and Deepgram's live socket closes after ~10s without audio. Real
  Twilio streams silence frames; a fake that goes quiet deadlocks the
  pipeline on turn one.
- Receives the agent's outbound `media` frames and buffers them; ignores
  unknown event types; flushes its agent-audio buffer on `{"event":
  "clear"}` (emitted by the serializer on interruption frames).

**Turn-taking — two triggers, not one:**

1. Trailing silence on the agent's outbound audio (energy threshold over
   decoded mu-law, ~1.5s) means "agent finished speaking" → transcribe the
   buffered agent audio via the toolkit (so the run log shows both sides as
   text) and stream the next scripted line.
2. **No agent audio at all within T seconds of the payer line ending →
   advance to the next line anyway.** Required, not optional: `wait` and
   `transfer_to_rep` produce zero outbound audio by design, and BOTH
   committed scenarios contain a turn whose correct response is the silent
   `transfer_to_rep`. Silence-only turn-taking would deadlock on every
   passing trajectory. (This is also what real IVRs do — they don't wait
   forever.)

The loop is capped by the existing `scenario.max_turns`. The
deadlock-ERROR path guards only genuine hangs beyond both triggers and the
turn cap. Silence threshold, window, and T are constants recorded in the
run output so later tuning is evidence-based; the turn-taker takes an
injectable clock so its unit tests are deterministic (per the fake-clock
rule).

**Process model and the one agent-side hook:** the eval spawns the agent app
in-process (uvicorn served as an asyncio task in the eval process). The
text E2E reaches the session because its eval constructs the runner itself;
in the audio path the runner is a local inside `main.py`'s `ws()` handler,
so the spec adds **one observability hook** (~3 lines): `main.py` registers
the runner in `app.state.runners[call_sid]` on start and removes it on
disconnect. The decision and audio paths are untouched; "agent unmodified"
is amended to "agent's decision/audio path unmodified; one registry hook."
Call termination: nothing on the agent side closes a successful call
(`HangupIntent` is a deliberate no-op), so the caller polls `session.done`
via the registry handle and then closes the websocket, triggering the
existing disconnect → cancel → `runner.stop()` chain.

**Twilio auto-hangup:** pipecat's `TwilioFrameSerializer` defaults
`auto_hang_up=True` and POSTs a call-termination request to api.twilio.com
on `EndFrame` — for a fabricated `callSid`, a guaranteed 404 logged at
error level (handled, non-fatal). To keep run logs honest, `main.py` gains
a `TWILIO_AUTO_HANGUP` env flag (default true, preserving production
behavior); the eval sets it false. Without this, "no phone network" carries
an asterisk and every run log contains a misleading error.

**Scoring:** `score_scenario` unchanged (completion reason + benefits).
Timing recorded, not asserted, appended to `eval_results/` as
`e2e_audio-timing-<ts>.jsonl`: per-turn latency (payer-line-end → first agent
audio frame), per-turn agent speech duration, total call duration. Per-turn
agent audio is also written under `eval_results/` (turn-taking misfires are
diagnosed by listening, not by guessing from text). Note the IVR-phase agent
audio currently comes from the TEMP DTMF stand-in (the actuator *speaks*
"Pressing zero."); when real DTMF tones land (task #2), energy-based
turn-taking still works (tones are audio) but transcribed logs for those
turns become garbage — that milestone touches this surface.

**Corpus:** the existing 2 `Scenario` entries, untouched. New scenarios
(e.g. a barge-in scenario where the payer interrupts mid-agent-speech) are
explicitly deferred to a follow-up after the plumbing works.

**CLI registration:** `e2e-audio` layer, **excluded from `all`** —
manual-dispatch only. It burns ElevenLabs + Deepgram on both sides of every
turn (~$1/run scale) and its latency variance makes it unsuitable for an
unattended gate until budgets are pinned.

## Error handling

- Provider failures in the *component* surfaces raise typed errors from the
  `AgentError` taxonomy → the runner's existing retry/ERROR path. Never a
  silent FAIL (the lesson from the transient-Groq bug, PR #46).
- The audio E2E scorer **returns** `CaseResult(outcome=ERROR)` rather than
  raising: per the runner's contract, returned ERRORs are final and not
  retried, and auto-retrying a multi-dollar-cents audio scenario on a
  websocket blip is the wrong default. The partial two-sided transcript and
  timing data are persisted to the `eval_results/` artifact files keyed by
  case id (a `CaseResult.error` string can't hold them).
- Harvest script: idempotent, refuses to overwrite without `--force`.
- Audit enforcement is mechanical, not procedural: the harvest writes every
  case's `rationale` prefixed `UNAUDITED:`, and the corpus loader rejects
  any case carrying the prefix. The audit removes the prefix case by case as
  each label is verified. (A gitignored staging file alone only prevents
  *accidental* commits; the loader check makes an unaudited corpus unusable.)

## Testing

Offline (counted toward the 90% floor, `# pyright: strict`):

- `degrade`: output sample rate, mu-law round-trip properties, length
  preservation within codec framing.
- Mu-law 20ms framing/chunking and base64 Twilio `media` message
  construction (exact JSON shape against a recorded real Twilio message).
- Turn-taker fed synthetic PCM (tone-then-silence, silence-then-tone,
  never-silent, and the no-audio-at-all timeout path) with an injectable
  clock — deterministic, no real audio, no wall-clock sleeps.
- Noisy-corpus loader: schema, `source_id` linkage, `UNAUDITED:` rejection.

Live (coverage-omitted, verified by being run, output committed as PR
evidence):

- The harvest itself (M-voice/A) — audited corpus + first clean-vs-noisy
  measurement.
- The audio E2E (M-voice/B) — both scenarios green + timing JSONL from at
  least 3 runs to show variance.

## Sequencing

1. **M-voice/A** — own worktree/PR: toolkit + harvest + audit + noisy layers
   + RLVR demotion docs. Measurement happens before the PR merges (the
   numbers go in the PR body as evidence).
2. **Gate decision** — user call, recorded in the RLVR spec status.
3. **M-voice/B** — own worktree/PR: caller + turn-taking + e2e-audio layer.
   Depends on the toolkit only, so it can start once A's toolkit lands.

Usual loop per milestone: plan → senior review → subagent-driven build →
simplify → code review → verify → commit → PR → self-review.

## Non-goals

- Asserted latency budgets (follow-up after baselines).
- Barge-in / interruption scenarios in the audio E2E corpus (follow-up).
- Real-PSTN automated calls, second Twilio number, Coval-style vendor
  platforms.
- TTS naturalness evaluation (MOS, neural MOS predictors) — the agent's
  listener is an IVR or a rep; intelligibility is covered by the round-trip
  construction itself.
- Background-noise overlays in `degrade` — deferred, but with the
  pre-committed sub-5%-WER trigger that forces them in if codec-only proves
  too gentle.
- Nightly/CI scheduling of the audio surface.

## Risks

- **Deepgram on TTS audio may be too clean — likely, in fact.** Mu-law
  round-trip is roughly 38dB SNR, nearly transparent to nova-3 on studio TTS;
  the dominant degradation (8kHz bandwidth) mostly costs fricatives. The
  expected failure mode is near-verbatim text with only normalization
  differences. Mitigation is the pre-committed sub-5%-WER re-harvest trigger
  (above), which is decidable by computation, not by impression.
- **Turn-taking fragility.** Energy-threshold silence detection can misfire
  on TTS pauses. Mitigation: threshold constants surfaced in run output;
  scenario ERROR (not FAIL) on turn-taking deadlock, with the two-sided
  transcript for diagnosis.
- **In-process uvicorn + pipeline + caller in one event loop** may have
  timing interactions a real deployment doesn't. Accepted for a learning
  project; the manual phone runs remain the full-realism check.
- **Cost creep.** Both live surfaces are run-on-demand with frozen artifacts;
  nothing live runs in CI.
