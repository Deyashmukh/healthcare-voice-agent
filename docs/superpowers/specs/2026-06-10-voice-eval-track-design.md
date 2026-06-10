# Voice-Layer Eval Track — Design Spec

Date: 2026-06-10
Status: Draft (pending senior review)

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
   exercising VAD, STT, TTS, turn-taking, and timing end to end with the agent
   completely unmodified.

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
- **RLVR gate:** if the noisy pass rate is within 10 points of the clean pass
  rate, RLVR is archived for this project. If it craters, RLVR is re-scoped
  around noise robustness with the noisy corpus as training signal. The
  10-point threshold is provisional pending first measurement, per project
  convention (don't pin numbers ahead of measurement).

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
- `transcribe(audio: bytes) -> str` — Deepgram, configured identically to
  production (import the model/settings from the production config path;
  do not copy constants). Live wrapper, coverage-omitted.

### M-voice/A — noisy corpus harvest + gate

**Harvest script** (`agent/eval/audio/harvest.py`, live, coverage-omitted):
for each of the 12 component cases (6 IVR + 6 rep), take each `user`-role
turn's text, run synthesize → degrade → transcribe, and write
`corpus/cases_noisy.jsonl` next to the clean corpus. Same case schema; ids
suffixed `-noisy`; a `source_id` field links back to the clean case.
Multi-turn histories: only `user` turns are re-transcribed (they are the
"heard" side); assistant/tool turns pass through unchanged.

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
corpus path differs. Included in `all`.

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
  `callSid` matching Twilio's JSON shapes.
- Streams payer lines as base64 mu-law `media` messages in 20ms frames at
  real-time pace (the pipeline's VAD assumes real-time audio).
- Receives the agent's outbound `media` frames and buffers them.

**Turn-taking:** trailing silence on the agent's outbound audio (energy
threshold over decoded mu-law, ~1.5s) means "agent finished speaking." The
caller then transcribes the buffered agent audio via the toolkit (so the run
log shows both sides as text) and streams the next scripted line. Scenario
scripts advance exactly as in the text E2E — one line per agent turn, final
line repeated if the agent needs extra turns. Silence threshold and window
are constants tuned during first live runs; both are recorded in the run
output so later tuning is evidence-based.

**Process model:** the eval spawns the agent app in-process (uvicorn served
as an asyncio task in the eval process), so `score_scenario` can reach the
live `CallSession` object exactly as the text E2E does. No subprocess, no
IPC.

**Scoring:** `score_scenario` unchanged (completion reason + benefits).
Timing recorded, not asserted, appended to `eval_results/` as
`e2e_audio-timing-<ts>.jsonl`: per-turn latency (payer-line-end → first agent
audio frame), per-turn agent speech duration, total call duration.

**Corpus:** the existing 2 `Scenario` entries, untouched. New scenarios
(e.g. a barge-in scenario where the payer interrupts mid-agent-speech) are
explicitly deferred to a follow-up after the plumbing works.

**CLI registration:** `e2e-audio` layer, **excluded from `all`** —
manual-dispatch only. It burns ElevenLabs + Deepgram on both sides of every
turn (~$1/run scale) and its latency variance makes it unsuitable for an
unattended gate until budgets are pinned.

## Error handling

- Provider failures (ElevenLabs, Deepgram) raise typed errors from the
  `AgentError` taxonomy → the runner's existing retry/ERROR path. Never a
  silent FAIL (the lesson from the transient-Groq bug, PR #46).
- Websocket disconnect mid-scenario → scenario ERROR with the partial
  two-sided transcript attached.
- Harvest script: idempotent, refuses to overwrite without `--force`.
- The audit step cannot be skipped: harvest output lands in a staging file
  (`cases_noisy.staging.jsonl`, gitignored); only the human-audited rename to
  `cases_noisy.jsonl` is committable.

## Testing

Offline (counted toward the 90% floor, `# pyright: strict`):

- `degrade`: output sample rate, mu-law round-trip properties, length
  preservation within codec framing.
- Mu-law 20ms framing/chunking and base64 Twilio `media` message
  construction (exact JSON shape against a recorded real Twilio message).
- Silence-detection turn-taker fed synthetic PCM (tone-then-silence,
  silence-then-tone, never-silent) — deterministic, no real audio.
- Noisy-corpus loader: schema, `source_id` linkage, staging-file rejection.

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
- Background-noise overlays in `degrade` (deferred until codec-only proves
  too gentle).
- Nightly/CI scheduling of the audio surface.

## Risks

- **Deepgram on TTS audio may be too clean.** ElevenLabs speech is clearer
  than real humans on real phones; the noisy corpus may understate production
  noise. Mitigation: the audit step observes the actual corruption level; if
  Deepgram returns near-verbatim text, add the deferred noise overlay to
  `degrade` and re-harvest before declaring the RLVR gate decided.
- **Turn-taking fragility.** Energy-threshold silence detection can misfire
  on TTS pauses. Mitigation: threshold constants surfaced in run output;
  scenario ERROR (not FAIL) on turn-taking deadlock, with the two-sided
  transcript for diagnosis.
- **In-process uvicorn + pipeline + caller in one event loop** may have
  timing interactions a real deployment doesn't. Accepted for a learning
  project; the manual phone runs remain the full-realism check.
- **Cost creep.** Both live surfaces are run-on-demand with frozen artifacts;
  nothing live runs in CI.
