# RLVR Phone-Agent Capability — Design Spec

Date: 2026-06-07
Status: Draft (revised after senior staff review — "take the cut: single-domain first")

## Goal (revised)

Train a small open-weights model (~9B, Qwen family) to do the phone-agent skill —
navigate an IVR menu via tool calls, decide the human-agent transition, and run a
goal-directed conversation with structured extraction — and **prove it beats its
own base model on a genuinely held-out set, on one domain (healthcare), before
committing to anything bigger.** RLVR (GRPO) supplies the reward on the verifiable
parts; SFT and an LLM judge cover the parts that aren't verifiable.

The general, multi-domain version (a model good at *any* IVR-navigate-transition-
converse task) remains the north star — but it is **earned by a gate, not assumed
up front.** This spec scopes the single-domain proof and the gate. Generality is
deferred behind it.

## Why, and the honest framing

Two real motivations: a reusable, self-hostable capability/asset; and learning
RLVR end-to-end by closing the loop the eval harness already half-built
(eval → reward → GRPO → trained policy → re-eval).

**"More effective" is a hypothesis to measure, not an assumption to engineer
toward.** The single most likely outcome — stated plainly because the senior
review insisted on it — is "the 9B base was already decent at tool use, so the
lift is solid-but-modest, or it just matches." That is a valid finding. The whole
point of the single-domain-first cut is to discover whether there is headroom
worth chasing *cheaply*, before spending the multi-domain budget.

## The central constraint: verifiable vs unverifiable (and what RLVR actually trains)

RLVR only works where the reward is automatically checkable. Being precise about
coverage (the senior review caught the spec overselling this):

- **RLVR trains (verifiable, per-turn):** IVR tool choice + args (`score_ivr`),
  the per-turn transition *decision* (press the rep digit; emit `transfer_to_rep`)
  when the turn's context makes it the right call, and structured extraction
  (`score_rep`). These are pure functions of `(case, model_output)` — verified in
  the code, no live LLM in the reward.
- **RLVR does NOT train:** trajectory-level timing ("did the handoff fire at the
  *right moment* across the whole call"), multi-turn conversational coherence, or
  conversational warmth. Per-turn rewards can't see cross-turn state, and warmth
  has no verifiable signal at all.
- **Therefore:** the transition *as a per-turn tool-call decision* is in RLVR's
  scope; the transition *as right-moment trajectory behavior* and all of the
  conversational quality are handled by **SFT warm-start** (seed the style + the
  multi-turn behavior) and an **LLM judge** guardrail (so verifiable RLVR can't
  silently reward-hack warmth into curt-but-correct). Trajectory RL is a possible
  later stretch, explicitly deferred.

## Anti-leakage (hard constraint, designed in now)

The effectiveness claim only counts on a held-out set that is **not from the same
generator as the training data.** A same-generator "held-out split" shares the
generator's phrasing tics, menu templates, and label conventions — a model that
learns those quirks looks like it generalizes when it has only memorized the
generator. So:

- The **held-out eval set comes from a different source** than the training data:
  human-authored, OR a different generator model/family, OR real IVR transcripts.
  Not the train pipeline with a different seed.
- The held-out set's labels are **not derived by the train generator's heuristics**
  (independently authored/audited).
- The shared-scorer guard (the eval and the trainer import the same comparison
  logic, but the *grading* split is never seen by the trainer) is necessary but
  not sufficient — it does not address generator leakage, which this constraint
  does.

## Reward adapter (training and grading deliberately diverge)

The eval scorers are built for *grading*, not for dense RL reward. The reward
adapter wraps the same core comparison logic but diverges where training needs it
to (the senior review flagged "one place, no drift" as actively wrong here):

- **Partial credit, not PASS/FAIL.** `score_rep` exact-dict-equality collapses
  "4 of 5 fields right" and "0 right" to reward 0 — useless gradient. The adapter
  exposes per-field F1 for extraction and arg-level partial credit for tool calls.
- **No-tool-call → reward 0, not a raise.** `score_ivr` *raises* `NoToolCallError`
  on an empty response (correct for the eval: a transient Groq glitch). In
  training, a base model emitting no parseable tool call is normal early behavior
  and the most informative negative — the adapter maps it to reward 0 (a penalty
  to learn away from), explicitly diverging from the eval's raise-and-retry.
- The *core* tool-name / arg-match / field-match logic stays shared with the eval
  so train-time and grade-time semantics don't drift on the parts that should match.

## Architecture — phased arc (single-domain first; generality gated)

Each phase is its own spec → plan → build cycle. Detailed scope below is the
**vertical slice only**; later phases are sketched.

- **Vertical slice (FIRST, detailed below).** Thin end-to-end RLVR on a tiny model
  + the existing healthcare IVR cases. Proves the serverless-GPU + GRPO + reward
  plumbing and pins the data format. A few dollars, a day.
- **Phase 0 — Focused healthcare dataset.** Generate a bounded, healthcare-domain
  corpus (IVR menus + rep dialogues) with verifiable labels, PLUS a separately-
  sourced held-out set (anti-leakage constraint). Gated by a **label-quality
  audit** (human-check a sample; if oracle label accuracy is low, stop — the rest
  is built on sand). Smaller than the original "hundreds-thousands multi-domain"
  ambition because it's one domain.
- **Phase 1 — SFT warm-start.** LoRA SFT of Qwen ~9B on the dataset: context →
  correct tool call / correct rep output. Seeds style + multi-turn behavior.
- **Phase 2 — RLVR (GRPO).** From the SFT checkpoint, GRPO with the partial-credit
  reward adapter over the verifiable rewards.
- **Phase 2.5 — THE GATE.** Measure trained-vs-base on the *separately-sourced*
  held-out healthcare set. **Real, leak-free lift over the base by a meaningful
  margin is required to proceed.** If it only matches the base, that is the
  finding; the multi-domain arc does NOT start.
- **Phase 3 — Judge guardrail.** Build the deferred LLM judge as a warmth eval
  (and optionally a reward term) so the RLVR run is checked for conversational
  regression, not just verifiable-metric gain.
- **Phase 4 — Serve + integrate + measure.** vLLM-serve the model; write
  `QwenPhoneAgentClient` satisfying `IVRLLMClient`/`RepLLMClient`. NOTE this seam
  is **two custom tool-call parsers we own**, not one pluggable client: a vLLM
  tool-call parser matching Qwen's chat template (e.g. `--enable-auto-tool-choice
  --tool-call-parser hermes`) for IVR, and guided/JSON-schema decoding for the rep
  structured output (a different mechanism from Anthropic's native `messages.parse`,
  with different failure modes). Then run the existing eval harness trained-vs-base
  on held-out healthcare.
- **[GATED] Phase 5+ — Generality.** Multi-domain dataset, the four-way comparison
  (trained vs base vs larger-same-family vs production), cross-domain held-out.
  Only if Phase 2.5 passes.

## Compute

Serverless GPU (Modal or RunPod), single 80GB GPU (H100/A100) with LoRA. Note
this is **tight, not comfortable**, for 9B GRPO: policy + frozen ref + vLLM
rollout engine + KV cache co-resident on one card is the memory cliff people hit.
LoRA (no full-weight optimizer state) is what makes it fit. **Confirm the 9B
memory budget on a short smoke run before any real GRPO spend.** The slice uses a
tiny model precisely so this is discovered cheaply.

## First sub-project (detailed): the vertical slice

A pipeline proof, not a capability proof.

- **Model:** smallest viable instruct model (Qwen ~0.5–1.7B).
- **Data:** the 6 IVR cases in `agent/eval/ivr_tool_choice/corpus/` + a handful of
  hand-written variants the trainer NEVER sees (a miniature held-out, so the slice
  doesn't rehearse the leakage flaw).
- **Reward:** the partial-credit adapter over `score_ivr` (arg-level partial;
  no-tool-call → 0, not raise).
- **Training:** TRL `GRPOTrainer` (LoRA) on the serverless GPU.
- **Success criteria (plumbing only):**
  1. The serverless GPU job launches and runs GRPO to completion.
  2. The reward adapter plugs in and mean reward *moves* over steps (the plumbing
     demonstrably shapes behavior).
  3. The trained tiny model does **not regress** vs its base on the held-out
     variants (we explicitly do NOT claim capability here — 6 cases prove nothing
     about capability).
  4. The data/rollout/reward interfaces are pinned, defining the format Phase 0
     must match.
- **Non-goal:** beating any production model, capability, or generality.

## Non-goals (this track)

- Production deployment / autoscaling serving / HIPAA-grade infra.
- Multi-turn trajectory RL (deferred stretch).
- The multi-domain generality, the four-way comparison, and large synthetic
  datasets — all GATED behind Phase 2.5, not in this scope.
- Beating a frontier hosted model on conversational polish.

## Risks (named)

- **Most likely result is "matches the base"** — the project's own admission; the
  single-domain gate exists to learn this cheaply instead of expensively.
- **Phase 0 label quality** is the real research risk — auto-derived labels need an
  oracle whose competence caps the model and can leak; the label-quality audit
  gate addresses it.
- **Reward sparsity/hacking** — handled by the partial-credit adapter (density) and
  SFT + judge (warmth).
- **Small-model GRPO instability** — surfaced cheaply by the slice.
- **Memory cliff on 9B GRPO** — smoke-budget before real spend.
- **Serving-seam parsing** — two custom parsers, scoped into Phase 4, not
  discovered during integration.

## Testing / validation

- The eval harness is reused as the scoreboard; the reward adapter shares the core
  comparison logic with the scorers but diverges deliberately (partial credit,
  no-tool-call). The grading split is separately-sourced and never seen by the
  trainer (anti-leakage).
- Hermetic unit tests for every non-GPU piece: the reward adapter (scalar + partial
  credit + no-tool-call mapping), dataset schema/loader, rollout formatting, and
  `QwenPhoneAgentClient` protocol conformance + tool-call parsing. GPU scripts are
  smoke-tested on the tiny model.
- **Phase 2.5 is the definitive validation** of the single-domain effectiveness
  hypothesis and the gate for everything beyond it.

## Open decisions (deferred to per-phase specs)

- Exact Qwen model IDs (slice tiny model + ~9B real model) — confirm at build time
  (cutoff predates mid-2026 releases).
- Modal vs RunPod — decide at the slice spec by TRL/vLLM setup ergonomics.
- LoRA rank / GRPO hyperparameters — from TRL defaults, tuned empirically.
- Judge as reward term vs guardrail-only — decided after Phase 2 shows how much
  warmth degrades under pure-verifiable RLVR.
- Held-out source — human-authored vs different-generator vs real transcripts —
  decided at the Phase 0 spec (whichever is cheapest that genuinely breaks
  generator correlation).
