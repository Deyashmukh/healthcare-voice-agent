# RLVR Vertical Slice — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the RLVR pipeline end-to-end on the cheapest possible setup — `Qwen/Qwen2.5-0.5B-Instruct` (tiny, deliberately NOT the target model), the existing 6 IVR cases, TRL GRPO with our verifiable reward, on Modal serverless GPU — establishing that the GPU+GRPO+reward plumbing works and pinning the data/reward format before any real spend. (The real-run target after this slice is **Qwen3.5-9B**, locked with the user; the slice stays tiny on purpose, to catch pipeline bugs for ~$2 before the 9B spend.)

**Architecture:** A pure, hermetically-tested **reward adapter** (parse the model's Hermes `<tool_call>` text → score against expected tool+args, with partial credit; no-tool-call → 0) and a pure **dataset builder** (the IVR corpus → GRPO `prompt`/`ground_truth` records) live in a new `training/` area under the project's normal pyright/ruff/pytest discipline. A **Modal training script** loads `Qwen/Qwen2.5-0.5B-Instruct`, builds a `datasets.Dataset`, runs `trl.GRPOTrainer` (LoRA + vLLM rollout) with the reward adapter, and scores trained-vs-base on a held-out set the trainer never sees. The reward adapter + dataset builder are local + tested; the Modal script's GPU run is the one paid, paused-for-user step.

**Tech Stack:** Python 3.12 (local: pure stdlib + `modal` SDK); Modal image: `trl[vllm]`, `vllm`, `transformers`, `peft`, `datasets`. Reward adapter reuses the *concept* of `agent/eval/ivr_tool_choice/_score.py` (tool-name exact, arg partial) but operates on raw completion text.

**Scope (this is a PLUMBING proof, not a capability proof):** success = GRPO runs to completion on Modal, mean reward moves, and the trained tiny model does not regress on held-out variants. It does NOT claim capability — 6 cases prove nothing about capability (per the spec).

---

## Pre-req the USER owns (paused step, Task 6)
A Modal account + a Modal token configured locally (`modal token new`) and a Modal Secret named `huggingface-secret` holding `HF_TOKEN`. The plan builds everything else first; only Task 6 needs this and a few dollars of GPU.

## File Structure

- `training/__init__.py` — empty marker.
- `training/reward.py` — `parse_tool_call(text)`, `ivr_reward(completion, expected_tool, expected_args)`, and `ivr_reward_func(completions, ground_truth, **kwargs)` (the GRPO-shaped wrapper). Pure, no heavy deps.
- `training/dataset.py` — `build_ivr_records(cases)` → `list[dict]` with `prompt` (chat messages) + `ground_truth` (JSON string). Pure.
- `training/corpus/heldout_ivr.jsonl` — hand-written held-out IVR cases the trainer never sees.
- `training/train_slice.py` — the Modal app + GRPO training + trained-vs-base held-out scoring. Imports `modal` at top; `trl`/`vllm`/etc. imported INSIDE the remote function (run in the cloud image). Coverage-omitted.
- `tests/training/__init__.py`, `tests/training/test_reward.py`, `tests/training/test_dataset.py` — hermetic.
- `pyproject.toml` — add a `training` dependency group (`modal`); coverage-omit `training/train_slice.py`.

Every new `.py` under `training/` and `tests/` starts with `# pyright: strict`.

---

## Task 1: Reward adapter (pure) + tests — the load-bearing piece

**Files:** Create `training/__init__.py` (empty), `training/reward.py`; Test `tests/training/__init__.py` (empty), `tests/training/test_reward.py`.

- [ ] **Step 1: Write the failing test** `tests/training/test_reward.py`:

```python
# pyright: strict
"""Unit tests for the GRPO reward adapter (pure, offline)."""

from __future__ import annotations

import json

from training.reward import ivr_reward, ivr_reward_func, parse_tool_call

_CALL = '<tool_call>\n{"name": "send_dtmf", "arguments": {"digits": "2"}}\n</tool_call>'


def test_parse_tool_call_extracts_name_and_args() -> None:
    parsed = parse_tool_call(f"sure, pressing 2. {_CALL}")
    assert parsed is not None
    assert parsed[0] == "send_dtmf"
    assert parsed[1] == {"digits": "2"}


def test_parse_tool_call_returns_none_when_absent() -> None:
    assert parse_tool_call("I am not sure what to do here.") is None


def test_parse_tool_call_returns_none_on_malformed_json() -> None:
    assert parse_tool_call("<tool_call>\n{not json}\n</tool_call>") is None


def test_reward_full_on_correct_tool_and_args() -> None:
    assert ivr_reward(_CALL, "send_dtmf", {"digits": "2"}) == 1.0


def test_reward_zero_on_no_tool_call() -> None:
    # No parseable tool call is a real penalty in training (the model must learn
    # away from it), NOT an error/raise like the eval scorer does.
    assert ivr_reward("uh, hello?", "send_dtmf", {"digits": "2"}) == 0.0


def test_reward_zero_on_wrong_tool() -> None:
    call = '<tool_call>\n{"name": "wait", "arguments": {}}\n</tool_call>'
    assert ivr_reward(call, "send_dtmf", {"digits": "2"}) == 0.0


def test_reward_partial_on_right_tool_wrong_arg() -> None:
    # right tool (0.5 base) + 0 of 1 args correct = 0.5 — dense signal vs the
    # eval scorer's binary FAIL.
    call = '<tool_call>\n{"name": "send_dtmf", "arguments": {"digits": "9"}}\n</tool_call>'
    assert ivr_reward(call, "send_dtmf", {"digits": "2"}) == 0.5


def test_reward_full_when_no_args_expected_and_tool_matches() -> None:
    call = '<tool_call>\n{"name": "wait", "arguments": {}}\n</tool_call>'
    assert ivr_reward(call, "wait", {}) == 1.0


def test_reward_json_number_arg_matches_string_expected() -> None:
    # Same provider quirk the eval scorer handles: compare as strings.
    call = '<tool_call>\n{"name": "send_dtmf", "arguments": {"digits": 2}}\n</tool_call>'
    assert ivr_reward(call, "send_dtmf", {"digits": "2"}) == 1.0


def test_reward_func_maps_over_a_batch() -> None:
    gt = json.dumps({"tool": "send_dtmf", "args": {"digits": "2"}})
    rewards = ivr_reward_func(
        completions=[_CALL, "no idea"],
        ground_truth=[gt, gt],
    )
    assert rewards == [1.0, 0.0]


def test_reward_func_handles_conversational_completions() -> None:
    # GRPO passes message-dict lists in conversational mode; take the last content.
    gt = json.dumps({"tool": "wait", "args": {}})
    call = '<tool_call>\n{"name": "wait", "arguments": {}}\n</tool_call>'
    rewards = ivr_reward_func(
        completions=[[{"role": "assistant", "content": call}]],
        ground_truth=[gt],
    )
    assert rewards == [1.0]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd <worktree> && uv run pytest tests/training/test_reward.py -q` → FAIL at import (`No module named 'training.reward'`).

- [ ] **Step 3: Write `training/reward.py`**

```python
# pyright: strict
"""GRPO reward adapter for the IVR tool-choice slice.

Parses the model's Hermes-style `<tool_call>{...}</tool_call>` completion and
scores it against the expected tool + args with PARTIAL CREDIT (dense reward for
GRPO), deliberately diverging from the eval scorer in two ways the spec calls
for: (1) partial credit instead of PASS/FAIL, (2) a missing/garbled tool call
maps to reward 0.0 (a penalty to learn away from) rather than raising — in
training, no-tool-call is normal early behavior, not a provider glitch.
"""

from __future__ import annotations

import json
import re

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_call(text: str) -> tuple[str, dict[str, object]] | None:
    """Extract (name, arguments) from the first Hermes tool call, or None."""
    match = _TOOL_CALL_RE.search(text)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return None
    name = payload.get("name")
    args = payload.get("arguments", {})
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    return name, args  # pyright: ignore[reportUnknownVariableType]


def ivr_reward(completion: str, expected_tool: str, expected_args: dict[str, object]) -> float:
    """Scalar reward in [0, 1]: 0 for no/wrong tool, 0.5 for right tool, plus up
    to 0.5 scaled by the fraction of expected args matched (string-compared, the
    same provider-quirk tolerance the eval scorer uses)."""
    parsed = parse_tool_call(completion)
    if parsed is None:
        return 0.0
    name, args = parsed
    if name != expected_tool:
        return 0.0
    if not expected_args:
        return 1.0
    matched = sum(1 for k, v in expected_args.items() if str(args.get(k)) == str(v))
    return 0.5 + 0.5 * (matched / len(expected_args))


def ivr_reward_func(
    completions: list[object],
    ground_truth: list[str],
    **kwargs: object,
) -> list[float]:
    """GRPO reward-function shape: called with `completions` + the `ground_truth`
    dataset column (a JSON string `{"tool":..., "args":...}` per sample); returns
    one float per completion. `completions` are strings (standard format) or
    message-dict lists (conversational) — take the last assistant content."""
    rewards: list[float] = []
    for completion, gt in zip(completions, ground_truth, strict=True):
        text = completion if isinstance(completion, str) else _last_content(completion)
        spec = json.loads(gt)
        rewards.append(ivr_reward(text, spec["tool"], spec["args"]))
    return rewards


def _last_content(completion: object) -> str:
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            content = last.get("content", "")  # pyright: ignore[reportUnknownMemberType]
            if isinstance(content, str):
                return content
    return ""
```

- [ ] **Step 4: Run + pass** `uv run pytest tests/training/test_reward.py -q` → 11 PASS.

- [ ] **Step 5: Lint + type-check** `uv run ruff check training/ tests/training/ && uv run pyright training/reward.py tests/training/test_reward.py` → clean.

- [ ] **Step 6: Commit**

```bash
git add training/__init__.py training/reward.py tests/training/__init__.py tests/training/test_reward.py
git commit -m "feat(rlvr): GRPO reward adapter with partial credit (vertical slice)"
```

---

## Task 2: Dataset builder (pure) + held-out variants + tests

**Files:** Create `training/dataset.py`, `training/corpus/heldout_ivr.jsonl`; Test `tests/training/test_dataset.py`.

- [ ] **Step 1: Write the held-out variants** `training/corpus/heldout_ivr.jsonl` (the trainer never sees these; they're for the trained-vs-base check). Reuse the `IVREvalCase` JSON shape so the existing loader can read them:

```jsonl
{"id": "ho-billing-press-3", "payer": "generic", "history": [{"role": "user", "content": "Welcome. For pharmacy press 1, for billing press 3."}], "expected_tool": "send_dtmf", "expected_args": {"digits": "3"}, "rationale": "billing is option 3"}
{"id": "ho-greeting", "payer": "generic", "history": [{"role": "user", "content": "Thank you for calling member services."}], "expected_tool": "wait", "rationale": "opening greeting, no menu"}
{"id": "ho-member-id", "payer": "generic", "history": [{"role": "user", "content": "Please key in your subscriber ID now."}], "expected_tool": "speak", "rationale": "identifier request"}
```

- [ ] **Step 2: Write the failing test** `tests/training/test_dataset.py`:

```python
# pyright: strict
"""Unit tests for the GRPO dataset builder (pure, offline)."""

from __future__ import annotations

import json

from agent.eval._loader import load_cases
from agent.eval._types import IVREvalCase
from agent.eval.ivr_tool_choice._score import CORPUS
from training.dataset import HELDOUT_CORPUS, build_ivr_records


def test_records_have_prompt_and_ground_truth() -> None:
    cases = load_cases(CORPUS, IVREvalCase)
    records = build_ivr_records(cases)
    assert len(records) == len(cases)
    rec = records[0]
    assert isinstance(rec["prompt"], list)  # chat messages
    assert rec["prompt"][0]["role"] == "system"
    assert rec["prompt"][-1]["role"] == "user"
    spec = json.loads(rec["ground_truth"])
    assert spec["tool"] == cases[0].expected_tool
    assert spec["args"] == cases[0].expected_args


def test_system_prompt_lists_the_tools_and_hermes_format() -> None:
    records = build_ivr_records(load_cases(CORPUS, IVREvalCase))
    system = records[0]["prompt"][0]["content"]
    assert "send_dtmf" in system and "transfer_to_rep" in system
    assert "<tool_call>" in system  # the model is told the output format


def test_user_message_is_the_latest_menu_transcript() -> None:
    cases = load_cases(CORPUS, IVREvalCase)
    records = build_ivr_records(cases)
    assert records[0]["prompt"][-1]["content"] == cases[0].history[-1].content


def test_heldout_corpus_path_exists_and_loads() -> None:
    cases = load_cases(HELDOUT_CORPUS, IVREvalCase)
    assert len(cases) >= 3
    assert all(c.id.startswith("ho-") for c in cases)
```

- [ ] **Step 3: Run to verify it fails** `uv run pytest tests/training/test_dataset.py -q` → FAIL at import.

- [ ] **Step 4: Write `training/dataset.py`**

```python
# pyright: strict
"""Build GRPO training records from IVR eval cases.

Each record is `{"prompt": <chat messages>, "ground_truth": <json string>}` —
the exact shape TRL's GRPOTrainer wants (a `prompt` column; every other column
is passed to the reward function by name). The tool list + Hermes output format
are baked into the system message as TEXT so the model sees them regardless of
how the chat template handles `tools=` (robust + self-contained for the slice).
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.eval._types import IVREvalCase

HELDOUT_CORPUS = Path(__file__).parent / "corpus" / "heldout_ivr.jsonl"

_TOOLS = (
    "send_dtmf(digits, purpose='menu'|'rep') — press DTMF keys to navigate.\n"
    "speak(text) — say the literal patient identifier when asked.\n"
    "wait() — acknowledge a greeting / hold / non-actionable filler.\n"
    "transfer_to_rep() — a human rep has arrived.\n"
    "complete_call(reason) — the IVR is closing.\n"
    "fail_with_reason(reason) — the IVR is a dead end."
)

_SYSTEM = (
    "You are navigating an automated phone menu (IVR) to reach a benefits "
    "representative. Choose exactly ONE tool per turn.\n\nTools:\n"
    f"{_TOOLS}\n\n"
    "Respond with a single tool call in this exact format:\n"
    '<tool_call>\n{"name": <tool>, "arguments": {<args>}}\n</tool_call>'
)


def build_ivr_records(cases: list[IVREvalCase]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for case in cases:
        menu = case.history[-1].content if case.history else ""
        records.append(
            {
                "prompt": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": menu},
                ],
                "ground_truth": json.dumps({"tool": case.expected_tool, "args": case.expected_args}),
            }
        )
    return records
```

- [ ] **Step 5: Run + pass** `uv run pytest tests/training/test_dataset.py -q` → 4 PASS.

- [ ] **Step 6: Lint + type-check + full suite**

Run: `uv run ruff check training/ tests/training/ && uv run ruff format --check training/ tests/training/ && uv run pyright training/ tests/training/ && uv run pytest tests/`
Expected: clean; full suite green (training tests are hermetic, count toward coverage).

- [ ] **Step 7: Commit**

```bash
git add training/dataset.py training/corpus/heldout_ivr.jsonl tests/training/test_dataset.py
git commit -m "feat(rlvr): GRPO dataset builder + held-out IVR variants (vertical slice)"
```

---

## Task 3: Dependencies + coverage config

**Files:** Modify `pyproject.toml`.

The reward adapter + dataset builder are pure (no new runtime deps). The Modal script needs the `modal` SDK locally to launch; `trl`/`vllm`/etc. live only in the Modal image. Keep `modal` out of the agent runtime via a dependency group.

- [ ] **Step 1: Add a `training` dependency group.** In `pyproject.toml`, add (`modal` to launch; `transformers`+`torch` for the local pre-flight in Task 5 — a CPU/MPS generate of a 0.5B is feasible locally):

```toml
[dependency-groups]
training = ["modal>=1.0", "transformers>=4.50", "torch>=2.4"]
```

(Install locally with `uv sync --group training` when working on Task 4+. Confirm current version floors at install time. `trl`/`vllm`/`peft`/`datasets` are NOT here — they live only in the Modal image.)

- [ ] **Step 2: Coverage-omit the cloud training script.** Add `"training/train_slice.py"` to the `[tool.coverage.run] omit` list (it imports `modal` + cloud-only deps and runs on GPU, never under `pytest`). The pure `training/reward.py` + `training/dataset.py` are NOT omitted.

- [ ] **Step 3: Verify the floor holds** `uv run pytest tests/` → green, floor reached.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore(rlvr): training dependency group + omit cloud train script"
```

---

## Task 4: The Modal GRPO training script (built locally; GPU run is Task 6)

**Files:** Create `training/train_slice.py`.

Built and lint/type/import-checked locally; the actual GPU run is the paused Task 6. `trl`/`vllm`/`datasets`/`peft` are imported INSIDE the remote function so the file imports locally with only `modal` installed.

- [ ] **Step 1: Resolve the dependency pinset FIRST.** The TRL ecosystem has a version conflict between primary sources (PyPI `trl==1.5.1` vs Modal's GRPO example `trl[vllm]==0.28.0` + `datasets==3.5.1`, which are mutually incompatible lines). Pick the **Modal-tested set** as the baseline (it is verified to run GRPO on Modal): `trl[vllm]==0.28.0`, `vllm==0.12.0`, `transformers==4.57`, `datasets==3.5.1`, `peft` (latest compatible). If that set fails to install or run on Modal, fall back to the latest `trl` line and resolve the whole stack against *that* line's `pyproject.toml` — do NOT mix the two pinsets. Record the working pins in a comment at the top of the file.

- [ ] **Step 2: Write `training/train_slice.py`**

```python
# pyright: strict
"""RLVR vertical slice — TRL GRPO on Qwen2.5-0.5B on Modal. Pipeline proof only.

Run: `modal run training/train_slice.py` (needs a Modal account/token + a Modal
Secret `huggingface-secret` with HF_TOKEN). Not a pytest test; coverage-omitted.

Dep pinset (Modal-tested GRPO line — see Task 4 Step 1):
    trl[vllm]==0.28.0, vllm==0.12.0, transformers==4.57, datasets==3.5.1, peft
"""

from __future__ import annotations

import modal

_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "trl[vllm]==0.28.0",
        "vllm==0.12.0",
        "transformers==4.57",
        "datasets==3.5.1",
        "peft",
        "accelerate",
    )
)

app = modal.App("rlvr-slice")
checkpoints = modal.Volume.from_name("rlvr-slice-ckpt", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/ckpt": checkpoints},
)
def train() -> dict[str, float]:
    import json

    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    # The pure helpers are added to the image so the cloud run scores with the
    # SAME reward + dataset code the local tests cover (see Task 4 Step 3).
    from training.dataset import HELDOUT_CORPUS, build_ivr_records
    from training.reward import ivr_reward
    from agent.eval._loader import load_cases
    from agent.eval._types import IVREvalCase
    from agent.eval.ivr_tool_choice._score import CORPUS

    train_records = build_ivr_records(load_cases(CORPUS, IVREvalCase))
    dataset = Dataset.from_list(train_records)

    def reward_func(completions: list[object], ground_truth: list[str], **_: object) -> list[float]:
        out: list[float] = []
        for completion, gt in zip(completions, ground_truth, strict=True):
            text = completion if isinstance(completion, str) else completion[-1]["content"]
            spec = json.loads(gt)
            out.append(ivr_reward(text, spec["tool"], spec["args"]))
        return out

    config = GRPOConfig(
        output_dir="/ckpt/grpo",
        learning_rate=1e-5,            # LoRA LR (≈10x the GRPO base 1e-6)
        num_generations=8,
        # 16 completions/step = 2 DISTINCT prompts × 8 generations. With only ONE
        # distinct prompt per step (the default 8×8) the group-relative advantage
        # collapses to ~0 and reward never moves — the exact signal we're paying
        # to observe. 16 % num_generations(8) == 0 satisfies TRL's divisibility rule.
        per_device_train_batch_size=16,
        max_completion_length=128,
        num_train_epochs=20,           # tiny dataset: many passes for a visible move
        use_vllm=True,
        # Single GPU: run vLLM IN-PROCESS. The default vllm_mode="server" expects a
        # separately-launched `trl vllm-serve` on a DIFFERENT GPU and will hang
        # trying to reach a server that doesn't exist.
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.3,  # headroom for policy + ref + KV cache
        logging_steps=1,
        report_to=[],
    )
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM")

    trainer = GRPOTrainer(
        model=_MODEL,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_func,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model("/ckpt/grpo-final")
    checkpoints.commit()

    base = _heldout_mean_reward(_MODEL, None)
    trained = _heldout_mean_reward(_MODEL, "/ckpt/grpo-final")
    print(f"held-out mean reward — base={base:.3f} trained={trained:.3f}")
    return {"base": base, "trained": trained}


def _heldout_mean_reward(model_id: str, adapter_path: str | None) -> float:
    """Generate one completion per held-out case (greedy) and mean the reward.
    Uses transformers generate (not vLLM) to keep the eval path simple."""
    import json

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from training.dataset import HELDOUT_CORPUS, build_ivr_records
    from training.reward import ivr_reward
    from agent.eval._loader import load_cases
    from agent.eval._types import IVREvalCase

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    records = build_ivr_records(load_cases(HELDOUT_CORPUS, IVREvalCase))
    rewards: list[float] = []
    for rec in records:
        text = tok.apply_chat_template(rec["prompt"], tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to("cuda")
        out = model.generate(**ids, max_new_tokens=128, do_sample=False)
        completion = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        spec = json.loads(rec["ground_truth"])
        rewards.append(ivr_reward(completion, spec["tool"], spec["args"]))
    return sum(rewards) / len(rewards)


@app.local_entrypoint()
def main() -> None:
    result = train.remote()
    print(f"RESULT: {result}")
```

- [ ] **Step 3: Make the local package importable in the Modal image.** The remote function imports `training.*` and `agent.eval.*`. Add the repo source to the image so those imports resolve in the cloud. Append to the `image` definition:

```python
    .add_local_python_source("training", "agent")
```

(Confirm the exact current Modal method name for adding local source — `add_local_python_source` vs `add_local_dir` — at build time; the goal is that `import training.reward` and `import agent.eval...` work inside the container.)

- [ ] **Step 4: Header + pyright config (cloud-only deps can't be strict).** `train_slice.py` touches `trl`/`vllm`/`peft`/`datasets`/`torch`/`transformers`, none installed locally — under `# pyright: strict` every symbol from them is `Unknown` and strict flags dozens of `reportUnknown*` errors that per-import ignores can't fix. So:
  - Give `train_slice.py` a **`# pyright: basic`** header (not strict).
  - In `pyproject.toml`'s `[tool.pyright]`, add `training` to `include` (so the PURE `reward.py`/`dataset.py` get strict-checked) and add `training/train_slice.py` to `exclude` (skip the cloud file entirely; it's validated by import-smoke + the run, not the type gate).

- [ ] **Step 5: Lint + import smoke (no GPU, no Modal run)**

Run:
```
uv sync --group training
uv run ruff check training/train_slice.py && uv run ruff format --check training/train_slice.py
uv run pyright training/reward.py training/dataset.py
uv run python -c "import training.train_slice; print('import OK')"
```
Expected: ruff clean; pyright clean on the pure files; import smoke prints OK (only `modal` is imported at module top; the heavy imports are inside the functions, so the local import succeeds without them).

- [ ] **Step 6: Commit**

```bash
git add training/train_slice.py pyproject.toml
git commit -m "feat(rlvr): Modal GRPO training script for the vertical slice"
```

---

## Task 5: Offline verification + local pre-flight gate

**Files:** Create `training/preflight.py`.

- [ ] **Step 1: Full offline gate.** Run:
```
uv run ruff check . && uv run ruff format --check . && uv run pyright training/reward.py training/dataset.py tests/training/ && uv run pytest tests/
```
Expected: all clean; full suite green including `tests/training/` (reward + dataset hermetic tests); coverage floor held with `train_slice.py` omitted.

- [ ] **Step 2: Write `training/preflight.py`** — a LOCAL check (no Modal, no GPU) that the base 0.5B actually emits parseable `<tool_call>` on our prompts. This is the single highest-value zero-cost de-risk: if the base model rarely emits the format, every GRPO rollout scores 0, reward can't move, and the paid run proves nothing. Catch that for free first.

```python
# pyright: basic
"""Local pre-flight: does Qwen2.5-0.5B emit parseable <tool_call> on our prompts?
Runs on CPU/MPS, no Modal, no GPU spend. Run: `uv run python -m training.preflight`.
"""

from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.eval._loader import load_cases
from agent.eval._types import IVREvalCase
from agent.eval.ivr_tool_choice._score import CORPUS
from training.dataset import build_ivr_records
from training.reward import parse_tool_call

_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> None:
    tok = AutoTokenizer.from_pretrained(_MODEL)
    model = AutoModelForCausalLM.from_pretrained(_MODEL)
    records = build_ivr_records(load_cases(CORPUS, IVREvalCase))
    parsed_ok = 0
    for rec in records:
        text = tok.apply_chat_template(rec["prompt"], tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt")
        out = model.generate(**ids, max_new_tokens=128, do_sample=False)
        completion = tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True)
        ok = parse_tool_call(completion) is not None
        parsed_ok += int(ok)
        print(f"  parseable={ok}  out={completion[:120]!r}")
    rate = parsed_ok / len(records)
    print(f"\nbase parse rate: {parsed_ok}/{len(records)} = {rate:.0%}")
    print("GATE: proceed only if a non-trivial fraction parse; if ~0%, switch the")
    print("dataset prompt to the tokenizer's tools= param (native Hermes) before paying.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the pre-flight + gate.** Run: `uv sync --group training && uv run python -m training.preflight`. (Slow on CPU/MPS but free — a 0.5B × 6 short generations is a couple of minutes.) **GATE:** if the base parse rate is ~0%, do NOT pay for the Modal run — the text-format prompt isn't eliciting `<tool_call>` from the tiny model; rework `dataset.py` to build prompts via `tokenizer.apply_chat_template(..., tools=<schemas>)` (Qwen's native Hermes formatting) and re-run pre-flight until a non-trivial fraction parse. Only then proceed to Task 6.

- [ ] **Step 4: Commit the pre-flight.** (`training/preflight.py` is heavy-dep + local-only — exclude it from pyright strict alongside `train_slice.py` in `pyproject.toml`, and coverage-omit it.)

```bash
git add training/preflight.py pyproject.toml
git commit -m "feat(rlvr): local pre-flight gate (base emits parseable tool calls)"
```

---

## Task 6: PAUSE — run the slice on Modal (user-owned, ~$2-5)

The first paid step. Hand to the user; do NOT run autonomously.

- [ ] **Step 1 (user):** `modal token new` (one-time), then create the Secret: `modal secret create huggingface-secret HF_TOKEN=<hf token>`.
- [ ] **Step 2 (user or, with the token present, the agent):** `modal run training/train_slice.py`. Watch the GRPO logs: **mean reward should move upward over steps** (success criterion #2).
- [ ] **Step 3:** Read the printed `RESULT: {'base': X, 'trained': Y}`. Success criteria: the job ran GRPO to completion (#1); **mean reward moved upward over steps (#2) — this is the real signal the slice exists to show**; and `trained >= base` on the held-out variants (#3). **Criterion #3 is a non-regression SANITY check, NOT evidence of capability** — 3 greedy cases on a 0.5B is near-noise (one case flipping swings the mean by 0.33), so a slightly-lower `trained` is not a failure, only a flat/crashed reward curve (#2) or a server-mode hang (#1) is. (Optional, same dollar: have `_heldout_mean_reward` sample n=4 at temperature ~0.8 and average, to make the readout less noisy.) Record base vs trained + the reward-curve observation in `docs/superpowers/notes/eval-baselines.md` (append an "RLVR slice" section).
- [ ] **Step 4:** Commit the recorded result (docs only). If GRPO crashed / reward never moved / trained < base, that is a real finding about the pipeline or pins — triage before declaring the slice done (per the spec, the slice exists to surface exactly this cheaply).

---

## Self-Review (completed during planning)

- **Spec coverage:** the slice section of the spec (model, data = existing 6 + held-out variants, partial-credit reward over `score_ivr`, TRL GRPOTrainer + LoRA on serverless GPU, the four success criteria) → Tasks 1 (reward, partial credit + no-tool-call→0), 2 (data + held-out the trainer never sees), 3 (deps/omit), 4 (Modal GRPO + trained-vs-base held-out scoring), 6 (the run + the 4 criteria). The anti-leakage principle is honored at slice scale (held-out variants are hand-written, not generated by any train pipeline). ✔
- **Placeholder scan:** complete code for reward, dataset, tests, and the Modal script. The two "confirm at build time" notes (the `modal` add-local-source method name; the local-pyright ignores for cloud-only imports) are genuine version-confirmation gates flagged by the API research, not hand-waves — the surrounding code is complete. ✔
- **Type consistency:** `parse_tool_call -> tuple[str, dict]|None`, `ivr_reward(completion, expected_tool, expected_args) -> float`, `ivr_reward_func(completions, ground_truth, **kwargs) -> list[float]`, `build_ivr_records(cases) -> list[dict]`, `HELDOUT_CORPUS`/`CORPUS` paths — all consistent across reward.py, dataset.py, the tests, and train_slice.py's inline reward_func (which mirrors `ivr_reward_func`). ✔
- **Known build-time risks (flagged, not hidden):** the TRL version pinset conflict (Task 4 Step 1 resolves it explicitly); pyright-strict can't resolve cloud-only imports (Task 4 Step 4 → `# pyright: basic` + exclude); Modal's `add_local_python_source` exact name (Task 4 Step 3). Fast-moving-stack realities to re-confirm at build time.

## Senior review changelog

Revised after an independent senior-staff ML review (verdict: proceed with changes; APIs cross-checked against current TRL/Modal docs). Two run-blockers fixed before any code:
- **MF-1 (would hang the paid run):** `GRPOConfig(use_vllm=True)` defaults to `vllm_mode="server"`, which needs a separate `trl vllm-serve` on a different GPU. Added `vllm_mode="colocate"` + `vllm_tensor_parallel_size=1` + `vllm_gpu_memory_utilization=0.3` for single-GPU.
- **MF-2 (would produce no GRPO signal):** `per_device_train_batch_size=8` with `num_generations=8` = one distinct prompt per step → group-relative advantage ~0 → reward never moves. Raised to 16 (2 prompts × 8 generations; 16 % 8 == 0).
- **SF-1 (highest-value de-risk):** added a free LOCAL pre-flight (Task 5) that confirms the base 0.5B emits parseable `<tool_call>` before any spend — if it's ~0%, switch to the tokenizer's native `tools=` Hermes formatting first.
- **SF-2:** reframed success criterion #3 as a non-regression sanity check (3 greedy cases is near-noise), with the reward-curve move (#2) as the real signal.
- **SF-3:** `train_slice.py`/`preflight.py` use `# pyright: basic` + pyright `exclude` (strict can't resolve cloud/heavy deps); `training/` added to pyright `include` for the pure files.
- Validated by the reviewer: the reward-func kwarg signature, the `trl[vllm]==0.28.0` Modal-tested pinset, the Modal API surface (`add_local_python_source`, Volume, Secret, local_entrypoint), and the partial-credit math.
