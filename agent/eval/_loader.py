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
