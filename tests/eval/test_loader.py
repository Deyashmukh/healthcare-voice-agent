# pyright: strict
"""Unit tests for the JSONL corpus loader."""

from __future__ import annotations

import json
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


def test_load_cases_rejects_unknown_field(tmp_path: Path) -> None:
    """A misspelled / stale corpus key must fail loudly (extra='forbid'), not be
    silently dropped and scored against defaults."""
    valid = IVREvalCase(id="c1", payer="p", history=[], expected_tool="wait", rationale="r")
    bad = valid.model_dump()
    bad["expcted_args"] = {"digits": "2"}  # typo of expected_args
    path = _write(tmp_path / "corpus.jsonl", [json.dumps(bad)])
    with pytest.raises(CorpusError) as exc:
        load_cases(path, IVREvalCase)
    assert "line 1" in str(exc.value)
