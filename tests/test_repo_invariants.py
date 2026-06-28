"""Repo-wide hard-rule invariants (CLAUDE.md), enforced as tests.

These statically scan ``src/`` and pass vacuously until it lands, then bite as
code arrives. ``ruff`` (DTZ ruleset) is the primary UTC guard; the datetime test
here is a backstop so the rule holds even if ruff is skipped.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def _src_files() -> list[Path]:
    return list(SRC.rglob("*.py")) if SRC.exists() else []


def test_no_test_locked_reference_in_src() -> None:
    bad = [f for f in _src_files() if "test_locked" in f.read_text(encoding="utf-8")]
    assert not bad, f"src/ must never reference data/test_locked/: {bad}"


def test_no_naive_datetime_in_src() -> None:
    naive = re.compile(r"datetime\.(now|utcnow)\(\s*(?:None|tz\s*=\s*None)?\s*\)")
    bad = [f for f in _src_files() if naive.search(f.read_text(encoding="utf-8"))]
    assert not bad, f"src/ must use tz-aware UTC datetimes only: {bad}"


def test_src_never_imports_scripts() -> None:
    imp = re.compile(r"^\s*(?:from|import)\s+scripts\b", re.MULTILINE)
    bad = [f for f in _src_files() if imp.search(f.read_text(encoding="utf-8"))]
    assert not bad, f"src/ must never import from scripts/: {bad}"
