"""Regression guards for scripts/search_predictor.py's constants.py safety net.

The search loop mutates constants.py directly and unattended. A prior real run corrupted
constants.py when an iteration was interrupted mid-flight (fixed in commit c2071f7). These
tests pin the invariant: whatever the failure mode — invalid candidate, non-parseable
patch, judge veto, no-improvement — constants.py is left byte-identical to its
pre-iteration state, and the transient .searchbak snapshot is cleaned up.

The tests never touch the real repo constants.py: search_predictor's module-level path
constants are monkeypatched to a temp copy, and every subprocess/LLM boundary (propose /
run_training / judge) is stubbed so no `claude` CLI or GPU is required.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "search_predictor.py"

# A minimal but syntactically real constants.py surrogate carrying only the fields the
# patcher targets. Byte-for-byte identity is what the tests assert on.
SURROGATE_CONSTANTS = '''"""surrogate constants for tests"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    LOOKBACK: int = 1_440
    SEARCH_CONFIRM_SEEDS: int = 3


@dataclass(frozen=True)
class PredictorConfig:
    D_MODEL: int = 128
    N_HEADS: int = 8
    N_LAYERS: int = 3
    D_FF: int = 256
    DROPOUT: float = 0.1
    PATCH_SIZE: int = 16
    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY: float = 1e-2
    WARMUP_FRAC: float = 0.05
    DIRECTION_PENALTY_LAMBDA: float = 1.75


DATA = DataConfig()
PREDICTOR = PredictorConfig()
'''


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("search_predictor_under_test", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """search_predictor module with all disk paths redirected into tmp_path."""
    mod = _load_module()
    constants_path = tmp_path / "constants.py"
    constants_path.write_text(SURROGATE_CONSTANTS, encoding="utf-8")

    monkeypatch.setattr(mod, "CONSTANTS_PATH", constants_path)
    monkeypatch.setattr(mod, "CONSTANTS_BAK_PATH", tmp_path / "constants.py.searchbak")
    monkeypatch.setattr(mod, "LOG_PATH", tmp_path / "search_log.jsonl")
    monkeypatch.setattr(mod, "KILL_FLAG_PATH", tmp_path / "SEARCH_KILL_SWITCH.flag")

    # current_values reloads the top-level `constants` package; stub it to read the
    # surrogate's baseline instead so no real import is required.
    monkeypatch.setattr(
        mod, "current_values",
        lambda: {
            "LEARNING_RATE": 3e-4, "WEIGHT_DECAY": 1e-2, "WARMUP_FRAC": 0.05,
            "DROPOUT": 0.1, "DIRECTION_PENALTY_LAMBDA": 1.75, "D_MODEL": 128,
            "N_HEADS": 8, "N_LAYERS": 3, "D_FF": 256, "PATCH_SIZE": 16, "LOOKBACK": 1440,
        },
    )
    return mod


def _run(mod: Any, monkeypatch: pytest.MonkeyPatch, proposal: dict[str, Any],
         *, val: float | None = 0.1, judge_ok: bool = True) -> None:
    monkeypatch.setattr(mod, "propose", lambda *a, **k: proposal)
    monkeypatch.setattr(mod, "run_training", lambda seed: val)
    monkeypatch.setattr(mod, "judge", lambda name, changes, move: (judge_ok, "stub"))
    mod.run_iteration(iteration=1, confirm_seeds=1)


def _read(mod: Any) -> str:
    text: str = mod.CONSTANTS_PATH.read_text(encoding="utf-8")
    return text


def test_invalid_candidate_leaves_constants_untouched(sp: Any, monkeypatch: Any) -> None:
    before = _read(sp)
    # N_HEADS=3 does not evenly divide D_MODEL=128 -> rejected before any write.
    _run(sp, monkeypatch, {"move": "CAPACITY", "changes": {"N_HEADS": 3}})
    assert _read(sp) == before


def test_out_of_bounds_value_leaves_constants_untouched(sp: Any, monkeypatch: Any) -> None:
    before = _read(sp)
    _run(sp, monkeypatch, {"move": "SCHEDULE", "changes": {"LEARNING_RATE": 9.9}})
    assert _read(sp) == before


def test_judge_veto_reverts_constants(sp: Any, monkeypatch: Any) -> None:
    before = _read(sp)
    _run(sp, monkeypatch, {"move": "CAPACITY", "changes": {"D_MODEL": 256}},
         val=0.1, judge_ok=False)
    assert _read(sp) == before
    assert not sp.CONSTANTS_BAK_PATH.exists()  # snapshot cleaned up


def test_no_improvement_reverts_constants(sp: Any, monkeypatch: Any) -> None:
    # Seed a prior kept best of 0.05 so a worse val (0.2) triggers the no-improvement path.
    sp.LOG_PATH.write_text(
        '{"iteration": 0, "kept": true, "val_total": 0.05}\n', encoding="utf-8"
    )
    before = _read(sp)
    _run(sp, monkeypatch, {"move": "CAPACITY", "changes": {"D_MODEL": 256}}, val=0.2)
    assert _read(sp) == before


def test_non_parseable_patch_reverts_constants(sp: Any, monkeypatch: Any) -> None:
    """If patch_constants ever produces syntactically broken output, the ast.parse guard
    must revert before any training runs. Force it by making patch_constants emit garbage."""
    before = _read(sp)
    monkeypatch.setattr(sp, "patch_constants", lambda text, fields: text + "\ndef (:\n")
    training_calls: list[int] = []

    def _fake_training(seed: int) -> float:
        training_calls.append(seed)
        return 0.1

    monkeypatch.setattr(sp, "run_training", _fake_training)
    monkeypatch.setattr(sp, "propose",
                        lambda *a, **k: {"move": "CAPACITY", "changes": {"D_MODEL": 256}})
    monkeypatch.setattr(sp, "judge", lambda name, changes, move: (True, "stub"))
    sp.run_iteration(iteration=1, confirm_seeds=1)
    assert _read(sp) == before
    assert training_calls == []  # never trained against a broken file
    assert not sp.CONSTANTS_BAK_PATH.exists()


def test_interrupt_mid_training_restores_and_logs(sp: Any, monkeypatch: Any) -> None:
    """An exception after the patch but before a keep/revert decision must (a) restore
    constants.py byte-identically and (b) leave an audit-trail log entry so the durable
    history records the interrupted attempt."""
    before = _read(sp)
    monkeypatch.setattr(sp, "propose",
                        lambda *a, **k: {"move": "CAPACITY", "changes": {"D_MODEL": 256}})

    def _boom(seed: int) -> float:
        raise KeyboardInterrupt

    monkeypatch.setattr(sp, "run_training", _boom)
    monkeypatch.setattr(sp, "judge", lambda name, changes, move: (True, "stub"))
    with pytest.raises(KeyboardInterrupt):
        sp.run_iteration(iteration=1, confirm_seeds=1)
    assert _read(sp) == before
    assert not sp.CONSTANTS_BAK_PATH.exists()
    entries = [
        json.loads(line)
        for line in sp.LOG_PATH.read_text(encoding="utf-8").splitlines()
    ]
    assert any("interrupted" in e.get("reason", "") for e in entries)


def test_successful_keep_applies_and_cleans_snapshot(sp: Any, monkeypatch: Any) -> None:
    _run(sp, monkeypatch, {"move": "CAPACITY", "changes": {"D_MODEL": 256}},
         val=0.1, judge_ok=True)
    after = _read(sp)
    assert "D_MODEL: int = 256  # search_predictor.py" in after
    assert not sp.CONSTANTS_BAK_PATH.exists()


def teardown_module(module: Any) -> None:
    sys.modules.pop("search_predictor_under_test", None)
