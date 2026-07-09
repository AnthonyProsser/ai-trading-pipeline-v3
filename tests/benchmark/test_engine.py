"""Tests for src/benchmark/engine.py — evaluate one checkpoint on its fold TEST slice.

``eval_slice_bounds`` is the testable seam (pure index math, covered under the
test-first commit); ``run_benchmark_job`` is the CSV -> fold -> model/scaler ->
trading-sim -> result-write orchestration, exercised here end-to-end against a tiny
synthetic dataset + a real (small) PatchTST checkpoint so the wiring is pinned without
the ~4.2M-row Kraken parse. Guard clauses (bad tag / semantics mismatch / no lookback)
are checked directly — they must refuse before any data or model load.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from constants import DATA, PREDICTOR
from src.benchmark import engine
from src.benchmark.registry import read_benchmark_result
from src.data.scaler import PerFoldMinMaxScaler
from src.data.walk_forward import Fold
from src.predictor.model import PatchTST

_STEM = "abc1234-s001e4348-c487b9e2e-fold0"
_LOOKBACK = 32
_NFEAT = DATA.NUM_INPUT_FEATURES  # idea-05-swinglevels: model input width (7)


class _StubRunner:
    def __init__(self, checkpoint_dir: Path, benchmark_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.benchmark_dir = benchmark_dir
        self.events: list[dict[str, object]] = []

    def broadcast(self, payload: dict[str, object]) -> None:
        self.events.append(payload)


def _write_checkpoint(
    checkpoint_dir: Path, stem: str, *, semantics: str, lookback: object = _LOOKBACK
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = PatchTST(lookback=_LOOKBACK)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "lookback": lookback,
            "constants_sha256": "a" * 64,
            "trained_through_ts_utc": "2020-01-05T00:00:00Z",
            "train_q90_coverage": 0.9,
            "target_semantics": semantics,
        },
        checkpoint_dir / f"{stem}{PREDICTOR.CHECKPOINT_WEIGHTS_SUFFIX}",
    )
    scaler = PerFoldMinMaxScaler(
        fold_start=np.datetime64("2020-01-01", "m"), fold_end=np.datetime64("2020-02-01", "m")
    )
    scaler.fit(np.random.default_rng(0).normal(size=(64, _NFEAT)))
    with open(checkpoint_dir / f"{stem}{PREDICTOR.CHECKPOINT_SCALER_SUFFIX}", "wb") as fh:
        pickle.dump(scaler, fh)


# --- eval_slice_bounds (pure index math) --------------------------------------

def test_eval_slice_bounds_targets_stay_inside_test_slice() -> None:
    fold = Fold(0, 0, 150_000, 150_000, 200_000, 200_000, 210_000)
    start, end = engine.eval_slice_bounds(fold, lookback=1_440)
    assert (start, end) == (200_000 - 1_440, 210_000)
    # The first window's target row is exactly test_start (lookback rows precede it).
    assert start + 1_440 == fold.test_start


def test_eval_slice_bounds_rejects_fold_with_insufficient_history() -> None:
    fold = Fold(0, 0, 10, 10, 20, 20, 30)
    with pytest.raises(ValueError):
        engine.eval_slice_bounds(fold, lookback=1_440)


# --- run_benchmark_job guard clauses ------------------------------------------

def test_run_benchmark_job_rejects_non_tag_stem(tmp_path: Path) -> None:
    runner = _StubRunner(tmp_path / "cp", tmp_path / "bench")
    with pytest.raises(ValueError, match="not a run-tag"):
        engine.run_benchmark_job(runner, "not-a-run-tag")


def test_run_benchmark_job_rejects_semantics_mismatch(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "cp"
    _write_checkpoint(checkpoint_dir, _STEM, semantics="per_step_logret")
    runner = _StubRunner(checkpoint_dir, tmp_path / "bench")
    with pytest.raises(ValueError, match="target_semantics"):
        engine.run_benchmark_job(runner, _STEM)


def test_run_benchmark_job_rejects_missing_lookback(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "cp"
    _write_checkpoint(
        checkpoint_dir, _STEM, semantics=PREDICTOR.TARGET_SEMANTICS, lookback=None
    )
    runner = _StubRunner(checkpoint_dir, tmp_path / "bench")
    with pytest.raises(ValueError, match="lookback"):
        engine.run_benchmark_job(runner, _STEM)


# --- run_benchmark_job happy path (synthetic data + real tiny checkpoint) ------

def test_run_benchmark_job_end_to_end_writes_result(
    tmp_path: Path, monkeypatch: Any
) -> None:
    checkpoint_dir = tmp_path / "cp"
    benchmark_dir = tmp_path / "bench"
    _write_checkpoint(checkpoint_dir, _STEM, semantics=PREDICTOR.TARGET_SEMANTICS)
    runner = _StubRunner(checkpoint_dir, benchmark_dir)

    n = 260
    rng = np.random.default_rng(1)
    features = rng.normal(0.0, 0.01, size=(n, _NFEAT)).astype(np.float64)
    feature_ts = (
        np.datetime64("2020-01-01T00:00", "m") + np.arange(n).astype("timedelta64[m]")
    ).astype("datetime64[s]")
    # One tiny fold whose TEST slice [150, 260) leaves >= lookback rows of history.
    fold = Fold(0, 0, 100, 100, 150, 150, n)

    # Redirect the CSV path to a dummy that merely needs to exist; _load_features is
    # patched so its contents are never parsed. make_folds/carve_locked_test are pinned
    # to the tiny fold above instead of the 78-fold production geometry.
    fake_root = tmp_path / "root"
    (fake_root / DATA.KRAKEN_HISTORY_OUT_DIR).mkdir(parents=True)
    (fake_root / DATA.KRAKEN_HISTORY_OUT_DIR / DATA.KRAKEN_HISTORY_CSV_NAME).write_text("x")
    monkeypatch.setattr(engine, "REPO_ROOT", fake_root)
    monkeypatch.setattr(engine, "_load_features", lambda _p: (feature_ts, features))
    monkeypatch.setattr(engine, "carve_locked_test", lambda total: total)
    monkeypatch.setattr(engine, "make_folds", lambda _n: [fold])

    engine.run_benchmark_job(runner, _STEM)

    result = read_benchmark_result(benchmark_dir, _STEM)
    assert result is not None
    assert result["stem"] == _STEM
    assert result["fold_index"] == 0
    for group in ("trading", "baselines", "statistical", "economic", "eval"):
        assert group in result
    trading = result["trading"]
    assert isinstance(trading, dict)
    assert "net_return" in trading and "trade_count" in trading
    eval_meta = result["eval"]
    assert isinstance(eval_meta, dict)
    assert str(eval_meta["start_ts_utc"]).endswith("Z")
    # A benchmark_complete event was broadcast at the end.
    assert any(e.get("type") == "benchmark_complete" for e in runner.events)
