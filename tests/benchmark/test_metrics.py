"""Tests for src/benchmark/metrics.py — the salvaged eval_predictor metric layer.

These are the reviewed metric tests migrated from tests/scripts/test_eval_predictor.py
(the functions themselves migrate from scripts/eval_predictor.py to
src/benchmark/metrics.py so the benchmark app can consume them without importing from
scripts/, per the src-never-imports-scripts rule). Expectations are unchanged — the
migration must be behavior-preserving.

Committed before src/benchmark/metrics.py, per the test-first discipline.
"""
from __future__ import annotations

import pytest
import torch

from constants import DATA, EXECUTION, PREDICTOR
from src.benchmark.metrics import (
    excursion_metrics,
    statistical_metrics,
    target_to_model_space,
)

_CLOSE = DATA.FEATURE_NAMES.index("close_logret")
_Q10 = PREDICTOR.QUANTILES.index(0.10)
_Q50 = PREDICTOR.QUANTILES.index(0.50)
_Q90 = PREDICTOR.QUANTILES.index(0.90)
_NDIM = PREDICTOR.NUM_OUTPUT_DIMS
_NQ = len(PREDICTOR.QUANTILES)


def _make_pred(close_q50: list[float], pad: float = 1.0) -> torch.Tensor:
    """(1, H, DIM, Q) prediction tensor with the given per-step close q50 values.

    q10/q90 are placed a fixed pad below/above q50 so calibration/coverage are
    deterministic; non-close dims are zero.
    """
    horizon = len(close_q50)
    pred = torch.zeros(1, horizon, _NDIM, _NQ)
    for t, v in enumerate(close_q50):
        pred[0, t, _CLOSE, _Q10] = v - pad
        pred[0, t, _CLOSE, _Q50] = v
        pred[0, t, _CLOSE, _Q90] = v + pad
    return pred


def _make_target(close_steps: list[float]) -> torch.Tensor:
    """(1, H, DIM) raw per-step log-return target with the given close deltas."""
    horizon = len(close_steps)
    target = torch.zeros(1, horizon, _NDIM)
    for t, v in enumerate(close_steps):
        target[0, t, _CLOSE] = v
    return target


# --- target_to_model_space -------------------------------------------------

def test_target_to_model_space_per_step_is_identity() -> None:
    target = _make_target([0.1, -0.2, 0.3])
    out = target_to_model_space(target, "per_step_logret")
    assert torch.equal(out, target)


def test_target_to_model_space_cumulative_is_cumsum() -> None:
    target = _make_target([0.1, -0.2, 0.3])
    out = target_to_model_space(target, "cumulative_logret")
    expected = torch.cumsum(target, dim=1)
    assert torch.allclose(out, expected)


# --- excursion_metrics (the Krafer-style metric) ---------------------------

def test_excursion_captured_fraction_per_step() -> None:
    # Predicted total move = -0.30 (down). Realized cumulative path = [-.10, -.15, +.05]:
    # best favorable (downward) excursion = 0.15 -> 0.15 / 0.30 = 0.50 of the move played out.
    pred = _make_pred([-0.30, 0.0, 0.0])  # per-step q50 cumsum -> final -0.30
    target = _make_target([-0.10, -0.05, 0.20])  # cumsum -> [-.10, -.15, +.05]
    m = excursion_metrics(pred, target, "per_step_logret", fee_threshold=EXECUTION.FEE_THRESHOLD)
    assert m["n_used"] == 1
    assert m["mean_captured_fraction"] == pytest.approx(0.5, abs=1e-6)
    # Adverse excursion: price went +0.05 against the short -> 0.05 / 0.30.
    assert m["mean_adverse_ratio"] == pytest.approx(0.05 / 0.30, abs=1e-6)


def test_excursion_captured_fraction_cumulative() -> None:
    # Same realized path, but the model emits the cumulative path directly.
    pred = _make_pred([-0.10, -0.20, -0.30])  # cumulative q50 path, final -0.30
    target = _make_target([-0.10, -0.05, 0.20])
    m = excursion_metrics(pred, target, "cumulative_logret", fee_threshold=EXECUTION.FEE_THRESHOLD)
    assert m["mean_captured_fraction"] == pytest.approx(0.5, abs=1e-6)


def test_excursion_excludes_sub_fee_predictions() -> None:
    # Predicted move below the round-trip fee is not tradeable -> excluded from the mean.
    tiny = EXECUTION.FEE_THRESHOLD / 2.0
    pred = _make_pred([tiny, 0.0, 0.0])
    target = _make_target([0.01, 0.0, 0.0])
    m = excursion_metrics(pred, target, "per_step_logret", fee_threshold=EXECUTION.FEE_THRESHOLD)
    assert m["n_used"] == 0
    assert m["mean_captured_fraction"] != m["mean_captured_fraction"]  # NaN when nothing tradeable


# --- statistical_metrics ---------------------------------------------------

def test_statistical_metrics_keys_and_coverage() -> None:
    # q90 sits a full pad above every target -> coverage 1.0; target inside [q10,q90] -> cal 1.0.
    pred = _make_pred([0.0, 0.0])
    target_model = _make_target([0.0, 0.0])  # already model-space for per_step
    m = statistical_metrics(pred, target_model)
    assert set(m) >= {"q90_coverage", "calibration_rate", "directional_accuracy", "sharpness_close"}
    assert m["q90_coverage"] == pytest.approx(1.0)
    # Sharpness = mean (q90 - q10) on close = 2 * pad = 2.0.
    assert m["sharpness_close"] == pytest.approx(2.0, abs=1e-6)
    # Pinball over (B=1,H=2,D=5,Q=3)=30 elements: only the close dim is non-zero. Per step
    # the close pins [q10,q50,q90]=[-1,0,1] vs target 0 -> errors give 0.1+0+0.1=0.2; two
    # steps -> 0.4 summed, meaned over all 30 elements = 0.4/30.
    assert m["pinball"] == pytest.approx(0.4 / 30, abs=1e-6)


def test_statistical_da_scores_final_horizon_step_only() -> None:
    """DA reads ONLY the final horizon step — the move a hold-to-horizon trade actually
    spans. Intermediate steps carry tradeable q50s with the WRONG sign; if they were
    counted (the old all-steps semantics), DA would be 1/3 instead of 1.0."""
    fee = EXECUTION.FEE_THRESHOLD
    pred = _make_pred([-2 * fee, -2 * fee, 2 * fee], pad=10.0)
    target_model = _make_target([1.0, 1.0, 1.0])  # model-space: positive at every step
    m = statistical_metrics(pred, target_model)
    assert m["directional_accuracy"] == pytest.approx(1.0)
