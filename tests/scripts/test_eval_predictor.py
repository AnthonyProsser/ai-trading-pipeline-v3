"""Tests for scripts/eval_predictor.py — the predictor bake-off harness.

Covers the pure metric layer, which is where the correctness risk lives:
  * target_to_model_space  — per-step vs cumulative reconciliation
  * excursion_metrics       — the Krafer-style "fraction of predicted move captured"
                              metric, paired with adverse excursion
  * statistical_metrics     — coverage/calibration/sharpness/pinball wrappers
  * format_comparison       — side-by-side table

Torch orchestration (evaluate against a real fold) is exercised end-to-end on GPU
by the operator; the pure functions are pinned here with hand-built tensors so a
regression in the metric math is caught without a training run.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from constants import DATA, EXECUTION, PREDICTOR  # noqa: E402

SCRIPT_PATH = REPO_ROOT / "scripts" / "eval_predictor.py"
_CLOSE = DATA.FEATURE_NAMES.index("close_logret")
_Q10 = PREDICTOR.QUANTILES.index(0.10)
_Q50 = PREDICTOR.QUANTILES.index(0.50)
_Q90 = PREDICTOR.QUANTILES.index(0.90)
_NDIM = PREDICTOR.NUM_OUTPUT_DIMS
_NQ = len(PREDICTOR.QUANTILES)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("eval_predictor_under_test", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def ep() -> Any:
    return _load_module()


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

def test_target_to_model_space_per_step_is_identity(ep: Any) -> None:
    target = _make_target([0.1, -0.2, 0.3])
    out = ep.target_to_model_space(target, "per_step_logret")
    assert torch.equal(out, target)


def test_target_to_model_space_cumulative_is_cumsum(ep: Any) -> None:
    target = _make_target([0.1, -0.2, 0.3])
    out = ep.target_to_model_space(target, "cumulative_logret")
    expected = torch.cumsum(target, dim=1)
    assert torch.allclose(out, expected)


# --- excursion_metrics (the Krafer-style metric) ---------------------------

def test_excursion_captured_fraction_per_step(ep: Any) -> None:
    # Predicted total move = -0.30 (down). Realized cumulative path = [-.10, -.15, +.05]:
    # best favorable (downward) excursion = 0.15 -> 0.15 / 0.30 = 0.50 of the move played out.
    pred = _make_pred([-0.30, 0.0, 0.0])  # per-step q50 cumsum -> final -0.30
    target = _make_target([-0.10, -0.05, 0.20])  # cumsum -> [-.10, -.15, +.05]
    m = ep.excursion_metrics(pred, target, "per_step_logret", fee_threshold=EXECUTION.FEE_THRESHOLD)
    assert m["n_used"] == 1
    assert m["mean_captured_fraction"] == pytest.approx(0.5, abs=1e-6)
    # Adverse excursion: price went +0.05 against the short -> 0.05 / 0.30.
    assert m["mean_adverse_ratio"] == pytest.approx(0.05 / 0.30, abs=1e-6)


def test_excursion_captured_fraction_cumulative(ep: Any) -> None:
    # Same realized path, but the model emits the cumulative path directly.
    pred = _make_pred([-0.10, -0.20, -0.30])  # cumulative q50 path, final -0.30
    target = _make_target([-0.10, -0.05, 0.20])
    m = ep.excursion_metrics(
        pred, target, "cumulative_logret", fee_threshold=EXECUTION.FEE_THRESHOLD
    )
    assert m["mean_captured_fraction"] == pytest.approx(0.5, abs=1e-6)


def test_excursion_excludes_sub_fee_predictions(ep: Any) -> None:
    # Predicted move below the round-trip fee is not tradeable -> excluded from the mean.
    tiny = EXECUTION.FEE_THRESHOLD / 2.0
    pred = _make_pred([tiny, 0.0, 0.0])
    target = _make_target([0.01, 0.0, 0.0])
    m = ep.excursion_metrics(pred, target, "per_step_logret", fee_threshold=EXECUTION.FEE_THRESHOLD)
    assert m["n_used"] == 0
    assert m["mean_captured_fraction"] != m["mean_captured_fraction"]  # NaN when nothing tradeable


# --- statistical_metrics ---------------------------------------------------

def test_statistical_metrics_keys_and_coverage(ep: Any) -> None:
    # q90 sits a full pad above every target -> coverage 1.0; target inside [q10,q90] -> cal 1.0.
    pred = _make_pred([0.0, 0.0])
    target_model = _make_target([0.0, 0.0])  # already model-space for per_step
    m = ep.statistical_metrics(pred, target_model)
    assert set(m) >= {"q90_coverage", "calibration_rate", "directional_accuracy", "sharpness_close"}
    assert m["q90_coverage"] == pytest.approx(1.0)
    # Sharpness = mean (q90 - q10) on close = 2 * pad = 2.0.
    assert m["sharpness_close"] == pytest.approx(2.0, abs=1e-6)
    # Pinball over (B=1,H=2,D=5,Q=3)=30 elements: only the close dim is non-zero. Per step
    # the close pins [q10,q50,q90]=[-1,0,1] vs target 0 -> errors give 0.1+0+0.1=0.2; two
    # steps -> 0.4 summed, meaned over all 30 elements = 0.4/30.
    assert m["pinball"] == pytest.approx(0.4 / 30, abs=1e-6)


# --- aggregate_seeds -------------------------------------------------------

def test_aggregate_seeds_mean_and_std(ep: Any) -> None:
    seed_results = [
        {"statistical": {"q90_coverage": 0.90, "pinball": 0.02},
         "economic": {"mean_captured_fraction": 0.4, "n_used": 100}},
        {"statistical": {"q90_coverage": 0.92, "pinball": 0.04},
         "economic": {"mean_captured_fraction": 0.6, "n_used": 100}},
    ]
    agg = ep.aggregate_seeds("main", "per_step_logret", seed_results)
    assert agg["label"] == "main"
    assert agg["semantics"] == "per_step_logret"
    assert agg["seeds"] == 2
    # Means over the two seeds.
    assert agg["statistical"]["q90_coverage"] == pytest.approx(0.91)
    assert agg["economic"]["mean_captured_fraction"] == pytest.approx(0.5)
    # Population std (ddof=0): std([0.90, 0.92]) = 0.01.
    assert agg["statistical_std"]["q90_coverage"] == pytest.approx(0.01, abs=1e-9)
    assert agg["economic_std"]["mean_captured_fraction"] == pytest.approx(0.1, abs=1e-9)


def test_aggregate_seeds_skips_nan_seed(ep: Any) -> None:
    # A seed with zero tradeable windows yields NaN economic metrics; it must be skipped,
    # not allowed to poison the mean across the other seeds.
    seed_results = [
        {"statistical": {"q90_coverage": 0.90},
         "economic": {"mean_captured_fraction": float("nan"), "n_used": 0}},
        {"statistical": {"q90_coverage": 0.90},
         "economic": {"mean_captured_fraction": 0.5, "n_used": 100}},
    ]
    agg = ep.aggregate_seeds("main", "per_step_logret", seed_results)
    assert agg["economic"]["mean_captured_fraction"] == pytest.approx(0.5)  # NaN seed skipped
    # n_used is a count (fold-fixed), reported once — never averaged or std'd.
    assert agg["economic"]["n_used"] == 100
    assert "n_used" not in agg["economic_std"]


def test_aggregate_seeds_single_seed_zero_std(ep: Any) -> None:
    seed_results = [
        {"statistical": {"q90_coverage": 0.9}, "economic": {"mean_captured_fraction": 0.5}},
    ]
    agg = ep.aggregate_seeds("fable-5", "cumulative_logret", seed_results)
    assert agg["seeds"] == 1
    assert agg["statistical_std"]["q90_coverage"] == pytest.approx(0.0)


# --- format_comparison -----------------------------------------------------

def test_format_comparison_includes_labels_and_metrics(ep: Any) -> None:
    results = [
        {"label": "main", "semantics": "per_step_logret",
         "statistical": {"q90_coverage": 0.988, "sharpness_close": 0.01},
         "economic": {"mean_captured_fraction": 0.4, "mean_adverse_ratio": 0.3, "n_used": 100}},
        {"label": "fable-5", "semantics": "cumulative_logret",
         "statistical": {"q90_coverage": 0.905, "sharpness_close": 0.02},
         "economic": {"mean_captured_fraction": 0.6, "mean_adverse_ratio": 0.2, "n_used": 100}},
    ]
    table = ep.format_comparison(results)
    assert "main" in table and "fable-5" in table
    assert "q90_coverage" in table
    assert "captured" in table.lower()
