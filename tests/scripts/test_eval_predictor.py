"""Tests for scripts/eval_predictor.py — the predictor training bake-off harness.

After the benchmark app landed, this script keeps only the ``train-eval`` role, so
only its seed-aggregation logic (``aggregate_seeds``) is unit-tested here. The reviewed
metric layer (``target_to_model_space`` / ``statistical_metrics`` / ``excursion_metrics``)
moved to ``src/benchmark/metrics.py`` and is tested in ``tests/benchmark/test_metrics.py``;
the ``format_comparison`` table went with the retired ``compare`` subcommand.

Torch orchestration (training against a real fold) is exercised end-to-end on GPU by the
operator; ``aggregate_seeds`` is pinned here with hand-built records so a regression in the
NaN-tolerant aggregation is caught without a training run.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_PATH = REPO_ROOT / "scripts" / "eval_predictor.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("eval_predictor_under_test", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def ep() -> Any:
    return _load_module()


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


def test_aggregate_seeds_includes_timing_group(ep: Any) -> None:
    # A `timing` group (train speed) is aggregated like any other group when present, and
    # its absence must not break aggregation.
    seed_results = [
        {"statistical": {"q90_coverage": 0.90}, "economic": {"mean_captured_fraction": 0.4},
         "timing": {"wall_seconds": 10.0, "seconds_per_epoch": 1.0, "epochs": 10.0}},
        {"statistical": {"q90_coverage": 0.90}, "economic": {"mean_captured_fraction": 0.6},
         "timing": {"wall_seconds": 12.0, "seconds_per_epoch": 1.2, "epochs": 10.0}},
    ]
    agg = ep.aggregate_seeds("new", "cumulative_logret", seed_results)
    assert agg["timing"]["wall_seconds"] == pytest.approx(11.0)
    assert agg["timing"]["seconds_per_epoch"] == pytest.approx(1.1)
    assert agg["timing_std"]["wall_seconds"] == pytest.approx(1.0, abs=1e-9)


def test_aggregate_seeds_single_seed_zero_std(ep: Any) -> None:
    seed_results = [
        {"statistical": {"q90_coverage": 0.9}, "economic": {"mean_captured_fraction": 0.5}},
    ]
    agg = ep.aggregate_seeds("fable-5", "cumulative_logret", seed_results)
    assert agg["seeds"] == 1
    assert agg["statistical_std"]["q90_coverage"] == pytest.approx(0.0)


def test_aggregate_seeds_includes_trading_group(ep: Any) -> None:
    # The fixed-instrument simulated-PnL group aggregates like any other group so the
    # bake-off decider can read seed-mean net_return with seed-noise std.
    seed_results = [
        {"statistical": {"q90_coverage": 0.90},
         "trading": {"net_return": 0.10, "trade_count": 40.0}},
        {"statistical": {"q90_coverage": 0.90},
         "trading": {"net_return": 0.30, "trade_count": 44.0}},
    ]
    agg = ep.aggregate_seeds("h240", "cumulative_logret", seed_results)
    assert agg["trading"]["net_return"] == pytest.approx(0.20)
    assert agg["trading_std"]["net_return"] == pytest.approx(0.10, abs=1e-9)
    assert agg["trading"]["trade_count"] == pytest.approx(42.0)


# --- fixed_instrument_summary (simulated net PnL on the fold VAL slice) -----

def test_fixed_instrument_summary_matches_hand_ledger(ep: Any) -> None:
    """The bake-off's trading summary must reproduce the benchmark app's fixed
    instrument exactly: enter when the FINAL-step cumulative close |q50| clears the
    fee AND [q10,q90] doesn't straddle zero; hold `H` origins (non-overlapping); net
    = direction x realised final move − one round-trip fee."""
    torch = pytest.importorskip("torch")
    from constants import DATA, EXECUTION, PREDICTOR

    close = DATA.FEATURE_NAMES.index("close_logret")
    q10 = PREDICTOR.QUANTILES.index(0.10)
    q50 = PREDICTOR.QUANTILES.index(0.50)
    q90 = PREDICTOR.QUANTILES.index(0.90)
    fee = EXECUTION.FEE_THRESHOLD

    n, horizon = 4, 2
    pred = torch.zeros(n, horizon, PREDICTOR.NUM_OUTPUT_DIMS, len(PREDICTOR.QUANTILES))
    # Origin 0: confident long (interval one-sided, |q50| > fee).
    pred[0, -1, close, q10] = fee
    pred[0, -1, close, q50] = 3 * fee
    pred[0, -1, close, q90] = 5 * fee
    # Origin 1: also a confident long — but it falls inside origin 0's hold (skipped).
    pred[1, -1, close, q10] = fee
    pred[1, -1, close, q50] = 3 * fee
    pred[1, -1, close, q90] = 5 * fee
    # Origin 2: confident long again — realised move is negative (wrong side).
    pred[2, -1, close, q10] = fee
    pred[2, -1, close, q50] = 3 * fee
    pred[2, -1, close, q90] = 5 * fee
    # Origin 3: interval straddles zero -> no trade.
    pred[3, -1, close, q10] = -1.0
    pred[3, -1, close, q50] = 3 * fee
    pred[3, -1, close, q90] = 1.0

    # Raw per-step targets; realised final move = cumsum over the horizon dim.
    target_raw = torch.zeros(n, horizon, PREDICTOR.NUM_OUTPUT_DIMS)
    target_raw[0, :, close] = torch.tensor([0.03, 0.02])  # cum +0.05 (right side)
    target_raw[2, :, close] = torch.tensor([-0.01, -0.03])  # cum -0.04 (wrong side)

    summary = ep.fixed_instrument_summary(pred, target_raw, "cumulative_logret")

    assert summary["trade_count"] == 2  # origin 0 and origin 2; origin 1 inside the hold
    assert summary["net_return"] == pytest.approx(0.05 - fee + (-0.04 - fee), abs=1e-6)
    assert summary["directional_hit_rate"] == pytest.approx(0.5)
    assert summary["hit_rate"] == pytest.approx(0.5)  # +0.05 clears the fee, -0.04 doesn't
    # Null baseline: add-one p-value is finite and in (0, 1] whenever trades exist.
    assert 0.0 < summary["p_value"] <= 1.0
    for key in ("sharpe", "max_drawdown", "null_mean", "null_std"):
        assert key in summary


def test_fixed_instrument_summary_no_trades_is_nan_pvalue(ep: Any) -> None:
    # All-zero predictions -> no signal clears the gate -> zero trades, NaN p-value
    # (nothing to test), zero net return. Must not raise.
    torch = pytest.importorskip("torch")
    from constants import PREDICTOR

    pred = torch.zeros(3, 2, PREDICTOR.NUM_OUTPUT_DIMS, len(PREDICTOR.QUANTILES))
    target_raw = torch.zeros(3, 2, PREDICTOR.NUM_OUTPUT_DIMS)
    summary = ep.fixed_instrument_summary(pred, target_raw, "cumulative_logret")
    assert summary["trade_count"] == 0
    assert summary["net_return"] == 0.0
    assert summary["p_value"] != summary["p_value"]  # NaN


def test_fixed_instrument_summary_cumsum_for_per_step_semantics(ep: Any) -> None:
    # On a per-step branch the q50 path must be cumsummed before the final-step gate;
    # a per-step forecast of +3*fee each step over H=2 is a +6*fee cumulative move.
    torch = pytest.importorskip("torch")
    from constants import DATA, EXECUTION, PREDICTOR

    close = DATA.FEATURE_NAMES.index("close_logret")
    fee = EXECUTION.FEE_THRESHOLD
    pred = torch.zeros(1, 2, PREDICTOR.NUM_OUTPUT_DIMS, len(PREDICTOR.QUANTILES))
    for qi in range(len(PREDICTOR.QUANTILES)):  # one-sided interval at every quantile
        pred[0, :, close, qi] = 3 * fee
    target_raw = torch.zeros(1, 2, PREDICTOR.NUM_OUTPUT_DIMS)
    target_raw[0, :, close] = 0.01

    summary = ep.fixed_instrument_summary(pred, target_raw, "per_step_logret")
    assert summary["trade_count"] == 1
    assert summary["net_return"] == pytest.approx(0.02 - fee, abs=1e-6)
