"""Tests for src/benchmark/trading_sim.py — the fixed-instrument trading simulation.

The trading rule is the benchmark's measuring stick, applied identically to every
model (no per-model rule search — garden-of-forking-paths guard): trade only when the
predicted final-horizon cumulative close move clears the round-trip fee AND the
[q10, q90] interval does not straddle zero; hold exactly HORIZON steps; pay
EXECUTION.FEE_THRESHOLD per round trip. Null baselines (buy-and-hold, random-entry at
matched trade frequency) live here too, so every model's PnL is read against noise.

All numbers hand-computed on tiny synthetic arrays. Committed before
src/benchmark/trading_sim.py, per the test-first discipline.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from constants import DATA, EXECUTION, PREDICTOR
from src.benchmark.trading_sim import (
    buy_and_hold,
    extract_signals,
    ledger_stats,
    random_entry_null,
    simulate_trades,
)
from src.data.walk_forward import Fold

_CLOSE = DATA.FEATURE_NAMES.index("close_logret")
_Q10 = PREDICTOR.QUANTILES.index(0.10)
_Q50 = PREDICTOR.QUANTILES.index(0.50)
_Q90 = PREDICTOR.QUANTILES.index(0.90)
_NDIM = PREDICTOR.NUM_OUTPUT_DIMS
_NQ = len(PREDICTOR.QUANTILES)
_FEE = EXECUTION.FEE_THRESHOLD


def _pred_with_final_close(rows: list[tuple[float, float, float]]) -> torch.Tensor:
    """(N, H=2, DIM, Q) cumulative-space predictions; each row sets the FINAL-step
    close (q10, q50, q90). Earlier steps stay zero — the rule only reads the final step."""
    pred = torch.zeros(len(rows), 2, _NDIM, _NQ)
    for i, (q10, q50, q90) in enumerate(rows):
        pred[i, -1, _CLOSE, _Q10] = q10
        pred[i, -1, _CLOSE, _Q50] = q50
        pred[i, -1, _CLOSE, _Q90] = q90
    return pred


# --- extract_signals (the confidence gate) ----------------------------------

def test_extract_signals_gate_logic() -> None:
    big = _FEE * 3
    pred = _pred_with_final_close([
        (big / 2, big, big * 2),          # long: clears fee, interval positive
        (-0.01, big, big * 2),            # straddles zero -> no trade despite big q50
        (0.001, _FEE / 2, 0.006),         # sub-fee predicted move -> no trade
        (-big * 2, -big, -big / 2),       # short: clears fee, interval negative
    ])
    signals = extract_signals(pred, fee_threshold=_FEE)
    assert signals.dtype == np.int8
    assert signals.tolist() == [1, 0, 0, -1]


def test_extract_signals_zero_boundary_is_not_a_trade() -> None:
    # An interval touching zero exactly (q10 == 0) still counts as straddling: the
    # model is not confidently one-sided, so the gate must stay closed.
    big = _FEE * 3
    pred = _pred_with_final_close([(0.0, big, big * 2)])
    assert extract_signals(pred, fee_threshold=_FEE).tolist() == [0]


# --- simulate_trades (non-overlapping, hold-to-horizon) ----------------------

def test_simulate_trades_non_overlap_and_fees() -> None:
    directions = np.array([1, 1, 0, 1], dtype=np.int8)
    realized_final = np.array([0.05, 0.99, 0.0, 0.03])  # index 1 must be skipped
    ledger = simulate_trades(directions, realized_final, horizon=2, fee=0.01)
    assert ledger.entries.tolist() == [0, 3]  # in-position at 1; no signal at 2
    assert ledger.net_returns.tolist() == pytest.approx([0.04, 0.02])


def test_simulate_trades_short_direction() -> None:
    directions = np.array([-1], dtype=np.int8)
    realized_final = np.array([-0.05])  # price fell; short gains +0.05 gross
    ledger = simulate_trades(directions, realized_final, horizon=2, fee=0.01)
    assert ledger.net_returns.tolist() == pytest.approx([0.04])


def test_simulate_trades_no_signals_yields_empty_ledger() -> None:
    ledger = simulate_trades(
        np.zeros(5, dtype=np.int8), np.zeros(5), horizon=2, fee=0.01
    )
    assert ledger.entries.size == 0
    assert ledger.net_returns.size == 0


# --- ledger_stats -------------------------------------------------------------

def test_ledger_stats_known_values() -> None:
    nets = np.array([0.04, 0.02, -0.01])
    stats = ledger_stats(nets)
    assert stats["net_return"] == pytest.approx(0.05)
    assert stats["trade_count"] == 3
    assert stats["hit_rate"] == pytest.approx(2 / 3)
    # Sample std (ddof=1) of [0.04, 0.02, -0.01] = 0.0251661...; sharpe = mean/std.
    assert stats["sharpe"] == pytest.approx(np.mean(nets) / np.std(nets, ddof=1))
    # Equity path [0.04, 0.06, 0.05]: peak 0.06 -> trough 0.05 -> max drawdown 0.01.
    assert stats["max_drawdown"] == pytest.approx(0.01)


def test_ledger_stats_drawdown_from_zero_start() -> None:
    # First trade loses: equity dips below the starting 0 -> drawdown counts from 0.
    stats = ledger_stats(np.array([-0.03, 0.01]))
    assert stats["max_drawdown"] == pytest.approx(0.03)


def test_ledger_stats_empty_ledger() -> None:
    stats = ledger_stats(np.array([]))
    assert stats["net_return"] == 0.0
    assert stats["trade_count"] == 0
    assert stats["max_drawdown"] == 0.0
    assert math.isnan(stats["sharpe"])
    assert math.isnan(stats["hit_rate"])


def test_ledger_stats_single_trade_has_nan_sharpe() -> None:
    # One trade has no return dispersion; a 0/0 sharpe must be NaN, not raise.
    assert math.isnan(ledger_stats(np.array([0.02]))["sharpe"])


# --- null baselines -----------------------------------------------------------

def test_buy_and_hold_pays_one_round_trip() -> None:
    assert buy_and_hold(0.10, fee=_FEE) == pytest.approx(0.10 - _FEE)


def test_random_entry_null_is_deterministic_and_counts_trades() -> None:
    # Flat market: every trade nets exactly -fee regardless of direction, so each null
    # draw must equal -(trade_count * fee) — this pins both determinism AND that every
    # draw places exactly the matched trade count.
    realized_final = np.zeros(100)
    a = random_entry_null(
        n_origins=100, trade_count=5, realized_final=realized_final,
        horizon=10, fee=0.01, draws=20, seed=0, model_net=-0.02,
    )
    b = random_entry_null(
        n_origins=100, trade_count=5, realized_final=realized_final,
        horizon=10, fee=0.01, draws=20, seed=0, model_net=-0.02,
    )
    assert a == b  # same seed -> identical result
    assert a["null_mean"] == pytest.approx(-0.05)
    assert a["null_std"] == pytest.approx(0.0)
    assert a["null_draws"] == 20


def test_random_entry_null_p_value_bounds() -> None:
    rng = np.random.default_rng(1)
    realized_final = rng.normal(0.0, 0.01, size=500)
    # A model PnL far above any null draw -> smallest attainable p = 1/(draws+1).
    high = random_entry_null(
        n_origins=500, trade_count=10, realized_final=realized_final,
        horizon=15, fee=0.0, draws=50, seed=0, model_net=1e9,
    )
    assert high["p_value"] == pytest.approx(1 / 51)
    # A model PnL below every null draw -> p = 1.0 (all draws beat it).
    low = random_entry_null(
        n_origins=500, trade_count=10, realized_final=realized_final,
        horizon=15, fee=0.0, draws=50, seed=0, model_net=-1e9,
    )
    assert low["p_value"] == pytest.approx(1.0)


def test_random_entry_null_zero_trades_is_nan() -> None:
    out = random_entry_null(
        n_origins=100, trade_count=0, realized_final=np.zeros(100),
        horizon=15, fee=0.01, draws=10, seed=0, model_net=0.0,
    )
    assert math.isnan(out["p_value"])


# --- eval slice bounds (engine helper) -----------------------------------------

def test_eval_slice_bounds_targets_stay_inside_test_slice() -> None:
    from src.benchmark.engine import eval_slice_bounds

    fold = Fold(0, 0, 150_000, 150_000, 200_000, 200_000, 210_000)
    start, end = eval_slice_bounds(fold, lookback=1_440)
    assert (start, end) == (200_000 - 1_440, 210_000)
    # Window origins i in [0, n_windows) within the slice give targets
    # [i+lookback, i+lookback+horizon) — the first target row is exactly test_start.
    assert start + 1_440 == fold.test_start


def test_eval_slice_bounds_rejects_fold_with_insufficient_history() -> None:
    from src.benchmark.engine import eval_slice_bounds

    fold = Fold(0, 0, 10, 10, 20, 20, 30)
    with pytest.raises(ValueError):
        eval_slice_bounds(fold, lookback=1_440)
