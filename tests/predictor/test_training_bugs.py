"""Predictor training-loop regression tests (predictor-training.md §"Three
regression tests"). Each one encodes a v2 bug that lived in this exact loop:

1. variance-floor  — loss collapsed to a constant → negative NLL (assert loss > 0)
2. trend-loss      — constant candles must yield a known non-zero direction penalty
3. patience        — EARLY_STOPPING_PATIENCE exposed in constants.py and respected

Committed before the loss / training-loop implementation, per test_discipline.
The two loss tests skip until torch is added as a dependency; the patience test
is torch-free and runs now.
"""
from __future__ import annotations

import pytest

from constants import EXECUTION, PREDICTOR
from src.predictor.early_stopping import EarlyStopper


def test_early_stopping_patience_exposed_and_respected() -> None:
    # Bug #3: premature early stopping. Patience must be a constants.py value > 1,
    # and the stopper must not fire until `patience` consecutive non-improving steps.
    patience = PREDICTOR.EARLY_STOPPING_PATIENCE
    assert patience > 1

    stopper = EarlyStopper(patience)
    assert stopper.step(1.0) is False  # first observation: improvement from +inf

    plateau = [stopper.step(1.0) for _ in range(patience)]
    assert plateau[: patience - 1] == [False] * (patience - 1)  # no premature stop
    assert plateau[patience - 1] is True  # fires exactly at patience


def test_early_stopping_resets_after_improvement() -> None:
    # A plateau that is broken by a real improvement restarts the full patience window.
    stopper = EarlyStopper(patience=2)
    assert stopper.step(1.0) is False  # improvement from +inf
    assert stopper.step(1.0) is False  # bad 1
    assert stopper.step(0.5) is False  # improvement resets the counter
    assert stopper.step(0.5) is False  # bad 1 (post-reset)
    assert stopper.step(0.5) is True   # bad 2 -> fires


def test_early_stopping_min_delta_requires_real_improvement() -> None:
    # A decrease smaller than min_delta does not count as improvement.
    stopper = EarlyStopper(patience=1, min_delta=0.1)
    assert stopper.step(1.0) is False
    assert stopper.step(0.95) is True  # 0.95 < 1.0 but not by 0.1 -> non-improving -> fires


def test_early_stopping_num_bad_property_tracks_bad_streak() -> None:
    # training-dashboard.md §"H — Patience bar" reads the live bad-streak count to
    # render the patience gauge; it must be a public, readable counter.
    stopper = EarlyStopper(patience=3)
    assert stopper.num_bad == 0
    stopper.step(1.0)  # improvement from +inf
    assert stopper.num_bad == 0
    stopper.step(1.0)  # no improvement
    assert stopper.num_bad == 1
    stopper.step(1.0)
    assert stopper.num_bad == 2
    stopper.step(0.5)  # improves -> resets
    assert stopper.num_bad == 0


def test_variance_floor_loss_strictly_positive() -> None:
    # Bug #1: output collapsed to a CONSTANT drove NLL negative. The failure mode is a
    # constant prediction, so the test pins that case (a random prediction could be > 0
    # even in a broken loss). The pinball + net-of-fee direction loss stays strictly > 0.
    torch = pytest.importorskip("torch")
    from src.predictor.loss import predictor_loss

    gen = torch.Generator().manual_seed(0)
    pred_shape = (8, PREDICTOR.HORIZON, PREDICTOR.NUM_OUTPUT_DIMS, len(PREDICTOR.QUANTILES))
    tgt_shape = (8, PREDICTOR.HORIZON, PREDICTOR.NUM_OUTPUT_DIMS)
    const_pred = torch.zeros(pred_shape)  # collapsed / constant output
    for _ in range(PREDICTOR.VARIANCE_FLOOR_FIRST_N_STEPS):
        target = torch.randn(tgt_shape, generator=gen) * 0.01
        loss = predictor_loss(const_pred, target)
        assert loss.total.item() > 0.0
        assert loss.pinball.item() >= 0.0
        assert loss.direction.item() >= 0.0


def test_trend_loss_constant_candles_known_baseline() -> None:
    # Bug #2 (historical, pre-vol-pivot): flat input collapsed the trend loss to zero
    # (no gradient). Constant candles -> zero log-return target -> directional PnL is
    # 0, so the penalty floors at FEE_THRESHOLD: a flat market cannot beat round-trip
    # fees. `direction_penalty` itself is unchanged (kept for its own tests below) --
    # this still pins its known, non-zero baseline in isolation.
    torch = pytest.importorskip("torch")
    from src.predictor.loss import direction_penalty

    pred_shape = (8, PREDICTOR.HORIZON, PREDICTOR.NUM_OUTPUT_DIMS, len(PREDICTOR.QUANTILES))
    pred = torch.randn(pred_shape)  # any prediction
    target = torch.zeros((8, PREDICTOR.HORIZON, PREDICTOR.NUM_OUTPUT_DIMS))  # flat candles

    penalty = direction_penalty(pred, target)
    assert penalty.item() == pytest.approx(EXECUTION.FEE_THRESHOLD)
    assert penalty.item() > 0.0


def test_trend_loss_retired_in_predictor_loss_under_vol_pivot() -> None:
    # REPURPOSED (2026-07-09, vol pivot, branch pivot-vol-01): direction is falsified
    # and the term is meaningless for a variance target, so DIRECTION_PENALTY_LAMBDA is
    # now pinned to 0.0 in constants.py. The composite predictor_loss must therefore
    # contribute EXACTLY zero direction loss for the very scenario (flat candles) that
    # used to floor non-zero at FEE_THRESHOLD directly -- proving the term is inert in
    # the actual training loss, not merely re-weighted small.
    torch = pytest.importorskip("torch")
    from src.predictor.loss import predictor_loss

    pred_shape = (8, PREDICTOR.HORIZON, PREDICTOR.NUM_OUTPUT_DIMS, len(PREDICTOR.QUANTILES))
    pred = torch.randn(pred_shape)  # any prediction
    target = torch.zeros((8, PREDICTOR.HORIZON, PREDICTOR.NUM_OUTPUT_DIMS))  # flat candles

    comp = predictor_loss(pred, target)  # default lambda_ = PREDICTOR.DIRECTION_PENALTY_LAMBDA
    assert PREDICTOR.DIRECTION_PENALTY_LAMBDA == 0.0
    assert comp.direction.item() == 0.0


def test_direction_penalty_final_horizon_cumulative() -> None:
    # The penalty reads ONLY the final-horizon cumulative close q50 vs. the cumulative
    # realised close move: fee is a per-trade round-trip cost, comparable to the whole
    # horizon move, not to a single 1-minute step. A correct-sign forecast clearing the
    # fee is penalty-free; the wrong sign pays relu(fee + |q50_cum|).
    torch = pytest.importorskip("torch")
    from constants import DATA
    from src.predictor.loss import direction_penalty

    close = DATA.FEATURE_NAMES.index("close_logret")
    q50 = PREDICTOR.QUANTILES.index(0.50)
    fee = EXECUTION.FEE_THRESHOLD
    pred_shape = (2, PREDICTOR.HORIZON, PREDICTOR.NUM_OUTPUT_DIMS, len(PREDICTOR.QUANTILES))

    pred = torch.zeros(pred_shape)
    pred[:, -1, close, q50] = 2 * fee  # cumulative forecast: up, clears the fee
    target_cum = torch.zeros((2, PREDICTOR.HORIZON, PREDICTOR.NUM_OUTPUT_DIMS))
    target_cum[:, -1, close] = 1.0  # realised horizon move: up
    assert direction_penalty(pred, target_cum).item() == pytest.approx(0.0)

    target_cum[:, -1, close] = -1.0  # realised horizon move: down -> wrong sign
    assert direction_penalty(pred, target_cum).item() == pytest.approx(3 * fee)

    # Intermediate steps are NOT penalised: a large wrong-sign q50 at step 0 with a
    # flat final step still floors at the flat-market fee baseline.
    pred = torch.zeros(pred_shape)
    pred[:, 0, close, q50] = -10.0
    target_cum = torch.zeros((2, PREDICTOR.HORIZON, PREDICTOR.NUM_OUTPUT_DIMS))
    assert direction_penalty(pred, target_cum).item() == pytest.approx(fee)


def test_pinball_is_zero_when_pred_equals_cumulative_target() -> None:
    # predictor_loss trains against the CUMULATIVE REALIZED-VARIANCE path (vol pivot,
    # 2026-07-09): quantiles are not additive, so per-step quantiles cannot give a
    # calibrated interval for the h-step move, and the model now predicts variance
    # (squared log-return) rather than signed log-return. A prediction equal to the
    # squared-then-cumsummed target at every quantile has exactly zero pinball loss.
    torch = pytest.importorskip("torch")
    from src.predictor.loss import predictor_loss

    gen = torch.Generator().manual_seed(1)
    target = torch.randn((4, PREDICTOR.HORIZON, PREDICTOR.NUM_OUTPUT_DIMS), generator=gen) * 0.01
    cum = torch.cumsum(target * target, dim=1)
    pred = cum.unsqueeze(-1).expand(-1, -1, -1, len(PREDICTOR.QUANTILES)).contiguous()

    comp = predictor_loss(pred, target)
    assert comp.pinball.item() == pytest.approx(0.0, abs=1e-9)
