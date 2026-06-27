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


def test_variance_floor_loss_strictly_positive() -> None:
    # Bug #1: output collapsed to a constant drove NLL negative. The pinball +
    # direction loss is strictly > 0 across the first N steps of any real run.
    torch = pytest.importorskip("torch")
    from src.predictor.loss import predictor_loss

    gen = torch.Generator().manual_seed(0)
    pred_shape = (8, PREDICTOR.HORIZON, PREDICTOR.NUM_OUTPUT_DIMS, len(PREDICTOR.QUANTILES))
    tgt_shape = (8, PREDICTOR.HORIZON, PREDICTOR.NUM_OUTPUT_DIMS)
    for _ in range(PREDICTOR.VARIANCE_FLOOR_FIRST_N_STEPS):
        pred = torch.randn(pred_shape, generator=gen)
        target = torch.randn(tgt_shape, generator=gen) * 0.01
        loss = predictor_loss(pred, target)
        assert loss.total.item() > 0.0
        assert loss.pinball.item() >= 0.0
        assert loss.direction.item() >= 0.0


def test_trend_loss_constant_candles_known_baseline() -> None:
    # Bug #2: flat input collapsed the trend loss to zero (no gradient). Constant
    # candles → zero log-return target → directional PnL is 0, so the penalty floors
    # at FEE_THRESHOLD: a flat market cannot beat round-trip fees. Known, non-zero.
    torch = pytest.importorskip("torch")
    from src.predictor.loss import direction_penalty

    pred_shape = (8, PREDICTOR.HORIZON, PREDICTOR.NUM_OUTPUT_DIMS, len(PREDICTOR.QUANTILES))
    pred = torch.randn(pred_shape)  # any prediction
    target = torch.zeros((8, PREDICTOR.HORIZON, PREDICTOR.NUM_OUTPUT_DIMS))  # flat candles

    penalty = direction_penalty(pred, target)
    assert penalty.item() == pytest.approx(EXECUTION.FEE_THRESHOLD)
    assert penalty.item() > 0.0
