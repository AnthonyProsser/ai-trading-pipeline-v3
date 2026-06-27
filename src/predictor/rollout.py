"""Geometry enforcement for the predictor rollout sampler (predictor-contract.md
§"Geometry enforcement").

Independent quantile heads can emit physically impossible candles -- a predicted
high below the open/close, or a low above them. Before any candle reaches the
trader (production inference AND any Trader-side mock harness) it must satisfy,
for every emitted step and every quantile q:

    high_q >= max(open_q, close_q)
    low_q  <= min(open_q, close_q)

``enforce_geometry`` clamps a quantile tensor to that invariant deterministically.
``resample_until_valid`` wraps a *stochastic* sampler: it redraws a violating
candle up to ``PredictorConfig.GEOMETRY_RESAMPLE_CAP`` times, then falls back to
the deterministic clamp so the returned candle is always valid.

Tensor layout follows predictor-contract.md:
    pred : (batch, HORIZON, NUM_OUTPUT_DIMS, NUM_QUANTILES)
           dim order O,H,L,C,V (OhlcvCol); quantile order q10, q50, q90.

The clamp generalises the contract's named cases (q90 high, q10 low) to every
quantile so each emitted candle scenario is self-consistent. No magic numbers
live here; the resample cap is sourced from constants.py.
"""
from __future__ import annotations

from collections.abc import Callable

import torch

from constants import PREDICTOR
from src.data.validator import OhlcvCol


def is_geometry_valid(pred: torch.Tensor) -> bool:
    """True iff every (step, quantile) candle satisfies the geometry inequalities."""
    o = pred[..., OhlcvCol.OPEN, :]
    h = pred[..., OhlcvCol.HIGH, :]
    low = pred[..., OhlcvCol.LOW, :]
    c = pred[..., OhlcvCol.CLOSE, :]
    high_ok = bool(torch.all(h >= torch.maximum(o, c)).item())
    low_ok = bool(torch.all(low <= torch.minimum(o, c)).item())
    return high_ok and low_ok


def enforce_geometry(pred: torch.Tensor) -> torch.Tensor:
    """Return a copy of ``pred`` with high/low clamped per (step, quantile) candle.

    ``high := max(high, open, close)`` and ``low := min(low, open, close)`` for every
    quantile, so each emitted candle scenario is self-consistent. Open, close, and
    volume are left untouched; the input tensor is not mutated.

    Raises ``ValueError`` on non-finite (NaN/inf) input: ``torch.maximum(NaN, x)`` is
    ``NaN``, which would pass the clamp silently and feed a physically impossible candle
    to the trader. A non-finite prediction is an upstream training/inference bug and must
    fail loudly here, not be absorbed.
    """
    if not bool(torch.isfinite(pred).all().item()):
        raise ValueError("enforce_geometry received a non-finite (NaN/inf) prediction tensor")
    out = pred.clone()
    body_high = torch.maximum(out[..., OhlcvCol.OPEN, :], out[..., OhlcvCol.CLOSE, :])
    body_low = torch.minimum(out[..., OhlcvCol.OPEN, :], out[..., OhlcvCol.CLOSE, :])
    out[..., OhlcvCol.HIGH, :] = torch.maximum(out[..., OhlcvCol.HIGH, :], body_high)
    out[..., OhlcvCol.LOW, :] = torch.minimum(out[..., OhlcvCol.LOW, :], body_low)
    return out


def resample_until_valid(
    sampler: Callable[[], torch.Tensor],
    *,
    cap: int = PREDICTOR.GEOMETRY_RESAMPLE_CAP,
) -> torch.Tensor:
    """Draw candles from ``sampler`` until one satisfies geometry, then return it.

    Redraws up to ``cap`` times. If every draw violates geometry, the final draw is
    clamped via ``enforce_geometry`` so the result is geometry-valid (or raises if that
    final draw is non-finite). Used by stochastic samplers (e.g. a Trader-side mock
    harness); the deterministic predictor inference path calls ``enforce_geometry``
    directly.
    """
    candle = sampler()
    attempts = 1
    while not is_geometry_valid(candle) and attempts < cap:
        candle = sampler()
        attempts += 1
    if not is_geometry_valid(candle):
        candle = enforce_geometry(candle)
    return candle
