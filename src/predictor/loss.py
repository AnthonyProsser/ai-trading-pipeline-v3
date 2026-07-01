"""Predictor training loss (DECISIONS `loss`, `target`):

    L = pinball(q10, q50, q90 vs CUMULATIVE log-return path) + lambda * direction_penalty

- Targets are converted to the cumulative path (``target.cumsum`` over the horizon)
  inside ``predictor_loss``: the model predicts quantiles of the total (h+1)-step move
  because quantiles are not additive — per-step quantiles cannot yield a calibrated
  interval for the horizon move the trader actually trades.
- Pinball loss is evaluated independently per (step, dim, quantile).
- The direction penalty fires on the **close** dimension at the **final horizon step
  only**. ``FEE_THRESHOLD`` is a per-trade round-trip cost, so it is compared against
  the whole-horizon cumulative move, never a single 1-minute step: the previous
  per-step form demanded |q50| >= 0.62% per minute (~12x the true per-minute median
  scale), which turned every q50 into a fee-scaled sign flag and destroyed median
  calibration. Confining the penalty to the one (final step, q50, close) coordinate
  keeps the tradeable-direction pressure while leaving the rest of the quantile
  surface honest. On a flat market the directional PnL is 0 and the penalty floors at
  ``FEE_THRESHOLD`` — the calibrated baseline the trend-loss regression test asserts.

All numeric parameters come from `constants.py`; this module hardcodes none.
Tensor shapes:
    pred   : (batch, HORIZON, NUM_OUTPUT_DIMS, NUM_QUANTILES)  cumulative quantiles,
             quantile index 0=q10, 1=q50, 2=q90
    target : (batch, HORIZON, NUM_OUTPUT_DIMS)  per-step log-returns (raw, dim order
             O,H,L,C,V); predictor_loss cumsums them. pinball_loss/direction_penalty
             take an ALREADY-cumulative target.
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from constants import DATA, EXECUTION, PREDICTOR

_CLOSE_DIM = DATA.FEATURE_NAMES.index("close_logret")  # OHLCV close index (== 3)
_Q50 = PREDICTOR.QUANTILES.index(0.50)  # median quantile index (== 1)


class LossComponents(NamedTuple):
    """Loss broken out for separate W&B logging (predictor-training.md smoke run)."""

    pinball: torch.Tensor
    direction: torch.Tensor
    total: torch.Tensor


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: tuple[float, ...] = PREDICTOR.QUANTILES,
) -> torch.Tensor:
    """Mean pinball (quantile) loss over (batch, step, dim, quantile).

    Semantics-agnostic comparison — pred and target must be in the same space
    (cumulative log-returns on the training path).
    """
    taus = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device)
    error = target.unsqueeze(-1) - pred  # (B, H, D, Q)
    return torch.maximum(taus * error, (taus - 1.0) * error).mean()


def direction_penalty(
    pred: torch.Tensor,
    target: torch.Tensor,
    fee_threshold: float = EXECUTION.FEE_THRESHOLD,
) -> torch.Tensor:
    """Net-of-fee directional penalty on the final-horizon cumulative close move.

    ``relu(FEE_THRESHOLD - sign(target_close_cum) * q50_close_cum)`` at the last
    horizon step, averaged over the batch. ``target`` must be cumulative (predictor_loss
    passes the cumsummed path). Penalises a final-horizon median that disagrees with
    the realised move or fails to clear round-trip fees; floors at ``fee_threshold`` on
    a flat market.
    """
    q50_final = pred[:, -1, _CLOSE_DIM, _Q50]  # (B,)
    target_final = target[:, -1, _CLOSE_DIM]  # (B,)
    directional_pnl = torch.sign(target_final) * q50_final
    return torch.relu(fee_threshold - directional_pnl).mean()


def predictor_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    lambda_: float = PREDICTOR.DIRECTION_PENALTY_LAMBDA,
    fee_threshold: float = EXECUTION.FEE_THRESHOLD,
    quantiles: tuple[float, ...] = PREDICTOR.QUANTILES,
) -> LossComponents:
    """Composite predictor loss; returns components for separate logging.

    ``target`` is the raw per-step log-return window from the loaders; it is converted
    to the cumulative path here (single conversion boundary for the training path).
    """
    target_cum = torch.cumsum(target, dim=1)
    pinball = pinball_loss(pred, target_cum, quantiles)
    direction = direction_penalty(pred, target_cum, fee_threshold)
    total = pinball + lambda_ * direction
    return LossComponents(pinball=pinball, direction=direction, total=total)
