"""Predictor training loss (predictor-training.md §"Loss", predictor-contract.md
§"Training loss"; DECISIONS `loss`):

    L = pinball(q10, q50, q90 vs log-return target) + lambda * direction_penalty

- Pinball loss is evaluated independently per (step, dim, quantile).
- The direction penalty fires on the **close** dimension only (the trader's PnL
  convention) and subtracts ``FEE_THRESHOLD`` from the predicted directional PnL
  *inside* the loss, so the model learns net-of-fee moves. On a flat market the
  directional PnL is 0 and the penalty floors at ``FEE_THRESHOLD`` — the calibrated
  baseline the trend-loss regression test asserts against.

All numeric parameters come from `constants.py`; this module hardcodes none.
Tensor shapes follow `predictor-contract.md`:
    pred   : (batch, HORIZON, NUM_OUTPUT_DIMS, NUM_QUANTILES)  quantile index 0=q10,1=q50,2=q90
    target : (batch, HORIZON, NUM_OUTPUT_DIMS)                 log-returns, dim order O,H,L,C,V
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
    """Mean pinball (quantile) loss over (batch, step, dim, quantile)."""
    taus = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device)
    error = target.unsqueeze(-1) - pred  # (B, H, D, Q)
    return torch.maximum(taus * error, (taus - 1.0) * error).mean()


def direction_penalty(
    pred: torch.Tensor,
    target: torch.Tensor,
    fee_threshold: float = EXECUTION.FEE_THRESHOLD,
) -> torch.Tensor:
    """Net-of-fee directional penalty on the close dimension.

    ``relu(FEE_THRESHOLD - sign(target_close) * q50_close)``, averaged over
    (batch, step). Penalises q50 predictions that disagree with the realised move
    or fail to clear round-trip fees; floors at ``fee_threshold`` on a flat market.
    """
    q50_close = pred[:, :, _CLOSE_DIM, _Q50]  # (B, H)
    target_close = target[:, :, _CLOSE_DIM]  # (B, H)
    directional_pnl = torch.sign(target_close) * q50_close
    return torch.relu(fee_threshold - directional_pnl).mean()


def predictor_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    lambda_: float = PREDICTOR.DIRECTION_PENALTY_LAMBDA,
    fee_threshold: float = EXECUTION.FEE_THRESHOLD,
    quantiles: tuple[float, ...] = PREDICTOR.QUANTILES,
) -> LossComponents:
    """Composite predictor loss; returns components for separate logging."""
    pinball = pinball_loss(pred, target, quantiles)
    direction = direction_penalty(pred, target, fee_threshold)
    total = pinball + lambda_ * direction
    return LossComponents(pinball=pinball, direction=direction, total=total)
