"""Predictor training loss (DECISIONS `loss`, `target`):

    L = pinball(q10, q50, q90 vs CUMULATIVE log-return path) + lambda * direction_penalty
        + COVERAGE_PENALTY_WEIGHT * coverage_penalty

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
- The coverage penalty (amended into DECISIONS `loss` 2026-07-01) is the squared gap
  between smooth empirical batch coverage (sigmoid indicator whose width is
  ``COVERAGE_PENALTY_TEMPERATURE_FRAC`` x the batch's per-step close-target std) and
  the nominal tail levels, close dim only, averaged over horizon steps. The capped
  bake-off measured under-coverage on the cumulative model (q90_coverage 0.866 vs
  0.90, calibration_rate 0.747 vs 0.80): pinball's marginal calibration pressure is
  spread over every (dim, quantile) coordinate and is too weak at capped budgets,
  while this term optimizes the two calibration gate metrics directly. It is
  self-limiting — the gradient is proportional to the coverage gap and vanishes at
  nominal, so it accelerates calibration without fighting pinball's optimum (the true
  quantile minimises both).

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
_Q10 = PREDICTOR.QUANTILES.index(0.10)  # lower-tail quantile index (== 0)
_Q50 = PREDICTOR.QUANTILES.index(0.50)  # median quantile index (== 1)
_Q90 = PREDICTOR.QUANTILES.index(0.90)  # upper-tail quantile index (== 2)


class LossComponents(NamedTuple):
    """Loss broken out for separate W&B logging (predictor-training.md smoke run)."""

    pinball: torch.Tensor
    direction: torch.Tensor
    coverage: torch.Tensor
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


def coverage_penalty(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature_frac: float = PREDICTOR.COVERAGE_PENALTY_TEMPERATURE_FRAC,
    std_floor: float = PREDICTOR.COVERAGE_PENALTY_STD_FLOOR,
) -> torch.Tensor:
    """Squared gap between smooth empirical batch coverage and the nominal tail levels.

    For each tail quantile (the locked gate levels ``_Q10``/``_Q90``), coverage is
    estimated per horizon step as the batch mean of ``sigmoid((q_tau - y) / s)`` on the
    **close** dim (the dim every deploy gate and the trader read, same precedent as the
    direction penalty), where ``s = temperature_frac * std(y)`` per step — relative so
    the indicator stays proportionally sharp as the cumulative path's variance grows
    with the horizon. Penalty = sum over tails of the horizon-mean of
    ``(coverage - tau)^2``; the gradient widens under-covering intervals and vanishes
    at nominal coverage. ``target`` must be ALREADY-cumulative (predictor_loss passes
    the cumsummed path).

    Computed in fp32 even under bf16 autocast: ``(q - y) / s`` divides a cancelled
    small difference by a small scale, and a saturated sigmoid's gradient underflows
    to exactly zero in bf16 precisely where the penalty must push hardest.

    WIDTH-ONLY by construction: the first bake-off runs (weights 1.0 and 0.1, fold 0,
    3 seeds) showed tail gradients flowing through the monotone head's shared median
    anchor drag q50 systematically and collapse directional accuracy to a coin flip.
    Each tail is therefore evaluated as ``q_tail + (q50.detach() - q50)`` — the exact
    same VALUE (adds zero), but the anchor's gradient path cancels, so the penalty can
    only widen or narrow the softplus offsets, never move the median.
    """
    y = target[..., _CLOSE_DIM].float()  # (B, H)
    # correction=0: a batch of one window must not produce a NaN std.
    s = y.std(dim=0, keepdim=True, correction=0) * temperature_frac + std_floor  # (1, H)
    q50 = pred[..., _CLOSE_DIM, _Q50].float()  # (B, H)
    anchor_free = q50.detach() - q50  # exactly zero-valued; cancels the anchor gradient
    penalty = torch.zeros((), dtype=torch.float32, device=pred.device)
    for q_idx in (_Q10, _Q90):
        tau = PREDICTOR.QUANTILES[q_idx]
        q = pred[..., _CLOSE_DIM, q_idx].float() + anchor_free  # (B, H)
        smooth_coverage = torch.sigmoid((q - y) / s).mean(dim=0)  # (H,)
        penalty = penalty + ((smooth_coverage - tau) ** 2).mean()
    return penalty


def predictor_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    lambda_: float = PREDICTOR.DIRECTION_PENALTY_LAMBDA,
    fee_threshold: float = EXECUTION.FEE_THRESHOLD,
    quantiles: tuple[float, ...] = PREDICTOR.QUANTILES,
    coverage_weight: float = PREDICTOR.COVERAGE_PENALTY_WEIGHT,
) -> LossComponents:
    """Composite predictor loss; returns components for separate logging.

    ``target`` is the raw per-step log-return window from the loaders; it is converted
    to the cumulative path here (single conversion boundary for the training path).
    """
    target_cum = torch.cumsum(target, dim=1)
    pinball = pinball_loss(pred, target_cum, quantiles)
    direction = direction_penalty(pred, target_cum, fee_threshold)
    coverage = coverage_penalty(pred, target_cum)
    total = pinball + lambda_ * direction + coverage_weight * coverage
    return LossComponents(pinball=pinball, direction=direction, coverage=coverage, total=total)
