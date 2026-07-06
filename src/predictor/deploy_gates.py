"""Predictor deploy gates (predictor-contract.md §"Deploy gates", DECISIONS deploy_gates).

Three gates, all required simultaneously before a checkpoint replaces the active one:

  (a) q90 coverage on the locked test set within ±DEPLOY_GATE_COVERAGE_TOLERANCE of the
      training-time coverage,
  (b) directional accuracy > DEPLOY_GATE_DA_THRESHOLD, computed ONLY over predictions
      where |q50| > FEE_THRESHOLD (sub-fee moves aren't tradeable),
  (c) calibration rate (target inside [q10, q90]) in
      [DEPLOY_GATE_CAL_LOWER, DEPLOY_GATE_CAL_UPPER].

All metrics use the CLOSE dimension -- the tradeable signal, consistent with the loss's
close-dim direction penalty and the trader's close-based confidence gate. Thresholds come
from constants.py; this module hardcodes none.

The functions are pure comparisons: pred and target must be in the SAME space. Under
PredictorConfig.TARGET_SEMANTICS both are cumulative log-return paths (the collectors in
src/predictor/training.py and scripts/deploy_predictor.py cumsum the raw per-step
targets), so DA's |q50| > FEE_THRESHOLD filter compares the fee against the horizon move
it actually applies to, and coverage/calibration score the intervals the trader trades.

Tensor layout (predictor-contract.md):
    pred   : (..., NUM_OUTPUT_DIMS, NUM_QUANTILES)   quantile order q10, q50, q90
    target : (..., NUM_OUTPUT_DIMS)                  log-returns, same space as pred
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from constants import DATA, EXECUTION, PREDICTOR

_CLOSE = DATA.FEATURE_NAMES.index("close_logret")
_Q10 = PREDICTOR.QUANTILES.index(0.10)
_Q50 = PREDICTOR.QUANTILES.index(0.50)
_Q90 = PREDICTOR.QUANTILES.index(0.90)


class DeployGateResult(NamedTuple):
    q90_coverage: float
    coverage_delta: float
    directional_accuracy: float
    calibration_rate: float
    coverage_pass: bool
    da_pass: bool
    calibration_pass: bool
    all_passed: bool


def q90_coverage(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Empirical coverage at q90: fraction of close targets at or below the q90 prediction."""
    q90 = pred[..., _CLOSE, _Q90]
    return float((target[..., _CLOSE] <= q90).to(torch.float64).mean())


def directional_accuracy(
    pred: torch.Tensor, target: torch.Tensor, fee_threshold: float = EXECUTION.FEE_THRESHOLD
) -> float:
    """Sign agreement between q50 and the realised close move, over tradeable predictions only.

    Returns NaN when no prediction clears |q50| > fee_threshold (nothing tradeable to score).
    """
    q50 = pred[..., _CLOSE, _Q50]
    realised = target[..., _CLOSE]
    tradeable = q50.abs() > fee_threshold
    n_tradeable = int(tradeable.sum())
    if n_tradeable == 0:
        return float("nan")
    correct = (torch.sign(q50) == torch.sign(realised)) & tradeable
    return float(int(correct.sum()) / n_tradeable)


def calibration_rate(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of close targets inside the [q10, q90] interval (ideally ~0.80)."""
    q10 = pred[..., _CLOSE, _Q10]
    q90 = pred[..., _CLOSE, _Q90]
    realised = target[..., _CLOSE]
    inside = (realised >= q10) & (realised <= q90)
    return float(inside.to(torch.float64).mean())


def evaluate_deploy_gates(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    train_time_q90_coverage: float,
    fee_threshold: float = EXECUTION.FEE_THRESHOLD,
    da_pred: torch.Tensor | None = None,
    da_target: torch.Tensor | None = None,
) -> DeployGateResult:
    """Compute all three gates and whether they pass simultaneously.

    A NaN directional accuracy (no tradeable predictions) fails gate (b): ``nan > x`` is
    False, so the checkpoint does not deploy.

    ``da_pred``/``da_target`` restrict gate (b) to a sub-tensor — collectors with a
    horizon axis pass the FINAL-step slice, the cumulative move a hold-to-horizon trade
    actually spans — while coverage/calibration keep scoring the full path. When omitted,
    DA scores the same tensors as the other gates (pure-comparison behaviour unchanged).
    """
    coverage = q90_coverage(pred, target)
    coverage_delta = coverage - train_time_q90_coverage
    da = directional_accuracy(
        pred if da_pred is None else da_pred,
        target if da_target is None else da_target,
        fee_threshold,
    )
    calibration = calibration_rate(pred, target)

    coverage_pass = abs(coverage_delta) <= PREDICTOR.DEPLOY_GATE_COVERAGE_TOLERANCE
    da_pass = da > PREDICTOR.DEPLOY_GATE_DA_THRESHOLD
    calibration_pass = (
        PREDICTOR.DEPLOY_GATE_CAL_LOWER <= calibration <= PREDICTOR.DEPLOY_GATE_CAL_UPPER
    )
    return DeployGateResult(
        q90_coverage=coverage,
        coverage_delta=coverage_delta,
        directional_accuracy=da,
        calibration_rate=calibration,
        coverage_pass=coverage_pass,
        da_pass=da_pass,
        calibration_pass=calibration_pass,
        all_passed=coverage_pass and da_pass and calibration_pass,
    )
