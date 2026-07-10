"""Reviewed metric layer for model evaluation, migrated from scripts/eval_predictor.py.

These functions were written and review-hardened for the predictor bake-off harness
(PR #18: a HIGH NaN-poisoning bug was caught and fixed there) and are salvaged here
verbatim so the benchmark app can consume them — the app lives under src/ and so may
not reach into scripts/. The eval_predictor bake-off now imports them back from this
module (scripts -> src is the allowed direction).

Two families:

  Statistical (scored in the model's OWN target space):
    q90 coverage, [q10,q90] calibration, directional accuracy, close-interval
    sharpness, pinball. Coverage/calibration/DA are cross-semantics comparable;
    sharpness/pinball are NOT (per-step vs cumulative magnitudes differ).

  Economic (scored on the realised price path):
    "fraction of the predicted move that played out" (favorable excursion / predicted
    magnitude) PAIRED WITH adverse excursion (how far price ran the wrong way first).
    Sub-fee predicted moves (|move| <= EXECUTION.FEE_THRESHOLD) are untradeable and
    excluded — this grounds the cutoff in an existing constant and stops the metric
    being gamed by shrinking predictions toward zero.

All numeric parameters come from constants.py; this module hardcodes none.
"""
from __future__ import annotations

import torch

from constants import DATA, EXECUTION, PREDICTOR
from src.predictor.deploy_gates import calibration_rate, directional_accuracy, q90_coverage
from src.predictor.loss import pinball_loss

_CLOSE = DATA.FEATURE_NAMES.index("close_logret")
_Q10 = PREDICTOR.QUANTILES.index(0.10)
_Q50 = PREDICTOR.QUANTILES.index(0.50)
_Q90 = PREDICTOR.QUANTILES.index(0.90)


def target_to_model_space(target: torch.Tensor, semantics: str) -> torch.Tensor:
    """Map raw per-step log-return targets into the model's prediction space.

    ``per_step_logret`` predicts each step's move (identity); ``cumulative_logret``
    predicts the running-sum path, so the target is cumsummed over the horizon dim;
    ``cumulative_sqret`` (vol pivot, 2026-07-09) predicts the cumulative REALIZED-
    VARIANCE path, so the target is squared elementwise THEN cumsummed.
    """
    if semantics == "cumulative_logret":
        return torch.cumsum(target, dim=1)
    if semantics == "cumulative_absret":
        return torch.cumsum(torch.abs(target), dim=1)
    return target


def statistical_metrics(pred: torch.Tensor, target_model_space: torch.Tensor) -> dict[str, float]:
    """Coverage/calibration/DA (cross-semantics comparable) + sharpness/pinball (per-space).

    ``target_model_space`` must already be in the model's space (see target_to_model_space).
    """
    sharpness = float((pred[..., _CLOSE, _Q90] - pred[..., _CLOSE, _Q10]).mean())
    return {
        "q90_coverage": q90_coverage(pred, target_model_space),
        "calibration_rate": calibration_rate(pred, target_model_space),
        # DA over the FINAL horizon step only — the cumulative move a hold-to-horizon
        # trade actually spans (the quantity the fixed instrument acts on).
        "directional_accuracy": directional_accuracy(pred[:, -1], target_model_space[:, -1]),
        "sharpness_close": sharpness,
        "pinball": float(pinball_loss(pred, target_model_space)),
    }


def excursion_metrics(
    pred: torch.Tensor,
    target_raw: torch.Tensor,
    semantics: str,
    *,
    fee_threshold: float = EXECUTION.FEE_THRESHOLD,
) -> dict[str, float]:
    """Krafer-style "how much of the predicted move played out", paired with adverse run.

    Reference move = the q50 close forecast, expressed as a cumulative path (cumsum of the
    per-step median, or the median directly when it is already cumulative). For each
    sample the predicted total move sets the direction ``s`` and magnitude ``M``; over the
    realised cumulative path the favorable excursion is the best move in direction ``s`` and
    the adverse excursion is the worst move against it. Fractions are ``excursion / M``,
    averaged over samples whose ``M`` clears the round-trip fee.
    """
    q50_close = pred[..., _CLOSE, _Q50]  # (B, H)
    pred_cum = q50_close if semantics == "cumulative_logret" else torch.cumsum(q50_close, dim=1)
    realized_cum = torch.cumsum(target_raw[..., _CLOSE], dim=1)  # (B, H)

    final = pred_cum[:, -1]
    sign = torch.sign(final)
    magnitude = final.abs()
    tradeable = magnitude > fee_threshold
    n_used = int(tradeable.sum())
    if n_used == 0:
        return {"mean_captured_fraction": float("nan"), "mean_adverse_ratio": float("nan"),
                "median_captured_fraction": float("nan"), "n_used": 0}

    directional = sign.unsqueeze(1) * realized_cum  # (B, H); >0 = with the forecast
    favorable = directional.amax(dim=1).clamp_min(0.0)
    adverse = (-directional).amax(dim=1).clamp_min(0.0)
    captured = (favorable / magnitude)[tradeable]
    adverse_ratio = (adverse / magnitude)[tradeable]
    return {
        "mean_captured_fraction": float(captured.mean()),
        "mean_adverse_ratio": float(adverse_ratio.mean()),
        "median_captured_fraction": float(captured.median()),
        "n_used": n_used,
    }
