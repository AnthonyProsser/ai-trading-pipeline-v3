"""Coverage-penalty tests for src/predictor/loss.py (DECISIONS `loss` amendment).

Intended behavior encoded before implementation (test_discipline): the training loss
gains an explicit differentiable coverage penalty —

    L = pinball + lambda * direction + COVERAGE_PENALTY_WEIGHT * coverage_penalty

where coverage_penalty is the squared gap between smooth empirical batch coverage
(sigmoid indicator, width = COVERAGE_PENALTY_TEMPERATURE_FRAC x the batch's per-step
close-target std) and the nominal tail levels (0.10 / 0.90), on the CLOSE dimension
only (the dim every deploy gate and the trader read), averaged over horizon steps.
Rationale: the capped bake-off showed the cumulative model under-covers both tails
(q90_coverage 0.866 vs 0.90, calibration_rate 0.747 vs 0.80) — pinball's marginal
calibration pressure is too weak at capped budgets; this term optimizes the two gate
metrics directly and is self-limiting (gradient ~ gap, vanishing at nominal).
"""
from __future__ import annotations

import pytest

from constants import DATA, PREDICTOR

torch = pytest.importorskip("torch")

from src.predictor.loss import coverage_penalty, predictor_loss  # noqa: E402

_CLOSE = DATA.FEATURE_NAMES.index("close_logret")
_Q10 = PREDICTOR.QUANTILES.index(0.10)
_Q90 = PREDICTOR.QUANTILES.index(0.90)
_Z90 = 1.2815515655446004  # standard-normal 90th-percentile z-score

_B, _H, _D, _Q = 4096, PREDICTOR.HORIZON, PREDICTOR.NUM_OUTPUT_DIMS, len(PREDICTOR.QUANTILES)


def _gaussian_case(scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Cumulative close targets ~ N(0, sigma) with exact Gaussian quantile predictions,
    interval width multiplied by `scale` (1.0 = perfectly calibrated)."""
    gen = torch.Generator().manual_seed(0)
    sigma = 0.01
    target_cum = torch.randn((_B, _H, _D), generator=gen) * sigma
    pred = torch.zeros((_B, _H, _D, _Q))
    pred[..., _Q90] = sigma * _Z90 * scale
    pred[..., _Q10] = -sigma * _Z90 * scale
    return pred, target_cum


def test_constants_exposed_and_sane() -> None:
    # Magic numbers live in constants.py only; the three knobs must exist there.
    assert PREDICTOR.COVERAGE_PENALTY_WEIGHT >= 0.0
    assert 0.0 < PREDICTOR.COVERAGE_PENALTY_TEMPERATURE_FRAC <= 1.0
    assert PREDICTOR.COVERAGE_PENALTY_STD_FLOOR > 0.0


def test_calibrated_quantiles_near_zero_penalty() -> None:
    # Exact Gaussian tail quantiles => empirical coverage ~ nominal => penalty ~ 0
    # (up to sampling noise at B=4096 and the smooth indicator's small bias).
    pred, target_cum = _gaussian_case(scale=1.0)
    assert float(coverage_penalty(pred, target_cum)) < 1e-3


def test_narrow_intervals_penalised_much_harder_than_calibrated() -> None:
    # Halving the interval width under-covers both tails (the observed failure mode);
    # the penalty must be far above the calibrated baseline, not marginally above.
    pred, target_cum = _gaussian_case(scale=1.0)
    narrow_pred, _ = _gaussian_case(scale=0.5)
    calibrated = float(coverage_penalty(pred, target_cum))
    narrow = float(coverage_penalty(narrow_pred, target_cum))
    assert narrow > 10.0 * max(calibrated, 1e-6)


def test_gradient_widens_undercovering_intervals() -> None:
    # Under-coverage must produce gradients that WIDEN the interval: increasing q90
    # reduces the penalty (negative grad) and decreasing q10 reduces it (positive grad).
    pred, target_cum = _gaussian_case(scale=0.5)
    pred = pred.clone().requires_grad_()
    coverage_penalty(pred, target_cum).backward()
    assert pred.grad is not None
    assert float(pred.grad[..., _CLOSE, _Q90].sum()) < 0.0
    assert float(pred.grad[..., _CLOSE, _Q10].sum()) > 0.0


def test_close_dimension_only() -> None:
    # Gates and trader read close only; the penalty must ignore every other dim.
    pred, target_cum = _gaussian_case(scale=0.5)
    base = float(coverage_penalty(pred, target_cum))
    pred_perturbed = pred.clone()
    target_perturbed = target_cum.clone()
    for dim in range(_D):
        if dim == _CLOSE:
            continue
        pred_perturbed[..., dim, :] += 123.0
        target_perturbed[..., dim] -= 456.0
    assert float(coverage_penalty(pred_perturbed, target_perturbed)) == pytest.approx(base)


def test_flat_batch_is_finite() -> None:
    # A degenerate flat batch (zero target std) must not divide by zero.
    pred = torch.zeros((8, _H, _D, _Q))
    target_cum = torch.zeros((8, _H, _D))
    assert torch.isfinite(coverage_penalty(pred, target_cum))


def test_predictor_loss_includes_weighted_coverage_component() -> None:
    # LossComponents grows a `coverage` field and total = pinball + lambda*direction
    # + COVERAGE_PENALTY_WEIGHT*coverage. Existing consumers read fields by name.
    gen = torch.Generator().manual_seed(1)
    pred = torch.randn((8, _H, _D, _Q), generator=gen) * 0.01
    target = torch.randn((8, _H, _D), generator=gen) * 0.01
    comp = predictor_loss(pred, target)
    assert float(comp.coverage) >= 0.0
    expected = (
        float(comp.pinball)
        + PREDICTOR.DIRECTION_PENALTY_LAMBDA * float(comp.direction)
        + PREDICTOR.COVERAGE_PENALTY_WEIGHT * float(comp.coverage)
    )
    assert float(comp.total) == pytest.approx(expected, rel=1e-5)
