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


def test_gradient_is_width_only_zero_net_median_force() -> None:
    # Empirically diagnosed failure mode (bake-off runs at weight 1.0 and 0.1, fold 0,
    # 3 seeds): tail-coverage gradients flowing through the monotone head's shared
    # median anchor dragged q50 systematically, collapsing directional accuracy to a
    # coin flip (0.537 -> ~0.49). The penalty must therefore act on interval WIDTH
    # only: through the model head (q_tail = q50_anchor +/- softplus offsets, each
    # dq_tail/danchor = 1) the anchor's total gradient is grad_q10 + grad_q50 +
    # grad_q90, so exact cancellation requires grad_q50 == -(grad_q10 + grad_q90)
    # elementwise at the pred leaf.
    pred, target_cum = _gaussian_case(scale=0.5)
    pred = pred.clone().requires_grad_()
    coverage_penalty(pred, target_cum).backward()
    assert pred.grad is not None
    q50_idx = PREDICTOR.QUANTILES.index(0.50)
    anchor_force = (
        pred.grad[..., _CLOSE, _Q10]
        + pred.grad[..., _CLOSE, q50_idx]
        + pred.grad[..., _CLOSE, _Q90]
    )
    # fp32 rounding leaves a tiny residual (~1e-11) from the separate backward paths;
    # the uncancelled anchor force in the failing implementation was ~1e-4 — six
    # orders larger — so 1e-9 cleanly separates "cancelled" from "not cancelled".
    assert float(anchor_force.abs().max()) < 1e-9


def test_gradient_cancellation_holds_under_bf16_autocast() -> None:
    # Real training wraps the loss in torch.autocast(bf16) with fp32 leaf parameters
    # (train_one_fold). The anchor cancellation must hold on that path too, not only
    # in the pure-fp32 case above — the penalty casts its own inputs to fp32, and
    # gradients accumulate in the fp32 leaf's dtype.
    pred, target_cum = _gaussian_case(scale=0.5)
    pred = pred.clone().requires_grad_()  # fp32 leaf, as model params are under AMP
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        penalty = coverage_penalty(pred, target_cum)
    penalty.backward()
    assert pred.grad is not None
    q50_idx = PREDICTOR.QUANTILES.index(0.50)
    anchor_force = (
        pred.grad[..., _CLOSE, _Q10]
        + pred.grad[..., _CLOSE, q50_idx]
        + pred.grad[..., _CLOSE, _Q90]
    )
    assert float(anchor_force.abs().max()) < 1e-9


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


# --- Vol pivot (2026-07-09, branch pivot-vol-01): cumulative REALIZED-VARIANCE target ---
# Target reformulation: cumulative log-return path -> cumulative SQUARED log-return path
# (realized-variance path). Direction penalty retired (DIRECTION_PENALTY_LAMBDA = 0.0):
# direction is falsified, meaningless for a variance target.


def test_target_semantics_is_cumulative_sqret() -> None:
    assert PREDICTOR.TARGET_SEMANTICS == "cumulative_sqret"


def test_direction_penalty_lambda_retired_to_zero() -> None:
    assert PREDICTOR.DIRECTION_PENALTY_LAMBDA == 0.0


def test_predictor_loss_target_conversion_is_squared_then_cumsum() -> None:
    # predictor_loss's single conversion boundary: target_cum = cumsum(target ** 2),
    # NOT cumsum(target). A prediction equal to the squared-cumsum target at every
    # quantile must have exactly zero pinball loss.
    gen = torch.Generator().manual_seed(2)
    target = torch.randn((4, _H, _D), generator=gen) * 0.01
    target_sqcum = (target * target).cumsum(dim=1)
    pred = target_sqcum.unsqueeze(-1).expand(-1, -1, -1, _Q).contiguous()
    comp = predictor_loss(pred, target, lambda_=0.0)
    assert comp.pinball.item() == pytest.approx(0.0, abs=1e-9)

    # A prediction matching the OLD (unsquared, signed) cumulative boundary must now
    # score nonzero pinball -- the conversion is no longer plain cumsum.
    old_cum = target.cumsum(dim=1)
    pred_old = old_cum.unsqueeze(-1).expand(-1, -1, -1, _Q).contiguous()
    comp_old = predictor_loss(pred_old, target, lambda_=0.0)
    assert comp_old.pinball.item() > 1e-6


def test_direction_component_zero_exactly_when_lambda_zero() -> None:
    # Direction penalty is retired under the vol target: with lambda_=0.0 (the default,
    # PREDICTOR.DIRECTION_PENALTY_LAMBDA) the direction component must be an exact zero,
    # not merely weighted to zero -- no wasted compute/grad on a meaningless term.
    gen = torch.Generator().manual_seed(3)
    pred = torch.randn((4, _H, _D, _Q), generator=gen) * 0.01
    target = torch.randn((4, _H, _D), generator=gen) * 0.01
    comp = predictor_loss(pred, target, lambda_=0.0)
    assert comp.direction.item() == 0.0
    expected_total = comp.pinball.item() + PREDICTOR.COVERAGE_PENALTY_WEIGHT * comp.coverage.item()
    assert comp.total.item() == pytest.approx(expected_total, rel=1e-6)
