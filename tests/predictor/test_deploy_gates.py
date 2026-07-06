"""Deploy-gate tests (predictor-contract.md §"Deploy gates", DECISIONS deploy_gates).

All three gates must pass simultaneously before a checkpoint deploys:
  (a) q90 coverage within ±DEPLOY_GATE_COVERAGE_TOLERANCE of training-time coverage,
  (b) directional accuracy > DEPLOY_GATE_DA_THRESHOLD over predictions with
      |q50| > FEE_THRESHOLD,
  (c) calibration rate in [DEPLOY_GATE_CAL_LOWER, DEPLOY_GATE_CAL_UPPER].

Metrics are computed on the CLOSE dimension (the tradeable signal; consistent with the
loss's close-dim direction penalty and the trader's close-based confidence gate).

Committed before src/predictor/deploy_gates.py, per the test-first discipline.
"""
from __future__ import annotations

import math
from typing import Any

import pytest

from constants import DATA, EXECUTION, PREDICTOR

_CLOSE = DATA.FEATURE_NAMES.index("close_logret")
_Q10 = PREDICTOR.QUANTILES.index(0.10)
_Q50 = PREDICTOR.QUANTILES.index(0.50)
_Q90 = PREDICTOR.QUANTILES.index(0.90)
_FEE = EXECUTION.FEE_THRESHOLD


def _blank(n: int) -> tuple[Any, Any]:
    import torch

    pred = torch.zeros(n, PREDICTOR.NUM_OUTPUT_DIMS, len(PREDICTOR.QUANTILES))
    target = torch.zeros(n, PREDICTOR.NUM_OUTPUT_DIMS)
    return pred, target


def test_calibration_rate_counts_targets_inside_q10_q90() -> None:
    torch = pytest.importorskip("torch")
    from src.predictor.deploy_gates import calibration_rate

    pred, target = _blank(4)
    pred[:, _CLOSE, _Q10] = -1.0
    pred[:, _CLOSE, _Q90] = 1.0
    target[:, _CLOSE] = torch.tensor([0.0, 0.5, 2.0, -3.0])  # 2 inside, 2 outside
    assert calibration_rate(pred, target) == pytest.approx(0.5)


def test_q90_coverage_is_fraction_below_q90() -> None:
    torch = pytest.importorskip("torch")
    from src.predictor.deploy_gates import q90_coverage

    pred, target = _blank(4)
    pred[:, _CLOSE, _Q90] = 0.0
    target[:, _CLOSE] = torch.tensor([-1.0, -0.5, 1.0, 2.0])  # 2 <= q90, 2 above
    assert q90_coverage(pred, target) == pytest.approx(0.5)


def test_directional_accuracy_only_over_tradeable_predictions() -> None:
    torch = pytest.importorskip("torch")
    from src.predictor.deploy_gates import directional_accuracy

    pred, target = _blank(4)
    # |q50| > fee for indices 0,1,3; index 2 is sub-fee and excluded from DA.
    pred[:, _CLOSE, _Q50] = torch.tensor([2 * _FEE, 2 * _FEE, 0.5 * _FEE, -2 * _FEE])
    target[:, _CLOSE] = torch.tensor([1.0, -1.0, 1.0, -1.0])
    # Among the 3 tradeable: i0 correct, i1 wrong, i3 correct -> 2/3.
    assert directional_accuracy(pred, target, _FEE) == pytest.approx(2.0 / 3.0)


def test_directional_accuracy_nan_when_no_tradeable_predictions() -> None:
    pytest.importorskip("torch")
    from src.predictor.deploy_gates import directional_accuracy

    pred, target = _blank(3)
    pred[:, _CLOSE, _Q50] = 0.5 * _FEE  # all sub-fee
    assert math.isnan(directional_accuracy(pred, target, _FEE))


def test_evaluate_deploy_gates_all_pass() -> None:
    torch = pytest.importorskip("torch")
    from src.predictor.deploy_gates import evaluate_deploy_gates

    pred, target = _blank(20)
    pred[:, _CLOSE, _Q10] = -1.0
    pred[:, _CLOSE, _Q90] = 1.0
    pred[:, _CLOSE, _Q50] = 2 * _FEE
    target[:, _CLOSE] = torch.tensor([0.5] * 16 + [2.0] * 4)  # 16 inside/covered, 4 above

    result = evaluate_deploy_gates(pred, target, train_time_q90_coverage=0.80)

    assert result.calibration_rate == pytest.approx(0.80)
    assert result.q90_coverage == pytest.approx(0.80)
    assert result.directional_accuracy == pytest.approx(1.0)
    assert result.coverage_pass and result.da_pass and result.calibration_pass
    assert result.all_passed


def test_evaluate_deploy_gates_fails_on_low_da() -> None:
    torch = pytest.importorskip("torch")
    from src.predictor.deploy_gates import evaluate_deploy_gates

    pred, target = _blank(20)
    pred[:, _CLOSE, _Q10] = -1.0
    pred[:, _CLOSE, _Q90] = 1.0
    pred[:, _CLOSE, _Q50] = 2 * _FEE  # all tradeable, all predict "up"
    # coverage 0.80 + calibration 0.80 (16 in [-1,1] & <= q90, 4 above), but only 10/20 up.
    target[:, _CLOSE] = torch.tensor([0.5] * 6 + [-0.5] * 10 + [2.0] * 4)

    result = evaluate_deploy_gates(pred, target, train_time_q90_coverage=0.80)

    assert result.coverage_pass and result.calibration_pass
    assert result.directional_accuracy == pytest.approx(0.5)
    assert not result.da_pass
    assert not result.all_passed


def test_evaluate_deploy_gates_fails_on_calibration() -> None:
    pytest.importorskip("torch")
    from src.predictor.deploy_gates import evaluate_deploy_gates

    pred, target = _blank(10)
    pred[:, _CLOSE, _Q10] = -1.0
    pred[:, _CLOSE, _Q90] = 1.0
    pred[:, _CLOSE, _Q50] = -2 * _FEE  # predict "down"
    # target below q10 -> outside [q10, q90] (calibration 0), but <= q90 (coverage 1.0)
    # and sign matches q50<0 (DA 1.0): isolates the calibration gate.
    target[:, _CLOSE] = -2.0

    result = evaluate_deploy_gates(pred, target, train_time_q90_coverage=1.0)

    assert result.coverage_pass and result.da_pass
    assert not result.calibration_pass
    assert not result.all_passed


def test_evaluate_deploy_gates_nan_da_fails_gate() -> None:
    torch = pytest.importorskip("torch")
    from src.predictor.deploy_gates import evaluate_deploy_gates

    pred, target = _blank(20)
    pred[:, _CLOSE, _Q10] = -1.0
    pred[:, _CLOSE, _Q90] = 1.0
    pred[:, _CLOSE, _Q50] = 0.5 * _FEE  # all sub-fee -> no tradeable -> DA is NaN
    target[:, _CLOSE] = torch.tensor([0.5] * 16 + [2.0] * 4)

    result = evaluate_deploy_gates(pred, target, train_time_q90_coverage=0.80)

    assert math.isnan(result.directional_accuracy)
    assert result.coverage_pass and result.calibration_pass
    assert not result.da_pass  # nan > threshold is False
    assert not result.all_passed


def test_evaluate_deploy_gates_fails_on_coverage_drift() -> None:
    torch = pytest.importorskip("torch")
    from src.predictor.deploy_gates import evaluate_deploy_gates

    pred, target = _blank(20)
    pred[:, _CLOSE, _Q10] = -1.0
    pred[:, _CLOSE, _Q90] = 1.0
    pred[:, _CLOSE, _Q50] = 2 * _FEE
    target[:, _CLOSE] = torch.tensor([0.5] * 16 + [2.0] * 4)

    # actual coverage 0.80 vs train-time 0.70 -> delta 0.10 > tolerance 0.05
    result = evaluate_deploy_gates(pred, target, train_time_q90_coverage=0.70)

    assert not result.coverage_pass
    assert not result.all_passed


def test_evaluate_deploy_gates_da_slice_overrides() -> None:
    """``da_pred``/``da_target`` restrict ONLY the DA gate (the traded final-horizon-step
    move); coverage and calibration still score the full tensors (interval quality across
    the whole path). Collectors with a horizon axis pass the final-step slice here."""
    pytest.importorskip("torch")
    from src.predictor.deploy_gates import evaluate_deploy_gates

    pred, target = _blank(4)
    pred[:, _CLOSE, _Q10] = -1.0
    pred[:, _CLOSE, _Q90] = 1.0
    pred[:, _CLOSE, _Q50] = 2 * _FEE  # full tensors say "up"...
    target[:, _CLOSE] = -0.5  # ...but realised is down: all-steps DA would be 0.0

    da_pred, da_target = _blank(2)
    da_pred[:, _CLOSE, _Q50] = 2 * _FEE
    da_target[:, _CLOSE] = 0.5  # final-step slice: sign agrees -> DA 1.0

    result = evaluate_deploy_gates(
        pred, target, train_time_q90_coverage=1.0, da_pred=da_pred, da_target=da_target
    )
    assert result.directional_accuracy == pytest.approx(1.0)  # from the DA slice
    assert result.calibration_rate == pytest.approx(1.0)  # from the full tensors
    assert result.q90_coverage == pytest.approx(1.0)
    assert result.da_pass and result.all_passed
