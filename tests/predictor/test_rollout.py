"""Geometry-enforcement tests for the predictor rollout sampler
(predictor-contract.md §"Geometry enforcement").

Independent quantile heads can emit physically impossible candles: a predicted
high below the open/close, or a low above them. Before any candle reaches the
trader (production inference OR a Trader-side mock harness) it must satisfy, for
every emitted step ``s`` and every quantile ``q``:

    high_q  >= max(open_q, close_q)
    low_q   <= min(open_q, close_q)

The contract names the q90 high and q10 low explicitly; this generalises the rule
to every quantile so each emitted candle *scenario* is self-consistent (the q90
high and q10 low cases are thereby covered too). Violations from a stochastic
sampler are redrawn up to ``GEOMETRY_RESAMPLE_CAP`` times, then deterministically
clamped so the returned candle is always valid.

Committed before src/predictor/rollout.py, per the test-first discipline.
"""
from __future__ import annotations

from typing import Any

import pytest

from constants import PREDICTOR
from src.data.validator import OhlcvCol

_Q10 = PREDICTOR.QUANTILES.index(0.10)
_Q90 = PREDICTOR.QUANTILES.index(0.90)
_PRED_SHAPE = (2, PREDICTOR.HORIZON, PREDICTOR.NUM_OUTPUT_DIMS, len(PREDICTOR.QUANTILES))


def _all_valid(pred: Any) -> bool:
    """True iff every (step, quantile) candle satisfies the geometry inequalities."""
    import torch

    o = pred[:, :, OhlcvCol.OPEN, :]
    h = pred[:, :, OhlcvCol.HIGH, :]
    low = pred[:, :, OhlcvCol.LOW, :]
    c = pred[:, :, OhlcvCol.CLOSE, :]
    high_ok = bool(torch.all(h >= torch.maximum(o, c)).item())
    low_ok = bool(torch.all(low <= torch.minimum(o, c)).item())
    return high_ok and low_ok


def test_enforce_geometry_corrects_inconsistent_candles() -> None:
    torch = pytest.importorskip("torch")
    from src.predictor.rollout import enforce_geometry

    gen = torch.Generator().manual_seed(0)
    bad = torch.randn(_PRED_SHAPE, generator=gen)
    # Force a violation at every step/quantile: high far below the body, low far above it.
    bad[:, :, OhlcvCol.HIGH, :] = -10.0
    bad[:, :, OhlcvCol.LOW, :] = 10.0
    assert not _all_valid(bad)

    fixed = enforce_geometry(bad)

    # Every emitted candle now satisfies H >= max(O,C) and L <= min(O,C)...
    assert _all_valid(fixed)
    # ...including the two cases the contract names explicitly.
    o90, c90 = bad[:, :, OhlcvCol.OPEN, _Q90], bad[:, :, OhlcvCol.CLOSE, _Q90]
    assert torch.all(fixed[:, :, OhlcvCol.HIGH, _Q90] >= torch.maximum(o90, c90))
    o10, c10 = bad[:, :, OhlcvCol.OPEN, _Q10], bad[:, :, OhlcvCol.CLOSE, _Q10]
    assert torch.all(fixed[:, :, OhlcvCol.LOW, _Q10] <= torch.minimum(o10, c10))


def test_enforce_geometry_leaves_open_close_volume_untouched() -> None:
    torch = pytest.importorskip("torch")
    from src.predictor.rollout import enforce_geometry

    gen = torch.Generator().manual_seed(1)
    bad = torch.randn(_PRED_SHAPE, generator=gen)
    bad[:, :, OhlcvCol.HIGH, :] = -10.0
    bad[:, :, OhlcvCol.LOW, :] = 10.0

    fixed = enforce_geometry(bad)

    for dim in (OhlcvCol.OPEN, OhlcvCol.CLOSE, OhlcvCol.VOLUME):
        assert torch.equal(fixed[:, :, dim, :], bad[:, :, dim, :])


def test_enforce_geometry_is_idempotent_on_valid_input() -> None:
    torch = pytest.importorskip("torch")
    from src.predictor.rollout import enforce_geometry

    gen = torch.Generator().manual_seed(2)
    valid = torch.randn(_PRED_SHAPE, generator=gen)
    # Make it consistent: high = body_high + 1, low = body_low - 1.
    o, c = valid[:, :, OhlcvCol.OPEN, :], valid[:, :, OhlcvCol.CLOSE, :]
    valid[:, :, OhlcvCol.HIGH, :] = torch.maximum(o, c) + 1.0
    valid[:, :, OhlcvCol.LOW, :] = torch.minimum(o, c) - 1.0
    assert _all_valid(valid)

    fixed = enforce_geometry(valid)
    assert torch.equal(fixed, valid)


def test_enforce_geometry_does_not_mutate_input() -> None:
    torch = pytest.importorskip("torch")
    from src.predictor.rollout import enforce_geometry

    gen = torch.Generator().manual_seed(3)
    bad = torch.randn(_PRED_SHAPE, generator=gen)
    bad[:, :, OhlcvCol.HIGH, :] = -10.0
    snapshot = bad.clone()

    enforce_geometry(bad)
    assert torch.equal(bad, snapshot)  # returns a new tensor; no in-place edit


def test_resample_redraws_up_to_cap_then_clamps() -> None:
    torch = pytest.importorskip("torch")
    from src.predictor.rollout import resample_until_valid

    bad = torch.randn(_PRED_SHAPE, generator=torch.Generator().manual_seed(4))
    bad[:, :, OhlcvCol.HIGH, :] = -10.0
    bad[:, :, OhlcvCol.LOW, :] = 10.0

    calls = {"n": 0}

    def sampler() -> Any:
        calls["n"] += 1
        return bad.clone()

    out = resample_until_valid(sampler)

    assert calls["n"] == PREDICTOR.GEOMETRY_RESAMPLE_CAP  # exhausts the cap of redraws
    assert _all_valid(out)  # deterministic clamp guarantees validity after the cap


def test_resample_returns_first_valid_draw_without_clamping() -> None:
    torch = pytest.importorskip("torch")
    from src.predictor.rollout import resample_until_valid

    gen = torch.Generator().manual_seed(5)
    invalid = torch.randn(_PRED_SHAPE, generator=gen)
    invalid[:, :, OhlcvCol.HIGH, :] = -10.0
    invalid[:, :, OhlcvCol.LOW, :] = 10.0

    valid = torch.randn(_PRED_SHAPE, generator=gen)
    o, c = valid[:, :, OhlcvCol.OPEN, :], valid[:, :, OhlcvCol.CLOSE, :]
    valid[:, :, OhlcvCol.HIGH, :] = torch.maximum(o, c) + 1.0
    valid[:, :, OhlcvCol.LOW, :] = torch.minimum(o, c) - 1.0

    draws = [invalid, valid]
    calls = {"n": 0}

    def sampler() -> Any:
        t = draws[min(calls["n"], len(draws) - 1)]
        calls["n"] += 1
        return t

    out = resample_until_valid(sampler)

    assert calls["n"] == 2  # stops the moment a valid candle appears
    assert torch.equal(out, valid)  # returned as-is, not clamped
