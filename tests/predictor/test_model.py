"""PatchTST predictor model contract tests (predictor-training.md §"Architecture",
predictor-contract.md §Input/Output).

The model is an encoder-only Transformer with channel-mixing patch embedding. It
must, in a SINGLE forward pass (autoregression banned):

    (batch, lookback, NUM_INPUT_FEATURES)
        -> (batch, HORIZON, NUM_OUTPUT_DIMS, NUM_QUANTILES)

Locked invariants under test: the I/O shape contract, determinism given a fixed
input (eval mode), and the lookback-divisible-by-PATCH_SIZE assertion.

Committed before src/predictor/model.py, per the test-first discipline.
"""
from __future__ import annotations

import pytest

from constants import DATA, PREDICTOR

_IN = DATA.NUM_INPUT_FEATURES
_OUT = (PREDICTOR.HORIZON, PREDICTOR.NUM_OUTPUT_DIMS, len(PREDICTOR.QUANTILES))


def test_forward_output_shape_matches_contract() -> None:
    torch = pytest.importorskip("torch")
    from src.predictor.model import PatchTST

    model = PatchTST(lookback=DATA.LOOKBACK).eval()
    x = torch.randn(4, DATA.LOOKBACK, _IN)
    y = model(x)

    assert y.shape == (4, *_OUT)
    assert y.dtype == torch.float32
    assert bool(torch.isfinite(y).all())  # no NaN/inf out of a fresh forward pass


def test_single_forward_pass_is_deterministic() -> None:
    # predictor-contract.md: "deterministic given a fixed input". In eval mode
    # (dropout disabled) the same input must yield identical output.
    torch = pytest.importorskip("torch")
    from src.predictor.model import PatchTST

    torch.manual_seed(PREDICTOR.SEED)
    model = PatchTST(lookback=DATA.LOOKBACK).eval()
    x = torch.randn(2, DATA.LOOKBACK, _IN)

    assert torch.equal(model(x), model(x))


def test_lookback_must_be_divisible_by_patch_size() -> None:
    pytest.importorskip("torch")
    from src.predictor.model import PatchTST

    with pytest.raises(ValueError, match="divisible"):
        PatchTST(lookback=PREDICTOR.PATCH_SIZE * 3 + 1)


@pytest.mark.parametrize("lookback", [240, 720, 1440])
def test_sweep_lookbacks_produce_contract_shape(lookback: int) -> None:
    # lookback is swept over [240, 720, 1440]; each must keep the output contract
    # and an integer token count (lookback / PATCH_SIZE).
    torch = pytest.importorskip("torch")
    from src.predictor.model import PatchTST

    assert lookback % PREDICTOR.PATCH_SIZE == 0
    model = PatchTST(lookback=lookback).eval()
    y = model(torch.randn(2, lookback, _IN))
    assert y.shape == (2, *_OUT)


def test_num_tokens_is_lookback_over_patch_size() -> None:
    pytest.importorskip("torch")
    from src.predictor.model import PatchTST

    model = PatchTST(lookback=DATA.LOOKBACK)
    assert model.num_tokens == DATA.LOOKBACK // PREDICTOR.PATCH_SIZE


def test_quantile_outputs_are_monotone_by_construction() -> None:
    # The head parameterises quantiles as median +/- cumulative softplus offsets, so
    # q10 <= q50 <= q90 holds for EVERY output coordinate of any input — crossing
    # quantiles are impossible by construction, not just discouraged by the loss.
    torch = pytest.importorskip("torch")
    from src.predictor.model import PatchTST

    torch.manual_seed(PREDICTOR.SEED)
    model = PatchTST(lookback=PREDICTOR.PATCH_SIZE * 2).eval()
    y = model(torch.randn(8, PREDICTOR.PATCH_SIZE * 2, _IN))

    for lo in range(len(PREDICTOR.QUANTILES) - 1):
        assert bool(torch.all(y[..., lo] <= y[..., lo + 1]))


def test_forward_is_shift_invariant_and_volatility_scaled() -> None:
    # RevIN-style instance normalization: a constant shift of the input window must not
    # change the forecast, and rescaling the window by k must rescale the predicted
    # quantiles by ~k (interval width tracks current volatility by construction).
    # idea-01-timefeatures: RevIN now applies to the OHLCV prefix only -- clock columns
    # bypass it by design (they're bounded periodic signals, not meant to be shifted or
    # volatility-scaled), so only the OHLCV columns are shifted/rescaled here; the clock
    # columns are held fixed between the two forward passes being compared.
    torch = pytest.importorskip("torch")
    from src.predictor.model import PatchTST

    torch.manual_seed(PREDICTOR.SEED)
    model = PatchTST(lookback=PREDICTOR.PATCH_SIZE * 2).eval()
    n_out = PREDICTOR.NUM_OUTPUT_DIMS
    x = torch.randn(4, PREDICTOR.PATCH_SIZE * 2, _IN)

    x_shifted = x.clone()
    x_shifted[..., :n_out] += 5.0
    assert torch.allclose(model(x_shifted), model(x), atol=1e-4)

    x_scaled = x.clone()
    x_scaled[..., :n_out] *= 3.0
    assert torch.allclose(model(x_scaled), model(x) * 3.0, rtol=1e-3, atol=1e-5)


def test_forward_maps_nine_input_features_to_five_output_dims() -> None:
    # idea-01-timefeatures: NUM_INPUT_FEATURES (9: 5 OHLCV + 4 clock) is decoupled from
    # NUM_OUTPUT_DIMS (5: OHLCV only) -- the model still predicts only OHLCV.
    torch = pytest.importorskip("torch")
    from src.predictor.model import PatchTST

    assert DATA.NUM_INPUT_FEATURES == 9
    assert PREDICTOR.NUM_OUTPUT_DIMS == 5
    model = PatchTST(lookback=DATA.LOOKBACK).eval()
    y = model(torch.randn(2, DATA.LOOKBACK, _IN))
    assert y.shape == (2, PREDICTOR.HORIZON, 5, len(PREDICTOR.QUANTILES))


def test_constant_clock_columns_do_not_blow_up_output() -> None:
    # The last (NUM_INPUT_FEATURES - NUM_OUTPUT_DIMS) columns are clock features
    # (tod_sin/cos, dow_sin/cos). Within a lookback window shorter than a day/week they
    # are near-constant. If RevIN's per-window (x - mean) / sigma were applied to them,
    # a near-zero sigma would blow the output to inf/nan -- they must bypass RevIN.
    torch = pytest.importorskip("torch")
    from src.predictor.model import PatchTST

    torch.manual_seed(PREDICTOR.SEED)
    lookback = PREDICTOR.PATCH_SIZE * 2
    n_out = PREDICTOR.NUM_OUTPUT_DIMS
    model = PatchTST(lookback=lookback).eval()
    x = torch.randn(4, lookback, _IN)
    x[..., n_out:] = 0.5  # simulate a within-24h/week window: clock cols exactly constant

    y = model(x)

    assert bool(torch.isfinite(y).all())
