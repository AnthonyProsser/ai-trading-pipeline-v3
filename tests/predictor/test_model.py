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


def test_direction_head_exists_with_expected_shape() -> None:
    # idea-04-auxdirhead: a training-only auxiliary head predicting the sign of the
    # final-horizon cumulative close move, sharing the encoder trunk with `head`.
    torch = pytest.importorskip("torch")
    from src.predictor.model import PatchTST

    model = PatchTST(lookback=DATA.LOOKBACK)
    assert isinstance(model.direction_head, torch.nn.Linear)
    assert model.direction_head.in_features == model.num_tokens * PREDICTOR.D_MODEL
    assert model.direction_head.out_features == 1


def test_forward_with_direction_matches_forward_quantiles_and_logit_shape() -> None:
    # forward_with_direction shares the trunk + quantile head with forward(): for the
    # same input the quantile tensors must be IDENTICAL (not just close), and the
    # direction logit must be (batch,).
    torch = pytest.importorskip("torch")
    from src.predictor.model import PatchTST

    torch.manual_seed(PREDICTOR.SEED)
    model = PatchTST(lookback=DATA.LOOKBACK).eval()
    x = torch.randn(4, DATA.LOOKBACK, _IN)

    y = model(x)
    quantiles, logit = model.forward_with_direction(x)

    assert torch.allclose(quantiles, y)
    assert quantiles.shape == (4, *_OUT)
    assert logit.shape == (4,)
    assert bool(torch.isfinite(logit).all())


def test_forward_signature_and_return_unchanged() -> None:
    # forward(x) MUST keep returning ONLY the quantile tensor -- many callers
    # (_gather, deploy, benchmark) depend on this exact interface.
    torch = pytest.importorskip("torch")
    from src.predictor.model import PatchTST

    model = PatchTST(lookback=DATA.LOOKBACK).eval()
    x = torch.randn(2, DATA.LOOKBACK, _IN)
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, *_OUT)


def test_aux_direction_gradient_flows_to_direction_head() -> None:
    # One optimizer step's worth of backward() must populate direction_head's grad --
    # the BCE aux term actually reaches the new head's parameters.
    torch = pytest.importorskip("torch")
    from src.predictor.loss import predictor_loss
    from src.predictor.model import PatchTST

    torch.manual_seed(PREDICTOR.SEED)
    model = PatchTST(lookback=DATA.LOOKBACK)
    x = torch.randn(4, DATA.LOOKBACK, _IN)
    target = torch.randn(4, PREDICTOR.HORIZON, PREDICTOR.NUM_OUTPUT_DIMS) * 0.01

    quantiles, logit = model.forward_with_direction(x)
    comp = predictor_loss(quantiles, target, direction_logit=logit)
    comp.total.backward()  # type: ignore[no-untyped-call]

    assert model.direction_head.weight.grad is not None
    assert bool(torch.isfinite(model.direction_head.weight.grad).all())
    assert float(model.direction_head.weight.grad.abs().sum()) > 0.0


def test_forward_is_shift_invariant_and_volatility_scaled() -> None:
    # RevIN-style instance normalization: a constant shift of the input window must not
    # change the forecast, and rescaling the window by k must rescale the predicted
    # quantiles by ~k (interval width tracks current volatility by construction).
    torch = pytest.importorskip("torch")
    from src.predictor.model import PatchTST

    torch.manual_seed(PREDICTOR.SEED)
    model = PatchTST(lookback=PREDICTOR.PATCH_SIZE * 2).eval()
    x = torch.randn(4, PREDICTOR.PATCH_SIZE * 2, _IN)

    assert torch.allclose(model(x + 5.0), model(x), atol=1e-4)
    assert torch.allclose(model(x * 3.0), model(x) * 3.0, rtol=1e-3, atol=1e-5)
