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
