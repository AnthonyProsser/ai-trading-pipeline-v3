"""Predictor training-loop unit tests (predictor-training.md §"Smoke run" /
§"Training data").

The reusable training logic lives in src/predictor/training.py so it is testable and
importable (scripts/train_predictor.py is a thin CLI over it, per the
src-never-imports-scripts rule). Contracts under test:

- WindowDataset: lookback input window (scaled) + horizon target window (raw log-returns).
- build_fold_loaders: consumes a walk-forward Fold + per-fold scaler; the scaler is fit on
  the TRAIN slice only; loaders carry scaled inputs and raw targets.
- train_one_fold: receives (scaler, train_loader, val_loader) and NEVER builds a scaler;
  runs on EarlyStopper + predictor_loss, returns finite, separated loss components.

Committed before src/predictor/training.py, per the test-first discipline.
"""
from __future__ import annotations

import numpy as np
import pytest

from constants import DATA, PREDICTOR

_F = DATA.NUM_INPUT_FEATURES
_LOOKBACK = PREDICTOR.PATCH_SIZE * 2  # 32: small but divisible by PATCH_SIZE
_H = PREDICTOR.HORIZON


def _synthetic(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Random-walk-ish log-return features + aligned 1-minute timestamps."""
    rng = np.random.default_rng(PREDICTOR.SEED)
    feats = (rng.standard_normal((n, _F)) * 0.01).astype(np.float64)
    base = np.datetime64("2020-01-01T00:00")  # minute precision; scaler compares in [m]
    ts = base + np.arange(n).astype("timedelta64[m]")
    return feats, ts


def test_window_dataset_shapes_and_count() -> None:
    torch = pytest.importorskip("torch")
    from src.predictor.training import WindowDataset

    n = 120
    x = torch.zeros((n, _F))
    y = torch.zeros((n, _F))
    ds = WindowDataset(x, y, lookback=_LOOKBACK, horizon=_H)

    assert len(ds) == n - _LOOKBACK - _H + 1
    xi, yi = ds[0]
    assert xi.shape == (_LOOKBACK, _F)
    assert yi.shape == (_H, _F)


def test_window_dataset_returns_scaled_x_and_raw_y() -> None:
    torch = pytest.importorskip("torch")
    from src.predictor.training import WindowDataset

    n = 80
    x_scaled = torch.full((n, _F), 0.5)  # stand-in "scaled" inputs
    y_raw = torch.arange(n * _F, dtype=torch.float32).reshape(n, _F)  # distinct raw values
    ds = WindowDataset(x_scaled, y_raw, lookback=_LOOKBACK, horizon=_H)

    xi, yi = ds[3]
    assert torch.equal(xi, x_scaled[3 : 3 + _LOOKBACK])
    assert torch.equal(yi, y_raw[3 + _LOOKBACK : 3 + _LOOKBACK + _H])


def test_build_fold_loaders_scales_on_train_and_shapes() -> None:
    pytest.importorskip("torch")
    from src.data.walk_forward import Fold
    from src.predictor.training import build_fold_loaders

    feats, ts = _synthetic(300)
    fold = Fold(0, 0, 150, 150, 250, 250, 300)  # tiny hand-made fold
    scaler, train_loader, val_loader = build_fold_loaders(
        feats, ts, fold, lookback=_LOOKBACK, batch_size=16
    )

    # Scaler fit on the TRAIN slice only -> train inputs land in [0, 1].
    assert scaler.data_min_ is not None and scaler.data_max_ is not None
    xb, yb = next(iter(train_loader))
    assert xb.shape == (16, _LOOKBACK, _F)
    assert yb.shape == (16, _H, _F)
    assert float(xb.min()) >= 0.0 and float(xb.max()) <= 1.0 + 1e-6
    assert next(iter(val_loader))[0].shape[1:] == (_LOOKBACK, _F)


def test_train_one_fold_finite_separated_losses() -> None:
    pytest.importorskip("torch")
    from src.data.walk_forward import Fold
    from src.predictor.model import PatchTST
    from src.predictor.training import build_fold_loaders, train_one_fold

    feats, ts = _synthetic(300)
    fold = Fold(0, 0, 150, 150, 250, 250, 300)
    scaler, train_loader, val_loader = build_fold_loaders(
        feats, ts, fold, lookback=_LOOKBACK, batch_size=16
    )
    model = PatchTST(lookback=_LOOKBACK)

    metrics = train_one_fold(
        model, scaler, train_loader, val_loader, device="cpu", max_epochs=1
    )

    # Loss components are separated and finite; total strictly positive (variance-floor).
    for value in (
        metrics.train_pinball,
        metrics.train_direction,
        metrics.train_total,
        metrics.val_pinball,
        metrics.val_direction,
        metrics.val_total,
    ):
        assert np.isfinite(value)
    assert metrics.train_total > 0.0
    assert metrics.steps > 0


def test_make_run_tag_includes_all_components() -> None:
    from src.predictor.training import make_run_tag

    tag = make_run_tag(
        git_sha="abcdef1", constants_sha="0011223344", scaler_sha="aabbccddee", fold_id=7
    )
    assert "abcdef1" in tag
    assert "0011223344"[:8] in tag
    assert "aabbccddee"[:8] in tag
    assert "fold7" in tag
