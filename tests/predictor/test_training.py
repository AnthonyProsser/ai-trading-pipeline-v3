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

import copy

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


def test_make_run_tag_orders_components_per_spec() -> None:
    # predictor-training.md: "Run tag = git SHA + scaler hash + constants.py hash + fold ID".
    from src.predictor.training import make_run_tag

    tag = make_run_tag(
        git_sha="abcdef1", constants_sha="0011223344", scaler_sha="aabbccddee", fold_id=7
    )
    assert tag.startswith("abcdef1")
    assert "aabbccdd" in tag and "00112233" in tag
    assert tag.index("aabbccdd") < tag.index("00112233")  # scaler hash precedes constants hash
    assert tag.endswith("fold7")


def test_train_one_fold_rejects_non_positive_epochs() -> None:
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
    with pytest.raises(ValueError, match="max_epochs"):
        train_one_fold(model, scaler, train_loader, val_loader, device="cpu", max_epochs=0)


def test_window_dataset_too_short_raises() -> None:
    torch = pytest.importorskip("torch")
    from src.predictor.training import WindowDataset

    n = _LOOKBACK + _H - 1  # one row short of a single window
    x = torch.zeros((n, _F))
    with pytest.raises(ValueError, match="shorter than"):
        WindowDataset(x, x, lookback=_LOOKBACK, horizon=_H)


def test_train_one_fold_raises_on_empty_train_loader() -> None:
    pytest.importorskip("torch")
    from src.data.walk_forward import Fold
    from src.predictor.model import PatchTST
    from src.predictor.training import build_fold_loaders, train_one_fold

    feats, ts = _synthetic(110)
    # train slice -> 4 windows < batch 16 with drop_last -> 0 batches.
    fold = Fold(0, 0, 50, 50, 110, 110, 110)
    scaler, train_loader, val_loader = build_fold_loaders(
        feats, ts, fold, lookback=_LOOKBACK, batch_size=16
    )
    assert len(train_loader) == 0
    model = PatchTST(lookback=_LOOKBACK)
    with pytest.raises(RuntimeError, match="no batches"):
        train_one_fold(model, scaler, train_loader, val_loader, device="cpu", max_epochs=1)


def test_train_one_fold_max_steps_early_exit() -> None:
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
        model, scaler, train_loader, val_loader, device="cpu", max_epochs=5, max_steps=3
    )
    assert metrics.steps == 3


def test_train_one_fold_raises_on_nonfinite_val(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    import src.predictor.training as training_mod
    from src.data.walk_forward import Fold
    from src.predictor.model import PatchTST
    from src.predictor.training import build_fold_loaders, train_one_fold

    feats, ts = _synthetic(300)
    fold = Fold(0, 0, 150, 150, 250, 250, 300)
    scaler, train_loader, val_loader = build_fold_loaders(
        feats, ts, fold, lookback=_LOOKBACK, batch_size=16
    )
    model = PatchTST(lookback=_LOOKBACK)
    # Force a NaN validation loss; it must fail loudly, not be laundered as "no improvement".
    monkeypatch.setattr(
        training_mod, "_evaluate", lambda *a, **k: (float("nan"), float("nan"), float("nan"))
    )
    with pytest.raises(ValueError, match="non-finite validation loss"):
        train_one_fold(model, scaler, train_loader, val_loader, device="cpu", max_epochs=1)


def test_train_one_fold_restores_best_val_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Save policy = best-by-val-total: after training, the model holds the weights from
    the epoch with the lowest val_total, NOT the last epoch's weights."""
    torch = pytest.importorskip("torch")
    import src.predictor.training as training_mod
    from src.data.walk_forward import Fold
    from src.predictor.model import PatchTST
    from src.predictor.training import build_fold_loaders, train_one_fold

    feats, ts = _synthetic(300)
    fold = Fold(0, 0, 150, 150, 250, 250, 300)
    scaler, train_loader, val_loader = build_fold_loaders(
        feats, ts, fold, lookback=_LOOKBACK, batch_size=16
    )
    model = PatchTST(lookback=_LOOKBACK)

    # Scripted val_total: epoch 1 is best (1.0), epochs 2-3 strictly worse. Patience (10)
    # is far higher than 2, so all three epochs run and "last" != "best".
    scripted = iter([(0.1, 0.1, 1.0), (0.1, 0.1, 2.0), (0.1, 0.1, 3.0)])
    snapshots: list[dict[str, object]] = []

    def fake_eval(
        m: torch.nn.Module, loader: object, device: object, use_amp: object
    ) -> tuple[float, float, float]:
        snapshots.append(copy.deepcopy(m.state_dict()))
        return next(scripted)

    monkeypatch.setattr(training_mod, "_evaluate", fake_eval)
    train_one_fold(model, scaler, train_loader, val_loader, device="cpu", max_epochs=3)

    final = model.state_dict()
    best = snapshots[0]  # epoch-1 weights = lowest val_total
    last = snapshots[-1]  # epoch-3 weights
    assert all(torch.equal(final[k], best[k]) for k in final)  # restored to best
    assert any(not torch.equal(final[k], last[k]) for k in final)  # not the last epoch


def test_save_checkpoint_round_trips(tmp_path: object) -> None:
    """save_checkpoint writes run_tag-based weights (.pt) + scaler (.scaler.pkl) in the
    exact format deploy_predictor.py reads: weights load under weights_only=True with a
    "state_dict" key + provenance metadata; the scaler unpickles to equal fold stats."""
    import pickle
    from pathlib import Path

    torch = pytest.importorskip("torch")
    from src.data.walk_forward import Fold
    from src.predictor.model import PatchTST
    from src.predictor.training import build_fold_loaders, save_checkpoint

    out_dir = Path(str(tmp_path))
    feats, ts = _synthetic(300)
    fold = Fold(0, 0, 150, 150, 250, 250, 300)
    scaler, _, _ = build_fold_loaders(feats, ts, fold, lookback=_LOOKBACK, batch_size=16)
    model = PatchTST(lookback=_LOOKBACK)

    weights_path, scaler_path = save_checkpoint(
        model,
        scaler,
        out_dir,
        "deadbeef-saaaaaaaa-cbbbbbbbb-fold0",
        lookback=_LOOKBACK,
        constants_sha256="cafef00d",
        trained_through_ts_utc="2020-01-01T00:00:00Z",
        train_q90_coverage=0.83,
    )

    assert weights_path == out_dir / "deadbeef-saaaaaaaa-cbbbbbbbb-fold0.pt"
    assert scaler_path == out_dir / "deadbeef-saaaaaaaa-cbbbbbbbb-fold0.scaler.pkl"

    # weights_only=True is what deploy_predictor.py uses; the wrapper must survive it.
    loaded = torch.load(weights_path, map_location="cpu", weights_only=True)
    assert loaded["lookback"] == _LOOKBACK
    assert loaded["constants_sha256"] == "cafef00d"
    assert loaded["trained_through_ts_utc"] == "2020-01-01T00:00:00Z"
    assert loaded["train_q90_coverage"] == 0.83  # deploy gate (a) baseline, embedded
    PatchTST(lookback=_LOOKBACK).load_state_dict(loaded["state_dict"])  # state_dict is loadable

    with open(scaler_path, "rb") as fh:
        restored = pickle.load(fh)
    assert np.array_equal(restored.data_min_, scaler.data_min_)
    assert np.array_equal(restored.data_max_, scaler.data_max_)


def test_evaluate_q90_coverage_finite_in_range() -> None:
    """The training-time q90 coverage baseline (embedded in the checkpoint, computed on
    the VAL split) is a finite fraction in [0, 1], computed via deploy_gates.q90_coverage."""
    pytest.importorskip("torch")
    from src.data.walk_forward import Fold
    from src.predictor.model import PatchTST
    from src.predictor.training import build_fold_loaders, evaluate_q90_coverage

    feats, ts = _synthetic(300)
    fold = Fold(0, 0, 150, 150, 250, 250, 300)
    _, _, val_loader = build_fold_loaders(feats, ts, fold, lookback=_LOOKBACK, batch_size=16)
    model = PatchTST(lookback=_LOOKBACK)

    coverage = evaluate_q90_coverage(model, val_loader, device="cpu")
    assert np.isfinite(coverage)
    assert 0.0 <= coverage <= 1.0
