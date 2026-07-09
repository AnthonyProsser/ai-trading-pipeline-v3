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
_OUT_DIM = PREDICTOR.NUM_OUTPUT_DIMS  # idea-05-swinglevels: targets are OHLCV-only
_LOOKBACK = PREDICTOR.PATCH_SIZE * 2  # 32: small but divisible by PATCH_SIZE
_H = PREDICTOR.HORIZON

# Synthetic fold slice sizes derived from the windowing requirement
# (rows >= lookback + horizon) so this file stays green at any HORIZON; the "+ k"
# pads reproduce the window counts of the original H=15 literals.
_TR = _LOOKBACK + _H + 103  # standard train slice: 104 windows
_VA = _LOOKBACK + _H + 53  # standard val slice: 54 windows
_TE = _LOOKBACK + _H + 3  # test slice (never consumed by build_fold_loaders)
_N = _TR + _VA + _TE
_N2 = 2 * _TR + 2 * _VA  # two-fold layout; fold 1's test slice spans _VA rows
_TR_BIG = _LOOKBACK + _H + 203  # throttle test: 204 windows at batch 8
_N_BIG = _TR_BIG + _VA + _TE
_TR_TINY = _LOOKBACK + _H + 3  # 4 train windows < batch 16 with drop_last -> 0 batches
_VA_TINY = _LOOKBACK + _H + 13
_N_TINY = _TR_TINY + _VA_TINY


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

    n = _LOOKBACK + _H + 73  # 73 windows
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

    n = _LOOKBACK + _H + 33  # 33 windows
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

    feats, ts = _synthetic(_N)
    fold = Fold(0, 0, _TR, _TR, _TR + _VA, _TR + _VA, _N)  # tiny hand-made fold
    scaler, train_loader, val_loader = build_fold_loaders(
        feats, ts, fold, lookback=_LOOKBACK, batch_size=16
    )

    # Scaler fit on the TRAIN slice only -> train inputs land in [0, 1].
    assert scaler.data_min_ is not None and scaler.data_max_ is not None
    xb, yb = next(iter(train_loader))
    assert xb.shape == (16, _LOOKBACK, _F)
    assert yb.shape == (16, _H, _OUT_DIM)
    assert float(xb.min()) >= 0.0 and float(xb.max()) <= 1.0 + 1e-6
    assert next(iter(val_loader))[0].shape[1:] == (_LOOKBACK, _F)


def test_train_one_fold_finite_separated_losses() -> None:
    pytest.importorskip("torch")
    from src.data.walk_forward import Fold
    from src.predictor.model import PatchTST
    from src.predictor.training import build_fold_loaders, train_one_fold

    feats, ts = _synthetic(_N)
    fold = Fold(0, 0, _TR, _TR, _TR + _VA, _TR + _VA, _N)
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

    feats, ts = _synthetic(_N)
    fold = Fold(0, 0, _TR, _TR, _TR + _VA, _TR + _VA, _N)
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

    feats, ts = _synthetic(_N_TINY)
    # train slice -> 4 windows < batch 16 with drop_last -> 0 batches.
    fold = Fold(0, 0, _TR_TINY, _TR_TINY, _N_TINY, _N_TINY, _N_TINY)
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

    feats, ts = _synthetic(_N)
    fold = Fold(0, 0, _TR, _TR, _TR + _VA, _TR + _VA, _N)
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

    feats, ts = _synthetic(_N)
    fold = Fold(0, 0, _TR, _TR, _TR + _VA, _TR + _VA, _N)
    scaler, train_loader, val_loader = build_fold_loaders(
        feats, ts, fold, lookback=_LOOKBACK, batch_size=16
    )
    model = PatchTST(lookback=_LOOKBACK)
    # Force a NaN validation loss; it must fail loudly, not be laundered as "no improvement".
    monkeypatch.setattr(
        training_mod,
        "_evaluate",
        lambda *a, **k: (float("nan"), float("nan"), float("nan"), float("nan")),
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

    feats, ts = _synthetic(_N)
    fold = Fold(0, 0, _TR, _TR, _TR + _VA, _TR + _VA, _N)
    scaler, train_loader, val_loader = build_fold_loaders(
        feats, ts, fold, lookback=_LOOKBACK, batch_size=16
    )
    model = PatchTST(lookback=_LOOKBACK)

    # Scripted val_total: epoch 1 is best (1.0), epochs 2-3 strictly worse. Patience (10)
    # is far higher than 2, so all three epochs run and "last" != "best".
    scripted = iter([(0.1, 0.1, 0.0, 1.0), (0.1, 0.1, 0.0, 2.0), (0.1, 0.1, 0.0, 3.0)])
    snapshots: list[dict[str, object]] = []

    def fake_eval(
        m: torch.nn.Module, loader: object, device: object, use_amp: object
    ) -> tuple[float, float, float, float]:
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
    feats, ts = _synthetic(_N)
    fold = Fold(0, 0, _TR, _TR, _TR + _VA, _TR + _VA, _N)
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
    # Target-semantics guard: deploy refuses a checkpoint whose training target
    # convention doesn't match the running code's (predictor_checkpoint_save_format).
    assert loaded["target_semantics"] == PREDICTOR.TARGET_SEMANTICS
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

    feats, ts = _synthetic(_N)
    fold = Fold(0, 0, _TR, _TR, _TR + _VA, _TR + _VA, _N)
    _, _, val_loader = build_fold_loaders(feats, ts, fold, lookback=_LOOKBACK, batch_size=16)
    model = PatchTST(lookback=_LOOKBACK)

    coverage = evaluate_q90_coverage(model, val_loader, device="cpu")
    assert np.isfinite(coverage)
    assert 0.0 <= coverage <= 1.0


def test_evaluate_directional_accuracy_finite_or_nan() -> None:
    """DA is either NaN (nothing tradeable in the fold) or a fraction in [0, 1] — same
    contract as deploy_gates.directional_accuracy, evaluated over a whole loader."""
    pytest.importorskip("torch")
    from src.data.walk_forward import Fold
    from src.predictor.model import PatchTST
    from src.predictor.training import build_fold_loaders, evaluate_directional_accuracy

    feats, ts = _synthetic(_N)
    fold = Fold(0, 0, _TR, _TR, _TR + _VA, _TR + _VA, _N)
    _, _, val_loader = build_fold_loaders(feats, ts, fold, lookback=_LOOKBACK, batch_size=16)
    model = PatchTST(lookback=_LOOKBACK)

    da = evaluate_directional_accuracy(model, val_loader, device="cpu")
    assert np.isnan(da) or (0.0 <= da <= 1.0)


def test_evaluate_directional_accuracy_scores_final_step_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DA collector passes ONLY the final horizon step to
    deploy_gates.directional_accuracy — the move a hold-to-horizon trade actually
    spans — i.e. a (N, DIM, Q)/(N, DIM) slice, not the full (N, H, DIM, Q) path."""
    pytest.importorskip("torch")
    import src.predictor.training as training_mod
    from src.data.walk_forward import Fold
    from src.predictor.model import PatchTST
    from src.predictor.training import build_fold_loaders, evaluate_directional_accuracy

    feats, ts = _synthetic(_N)
    fold = Fold(0, 0, _TR, _TR, _TR + _VA, _TR + _VA, _N)
    _, _, val_loader = build_fold_loaders(feats, ts, fold, lookback=_LOOKBACK, batch_size=16)
    model = PatchTST(lookback=_LOOKBACK)

    seen: dict[str, int] = {}

    def spy(pred: object, target: object, *args: object, **kwargs: object) -> float:
        seen["pred_ndim"] = pred.ndim  # type: ignore[attr-defined]
        seen["target_ndim"] = target.ndim  # type: ignore[attr-defined]
        return 0.5

    monkeypatch.setattr(training_mod, "directional_accuracy", spy)
    evaluate_directional_accuracy(model, val_loader, device="cpu")
    assert seen["pred_ndim"] == 3  # (N, DIM, Q) — final step only
    assert seen["target_ndim"] == 2  # (N, DIM)


def test_train_one_fold_batch_payload_includes_fold_lr_grad_norm() -> None:
    """training-dashboard.md's SSE 'batch' message needs fold/epoch/lr/grad_norm on every
    per-batch log call, and the 'epoch' message needs patience + best_val_total."""
    pytest.importorskip("torch")
    from src.data.walk_forward import Fold
    from src.predictor.model import PatchTST
    from src.predictor.training import build_fold_loaders, train_one_fold

    feats, ts = _synthetic(_N)
    fold = Fold(0, 0, _TR, _TR, _TR + _VA, _TR + _VA, _N)
    scaler, train_loader, val_loader = build_fold_loaders(
        feats, ts, fold, lookback=_LOOKBACK, batch_size=16
    )
    model = PatchTST(lookback=_LOOKBACK)

    batch_payloads: list[dict[str, object]] = []
    epoch_payloads: list[dict[str, object]] = []

    def spy(payload: dict[str, object]) -> None:
        (batch_payloads if payload["split"] == "train" else epoch_payloads).append(payload)

    train_one_fold(
        model, scaler, train_loader, val_loader, device="cpu", max_epochs=1,
        log=spy, fold_index=5,
    )

    assert batch_payloads
    first = batch_payloads[0]
    assert first["fold"] == 5
    assert first["epoch"] == 0
    assert np.isfinite(first["lr"]) and first["lr"] > 0  # type: ignore[arg-type]
    assert np.isfinite(first["grad_norm"]) and first["grad_norm"] >= 0  # type: ignore[arg-type]

    assert epoch_payloads
    ep = epoch_payloads[0]
    assert ep["fold"] == 5
    assert ep["patience"] == 0  # first epoch always "improves" from +inf
    assert np.isfinite(ep["best_val_total"])  # type: ignore[arg-type]
    assert "train_total" in ep and "val_total" in ep


def test_train_one_fold_stop_event_halts_and_flags_stopped() -> None:
    """A pre-set stop_event halts training before any batch runs (checked at the top of
    the batch loop, before the forward/backward pass) — no torn optimizer step."""
    pytest.importorskip("torch")
    import threading

    from src.data.walk_forward import Fold
    from src.predictor.model import PatchTST
    from src.predictor.training import build_fold_loaders, train_one_fold

    feats, ts = _synthetic(_N)
    fold = Fold(0, 0, _TR, _TR, _TR + _VA, _TR + _VA, _N)
    scaler, train_loader, val_loader = build_fold_loaders(
        feats, ts, fold, lookback=_LOOKBACK, batch_size=16
    )
    model = PatchTST(lookback=_LOOKBACK)

    stop_event = threading.Event()
    stop_event.set()

    metrics = train_one_fold(
        model, scaler, train_loader, val_loader, device="cpu", max_epochs=5,
        stop_event=stop_event,
    )
    assert metrics.stopped_by_user is True
    assert metrics.steps == 0


def test_train_one_fold_save_event_triggers_callback_once_then_clears() -> None:
    """A set save_event fires on_save_request exactly once at the next safe batch
    boundary, then clears itself so it doesn't refire every subsequent batch."""
    pytest.importorskip("torch")
    import threading

    from src.data.walk_forward import Fold
    from src.predictor.model import PatchTST
    from src.predictor.training import build_fold_loaders, train_one_fold

    feats, ts = _synthetic(_N)
    fold = Fold(0, 0, _TR, _TR, _TR + _VA, _TR + _VA, _N)
    scaler, train_loader, val_loader = build_fold_loaders(
        feats, ts, fold, lookback=_LOOKBACK, batch_size=16
    )
    model = PatchTST(lookback=_LOOKBACK)

    save_event = threading.Event()
    save_event.set()
    calls = []

    train_one_fold(
        model, scaler, train_loader, val_loader, device="cpu", max_epochs=1,
        save_event=save_event, on_save_request=lambda: calls.append(1),
    )
    assert calls == [1]
    assert save_event.is_set() is False


def test_train_all_folds_walks_folds_and_emits_wire_events(tmp_path: object) -> None:
    """train_all_folds walks folds in order, emits batch/epoch/fold_complete events per
    the SSE schema, and writes a real checkpoint per fold."""
    pytest.importorskip("torch")
    from pathlib import Path

    from src.data.walk_forward import Fold
    from src.predictor.training import train_all_folds

    feats, ts = _synthetic(_N2)
    folds = [
        Fold(0, 0, _TR, _TR, _TR + _VA, _TR + _VA, _N),
        Fold(1, _TR, 2 * _TR, 2 * _TR, 2 * _TR + _VA, 2 * _TR + _VA, _N2),
    ]
    events: list[dict[str, object]] = []

    train_all_folds(
        feats, ts, folds, lookback=_LOOKBACK, device="cpu", batch_size=16, max_epochs=1,
        checkpoint_dir=Path(str(tmp_path)), git_sha="abc1234", constants_sha="deadbeef",
        log=events.append,
    )

    completes = [e for e in events if e["type"] == "fold_complete"]
    assert [e["fold"] for e in completes] == [0, 1]
    for e in completes:
        assert Path(str(e["checkpoint_path"])).exists()
        assert np.isfinite(e["q_coverage"])  # type: ignore[arg-type]
        # fold_complete carries the run-tag stem so a benchmark result can later be
        # joined back to this fold's training record (analysis endpoint).
        assert str(e["stem"]).startswith("abc1234-")
        assert str(e["stem"]).endswith(f"-fold{e['fold']}")

    batches = [e for e in events if e["type"] == "batch"]
    assert batches and all(e["fold"] in (0, 1) for e in batches)
    epochs = [e for e in events if e["type"] == "epoch"]
    assert epochs and all("max_epochs" in e for e in epochs)
    statuses = [e for e in events if e["type"] == "status"]
    assert any(e["state"] == "done" for e in statuses)  # natural run completion
    # "stopped" is reserved for a user-initiated stop; a run that finishes every fold
    # must be distinguishable so the UI can show a terminal "Done" state.
    assert not any(e["state"] == "stopped" for e in statuses)


def test_window_loader_matches_dataset_windowing_and_scaling() -> None:
    """The GPU-resident batch loader must produce the exact same window semantics as the
    reference WindowDataset path: scaled lookback inputs in [0,1] on TRAIN, raw horizon
    targets, drop_last on train, full coverage on val, and the identical window count.
    Speed path must not change the numbers."""
    pytest.importorskip("torch")
    from src.data.walk_forward import Fold
    from src.predictor.training import build_fold_loaders

    feats, ts = _synthetic(_N)
    fold = Fold(0, 0, _TR, _TR, _TR + _VA, _TR + _VA, _N)
    scaler, train_loader, val_loader = build_fold_loaders(
        feats, ts, fold, lookback=_LOOKBACK, batch_size=16
    )

    train_batches = list(iter(train_loader))
    assert len(train_loader) == len(train_batches)
    for xb, yb in train_batches:
        assert xb.shape == (16, _LOOKBACK, _F)
        assert yb.shape == (16, _H, _OUT_DIM)
        assert float(xb.min()) >= 0.0 and float(xb.max()) <= 1.0 + 1e-6

    val_rows = sum(int(xb.shape[0]) for xb, _ in val_loader)
    assert val_rows == len(val_loader.dataset)  # type: ignore[attr-defined]
    assert val_rows > 0


def test_window_loader_drop_last_preserved() -> None:
    """Shuffled train iteration with drop_last must cover exactly floor(N/bs)*bs windows —
    sanity that randperm windowing preserves drop_last semantics."""
    pytest.importorskip("torch")
    from src.data.walk_forward import Fold
    from src.predictor.training import build_fold_loaders

    feats, ts = _synthetic(_N)
    fold = Fold(0, 0, _TR, _TR, _TR + _VA, _TR + _VA, _N)
    _, train_loader, _ = build_fold_loaders(feats, ts, fold, lookback=_LOOKBACK, batch_size=8)
    seen = sum(int(xb.shape[0]) for xb, _ in train_loader)
    n_windows = len(train_loader.dataset)  # type: ignore[attr-defined]
    assert seen == (n_windows // 8) * 8


def test_train_one_fold_throttles_batch_logging() -> None:
    """Per-step logging is throttled to every BATCH_LOG_INTERVAL steps (plus a final
    flush), but the emitted 'batch' payload SCHEMA is unchanged (step/fold/epoch/
    pinball/direction/total/lr/grad_norm all present)."""
    pytest.importorskip("torch")
    from constants import TRAINING_UI
    from src.data.walk_forward import Fold
    from src.predictor.model import PatchTST
    from src.predictor.training import build_fold_loaders, train_one_fold

    feats, ts = _synthetic(_N_BIG)
    fold = Fold(0, 0, _TR_BIG, _TR_BIG, _TR_BIG + _VA, _TR_BIG + _VA, _N_BIG)
    scaler, train_loader, val_loader = build_fold_loaders(
        feats, ts, fold, lookback=_LOOKBACK, batch_size=8
    )
    model = PatchTST(lookback=_LOOKBACK)

    batch_payloads: list[dict[str, object]] = []

    def spy(payload: dict[str, object]) -> None:
        if payload["split"] == "train":
            batch_payloads.append(payload)

    metrics = train_one_fold(
        model, scaler, train_loader, val_loader, device="cpu", max_epochs=1, log=spy
    )

    interval = TRAINING_UI.BATCH_LOG_INTERVAL
    assert interval > 1  # actually throttling
    assert 0 < len(batch_payloads) <= metrics.steps
    assert len(batch_payloads) <= metrics.steps // interval + 2
    for key in ("step", "fold", "epoch", "pinball", "direction", "total", "lr", "grad_norm"):
        assert key in batch_payloads[0]
    logged_steps = [int(p["step"]) for p in batch_payloads]  # type: ignore[arg-type]
    assert logged_steps == sorted(logged_steps)


def test_lr_decays_on_val_plateau(monkeypatch: pytest.MonkeyPatch) -> None:
    """The LR schedule is warmup -> constant -> plateau decay: after LR_PLATEAU_PATIENCE
    consecutive non-improving validation epochs, the LR is multiplied by
    LR_PLATEAU_FACTOR (composing with — not fighting — the EarlyStopper, unlike a cosine
    schedule pinned to a MAX_EPOCHS horizon that early stopping never lets anneal)."""
    pytest.importorskip("torch")
    import src.predictor.training as training_mod
    from src.data.walk_forward import Fold
    from src.predictor.model import PatchTST
    from src.predictor.training import build_fold_loaders, train_one_fold

    feats, ts = _synthetic(_N)
    fold = Fold(0, 0, _TR, _TR, _TR + _VA, _TR + _VA, _N)
    scaler, train_loader, val_loader = build_fold_loaders(
        feats, ts, fold, lookback=_LOOKBACK, batch_size=16
    )
    model = PatchTST(lookback=_LOOKBACK)

    # Constant val loss: epoch 1 improves from +inf, every later epoch is a plateau.
    monkeypatch.setattr(training_mod, "_evaluate", lambda *a, **k: (0.1, 0.1, 0.0, 1.0))

    lrs: list[float] = []

    def spy(payload: dict[str, object]) -> None:
        if payload["split"] == "train":
            lrs.append(float(payload["lr"]))  # type: ignore[arg-type]

    max_epochs = PREDICTOR.LR_PLATEAU_PATIENCE + 3  # enough plateau epochs to decay once
    assert max_epochs < PREDICTOR.EARLY_STOPPING_PATIENCE  # early stop must not fire first
    train_one_fold(
        model, scaler, train_loader, val_loader, device="cpu", max_epochs=max_epochs, log=spy
    )

    peak = max(lrs)
    assert peak == pytest.approx(PREDICTOR.LEARNING_RATE)  # warmup reaches the base LR
    assert lrs[-1] == pytest.approx(peak * PREDICTOR.LR_PLATEAU_FACTOR)  # decayed once


def test_prod_batch_size_constants_present() -> None:
    """Production training batch size is a distinct frozen constant, not the smoke value."""
    assert PREDICTOR.PROD_BATCH_SIZE > PREDICTOR.SMOKE_BATCH_SIZE
    assert PREDICTOR.PROD_BATCH_SIZE_FALLBACK < PREDICTOR.PROD_BATCH_SIZE


def test_train_all_folds_stops_early_when_stop_event_set(tmp_path: object) -> None:
    """Setting stop_event after fold 0 completes prevents fold 1 from ever starting —
    no garbage fold_complete for an aborted fold."""
    pytest.importorskip("torch")
    import threading
    from pathlib import Path

    from src.data.walk_forward import Fold
    from src.predictor.training import train_all_folds

    feats, ts = _synthetic(_N2)
    folds = [
        Fold(0, 0, _TR, _TR, _TR + _VA, _TR + _VA, _N),
        Fold(1, _TR, 2 * _TR, 2 * _TR, 2 * _TR + _VA, 2 * _TR + _VA, _N2),
    ]
    stop_event = threading.Event()
    events: list[dict[str, object]] = []

    def log(payload: dict[str, object]) -> None:
        events.append(payload)
        if payload["type"] == "fold_complete":
            stop_event.set()

    train_all_folds(
        feats, ts, folds, lookback=_LOOKBACK, device="cpu", batch_size=16, max_epochs=1,
        checkpoint_dir=Path(str(tmp_path)), git_sha="abc1234", constants_sha="deadbeef",
        log=log, stop_event=stop_event,
    )

    completes = [e for e in events if e["type"] == "fold_complete"]
    assert [e["fold"] for e in completes] == [0]
    statuses = [e for e in events if e["type"] == "status"]
    assert any(e["state"] == "stopped" for e in statuses)
    assert not any(e["state"] == "done" for e in statuses)  # user stop is never "done"


def test_train_all_folds_promotes_finished_checkpoints_on_completion(
    tmp_path: object,
) -> None:
    """A full run (all folds to natural end) copies every fold's gate-evaluated
    checkpoint + scaler into finished_dir and emits a terminal state:'done'."""
    pytest.importorskip("torch")
    from pathlib import Path

    from src.data.walk_forward import Fold
    from src.predictor.training import train_all_folds

    feats, ts = _synthetic(_N2)
    folds = [
        Fold(0, 0, _TR, _TR, _TR + _VA, _TR + _VA, _N),
        Fold(1, _TR, 2 * _TR, 2 * _TR, 2 * _TR + _VA, 2 * _TR + _VA, _N2),
    ]
    ckpt_dir = Path(str(tmp_path)) / "checkpoints"
    finished_dir = Path(str(tmp_path)) / "finished"
    events: list[dict[str, object]] = []

    train_all_folds(
        feats, ts, folds, lookback=_LOOKBACK, device="cpu", batch_size=16, max_epochs=1,
        checkpoint_dir=ckpt_dir, git_sha="abc1234", constants_sha="deadbeef",
        log=events.append, finished_dir=finished_dir,
    )

    assert len(list(finished_dir.glob(f"*{PREDICTOR.CHECKPOINT_WEIGHTS_SUFFIX}"))) == 2
    assert len(list(finished_dir.glob(f"*{PREDICTOR.CHECKPOINT_SCALER_SUFFIX}"))) == 2
    done = [e for e in events if e.get("state") == "done"]
    assert done and done[-1]["promoted"] == 2


def test_train_all_folds_does_not_promote_on_user_stop(tmp_path: object) -> None:
    """A run stopped before natural completion promotes nothing — the finished dir is
    for models that actually finished training."""
    pytest.importorskip("torch")
    import threading
    from pathlib import Path

    from src.data.walk_forward import Fold
    from src.predictor.training import train_all_folds

    feats, ts = _synthetic(_N2)
    folds = [
        Fold(0, 0, _TR, _TR, _TR + _VA, _TR + _VA, _N),
        Fold(1, _TR, 2 * _TR, 2 * _TR, 2 * _TR + _VA, 2 * _TR + _VA, _N2),
    ]
    ckpt_dir = Path(str(tmp_path)) / "checkpoints"
    finished_dir = Path(str(tmp_path)) / "finished"
    stop_event = threading.Event()
    events: list[dict[str, object]] = []

    def log(payload: dict[str, object]) -> None:
        events.append(payload)
        if payload["type"] == "fold_complete":
            stop_event.set()  # stop after fold 0 -> fold 1 never starts

    train_all_folds(
        feats, ts, folds, lookback=_LOOKBACK, device="cpu", batch_size=16, max_epochs=1,
        checkpoint_dir=ckpt_dir, git_sha="abc1234", constants_sha="deadbeef",
        log=log, stop_event=stop_event, finished_dir=finished_dir,
    )

    assert not finished_dir.exists() or not list(finished_dir.iterdir())
    assert not any(e.get("state") == "done" for e in events)
    assert any(e.get("state") == "stopped" for e in events)


def test_train_all_folds_partial_promotion_warns_but_still_completes(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy failure during promotion (e.g. a Windows AV lock) must not sink the run:
    it warns with the accurate partial count and still emits the terminal "done"
    state, rather than surfacing as a bare exception."""
    pytest.importorskip("torch")
    from pathlib import Path

    import src.predictor.training as training_mod
    from src.data.walk_forward import Fold

    feats, ts = _synthetic(_N2)
    folds = [Fold(0, 0, _TR, _TR, _TR + _VA, _TR + _VA, _N)]
    finished_dir = Path(str(tmp_path)) / "finished"
    events: list[dict[str, object]] = []

    def boom(src: object, dst: object) -> None:
        raise OSError("checkpoint locked by another process")

    monkeypatch.setattr(training_mod.shutil, "copy2", boom)
    training_mod.train_all_folds(
        feats, ts, folds, lookback=_LOOKBACK, device="cpu", batch_size=16, max_epochs=1,
        checkpoint_dir=Path(str(tmp_path)) / "checkpoints", git_sha="abc1234",
        constants_sha="deadbeef", log=events.append, finished_dir=finished_dir,
    )

    assert any(
        e.get("type") == "alert" and e.get("level") == "warn"
        and "Promotion incomplete" in str(e.get("message"))
        for e in events
    )
    done = [e for e in events if e.get("state") == "done"]
    assert done and done[-1]["promoted"] == 0
