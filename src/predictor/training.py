"""Reusable predictor training logic (predictor-training.md §"Smoke run" /
§"Training data").

Lives under src/ (not the train script) so it is unit-testable and importable;
scripts/train_predictor.py is a thin CLI over it, honouring the
src-never-imports-scripts rule.

Data convention: the model consumes per-fold MinMax-**scaled** inputs but predicts
**raw** log-returns, because the loss's FEE_THRESHOLD term lives in raw log-return
units. So WindowDataset pairs a scaled lookback window with a raw horizon target.
build_fold_loaders fits the scaler on the fold's TRAIN slice only; train_one_fold
*receives* that scaler and never builds one. All numeric settings come from
constants.py.
"""
from __future__ import annotations

import copy
import hashlib
import math
import pickle
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset

from constants import PREDICTOR, TRAINING_UI
from src.data.scaler import PerFoldMinMaxScaler
from src.data.walk_forward import Fold
from src.predictor.deploy_gates import directional_accuracy, q90_coverage
from src.predictor.early_stopping import EarlyStopper
from src.predictor.loss import predictor_loss
from src.predictor.model import PatchTST
from src.predictor.rollout import enforce_geometry

LogFn = Callable[[dict[str, object]], None]
Loaders = tuple[
    PerFoldMinMaxScaler,
    "DataLoader[tuple[torch.Tensor, torch.Tensor]]",
    "DataLoader[tuple[torch.Tensor, torch.Tensor]]",
]


class FoldMetrics(NamedTuple):
    """Final per-fold losses (components kept separate per the smoke-run spec)."""

    train_pinball: float
    train_direction: float
    train_total: float
    val_pinball: float
    val_direction: float
    val_total: float
    steps: int
    epochs: int
    stopped_early: bool
    stopped_by_user: bool = False


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sliding windows: scaled (lookback, F) input + raw (horizon, F) log-return target."""

    def __init__(
        self, x_scaled: torch.Tensor, y_raw: torch.Tensor, lookback: int, horizon: int
    ) -> None:
        if x_scaled.shape[0] != y_raw.shape[0]:
            raise ValueError("x_scaled and y_raw must have the same number of rows")
        self._x = x_scaled
        self._y = y_raw
        self._lookback = lookback
        self._horizon = horizon
        self._n = x_scaled.shape[0] - lookback - horizon + 1
        if self._n <= 0:
            raise ValueError(
                f"slice of {x_scaled.shape[0]} rows is shorter than "
                f"lookback({lookback}) + horizon({horizon})"
            )

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._x[index : index + self._lookback]
        y = self._y[index + self._lookback : index + self._lookback + self._horizon]
        return x, y


def _to_f32(array: npt.NDArray[np.float64]) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def build_fold_loaders(
    features: npt.NDArray[np.float64],
    timestamps: npt.NDArray[np.datetime64],
    fold: Fold,
    *,
    lookback: int,
    batch_size: int,
) -> Loaders:
    """Fit a per-fold scaler on TRAIN, return (scaler, train_loader, val_loader).

    Inputs are scaled; targets stay raw. The scaler's fold window is [train_start,
    val_end), so its own assertion rejects any transform that strays outside the fold.
    """
    ts_m = timestamps.astype("datetime64[m]")
    scaler = PerFoldMinMaxScaler(fold_start=ts_m[fold.train_start], fold_end=ts_m[fold.val_end - 1])
    train_feats = features[fold.train_start : fold.train_end]
    val_feats = features[fold.val_start : fold.val_end]
    scaler.fit(train_feats)
    x_train = scaler.transform(train_feats, timestamps[fold.train_start : fold.train_end])
    x_val = scaler.transform(val_feats, timestamps[fold.val_start : fold.val_end])

    train_ds = WindowDataset(_to_f32(x_train), _to_f32(train_feats), lookback, PREDICTOR.HORIZON)
    val_ds = WindowDataset(_to_f32(x_val), _to_f32(val_feats), lookback, PREDICTOR.HORIZON)
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return scaler, train_loader, val_loader


def _warmup_cosine(
    optimizer: torch.optim.Optimizer, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(1, int(PREDICTOR.WARMUP_FRAC * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = float(step - warmup) / float(max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, float]:
    model.eval()
    pinball = direction = total = 0.0
    batches = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                comp = predictor_loss(model(x), y)
            pinball += float(comp.pinball)
            direction += float(comp.direction)
            total += float(comp.total)
            batches += 1
    if batches == 0:
        raise RuntimeError("val_loader yielded no batches — fold or batch_size misconfigured")
    return pinball / batches, direction / batches, total / batches


def train_one_fold(
    model: torch.nn.Module,
    scaler: PerFoldMinMaxScaler,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: str = "cpu",
    max_epochs: int = PREDICTOR.MAX_EPOCHS,
    log: LogFn | None = None,
    max_steps: int | None = None,
    fold_index: int = 0,
    stop_event: threading.Event | None = None,
    save_event: threading.Event | None = None,
    on_save_request: Callable[[], None] | None = None,
) -> FoldMetrics:
    """Train one fold on the supplied loaders. Receives the (already-fitted) per-fold
    scaler -- it never builds one. Uses AdamW + warmup-cosine, AMP (bf16) on CUDA,
    grad clipping, predictor_loss, and EarlyStopper; raises on a non-finite loss.

    `stop_event`/`save_event` (training-dashboard.md training UI controls) are polled at
    the top of each batch iteration -- a safe boundary between complete optimizer steps,
    so a snapshot/interrupt never observes a torn (mid-`.backward()`/`.step()`) tensor.
    A set `stop_event` halts immediately (`stopped_by_user=True` on the returned
    `FoldMetrics`); a set `save_event` fires `on_save_request()` once, then self-clears.
    """
    if max_epochs < 1:
        raise ValueError(f"max_epochs must be >= 1, got {max_epochs}")
    if scaler.data_min_ is None:
        raise ValueError(
            "train_one_fold received an unfitted scaler; build_fold_loaders must fit it"
        )

    dev = torch.device(device)
    model.to(dev)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PREDICTOR.LEARNING_RATE, weight_decay=PREDICTOR.WEIGHT_DECAY
    )
    total_steps = max(1, max_epochs * max(1, len(train_loader)))
    scheduler = _warmup_cosine(optimizer, total_steps)
    stopper = EarlyStopper(PREDICTOR.EARLY_STOPPING_PATIENCE)
    use_amp = PREDICTOR.USE_AMP and dev.type == "cuda"
    if len(train_loader) == 0:
        raise RuntimeError("train_loader yielded no batches — fold or batch_size misconfigured")

    step = 0
    epochs_run = 0
    stopped = False
    stopped_by_user = False
    train_pin = train_dir = train_tot = math.nan
    val_pin = val_dir = val_tot = math.nan
    # Save policy = best-by-val-total: retain the lowest-val-loss weights so the saved
    # checkpoint is the best epoch, not the last (early stopping leaves ~patience epochs
    # of drift past the minimum). DECISIONS.md §Predictor predictor_checkpoint_save_policy.
    best_val = math.inf
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        model.train()
        ep_pin = ep_dir = ep_tot = 0.0
        ep_batches = 0
        for x, y in train_loader:
            if stop_event is not None and stop_event.is_set():
                stopped_by_user = True
                break
            if save_event is not None and save_event.is_set():
                if on_save_request is not None:
                    on_save_request()
                save_event.clear()
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
                comp = predictor_loss(model(x), y)
            if not bool(torch.isfinite(comp.total)):
                raise ValueError(
                    f"non-finite training loss at step {step}: total={float(comp.total)}"
                )
            comp.total.backward()  # type: ignore[no-untyped-call]  # torch stubs untyped
            # clip_grad_norm_ returns the PRE-clip total norm -- exactly the "is clipping
            # engaging" signal the loss-component strip's grad-norm coloring needs.
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), PREDICTOR.GRAD_CLIP_NORM)
            )
            optimizer.step()
            scheduler.step()
            step += 1
            lr = float(scheduler.get_last_lr()[0])
            batch_pin, batch_dir, batch_tot = (
                float(comp.pinball),
                float(comp.direction),
                float(comp.total),
            )
            ep_pin += batch_pin
            ep_dir += batch_dir
            ep_tot += batch_tot
            ep_batches += 1
            if log is not None:
                payload: dict[str, object] = {
                    "split": "train",
                    "step": step,
                    "fold": fold_index,
                    "epoch": epoch,
                    "pinball": batch_pin,
                    "direction": batch_dir,
                    "total": batch_tot,
                    "lr": lr,
                    "grad_norm": grad_norm,
                }
                log(payload)
            if max_steps is not None and step >= max_steps:
                break

        if stopped_by_user:
            break

        # Report the epoch-averaged train loss (not the last batch) so it is comparable
        # with the epoch-averaged val loss from _evaluate.
        if ep_batches > 0:
            train_pin, train_dir, train_tot = (
                ep_pin / ep_batches,
                ep_dir / ep_batches,
                ep_tot / ep_batches,
            )

        val_pin, val_dir, val_tot = _evaluate(model, val_loader, dev, use_amp)
        if not math.isfinite(val_tot):
            raise ValueError(f"non-finite validation loss at epoch {epochs_run}: total={val_tot}")
        if val_tot < best_val:
            best_val = val_tot
            best_state = copy.deepcopy(model.state_dict())
        if log is not None:
            log(
                {
                    "split": "val",
                    "step": step,
                    "fold": fold_index,
                    "epoch": epoch,
                    "train_pinball": train_pin,
                    "train_direction": train_dir,
                    "train_total": train_tot,
                    "val_pinball": val_pin,
                    "val_direction": val_dir,
                    "val_total": val_tot,
                    "patience": stopper.num_bad,
                    "best_val_total": best_val,
                }
            )
        if stopper.step(val_tot):
            stopped = True
            break
        if max_steps is not None and step >= max_steps:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return FoldMetrics(
        train_pin, train_dir, train_tot, val_pin, val_dir, val_tot, step, epochs_run,
        stopped, stopped_by_user,
    )


def _collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `model` over every batch in `loader` with geometry enforcement, concatenated
    to CPU. Shared by evaluate_q90_coverage / evaluate_directional_accuracy so both
    metrics -- and any future gate metric -- read predictions the identical way deploy
    does (predictor-contract.md geometry enforcement at every inference step)."""
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            # preds -> CPU to match the raw targets (y stays on CPU); metrics compare there.
            preds.append(enforce_geometry(model(x.to(device))).cpu())
            targets.append(y)
    if not preds:
        raise RuntimeError("loader yielded no batches — cannot evaluate predictions")
    return torch.cat(preds), torch.cat(targets)


def evaluate_q90_coverage(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: str = "cpu",
) -> float:
    """Empirical q90 coverage over `loader`, computed exactly as the deploy gate does
    (geometry-enforced predictions, Close dim via `deploy_gates.q90_coverage`).

    Used to capture the training-time baseline for deploy gate (a) on the held-out VAL
    split, using the best-by-val-total weights. Embedding this number in the checkpoint
    lets `deploy_predictor.py` read it instead of taking a hand-typed `--train-coverage`.

    Side-effect: moves `model` to `device` in-place (same convention as train_one_fold).
    Callers pass the device the model already trains on, so this is a no-op in practice.
    """
    dev = torch.device(device)
    model.to(dev)
    preds, targets = _collect_predictions(model, loader, dev)
    return q90_coverage(preds, targets)


def evaluate_directional_accuracy(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: str = "cpu",
) -> float:
    """Directional accuracy over `loader` (deploy_gates.directional_accuracy: Close dim,
    sign agreement on tradeable |q50| > FEE_THRESHOLD predictions only). Feeds the
    training UI's fold-history DA column and the training_metrics.json export -- the
    same computation deploy_predictor.py runs on the locked test set, run here on a
    fold's VAL split as a live-training health signal. NaN when nothing in the fold is
    tradeable (see deploy_gates.directional_accuracy)."""
    dev = torch.device(device)
    model.to(dev)
    preds, targets = _collect_predictions(model, loader, dev)
    return directional_accuracy(preds, targets)


def scaler_sha(scaler: PerFoldMinMaxScaler) -> str:
    """Stable hash of the fitted scaler's fold statistics (part of the run tag).

    Hashes the raw array buffers (not pickle) so the tag is reproducible across
    Python versions. Lives here (not scripts/train_predictor.py) so train_all_folds
    can build run tags too, per the src-never-imports-scripts rule.
    """
    if scaler.data_min_ is None or scaler.data_max_ is None:
        raise ValueError("scaler_sha requires a fitted scaler")
    return hashlib.sha256(scaler.data_min_.tobytes() + scaler.data_max_.tobytes()).hexdigest()


def make_run_tag(*, git_sha: str, constants_sha: str, scaler_sha: str, fold_id: int) -> str:
    """W&B run tag = git SHA + scaler hash + constants.py hash + fold id (spec order)."""
    return f"{git_sha}-s{scaler_sha[:8]}-c{constants_sha[:8]}-fold{fold_id}"


def save_checkpoint(
    model: torch.nn.Module,
    scaler: PerFoldMinMaxScaler,
    out_dir: Path,
    run_tag: str,
    *,
    lookback: int,
    constants_sha256: str,
    trained_through_ts_utc: str,
    train_q90_coverage: float,
) -> tuple[Path, Path]:
    """Persist trained weights + fitted scaler under run_tag-based filenames.

    Weights are a wrapped dict — the CPU state_dict plus provenance (lookback, the
    constants.py hash, the ISO8601-UTC timestamp the model was trained through, and the
    training-time q90 coverage baseline for deploy gate (a)) — saved via torch.save so it
    loads cleanly under deploy_predictor.py's weights_only=True. The scaler is pickled
    (deploy reads it with pickle.load). The SHA256 manifest over both files +
    constants.py is built later by deploy_predictor. Returns (weights_path, scaler_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / f"{run_tag}{PREDICTOR.CHECKPOINT_WEIGHTS_SUFFIX}"
    scaler_path = out_dir / f"{run_tag}{PREDICTOR.CHECKPOINT_SCALER_SUFFIX}"
    cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(
        {
            "state_dict": cpu_state,
            "lookback": lookback,
            "constants_sha256": constants_sha256,
            "trained_through_ts_utc": trained_through_ts_utc,
            "train_q90_coverage": train_q90_coverage,
        },
        weights_path,
    )
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)
    return weights_path, scaler_path


def _trained_through_ts_utc(timestamps: npt.NDArray[np.datetime64], fold: Fold) -> str:
    """Last TRAIN candle, ISO8601 UTC -- matches scripts/train_predictor.py's inline
    computation exactly (predictor_checkpoint_save_format)."""
    return (
        str(np.datetime_as_string(timestamps[fold.train_end - 1].astype("datetime64[s]"), unit="s"))
        + "Z"
    )


def train_all_folds(
    features: npt.NDArray[np.float64],
    timestamps: npt.NDArray[np.datetime64],
    folds: Sequence[Fold],
    *,
    lookback: int,
    batch_size: int,
    device: str = "cpu",
    max_epochs: int = PREDICTOR.MAX_EPOCHS,
    checkpoint_dir: Path,
    git_sha: str,
    constants_sha: str,
    log: LogFn | None = None,
    stop_event: threading.Event | None = None,
    save_event: threading.Event | None = None,
) -> list[FoldMetrics]:
    """Walk-forward driver: trains `folds` in order, one fresh PatchTST per fold (no
    warm-start across folds -- matches the existing single-fold train_predictor.py
    behavior). Emits SSE-wire-schema events (training-dashboard.md "Live metric
    stream") via `log`: "batch" per training step, "epoch" per epoch-end (with ETAs),
    "fold_complete" + "alert" after each fold, "status" on run stop/completion.

    Designed to run inside a background thread (the training UI's in-process model):
    `stop_event` is checked before each fold starts AND inside train_one_fold's batch
    loop (so a stop mid-fold still exits promptly); `save_event` is forwarded into
    train_one_fold so "Save" checkpoints the live model without pausing training.
    A checkpoint is always written for a fold that produced any weights, including one
    interrupted by stop_event, matching the "stop = graceful checkpoint save" spec.

    Returns the FoldMetrics for every fold that actually ran (empty if stopped before
    the first fold started).
    """
    results: list[FoldMetrics] = []
    total_folds = len(folds)
    # Exponential moving average of completed-fold durations (training-dashboard.md
    # §"F — ETA strip": "TOTAL ETA uses exponential smoothing over recent fold
    # durations"). None until the first fold completes.
    fold_duration_ema: float | None = None

    def emit(payload: dict[str, object]) -> None:
        if log is not None:
            log(payload)

    for fold in folds:
        if stop_event is not None and stop_event.is_set():
            emit({"type": "status", "state": "stopped"})
            return results

        fold_start = time.monotonic()
        model = PatchTST(lookback=lookback)
        scaler, train_loader, val_loader = build_fold_loaders(
            features, timestamps, fold, lookback=lookback, batch_size=batch_size
        )
        run_tag = make_run_tag(
            git_sha=git_sha, constants_sha=constants_sha,
            scaler_sha=scaler_sha(scaler), fold_id=fold.index,
        )

        def on_save_request(
            _model: torch.nn.Module = model,
            _scaler: PerFoldMinMaxScaler = scaler,
            _run_tag: str = run_tag,
            _fold: Fold = fold,
        ) -> None:
            # Ad-hoc mid-fold save: no fresh q90-coverage pass over val_loader from
            # inside the hot training loop (expensive, and this checkpoint is a manual
            # snapshot, not the gate-evaluated artifact written at natural fold
            # completion below) -- train_q90_coverage is left NaN, consistent with
            # "this is not a deploy candidate" rather than a fabricated number.
            weights_path, _ = save_checkpoint(
                _model, _scaler, checkpoint_dir, _run_tag,
                lookback=lookback, constants_sha256=constants_sha,
                trained_through_ts_utc=_trained_through_ts_utc(timestamps, _fold),
                train_q90_coverage=float("nan"),
            )
            emit(
                {"type": "alert", "level": "save", "message": f"Checkpoint saved — {weights_path}"}
            )

        def wrapped_log(
            payload: dict[str, object], _fold: Fold = fold, _fold_start: float = fold_start,
            _fold_duration_ema: float | None = fold_duration_ema,
        ) -> None:
            if payload["split"] == "train":
                emit({**payload, "type": "batch", "total_folds": total_folds})
                return
            epoch_value = payload["epoch"]
            assert isinstance(epoch_value, int)
            epoch = epoch_value
            elapsed = time.monotonic() - _fold_start
            per_epoch_s = elapsed / max(1, epoch + 1)
            remaining_this_fold = max(0, max_epochs - (epoch + 1))
            fold_eta_s = per_epoch_s * remaining_this_fold
            remaining_folds = total_folds - _fold.index - 1
            total_eta_s = (
                fold_eta_s + _fold_duration_ema * remaining_folds
                if _fold_duration_ema is not None else None
            )
            enough = (epoch + 1) >= TRAINING_UI.ETA_MIN_EPOCHS_FOR_ESTIMATE
            emit({
                **payload,
                "type": "epoch",
                "max_epochs": max_epochs,
                "total_folds": total_folds,
                "max_patience": PREDICTOR.EARLY_STOPPING_PATIENCE,
                "epoch_eta_s": per_epoch_s if enough else None,
                "fold_eta_s": fold_eta_s if enough else None,
                "total_eta_s": total_eta_s if enough else None,
            })

        metrics = train_one_fold(
            model, scaler, train_loader, val_loader, device=device, max_epochs=max_epochs,
            log=wrapped_log, fold_index=fold.index, stop_event=stop_event,
            save_event=save_event, on_save_request=on_save_request,
        )

        q_cov = evaluate_q90_coverage(model, val_loader, device=device)
        da = evaluate_directional_accuracy(model, val_loader, device=device)
        weights_path, _ = save_checkpoint(
            model, scaler, checkpoint_dir, run_tag,
            lookback=lookback, constants_sha256=constants_sha,
            trained_through_ts_utc=_trained_through_ts_utc(timestamps, fold),
            train_q90_coverage=q_cov,
        )
        duration_s = time.monotonic() - fold_start
        alpha = TRAINING_UI.ETA_FOLD_DURATION_EMA_ALPHA
        fold_duration_ema = (
            duration_s if fold_duration_ema is None
            else alpha * duration_s + (1 - alpha) * fold_duration_ema
        )
        results.append(metrics)

        if metrics.stopped_early:
            emit({
                "type": "alert", "level": "warn",
                "message": (
                    f"Early stopping triggered — fold {fold.index}, epoch {metrics.epochs} "
                    f"(patience exhausted at {PREDICTOR.EARLY_STOPPING_PATIENCE})."
                ),
            })
        emit({
            "type": "fold_complete",
            "fold": fold.index,
            "train_loss": metrics.train_total,
            "val_loss": metrics.val_total,
            "da": da,
            "q_coverage": q_cov,
            "duration_s": duration_s,
            "checkpoint_path": str(weights_path),
        })

        if metrics.stopped_by_user:
            emit({"type": "status", "state": "stopped"})
            return results

    emit({"type": "status", "state": "stopped"})
    return results
