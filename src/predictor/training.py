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

import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset

from constants import PREDICTOR
from src.data.scaler import PerFoldMinMaxScaler
from src.data.walk_forward import Fold
from src.predictor.early_stopping import EarlyStopper
from src.predictor.loss import predictor_loss

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
) -> FoldMetrics:
    """Train one fold on the supplied loaders. Receives the (already-fitted) per-fold
    scaler -- it never builds one. Uses AdamW + warmup-cosine, AMP (bf16) on CUDA,
    grad clipping, predictor_loss, and EarlyStopper; raises on a non-finite loss."""
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
    train_pin = train_dir = train_tot = math.nan
    val_pin = val_dir = val_tot = math.nan

    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        model.train()
        ep_pin = ep_dir = ep_tot = 0.0
        ep_batches = 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
                comp = predictor_loss(model(x), y)
            if not bool(torch.isfinite(comp.total)):
                raise ValueError(
                    f"non-finite training loss at step {step}: total={float(comp.total)}"
                )
            comp.total.backward()  # type: ignore[no-untyped-call]  # torch stubs untyped
            torch.nn.utils.clip_grad_norm_(model.parameters(), PREDICTOR.GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            step += 1
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
                    "pinball": batch_pin,
                    "direction": batch_dir,
                    "total": batch_tot,
                }
                log(payload)
            if max_steps is not None and step >= max_steps:
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
        if log is not None:
            log(
                {
                    "split": "val",
                    "step": step,
                    "pinball": val_pin,
                    "direction": val_dir,
                    "total": val_tot,
                }
            )
        if stopper.step(val_tot):
            stopped = True
            break
        if max_steps is not None and step >= max_steps:
            break

    return FoldMetrics(
        train_pin, train_dir, train_tot, val_pin, val_dir, val_tot, step, epochs_run, stopped
    )


def make_run_tag(*, git_sha: str, constants_sha: str, scaler_sha: str, fold_id: int) -> str:
    """W&B run tag = git SHA + scaler hash + constants.py hash + fold id (spec order)."""
    return f"{git_sha}-s{scaler_sha[:8]}-c{constants_sha[:8]}-fold{fold_id}"
