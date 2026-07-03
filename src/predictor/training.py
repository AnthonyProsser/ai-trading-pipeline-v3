"""Reusable predictor training logic (predictor-training.md §"Smoke run" /
§"Training data").

Lives under src/ (not the train script) so it is unit-testable and importable;
scripts/train_predictor.py is a thin CLI over it, honouring the
src-never-imports-scripts rule.

Data convention: the model consumes per-fold MinMax-**scaled** inputs (and internally
re-normalises them per window — RevIN) but predicts **raw cumulative** log-returns,
because the loss's FEE_THRESHOLD term lives in raw log-return units. WindowDataset
pairs a scaled lookback window with a raw per-step horizon target; predictor_loss and
the gate-metric collectors convert targets to the cumulative path (the model's output
space, PredictorConfig.TARGET_SEMANTICS). build_fold_loaders fits the scaler on the
fold's TRAIN slice only; train_one_fold *receives* that scaler and never builds one.
All numeric settings come from constants.py.
"""
from __future__ import annotations

import copy
import hashlib
import math
import pickle
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from constants import PREDICTOR, TRAINING_UI
from src.data.scaler import PerFoldMinMaxScaler
from src.data.walk_forward import Fold
from src.predictor.deploy_gates import directional_accuracy, q90_coverage
from src.predictor.early_stopping import EarlyStopper
from src.predictor.loss import predictor_loss
from src.predictor.model import PatchTST
from src.predictor.rollout import enforce_geometry

# TF32 matmuls + cuDNN autotuner: free throughput on Ampere+ (the RTX 4060 is sm_89)
# with no change to model quality (bf16 AMP already trains in reduced precision, and the
# fixed-shape training loop lets cuDNN pick its fastest kernels once). Set at import so
# every entry point (training UI, train_predictor.py) benefits; both are no-ops on CPU.
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

LogFn = Callable[[dict[str, object]], None]
Batch = tuple[torch.Tensor, torch.Tensor]
Loaders = tuple[PerFoldMinMaxScaler, "_WindowLoader", "_WindowLoader"]


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


class _WindowLoader:
    """Device-resident batched sliding-window iterator — a drop-in replacement for a
    DataLoader over WindowDataset that eliminates the per-step host->device copy and
    Python-side per-sample slicing that dominated the old hot loop.

    The fold's scaled inputs and raw targets are tiny (~3 MB for a 200k-candle fold), so
    both tensors live on the training device up front. Each batch is formed by a single
    vectorised gather over precomputed window offsets — no DataLoader workers, no
    pin_memory, no blocking `.to(device)` per step. Windowing semantics are identical to
    WindowDataset: input = scaled `[i : i+lookback]`, target = raw
    `[i+lookback : i+lookback+horizon]`, `n = rows - lookback - horizon + 1` windows.
    `shuffle`/`drop_last` mirror the DataLoader flags the two splits used before.

    `.dataset` exposes the underlying WindowDataset purely so callers/tests can read the
    window count via `len(loader.dataset)` exactly as with a real DataLoader.
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        lookback: int,
        horizon: int,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        device: torch.device,
    ) -> None:
        self.dataset = WindowDataset(x, y, lookback, horizon)  # validates + carries len
        self._x = x.to(device)
        self._y = y.to(device)
        self._lookback = lookback
        self._horizon = horizon
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._device = device
        self._n = len(self.dataset)
        # Precomputed within-window offsets, reused every batch (never re-materialised).
        self._look_off = torch.arange(lookback, device=device)
        self._hor_off = torch.arange(horizon, device=device)

    def __len__(self) -> int:
        if self._drop_last:
            return self._n // self._batch_size
        return (self._n + self._batch_size - 1) // self._batch_size

    def __iter__(self) -> Iterator[Batch]:
        if self._shuffle:
            order = torch.randperm(self._n, device=self._device)
        else:
            order = torch.arange(self._n, device=self._device)
        bs = self._batch_size
        limit = (self._n // bs) * bs if self._drop_last else self._n
        for start in range(0, limit, bs):
            base = order[start : start + bs]
            # Vectorised gather: (batch, lookback/horizon) index grids into the fold tensor.
            x_idx = base[:, None] + self._look_off[None, :]
            y_idx = base[:, None] + (self._lookback + self._hor_off)[None, :]
            yield self._x[x_idx], self._y[y_idx]


def build_fold_loaders(
    features: npt.NDArray[np.float64],
    timestamps: npt.NDArray[np.datetime64],
    fold: Fold,
    *,
    lookback: int,
    batch_size: int,
    device: str = "cpu",
) -> Loaders:
    """Fit a per-fold scaler on TRAIN, return (scaler, train_loader, val_loader).

    Inputs are scaled; targets stay raw. The scaler's fold window is [train_start,
    val_end), so its own assertion rejects any transform that strays outside the fold.
    Loaders are device-resident (`_WindowLoader`): the whole fold's tensors are moved to
    `device` once and batched by vectorised gather, so the training loop never pays a
    per-step host->device copy. `device` defaults to "cpu"; callers that train on CUDA
    pass "cuda" so the fold lives GPU-resident.
    """
    dev = torch.device(device)
    ts_m = timestamps.astype("datetime64[m]")
    scaler = PerFoldMinMaxScaler(fold_start=ts_m[fold.train_start], fold_end=ts_m[fold.val_end - 1])
    train_feats = features[fold.train_start : fold.train_end]
    val_feats = features[fold.val_start : fold.val_end]
    scaler.fit(train_feats)
    x_train = scaler.transform(train_feats, timestamps[fold.train_start : fold.train_end])
    x_val = scaler.transform(val_feats, timestamps[fold.val_start : fold.val_end])

    train_loader = _WindowLoader(
        _to_f32(x_train), _to_f32(train_feats),
        lookback=lookback, horizon=PREDICTOR.HORIZON, batch_size=batch_size,
        shuffle=True, drop_last=True, device=dev,
    )
    val_loader = _WindowLoader(
        _to_f32(x_val), _to_f32(val_feats),
        lookback=lookback, horizon=PREDICTOR.HORIZON, batch_size=batch_size,
        shuffle=False, drop_last=False, device=dev,
    )
    return scaler, train_loader, val_loader


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _evaluate(
    model: torch.nn.Module,
    loader: _WindowLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, float, float]:
    model.eval()
    pinball = direction = coverage = total = 0.0
    batches = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                comp = predictor_loss(model(x), y)
            pinball += float(comp.pinball)
            direction += float(comp.direction)
            coverage += float(comp.coverage)
            total += float(comp.total)
            batches += 1
    if batches == 0:
        raise RuntimeError("val_loader yielded no batches — fold or batch_size misconfigured")
    return pinball / batches, direction / batches, coverage / batches, total / batches


def train_one_fold(
    model: torch.nn.Module,
    scaler: PerFoldMinMaxScaler,
    train_loader: _WindowLoader,
    val_loader: _WindowLoader,
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
    scaler -- it never builds one. Uses AdamW with linear warmup then a constant LR
    that decays by LR_PLATEAU_FACTOR after LR_PLATEAU_PATIENCE non-improving val
    epochs (a cosine schedule pinned to the MAX_EPOCHS horizon never annealed before
    early stopping fired, so real runs trained at near-peak LR throughout). AMP (bf16)
    on CUDA, grad clipping, predictor_loss, and EarlyStopper; raises on a non-finite
    loss.

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
    warmup_steps = max(1, int(PREDICTOR.WARMUP_FRAC * total_steps))
    # Plateau LR decay state: LR = LEARNING_RATE x warmup ramp x lr_scale, where
    # lr_scale halves (LR_PLATEAU_FACTOR) after LR_PLATEAU_PATIENCE consecutive
    # non-improving validation epochs. Managed manually (not a torch scheduler) so the
    # warmup ramp and the plateau decay compose without scheduler-interaction bugs.
    lr_scale = 1.0
    plateau_best = math.inf
    plateau_bad = 0
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

    # SSE 'batch' cadence: emitting a payload forces a CUDA sync (to read the loss
    # components as floats) plus a JSON broadcast; doing that every step starves the GPU
    # and dominated the old hot loop. Throttle to every LOG_INTERVAL-th step (schema
    # unchanged), always flushing the last step of the epoch so the trace ends on the
    # true value. DECISIONS.md 'training_ui_batch_log_interval'.
    log_interval = TRAINING_UI.BATCH_LOG_INTERVAL

    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        model.train()
        # On-GPU epoch accumulators: kept as device tensors so no per-step .item()/sync
        # is needed; converted to Python floats once at epoch end.
        ep_pin = torch.zeros((), device=dev)
        ep_dir = torch.zeros((), device=dev)
        ep_cov = torch.zeros((), device=dev)
        ep_tot = torch.zeros((), device=dev)
        ep_batches = 0
        n_batches = len(train_loader)
        for batch_idx, (x, y) in enumerate(train_loader):
            if stop_event is not None and stop_event.is_set():
                stopped_by_user = True
                break
            if save_event is not None and save_event.is_set():
                if on_save_request is not None:
                    on_save_request()
                save_event.clear()
            # Defensive no-op when the loader was built for `dev` (the normal case: the
            # _WindowLoader already yields device-resident tensors). Kept only to stay
            # correct if a caller passes a loader built for a different device.
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
                comp = predictor_loss(model(x), y)
            comp.total.backward()  # type: ignore[no-untyped-call]  # torch stubs untyped
            # clip_grad_norm_ returns the PRE-clip total norm -- exactly the "is clipping
            # engaging" signal the loss-component strip's grad-norm coloring needs. Kept as
            # a tensor; only read as a float when a batch payload is actually emitted.
            grad_norm_t = torch.nn.utils.clip_grad_norm_(
                model.parameters(), PREDICTOR.GRAD_CLIP_NORM
            )
            lr_now = (
                PREDICTOR.LEARNING_RATE
                * min(1.0, float(step + 1) / float(warmup_steps))
                * lr_scale
            )
            _set_lr(optimizer, lr_now)
            optimizer.step()
            step += 1
            ep_pin += comp.pinball.detach()
            ep_dir += comp.direction.detach()
            ep_cov += comp.coverage.detach()
            ep_tot += comp.total.detach()
            ep_batches += 1
            is_last_batch = (batch_idx == n_batches - 1) or (
                max_steps is not None and step >= max_steps
            )
            # One sync per emitted payload instead of one per step. The step's own
            # finiteness is verified here (the emitted 'total' is read anyway); a NaN
            # between emissions still surfaces at the epoch-boundary accumulator check
            # below, so a non-finite training loss can never pass silently.
            if log is not None and (step == 1 or step % log_interval == 0 or is_last_batch):
                batch_tot = float(comp.total)
                if not math.isfinite(batch_tot):
                    raise ValueError(
                        f"non-finite training loss at step {step}: total={batch_tot}"
                    )
                log(
                    {
                        "split": "train",
                        "step": step,
                        "fold": fold_index,
                        "epoch": epoch,
                        "pinball": float(comp.pinball),
                        "direction": float(comp.direction),
                        "coverage": float(comp.coverage),
                        "total": batch_tot,
                        "lr": lr_now,
                        "grad_norm": float(grad_norm_t),
                    }
                )
            if max_steps is not None and step >= max_steps:
                break

        if stopped_by_user:
            break

        # Report the epoch-averaged train loss (not the last batch) so it is comparable
        # with the epoch-averaged val loss from _evaluate. Single sync per epoch.
        if ep_batches > 0:
            train_pin = float(ep_pin) / ep_batches
            train_dir = float(ep_dir) / ep_batches
            train_cov = float(ep_cov) / ep_batches
            train_tot = float(ep_tot) / ep_batches
            if not math.isfinite(train_tot):
                raise ValueError(
                    f"non-finite training loss in epoch {epochs_run}: mean total={train_tot}"
                )

        val_pin, val_dir, val_cov, val_tot = _evaluate(model, val_loader, dev, use_amp)
        if not math.isfinite(val_tot):
            raise ValueError(f"non-finite validation loss at epoch {epochs_run}: total={val_tot}")
        if val_tot < best_val:
            best_val = val_tot
            best_state = copy.deepcopy(model.state_dict())
        # Plateau LR decay, evaluated on the same val_total the EarlyStopper watches but
        # with a shorter patience, so the LR gets a chance to un-stick a plateau before
        # early stopping ends the fold.
        if val_tot < plateau_best:
            plateau_best = val_tot
            plateau_bad = 0
        else:
            plateau_bad += 1
            if plateau_bad >= PREDICTOR.LR_PLATEAU_PATIENCE:
                lr_scale *= PREDICTOR.LR_PLATEAU_FACTOR
                plateau_bad = 0
        if log is not None:
            log(
                {
                    "split": "val",
                    "step": step,
                    "fold": fold_index,
                    "epoch": epoch,
                    "train_pinball": train_pin,
                    "train_direction": train_dir,
                    "train_coverage": train_cov,
                    "train_total": train_tot,
                    "val_pinball": val_pin,
                    "val_direction": val_dir,
                    "val_coverage": val_cov,
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
    loader: _WindowLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `model` over every batch in `loader` with geometry enforcement, concatenated
    to CPU. Targets are converted to the CUMULATIVE log-return path (the model's output
    space, PredictorConfig.TARGET_SEMANTICS) so gate metrics compare like with like.
    Shared by evaluate_q90_coverage / evaluate_directional_accuracy so both metrics --
    and any future gate metric -- read predictions the identical way deploy does
    (predictor-contract.md geometry enforcement at every inference step)."""
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            # Both sides -> CPU explicitly: the _WindowLoader keeps y device-resident, so
            # leaving it in place would compare a CPU pred against a CUDA target.
            preds.append(enforce_geometry(model(x.to(device))).cpu())
            targets.append(y.cumsum(dim=1).cpu())
    if not preds:
        raise RuntimeError("loader yielded no batches — cannot evaluate predictions")
    return torch.cat(preds), torch.cat(targets)


def evaluate_q90_coverage(
    model: torch.nn.Module,
    loader: _WindowLoader,
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
    loader: _WindowLoader,
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
    constants.py hash, the ISO8601-UTC timestamp the model was trained through, the
    training-time q90 coverage baseline for deploy gate (a), and the target-semantics
    tag deploy uses to refuse convention-mismatched checkpoints) — saved via torch.save
    so it loads cleanly under deploy_predictor.py's weights_only=True. The scaler is
    pickled (deploy reads it with pickle.load). The SHA256 manifest over both files +
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
            "target_semantics": PREDICTOR.TARGET_SEMANTICS,
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
    "fold_complete" + "alert" after each fold, "status" on user stop ("stopped") or
    natural completion of all folds ("done").

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
            features, timestamps, fold, lookback=lookback, batch_size=batch_size, device=device
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

    # Natural completion of every fold is "done", distinct from a user-initiated
    # "stopped", so the UI can render a terminal state instead of an ambiguous one.
    emit({"type": "status", "state": "done"})
    return results
