"""Benchmark engine: evaluate ONE checkpoint on its own walk-forward fold's TEST slice.

Honesty properties, in order of importance:

  * Out-of-sample: each per-fold checkpoint is scored on its fold's TEST slice
    (WALK_FORWARD_TEST candles) — data that never entered training OR early-stopping
    model selection (best-by-val-total picks on the VAL slice, so val is mildly
    selection-contaminated; test is not). The LOCKED test set is never read here — it
    stays reserved for the deploy gates and the 12x1w holdout evaluator, so the
    leaderboard cannot burn it.
  * Same inference path as deploy: the paired pickled scaler transforms via
    transform_inference (fold stats, window assertion legitimately skipped outside the
    training fold), predictions are geometry-enforced, targets cumsummed to the
    model's cumulative space (predictor_checkpoint_save_format).
  * Fixed instrument: signals/PnL/baselines all come from src/benchmark/trading_sim —
    identical for every model, no per-model tuning surface.

The full Kraken CSV parse (~4.2M rows) is slow, so features are computed once per
process and cached; every subsequent benchmark run on this server reuses them.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader

from constants import BENCHMARK, DATA, EXECUTION, PREDICTOR
from src.benchmark.metrics import excursion_metrics, statistical_metrics
from src.benchmark.registry import (
    parse_run_tag,
    read_checkpoint_meta,
    write_benchmark_result,
)
from src.benchmark.trading_sim import (
    buy_and_hold,
    extract_signals,
    ledger_stats,
    random_entry_null,
    simulate_trades,
)
from src.data.feature_pipeline import compute_features
from src.data.scaler import PerFoldMinMaxScaler
from src.data.validator import validate_candles
from src.data.walk_forward import (
    Fold,
    carve_locked_test,
    filter_by_historical_start,
    make_folds,
)
from src.predictor.model import PatchTST
from src.predictor.rollout import enforce_geometry
from src.predictor.training import WindowDataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_CLOSE = DATA.FEATURE_NAMES.index("close_logret")

FeatureArrays = tuple[npt.NDArray[np.datetime64], npt.NDArray[np.float64]]
# Process-level feature cache. Safe as an unguarded dict ONLY because BenchmarkRunner
# serializes jobs (one at a time), so _load_features is never entered concurrently. Add
# a lock here if that invariant ever changes (e.g. parallel benchmark jobs).
_FEATURE_CACHE: dict[str, FeatureArrays] = {}


class RunnerLike(Protocol):
    """What the engine needs from the app's BenchmarkRunner (avoids a circular import)."""

    checkpoint_dir: Path
    benchmark_dir: Path

    def broadcast(self, payload: dict[str, object]) -> None: ...


def eval_slice_bounds(fold: Fold, *, lookback: int) -> tuple[int, int]:
    """Feature-row span [start, end) whose sliding windows put every TARGET row inside
    the fold's test slice: origins begin lookback rows before test_start (lookback
    context is past data — legitimate input), and the last window's target ends at
    test_end. Raises if the dataset has fewer than lookback rows before the slice."""
    start = fold.test_start - lookback
    if start < 0:
        raise ValueError(
            f"fold {fold.index}: test_start {fold.test_start} < lookback {lookback} — "
            f"not enough history to form evaluation windows"
        )
    return start, fold.test_end


def _load_features(csv_path: Path) -> FeatureArrays:
    """Validated, HISTORICAL_START-filtered features + timestamps (process-cached).

    Identical pipeline to the real-data training path (src/training_ui/app.py) so the
    rebuilt fold boundaries match the ones the checkpoints were trained under.
    """
    key = str(csv_path)
    if key not in _FEATURE_CACHE:
        arr: npt.NDArray[np.float64] = np.loadtxt(
            csv_path, delimiter=",", usecols=(0, 1, 2, 3, 4, 5), dtype=np.float64
        )
        timestamps = arr[:, 0].astype("int64").astype("datetime64[s]")
        validated = validate_candles(timestamps, arr[:, 1:6].astype(np.float64))
        features = compute_features(validated.ohlcv)
        feature_ts = validated.timestamps[1:]
        feature_ts, features = filter_by_historical_start(feature_ts, features)
        _FEATURE_CACHE[key] = (feature_ts, features)
    return _FEATURE_CACHE[key]


def _gather_predictions(
    model: PatchTST,
    x_scaled: npt.NDArray[np.float64],
    y_raw: npt.NDArray[np.float64],
    *,
    lookback: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Geometry-enforced predictions + raw targets over all sliding windows (CPU)."""
    dataset = WindowDataset(
        torch.from_numpy(np.ascontiguousarray(x_scaled, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(y_raw, dtype=np.float32)),
        lookback,
        PREDICTOR.HORIZON,
    )
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        dataset, batch_size=PREDICTOR.PROD_BATCH_SIZE, shuffle=False
    )
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            preds.append(enforce_geometry(model(x_batch.to(device))).cpu())
            targets.append(y_batch)
    return torch.cat(preds), torch.cat(targets)


def run_benchmark_job(runner: RunnerLike, stem: str) -> None:
    """The default BenchmarkRunner job: evaluate `stem` and write its result JSON."""

    def emit(stage: str, **extra: object) -> None:
        runner.broadcast(
            {"type": "benchmark_progress", "stem": stem, "stage": stage, **extra}
        )

    tag = parse_run_tag(stem)
    if tag is None:
        raise ValueError(f"{stem}: not a run-tag checkpoint name")
    weights_path = runner.checkpoint_dir / f"{stem}{PREDICTOR.CHECKPOINT_WEIGHTS_SUFFIX}"
    scaler_path = runner.checkpoint_dir / f"{stem}{PREDICTOR.CHECKPOINT_SCALER_SUFFIX}"
    meta = read_checkpoint_meta(weights_path)
    semantics = meta.get("target_semantics")
    if semantics != PREDICTOR.TARGET_SEMANTICS:
        raise ValueError(
            f"{stem}: target_semantics {semantics!r} != {PREDICTOR.TARGET_SEMANTICS!r}"
        )
    lookback = meta.get("lookback")
    if not isinstance(lookback, int):
        raise ValueError(f"{stem}: checkpoint has no embedded lookback")

    emit("loading-model")
    loaded = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = loaded["state_dict"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchTST(lookback=lookback)
    # The model is built from the CURRENT constants.py; a state_dict trained under
    # architecture-affecting constants (D_MODEL, N_LAYERS, PATCH_SIZE, QUANTILES, ...)
    # that differ from now won't fit. Turn the raw RuntimeError into a diagnostic that
    # names the likely cause — the checkpoint's embedded constants hash — instead of a
    # generic "Benchmark failed". (The registry surfaces a `constants_match` flag so the
    # UI can warn before the user even starts a cross-constants run.)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise ValueError(
            f"{stem}: checkpoint weights do not fit the current architecture — its "
            f"constants_sha256 {meta.get('constants_sha256')!r} likely differs from the "
            f"running constants.py. ({exc})"
        ) from exc
    model.to(device)
    with open(scaler_path, "rb") as fh:
        scaler: PerFoldMinMaxScaler = pickle.load(fh)

    emit("loading-data")  # first run per process parses the full CSV (slow, ~1 min)
    csv_path = REPO_ROOT / DATA.KRAKEN_HISTORY_OUT_DIR / DATA.KRAKEN_HISTORY_CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found — real data required to benchmark")
    feature_ts, features = _load_features(csv_path)
    folds = make_folds(carve_locked_test(features.shape[0]))
    if not 0 <= tag.fold_index < len(folds):
        raise ValueError(
            f"{stem}: fold {tag.fold_index} out of range (0..{len(folds) - 1}) — "
            f"fold geometry may have changed since this checkpoint was trained"
        )
    fold = folds[tag.fold_index]
    start, end = eval_slice_bounds(fold, lookback=lookback)
    slice_feats = features[start:end]
    x_scaled = scaler.transform_inference(slice_feats)

    emit("inference", n_windows=int(end - start - lookback - PREDICTOR.HORIZON + 1))
    pred, target_raw = _gather_predictions(
        model, x_scaled, slice_feats, lookback=lookback, device=device
    )
    target_cum = torch.cumsum(target_raw, dim=1)

    emit("simulating")
    signals = extract_signals(pred)
    realized_final: npt.NDArray[np.float64] = (
        target_cum[:, -1, _CLOSE].to(torch.float64).numpy()
    )
    ledger = simulate_trades(
        signals, realized_final, horizon=PREDICTOR.HORIZON, fee=EXECUTION.FEE_THRESHOLD
    )
    trading = ledger_stats(ledger.net_returns, ledger.gross_returns)
    trading["signal_count"] = int(np.count_nonzero(signals))

    test_total_logret = float(
        np.sum(features[fold.test_start : fold.test_end, _CLOSE])
    )
    baselines: dict[str, float] = {
        "buy_and_hold_net": buy_and_hold(test_total_logret, fee=EXECUTION.FEE_THRESHOLD),
    }
    baselines.update(
        random_entry_null(
            n_origins=int(signals.shape[0]),
            trade_count=int(trading["trade_count"]),
            realized_final=realized_final,
            horizon=PREDICTOR.HORIZON,
            fee=EXECUTION.FEE_THRESHOLD,
            draws=BENCHMARK.NULL_DRAWS,
            seed=BENCHMARK.NULL_SEED,
            model_net=float(trading["net_return"]),
        )
    )

    emit("writing")
    record: dict[str, object] = {
        "stem": stem,
        "fold_index": tag.fold_index,
        "git_sha": tag.git_sha,
        "semantics": PREDICTOR.TARGET_SEMANTICS,
        "lookback": lookback,
        "constants_sha256_checkpoint": meta.get("constants_sha256"),
        "trained_through_ts_utc": meta.get("trained_through_ts_utc"),
        "eval": {
            "slice": "fold_test",
            "n_windows": int(pred.shape[0]),
            "start_ts_utc": _ts_utc(feature_ts, fold.test_start),
            "end_ts_utc": _ts_utc(feature_ts, fold.test_end - 1),
            "device": device.type,
        },
        "trading": trading,
        "baselines": baselines,
        "statistical": statistical_metrics(pred, target_cum),
        "economic": excursion_metrics(pred, target_raw, PREDICTOR.TARGET_SEMANTICS),
    }
    write_benchmark_result(runner.benchmark_dir, stem, record)
    runner.broadcast(
        {
            "type": "benchmark_complete",
            "stem": stem,
            "net_return": float(trading["net_return"]),
            "trade_count": int(trading["trade_count"]),
            "p_value": float(baselines["p_value"]),
        }
    )


def _ts_utc(timestamps: npt.NDArray[np.datetime64], index: int) -> str:
    """ISO8601-UTC 'Z' string for a feature-row timestamp (repo timestamp convention)."""
    return str(np.datetime_as_string(timestamps[index].astype("datetime64[s]"), unit="s")) + "Z"
