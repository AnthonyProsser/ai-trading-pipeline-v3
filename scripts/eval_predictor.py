"""eval_predictor.py — predictor training bake-off harness (cross-phase tooling).

Scope after the benchmark app landed (src/benchmark/): this script keeps ONLY the
``train-eval`` role — train N capped seeds of the CURRENT branch's predictor on one
fold and report seed-aggregated metrics. That is the one thing the benchmark app
deliberately does not do: the app scores ALREADY-TRAINED checkpoints, it never trains.
The former ``eval`` (score a pre-trained checkpoint) and ``compare`` (side-by-side
table) subcommands are retired — the app's per-model evaluation + leaderboard cover
them, ranking by simulated net-of-fee PnL rather than raw metrics.

The reviewed metric layer (``target_to_model_space`` / ``statistical_metrics`` /
``excursion_metrics``) moved to ``src/benchmark/metrics.py`` so both this script and the
app share one copy; this script imports it back (scripts -> src is the allowed
direction). Metric semantics are unchanged.

``train-eval`` still reads ``getattr(PREDICTOR, "TARGET_SEMANTICS", "per_step_logret")``
so it runs unchanged on either the per-step (pre-rework) or cumulative branch. A win
must survive seed noise (``--seeds`` capped runs, mean + std), not one lucky init.

All numeric parameters come from constants.py; this module hardcodes none.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from constants import DATA, PREDICTOR
from src.benchmark.metrics import (
    excursion_metrics,
    statistical_metrics,
    target_to_model_space,
)
from src.data.feature_pipeline import compute_features
from src.data.validator import validate_candles
from src.data.walk_forward import (
    Fold,
    carve_locked_test,
    filter_by_historical_start,
    make_folds,
)
from src.predictor.model import PatchTST
from src.predictor.rollout import enforce_geometry
from src.predictor.training import build_fold_loaders, train_one_fold

# Resolved once at import: absent on the pre-rework (per-step) branch, present on Fable-5.
SEMANTICS: str = str(getattr(PREDICTOR, "TARGET_SEMANTICS", "per_step_logret"))


# --------------------------------------------------------------------------- orchestration

def _load_real_candles(path: Path) -> tuple[npt.NDArray[np.datetime64], npt.NDArray[np.float64]]:
    arr: npt.NDArray[np.float64] = np.loadtxt(
        path, delimiter=",", usecols=(0, 1, 2, 3, 4, 5), dtype=np.float64
    )
    timestamps = arr[:, 0].astype("int64").astype("datetime64[s]")
    return timestamps, arr[:, 1:6].astype(np.float64)


def _prepare_fold(csv_path: Path, fold_index: int) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.datetime64], Fold
]:
    """Rebuild the identical walk-forward fold the training script uses (real-data path)."""
    timestamps, ohlcv = _load_real_candles(csv_path)
    validated = validate_candles(timestamps, ohlcv)
    features = compute_features(validated.ohlcv)
    feature_ts = validated.timestamps[1:]
    feature_ts, features = filter_by_historical_start(feature_ts, features)
    folds = make_folds(carve_locked_test(features.shape[0]))
    if not 0 <= fold_index < len(folds):
        raise SystemExit(f"[stop] fold {fold_index} out of range (0..{len(folds) - 1})")
    return features, feature_ts, folds[fold_index]


def _gather(
    model: PatchTST, val_loader: Iterable[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor]:
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            preds.append(enforce_geometry(model(x_batch)).cpu())
            targets.append(y_batch.cpu())
    return torch.cat(preds), torch.cat(targets)


def train_and_evaluate_seed(
    *,
    features: npt.NDArray[np.float64],
    feature_ts: npt.NDArray[np.datetime64],
    fold: Fold,
    lookback: int,
    batch_size: int,
    device: str,
    seed: int,
    max_epochs: int,
    max_steps: int | None,
) -> dict[str, dict[str, float]]:
    """Train one capped run on the fold and evaluate it on the held-out split.

    ``torch.manual_seed(seed)`` is set before both the model init and training so weight
    init and the loaders' batch shuffle vary per seed. The budget is intentionally capped
    (``max_epochs``/``max_steps``) — this is a fast directional signal, not a full run.
    """
    torch.manual_seed(seed)
    scaler, train_loader, val_loader = build_fold_loaders(
        features, feature_ts, fold, lookback=lookback, batch_size=batch_size, device=device
    )
    model = PatchTST(lookback=lookback)
    # Wall-clock for the whole train_one_fold call, INCLUDING its per-epoch validation passes
    # (not pure backward-pass time). Reported as a SECONDARY signal only — a faster model is
    # preferable at equal quality, never at the expense of the metrics above. CUDA kernels are
    # async, so we sync on BOTH sides of the timed region to avoid folding neighbouring seeds'
    # queued work into this measurement or stopping the clock before the GPU is done.
    dev_is_cuda = torch.device(device).type == "cuda"
    if dev_is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    fold_metrics = train_one_fold(
        model, scaler, train_loader, val_loader,
        device=device, max_epochs=max_epochs, max_steps=max_steps,
    )
    if dev_is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    epochs = max(1, fold_metrics.epochs)
    pred, target_raw = _gather(model, val_loader)
    target_model = target_to_model_space(target_raw, SEMANTICS)
    result = {
        "statistical": statistical_metrics(pred, target_model),
        "economic": excursion_metrics(pred, target_raw, SEMANTICS),
        "timing": {
            "wall_seconds": elapsed,
            "seconds_per_epoch": elapsed / epochs,
            "epochs": float(fold_metrics.epochs),
        },
    }
    # Release the seed's model + device-resident fold tensors before the next seed builds
    # its own — cheap insurance for an unattended multi-seed loop on CUDA.
    del model, scaler, train_loader, val_loader
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return result


def aggregate_seeds(
    label: str, semantics: str, seed_results: list[dict[str, dict[str, float]]]
) -> dict[str, object]:
    """Mean + population std of each metric across seed runs (a win must survive seed noise).

    NaN-tolerant: a seed with zero tradeable windows yields NaN economic metrics; those are
    skipped so one unlucky seed cannot poison the aggregate. ``n_used`` is a count (the val
    split is fold-fixed across seeds), so it is reported once (max) rather than averaged/std'd.
    Every metric group present in the seed records (``statistical``/``economic``/``timing``)
    is aggregated the same way; a group's absence is simply skipped.
    """
    def agg(group: str) -> tuple[dict[str, float], dict[str, float]]:
        keys = seed_results[0][group].keys()
        mean: dict[str, float] = {}
        std: dict[str, float] = {}
        for key in keys:
            # `.get(..., nan)` so a seed missing this group/key can't KeyError the whole
            # aggregate — the NaN is then skipped by the finite-filter below.
            values = [float(r.get(group, {}).get(key, float("nan"))) for r in seed_results]
            if key == "n_used":
                mean[key] = float(max(values))  # count, not a quality metric — no std
                continue
            finite = [v for v in values if not math.isnan(v)]
            mean[key] = float(np.mean(finite)) if finite else float("nan")
            std[key] = float(np.std(finite)) if finite else float("nan")
        return mean, std

    out: dict[str, object] = {
        "label": label,
        "semantics": semantics,
        "seeds": len(seed_results),
    }
    for group in seed_results[0]:
        mean, std = agg(group)
        out[group] = mean
        out[f"{group}_std"] = std
    return out


def run_bakeoff(
    *, label: str, fold_index: int, seeds: int, batch_size: int, device: str,
    max_epochs: int, max_steps: int | None,
) -> dict[str, object]:
    """Train ``seeds`` capped runs on one fold, evaluate each, return the seed-aggregated record."""
    csv_path = REPO_ROOT / DATA.KRAKEN_HISTORY_OUT_DIR / DATA.KRAKEN_HISTORY_CSV_NAME
    if not csv_path.exists():
        raise SystemExit(f"[stop] {csv_path} not found; real data required for a fair bake-off.")
    features, feature_ts, fold = _prepare_fold(csv_path, fold_index)
    lookback = DATA.LOOKBACK
    seed_results: list[dict[str, dict[str, float]]] = []
    failed = 0
    for i in range(seeds):
        seed = PREDICTOR.SEED + i
        print(f"[train-eval] {label}: seed {i + 1}/{seeds} (seed={seed}) ...")
        try:
            seed_results.append(train_and_evaluate_seed(
                features=features, feature_ts=feature_ts, fold=fold, lookback=lookback,
                batch_size=batch_size, device=device, seed=seed,
                max_epochs=max_epochs, max_steps=max_steps,
            ))
        except Exception as exc:  # one bad seed must not discard the rest of the bake-off
            failed += 1
            print(f"[train-eval] {label}: seed {i + 1}/{seeds} FAILED ({exc!r}) — dropping it.")
    if not seed_results:
        raise SystemExit(f"[stop] all {seeds} seeds failed; nothing to aggregate.")
    agg = aggregate_seeds(label, SEMANTICS, seed_results)
    agg["fold_index"] = fold_index
    agg["max_epochs"] = max_epochs
    agg["max_steps"] = max_steps
    agg["n_seeds_failed"] = failed
    return agg


# --------------------------------------------------------------------------- CLI

def _run_train_eval(args: argparse.Namespace) -> int:
    result = run_bakeoff(
        label=args.label, fold_index=args.fold, seeds=args.seeds,
        batch_size=args.batch_size, device=args.device,
        max_epochs=args.max_epochs, max_steps=args.max_steps,
    )
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"[train-eval] {args.label}: wrote {args.out}")
    return 0


def _batch_size_default() -> int:
    return int(getattr(PREDICTOR, "PROD_BATCH_SIZE", PREDICTOR.SMOKE_BATCH_SIZE))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    te = sub.add_parser(
        "train-eval", help="train N capped seeds on a fold + evaluate -> seed-aggregated JSON"
    )
    te.add_argument("--label", required=True, help="run label recorded in the result JSON")
    te.add_argument("--out", type=Path, required=True, help="result JSON path")
    te.add_argument("--fold", type=int, default=0)
    te.add_argument(
        "--seeds", type=int, default=DATA.SEARCH_CONFIRM_SEEDS,
        help="number of repeat-seed runs to average (default: SEARCH_CONFIRM_SEEDS)",
    )
    te.add_argument(
        # Default 1: a single epoch is the fastest budget that still exposes each branch's
        # early calibration/convergence behaviour — the intended "directional signal", not a
        # full run (PredictorConfig.MAX_EPOCHS=100). Operator raises it for a fuller compare.
        "--max-epochs", type=int, dest="max_epochs", default=1,
        help="capped training epochs per seed (raise for a fuller comparison)",
    )
    te.add_argument(
        "--max-steps", type=int, dest="max_steps", default=None,
        help="optional hard cap on optimizer steps per seed (fastest signal)",
    )
    te.add_argument("--batch-size", type=int, dest="batch_size", default=_batch_size_default())
    te.add_argument("--device", default="cpu")
    te.set_defaults(func=_run_train_eval)

    args = ap.parse_args()
    func: object = args.func
    assert callable(func)
    return int(func(args))


if __name__ == "__main__":
    sys.exit(main())
