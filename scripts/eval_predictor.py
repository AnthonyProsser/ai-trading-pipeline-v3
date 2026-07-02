"""eval_predictor.py — controlled predictor bake-off harness (cross-phase tooling).

Not a phase deliverable; the counterpart of scripts/search_predictor.py. Its job is to
turn "the Fable-5 rework is supposedly better" into a measured, side-by-side verdict on
the SAME held-out fold, so a branch is never merged on self-report.

The two branches predict in different target spaces — ``main`` emits per-step
log-return quantiles, ``fable-5-restructuring`` emits CUMULATIVE-path quantiles — so this
one file is written to run UNCHANGED on either branch: it reads
``getattr(PREDICTOR, "TARGET_SEMANTICS", "per_step_logret")`` and adapts.

Three subcommands:
  * ``train-eval`` — train N capped seeds on one fold IN THIS SCRIPT, evaluate each, and
    report the seed-aggregated metrics (mean + std). Self-contained: no pre-trained
    checkpoint needed, and the budget is capped (``--max-epochs``, default 1, or
    ``--max-steps``) so you get a fast directional signal without a full multi-hour run.
    A win must survive seed noise, not just beat on one lucky init.
  * ``eval`` — evaluate an ALREADY-trained ``--checkpoint`` on its held-out fold (use once
    a real full-training checkpoint exists; no training happens here).
  * ``compare`` — join result JSONs into a side-by-side table.

Typical fast loop: run ``train-eval`` on each branch (each under its own checked-out code,
writing a JSON), then ``compare`` the two JSONs.

Two metric families, reported together (see the module docstring rationale in the PR):

  Statistical (per-branch, scored in the model's OWN target space):
    q90 coverage, [q10,q90] calibration, directional accuracy, close-interval sharpness,
    pinball. Coverage/calibration/DA are cross-branch comparable; sharpness/pinball are
    NOT (per-step vs cumulative magnitudes differ) and are labelled per-branch only.

  Economic (cross-branch comparable, scored on the realised price path):
    "fraction of the predicted move that played out" (favorable excursion / predicted
    magnitude) PAIRED WITH adverse excursion (how far price ran the wrong way first).
    Sub-fee predicted moves (|move| <= EXECUTION.FEE_THRESHOLD) are untradeable and
    excluded — this both grounds the cutoff in an existing constant and stops the metric
    being gamed by shrinking predictions toward zero.

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

from constants import DATA, EXECUTION, PREDICTOR
from src.data.feature_pipeline import compute_features
from src.data.validator import validate_candles
from src.data.walk_forward import (
    Fold,
    carve_locked_test,
    filter_by_historical_start,
    make_folds,
)
from src.predictor.deploy_gates import calibration_rate, directional_accuracy, q90_coverage
from src.predictor.loss import pinball_loss
from src.predictor.model import PatchTST
from src.predictor.rollout import enforce_geometry
from src.predictor.training import build_fold_loaders, train_one_fold

_CLOSE = DATA.FEATURE_NAMES.index("close_logret")
_Q10 = PREDICTOR.QUANTILES.index(0.10)
_Q50 = PREDICTOR.QUANTILES.index(0.50)
_Q90 = PREDICTOR.QUANTILES.index(0.90)

# Resolved once at import: absent on the pre-rework (per-step) branch, present on Fable-5.
SEMANTICS: str = str(getattr(PREDICTOR, "TARGET_SEMANTICS", "per_step_logret"))


# --------------------------------------------------------------------------- metrics

def target_to_model_space(target: torch.Tensor, semantics: str) -> torch.Tensor:
    """Map raw per-step log-return targets into the model's prediction space.

    ``per_step_logret`` predicts each step's move (identity); ``cumulative_logret``
    predicts the running-sum path, so the target is cumsummed over the horizon dim.
    """
    if semantics == "cumulative_logret":
        return torch.cumsum(target, dim=1)
    return target


def statistical_metrics(pred: torch.Tensor, target_model_space: torch.Tensor) -> dict[str, float]:
    """Coverage/calibration/DA (cross-branch comparable) + sharpness/pinball (per-branch).

    ``target_model_space`` must already be in the model's space (see target_to_model_space).
    """
    sharpness = float((pred[..., _CLOSE, _Q90] - pred[..., _CLOSE, _Q10]).mean())
    return {
        "q90_coverage": q90_coverage(pred, target_model_space),
        "calibration_rate": calibration_rate(pred, target_model_space),
        "directional_accuracy": directional_accuracy(pred, target_model_space),
        "sharpness_close": sharpness,
        "pinball": float(pinball_loss(pred, target_model_space)),
    }


def excursion_metrics(
    pred: torch.Tensor,
    target_raw: torch.Tensor,
    semantics: str,
    *,
    fee_threshold: float = EXECUTION.FEE_THRESHOLD,
) -> dict[str, float]:
    """Krafer-style "how much of the predicted move played out", paired with adverse run.

    Reference move = the q50 close forecast, expressed as a cumulative path (cumsum of the
    per-step median, or the median directly when it is already cumulative). For each
    sample the predicted total move sets the direction ``s`` and magnitude ``M``; over the
    realised cumulative path the favorable excursion is the best move in direction ``s`` and
    the adverse excursion is the worst move against it. Fractions are ``excursion / M``,
    averaged over samples whose ``M`` clears the round-trip fee.
    """
    q50_close = pred[..., _CLOSE, _Q50]  # (B, H)
    pred_cum = q50_close if semantics == "cumulative_logret" else torch.cumsum(q50_close, dim=1)
    realized_cum = torch.cumsum(target_raw[..., _CLOSE], dim=1)  # (B, H)

    final = pred_cum[:, -1]
    sign = torch.sign(final)
    magnitude = final.abs()
    tradeable = magnitude > fee_threshold
    n_used = int(tradeable.sum())
    if n_used == 0:
        return {"mean_captured_fraction": float("nan"), "mean_adverse_ratio": float("nan"),
                "median_captured_fraction": float("nan"), "n_used": 0}

    directional = sign.unsqueeze(1) * realized_cum  # (B, H); >0 = with the forecast
    favorable = directional.amax(dim=1).clamp_min(0.0)
    adverse = (-directional).amax(dim=1).clamp_min(0.0)
    captured = (favorable / magnitude)[tradeable]
    adverse_ratio = (adverse / magnitude)[tradeable]
    return {
        "mean_captured_fraction": float(captured.mean()),
        "mean_adverse_ratio": float(adverse_ratio.mean()),
        "median_captured_fraction": float(captured.median()),
        "n_used": n_used,
    }


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


def evaluate_model(
    *, checkpoint: Path, label: str, fold_index: int, batch_size: int, device: str
) -> dict[str, object]:
    """Load a checkpoint, evaluate on its fold's held-out split, return the result record."""
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    meta: dict[str, object] = loaded if isinstance(loaded, dict) else {}
    state = meta.get("state_dict", loaded)
    if not isinstance(state, dict):
        raise SystemExit("[stop] checkpoint has no usable state_dict; corrupt or wrong file.")
    embedded = meta.get("target_semantics")
    if embedded is not None and embedded != SEMANTICS:
        raise SystemExit(
            f"[stop] checkpoint target_semantics {embedded!r} != this branch's {SEMANTICS!r} — "
            f"evaluate it on the branch it was trained on."
        )
    lookback = meta.get("lookback", DATA.LOOKBACK)
    if not isinstance(lookback, int):
        raise SystemExit(f"[stop] checkpoint lookback is not an int: {lookback!r}")

    model = PatchTST(lookback=lookback)
    model.load_state_dict(state)
    model.to(torch.device(device))

    csv_path = REPO_ROOT / DATA.KRAKEN_HISTORY_OUT_DIR / DATA.KRAKEN_HISTORY_CSV_NAME
    if not csv_path.exists():
        raise SystemExit(f"[stop] {csv_path} not found; real data required for a fair bake-off.")
    features, feature_ts, fold = _prepare_fold(csv_path, fold_index)
    _, _, val_loader = build_fold_loaders(
        features, feature_ts, fold,
        lookback=lookback, batch_size=batch_size, device=device,
    )
    pred, target_raw = _gather(model, val_loader)
    target_model = target_to_model_space(target_raw, SEMANTICS)

    return {
        "label": label,
        "semantics": SEMANTICS,
        "checkpoint": str(checkpoint),
        "lookback": lookback,
        "fold_index": fold_index,
        "n_windows": int(pred.shape[0]),
        "statistical": statistical_metrics(pred, target_model),
        "economic": excursion_metrics(pred, target_raw, SEMANTICS),
    }


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
    is aggregated the same way; a group's absence (e.g. the checkpoint ``eval`` path has no
    ``timing``) is simply skipped.
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


# --------------------------------------------------------------------------- comparison

_STAT_ROWS = ["q90_coverage", "calibration_rate", "directional_accuracy", "sharpness_close",
              "pinball"]
_ECON_ROWS = ["mean_captured_fraction", "median_captured_fraction", "mean_adverse_ratio", "n_used"]
_TIMING_ROWS = ["wall_seconds", "seconds_per_epoch", "epochs"]


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_comparison(results: list[dict[str, object]]) -> str:
    """Render loaded result records as a side-by-side text table (labels = columns)."""
    labels = [str(r["label"]) for r in results]
    width = max([12, *(len(x) for x in labels)])
    header = "metric".ljust(24) + "".join(x.rjust(width + 2) for x in labels)
    lines = [header, "-" * len(header)]

    def row(name: str, group: str) -> str:
        cells = ""
        for r in results:
            metrics = r.get(group, {}) if isinstance(r, dict) else {}
            value = metrics.get(name, "-") if isinstance(metrics, dict) else "-"
            cells += _fmt(value).rjust(width + 2)
        return name.ljust(24) + cells

    lines.append(
        "semantics".ljust(24) + "".join(str(r["semantics"]).rjust(width + 2) for r in results)
    )
    lines.append("[statistical — coverage/cal/DA comparable; sharpness/pinball per-branch only]")
    lines += [row(n, "statistical") for n in _STAT_ROWS]
    lines.append("[economic — comparable across branches]")
    lines += [row(n, "economic") for n in _ECON_ROWS]
    if any("timing" in r for r in results):
        lines.append("[timing — secondary: faster is better only at equal quality]")
        lines += [row(n, "timing") for n in _TIMING_ROWS]
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI

def _run_eval(args: argparse.Namespace) -> int:
    result = evaluate_model(
        checkpoint=args.checkpoint, label=args.label, fold_index=args.fold,
        batch_size=args.batch_size, device=args.device,
    )
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"[eval] {args.label}: wrote {args.out}")
    return 0


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


def _run_compare(args: argparse.Namespace) -> int:
    results = [json.loads(p.read_text(encoding="utf-8")) for p in args.results]
    print(format_comparison(results))
    return 0


def _batch_size_default() -> int:
    return int(getattr(PREDICTOR, "PROD_BATCH_SIZE", PREDICTOR.SMOKE_BATCH_SIZE))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    te = sub.add_parser(
        "train-eval", help="train N capped seeds on a fold + evaluate -> seed-aggregated JSON"
    )
    te.add_argument("--label", required=True, help="column name in the comparison (e.g. main)")
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

    ev = sub.add_parser("eval", help="evaluate one pre-trained checkpoint on its held-out fold")
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--label", required=True, help="column name in the comparison (e.g. main)")
    ev.add_argument("--out", type=Path, required=True, help="result JSON path")
    ev.add_argument("--fold", type=int, default=0)
    ev.add_argument("--batch-size", type=int, dest="batch_size", default=_batch_size_default())
    ev.add_argument("--device", default="cpu")
    ev.set_defaults(func=_run_eval)

    cmp = sub.add_parser("compare", help="print a side-by-side table of result JSONs")
    cmp.add_argument("results", type=Path, nargs="+")
    cmp.set_defaults(func=_run_compare)

    args = ap.parse_args()
    func: object = args.func
    assert callable(func)
    return int(func(args))


if __name__ == "__main__":
    sys.exit(main())
