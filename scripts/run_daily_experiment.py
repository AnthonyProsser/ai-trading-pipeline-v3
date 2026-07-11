"""Headless full walk-forward training run on 1-HOUR candles -> checkpoints/
(idea-07-daily-hourly, experiments/idea_search/ledger.jsonl idea 7).

Sibling of scripts/run_full_training.py (tooling-full-run branch, not on this branch)
but resamples the raw 1-minute Kraken candles to 1-hour bars before building folds.
Tests the user thesis: does 1-day-ahead (HORIZON=24 hourly candles) directional
accuracy reach ~54% on hourly-resolution data? DATA.LOOKBACK/HORIZON/WALK_FORWARD_*/
LOCKED_TEST_CANDLES are all set to their hourly-experiment values on this branch (see
constants.py) -- this is a DIRECTIONAL experiment; target stays cumulative_logret
(main's default), loss/model architecture untouched.

`validate_candles` is intentionally NOT run: it reindexes onto a native 1-minute grid
(np.timedelta64(1, "m")) and would treat every ~59-minute gap between hourly bars as a
long interpolation gap, silently corrupting the data rather than validating it. Gap/
corruption handling is bypassed for this experiment; resampled candles are used as-is.

Usage:
    uv run python scripts/run_daily_experiment.py --device cuda
    uv run python scripts/run_daily_experiment.py --device cuda --folds 0-9
    uv run python scripts/run_daily_experiment.py --dry-run   # build folds, print count, no train
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from constants import DATA, PREDICTOR
from src.data.feature_pipeline import compute_features
from src.data.manifest import sha256_file
from src.data.resample import resample_to_hourly
from src.data.walk_forward import Fold, carve_locked_test, filter_by_historical_start, make_folds
from src.predictor.training import train_all_folds

# Per-round-trip MAKER fee estimate for the economics printout below (limit-order
# entries are realistic at the daily trading pace this experiment targets, vs the
# TAKER-based EXECUTION.FEE_THRESHOLD used during training/DA evaluation). NOT yet a
# locked DECISIONS.md value -- flagged unspecced in
# experiments/idea_search/ledger.jsonl idea 7 ("maker_fee": 0.002). Kept local to this
# reporting-only script rather than added to constants.py.
_MAKER_FEE_ROUND_TRIP = 0.0020


def _git_short_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip() or "nogit"
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "nogit"


def _load_real_candles(path: Path) -> tuple[npt.NDArray[np.datetime64], npt.NDArray[np.float64]]:
    arr: npt.NDArray[np.float64] = np.loadtxt(
        path, delimiter=",", usecols=(0, 1, 2, 3, 4, 5), dtype=np.float64
    )
    timestamps = arr[:, 0].astype("int64").astype("datetime64[s]")
    ohlcv = arr[:, 1:6].astype(np.float64)
    return timestamps, ohlcv


def _parse_folds(spec: str, n: int) -> list[int] | None:
    """'all' -> None (every fold); '0-19' -> [0..19]; '0,8,16' -> those indices."""
    if spec == "all":
        return None
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return [i for i in out if 0 <= i < n]


def _mean_abs_horizon_move(
    close_logret: npt.NDArray[np.float64], fold: Fold, horizon: int
) -> float:
    """Mean |cumulative close log-return| over non-overlapping HORIZON-step windows
    inside the fold's TEST slice -- the realized move size the fixed-fee economics
    (net_per_trade_at_maker below) are measured against."""
    seg = close_logret[fold.test_start : fold.test_end]
    n_windows = seg.shape[0] // horizon
    if n_windows == 0:
        return float("nan")
    cum = seg[: n_windows * horizon].reshape(n_windows, horizon).sum(axis=1)
    return float(np.abs(cum).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folds", default="all", help="'all', a range '0-19', or a list '0,8,16'")
    ap.add_argument("--max-epochs", type=int, default=PREDICTOR.MAX_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=PREDICTOR.PROD_BATCH_SIZE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    csv_path = REPO_ROOT / DATA.KRAKEN_HISTORY_OUT_DIR / DATA.KRAKEN_HISTORY_CSV_NAME
    if not csv_path.exists():
        raise SystemExit(f"[stop] {csv_path} not found; real data required.")
    minute_ts, minute_ohlcv = _load_real_candles(csv_path)
    hourly_ts, hourly_ohlcv = resample_to_hourly(minute_ts, minute_ohlcv)
    print(
        f"[resample] {minute_ts.shape[0]} 1-min candles -> {hourly_ts.shape[0]} 1-hour candles",
        flush=True,
    )

    features = compute_features(hourly_ohlcv)
    feature_ts = hourly_ts[1:]
    feature_ts, features = filter_by_historical_start(feature_ts, features)
    folds = make_folds(carve_locked_test(features.shape[0]))

    keep = _parse_folds(args.folds, len(folds))
    if keep is not None:
        folds = [f for f in folds if f.index in keep]

    print(
        f"[run_daily_experiment] {len(folds)} folds, lookback={DATA.LOOKBACK}, "
        f"horizon={PREDICTOR.HORIZON}, batch={args.batch_size}, "
        f"max_epochs={args.max_epochs}, device={args.device}",
        flush=True,
    )
    if args.dry_run:
        print("[dry-run] no training.", flush=True)
        return 0

    close_logret = features[:, DATA.FEATURE_NAMES.index("close_logret")]

    t0 = time.perf_counter()
    done = 0
    das: list[float] = []

    def log(payload: dict[str, object]) -> None:
        nonlocal done
        kind = payload.get("type")
        if kind == "fold_complete":
            done += 1
            da = payload.get("da")
            if isinstance(da, float):
                das.append(da)
            el = time.perf_counter() - t0
            print(
                f"[fold_complete] {done}/{len(folds)} fold={payload.get('fold')} "
                f"da={da} qcov={payload.get('q_coverage')} "
                f"dur_s={payload.get('duration_s')} elapsed_min={el / 60:.1f}",
                flush=True,
            )
        elif kind in ("status", "alert"):
            print(f"[{kind}] {payload}", flush=True)

    train_all_folds(
        features, feature_ts, folds,
        lookback=DATA.LOOKBACK, batch_size=args.batch_size, device=args.device,
        max_epochs=args.max_epochs, checkpoint_dir=REPO_ROOT / PREDICTOR.CHECKPOINT_DIR,
        git_sha=_git_short_sha(), constants_sha=sha256_file(REPO_ROOT / "constants.py"),
        log=log, finished_dir=None,
    )
    print(f"[done] {done} folds in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    moves = [_mean_abs_horizon_move(close_logret, fold, PREDICTOR.HORIZON) for fold in folds]
    valid_das = [d for d in das if np.isfinite(d)]
    valid_moves = [m for m in moves if np.isfinite(m)]
    mean_da = float(np.mean(valid_das)) if valid_das else float("nan")
    mean_abs_move = float(np.mean(valid_moves)) if valid_moves else float("nan")
    net_per_trade_at_maker = (2 * mean_da - 1) * mean_abs_move - _MAKER_FEE_ROUND_TRIP

    print(f"[summary] per-fold da = {das}", flush=True)
    print(f"[summary] mean da = {mean_da:.4f} ({len(valid_das)}/{len(folds)} folds)", flush=True)
    print(f"[summary] per-fold mean|move| = {moves}", flush=True)
    print(f"[summary] mean|move| = {mean_abs_move:.6f}", flush=True)
    print(f"[summary] net_per_trade_at_maker = {net_per_trade_at_maker:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
