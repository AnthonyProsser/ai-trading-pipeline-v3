"""Headless full walk-forward training run (all folds) -> checkpoints/finished/.

Thin wrapper over the already-tested src.predictor.training.train_all_folds, mirroring
src/training_ui/app.py::_default_run_training but without the FastAPI/SSE/threading
layer, so a full run can be launched unattended from the shell (overnight). Each fold's
gate-evaluated checkpoint is written to CHECKPOINT_DIR during the run and COPIED into
FINISHED_DIR on natural completion of the fold list (what the benchmark app reads).

Usage:
    uv run python scripts/run_full_training.py --device cuda
    uv run python scripts/run_full_training.py --device cuda --folds 0-19
    uv run python scripts/run_full_training.py --dry-run   # build folds, print count, no train
"""
from __future__ import annotations

import argparse
import inspect
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
from src.data.validator import validate_candles
from src.data.walk_forward import carve_locked_test, filter_by_historical_start, make_folds
from src.predictor.training import train_all_folds


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folds", default="all", help="'all', a range '0-19', or a list '0,8,16'")
    ap.add_argument("--max-epochs", type=int, default=PREDICTOR.MAX_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=PREDICTOR.PROD_BATCH_SIZE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    csv_path = REPO_ROOT / DATA.KRAKEN_HISTORY_OUT_DIR / DATA.KRAKEN_HISTORY_CSV_NAME
    if not csv_path.exists():
        raise SystemExit(f"[stop] {csv_path} not found; real data required.")
    timestamps, ohlcv = _load_real_candles(csv_path)
    validated = validate_candles(timestamps, ohlcv)
    # Branch-agnostic: idea branches may extend compute_features with a timestamps arg.
    if "timestamps" in inspect.signature(compute_features).parameters:
        features = compute_features(validated.ohlcv, validated.timestamps)
    else:
        features = compute_features(validated.ohlcv)
    feature_ts = validated.timestamps[1:]
    feature_ts, features = filter_by_historical_start(feature_ts, features)
    folds = make_folds(carve_locked_test(features.shape[0]))

    keep = _parse_folds(args.folds, len(folds))
    if keep is not None:
        folds = [f for f in folds if f.index in keep]

    print(f"[run_full_training] {len(folds)} folds, lookback={DATA.LOOKBACK}, "
          f"batch={args.batch_size}, max_epochs={args.max_epochs}, device={args.device}", flush=True)
    if args.dry_run:
        print("[dry-run] no training.", flush=True)
        return 0

    t0 = time.perf_counter()
    done = 0

    def log(payload: dict[str, object]) -> None:
        nonlocal done
        kind = payload.get("type")
        if kind == "fold_complete":
            done += 1
            el = time.perf_counter() - t0
            print(f"[fold_complete] {done}/{len(folds)} fold={payload.get('fold')} "
                  f"da={payload.get('da')} qcov={payload.get('q_coverage')} "
                  f"dur_s={payload.get('duration_s')} elapsed_min={el/60:.1f}", flush=True)
        elif kind in ("status", "alert"):
            print(f"[{kind}] {payload}", flush=True)

    train_all_folds(
        features, feature_ts, folds,
        lookback=DATA.LOOKBACK, batch_size=args.batch_size, device=args.device,
        max_epochs=args.max_epochs, checkpoint_dir=REPO_ROOT / PREDICTOR.CHECKPOINT_DIR,
        git_sha=_git_short_sha(), constants_sha=sha256_file(REPO_ROOT / "constants.py"),
        log=log, finished_dir=REPO_ROOT / PREDICTOR.FINISHED_DIR,
    )
    print(f"[done] {done} folds in {(time.perf_counter()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
