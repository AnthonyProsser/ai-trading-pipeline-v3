"""train_predictor.py — predictor training entry point and Phase 1 smoke gate.

`--smoke` runs 1 epoch on 1 fold to catch OOM / NaN / dataloader issues before the
4-week run. Batch size defaults to PredictorConfig.SMOKE_BATCH_SIZE (32); on a CUDA
OOM it drops to SMOKE_BATCH_SIZE_FALLBACK (16); if that still OOMs it STOPs and reports
(the card's "Azure A100 decision point") rather than silently proceeding.

Data:
  --synthetic  generate random-walk candles in-memory (a reduced fold) — a faithful
               GPU/plumbing smoke at the real batch/lookback footprint, used when the
               real CSV is unavailable.
  (default)    read data/raw/BTCUSD_1.csv (Kraken OHLCVT, no header:
               unix_seconds, open, high, low, close, volume, trades), validate, build
               features, then carve the locked test block and walk-forward folds.

Logging is W&B (--wandb online|offline|disabled, default offline); pinball, direction,
and total are logged separately. If wandb is not installed it degrades to stdout with a
warning (run `uv add wandb` to enable). This script is a thin CLI over
src/predictor/training.py — all reusable logic and every magic number live there / in
constants.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from constants import DATA, PREDICTOR
from src.data.feature_pipeline import compute_features
from src.data.manifest import sha256_file
from src.data.scaler import PerFoldMinMaxScaler
from src.data.validator import validate_candles
from src.data.walk_forward import Fold, carve_locked_test, make_folds
from src.predictor.model import PatchTST
from src.predictor.training import LogFn, build_fold_loaders, make_run_tag, train_one_fold

CandleArrays = tuple[npt.NDArray[np.datetime64], npt.NDArray[np.float64]]


def git_short_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip() or "nogit"
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "nogit"


def scaler_sha(scaler: PerFoldMinMaxScaler) -> str:
    """Stable hash of the fitted scaler's fold statistics (part of the run tag).

    Hashes the raw array buffers (not pickle) so the tag is reproducible across
    Python versions.
    """
    if scaler.data_min_ is None or scaler.data_max_ is None:
        raise ValueError("scaler_sha requires a fitted scaler")
    return hashlib.sha256(scaler.data_min_.tobytes() + scaler.data_max_.tobytes()).hexdigest()


def synthetic_candles(n: int, seed: int) -> CandleArrays:
    """Geometry-valid random-walk OHLCV on a contiguous 1-minute grid (passes the validator)."""
    rng = np.random.default_rng(seed)
    close = 30_000.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.001))
    open_ = np.empty(n, dtype=np.float64)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)
    wick = np.abs(rng.standard_normal(n)) * close * 0.0005
    high = body_high + wick
    low = np.maximum(body_low - wick, 1e-6)
    volume = np.abs(rng.standard_normal(n)) * 10.0 + 1.0
    ohlcv = np.column_stack([open_, high, low, close, volume]).astype(np.float64)
    base = np.datetime64("2020-01-01T00:00")
    timestamps = base + np.arange(n).astype("timedelta64[m]")
    return timestamps, ohlcv


def load_real_candles(path: Path) -> CandleArrays:
    """Read a Kraken OHLCVT CSV (no header) into (timestamps, OHLCV). Exercised once
    data/raw is populated; not run in --synthetic mode."""
    arr: npt.NDArray[np.float64] = np.loadtxt(
        path, delimiter=",", usecols=(0, 1, 2, 3, 4, 5), dtype=np.float64
    )
    timestamps = arr[:, 0].astype("int64").astype("datetime64[s]")
    ohlcv = arr[:, 1:6].astype(np.float64)
    return timestamps, ohlcv


def prepare(
    *, synthetic: bool, lookback: int, fold_index: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.datetime64], Fold]:
    """Load candles -> validate -> features, and pick the fold to train."""
    if synthetic:
        # Synthetic-only sizing (NOT a locked decision; real runs use the walk_forward
        # constants). Multiples of lookback guarantee enough windows per slice.
        train_n, val_n, test_n = lookback * 4, lookback * 2, lookback
        timestamps, ohlcv = synthetic_candles(train_n + val_n + test_n + 1, PREDICTOR.SEED)
        validated = validate_candles(timestamps, ohlcv)
        features = compute_features(validated.ohlcv)
        feature_ts = validated.timestamps[1:]
        fold = Fold(
            0, 0, train_n, train_n, train_n + val_n, train_n + val_n, train_n + val_n + test_n
        )
        return features, feature_ts, fold

    csv_path = REPO_ROOT / DATA.KRAKEN_HISTORY_OUT_DIR / "BTCUSD_1.csv"
    if not csv_path.exists():
        raise SystemExit(
            f"[stop] {csv_path} not found. Place BTCUSD_1.csv there first, "
            f"or pass --synthetic for a GPU/plumbing smoke without real data."
        )
    timestamps, ohlcv = load_real_candles(csv_path)
    validated = validate_candles(timestamps, ohlcv)
    features = compute_features(validated.ohlcv)
    feature_ts = validated.timestamps[1:]
    folds = make_folds(carve_locked_test(features.shape[0]))
    if not 0 <= fold_index < len(folds):
        raise SystemExit(f"[stop] fold {fold_index} out of range (0..{len(folds) - 1})")
    return features, feature_ts, folds[fold_index]


def make_logger(mode: str, project: str, run_tag: str) -> LogFn:
    def to_stdout(payload: dict[str, object]) -> None:
        print(f"[metric] {json.dumps(payload)}")

    if mode == "disabled":
        return to_stdout
    try:
        import wandb
    except ImportError:
        print(
            f"[warn] wandb not installed; --wandb {mode} degraded to stdout "
            f"(run `uv add wandb` to enable)."
        )
        return to_stdout
    run = wandb.init(project=project, name=run_tag, mode=mode)
    if run is None:  # some wandb configs return None; fall back to stdout
        return to_stdout

    def to_both(payload: dict[str, object]) -> None:
        to_stdout(payload)
        run.log(payload)

    return to_both


def run(args: argparse.Namespace) -> int:
    torch.manual_seed(PREDICTOR.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(PREDICTOR.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_epochs = 1 if args.smoke else PREDICTOR.MAX_EPOCHS

    features, feature_ts, fold = prepare(
        synthetic=args.synthetic, lookback=args.lookback, fold_index=args.fold
    )
    constants_hash = sha256_file(REPO_ROOT / "constants.py")
    git_sha = git_short_sha()

    batch_candidates = [args.batch_size]
    if args.batch_size != PREDICTOR.SMOKE_BATCH_SIZE_FALLBACK:
        batch_candidates.append(PREDICTOR.SMOKE_BATCH_SIZE_FALLBACK)

    for batch_size in batch_candidates:
        try:
            scaler, train_loader, val_loader = build_fold_loaders(
                features, feature_ts, fold, lookback=args.lookback, batch_size=batch_size
            )
            model = PatchTST(lookback=args.lookback)
            run_tag = make_run_tag(
                git_sha=git_sha, constants_sha=constants_hash,
                scaler_sha=scaler_sha(scaler), fold_id=fold.index,
            )
            logger = make_logger(args.wandb, PREDICTOR.WANDB_PROJECT, run_tag)
            print(
                f"[run] tag={run_tag} device={device} batch={batch_size} "
                f"lookback={args.lookback} fold={fold.index} epochs={max_epochs}"
            )
            metrics = train_one_fold(
                model, scaler, train_loader, val_loader, device=device,
                max_epochs=max_epochs, log=logger,
            )
            print(
                f"[done] steps={metrics.steps} train_total={metrics.train_total:.6f} "
                f"val_total={metrics.val_total:.6f} (pinball/direction logged separately)"
            )
            print("[smoke] PASSED — finite loss, no OOM/NaN" if args.smoke else "[train] complete")
            return 0
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[oom] batch {batch_size} exhausted VRAM")
            if batch_size == PREDICTOR.SMOKE_BATCH_SIZE_FALLBACK:
                print(
                    "[stop] still OOM at the fallback batch size — Azure A100 decision "
                    "point. Do NOT start the 4-week run on this config."
                )
                return 2
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="1 epoch, 1 fold pre-launch gate")
    ap.add_argument(
        "--synthetic", action="store_true",
        help="random-walk candles in-memory (no real data needed)",
    )
    ap.add_argument("--wandb", choices=("online", "offline", "disabled"), default="offline")
    ap.add_argument(
        "--batch-size", type=int, default=PREDICTOR.SMOKE_BATCH_SIZE, dest="batch_size"
    )
    ap.add_argument("--lookback", type=int, default=DATA.LOOKBACK)
    ap.add_argument("--fold", type=int, default=0, help="walk-forward fold index (real data)")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
