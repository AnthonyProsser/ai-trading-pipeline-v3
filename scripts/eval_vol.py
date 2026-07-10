"""Vol-pivot evaluator: does the model's realized-variance forecast beat persistence?

For every finished checkpoint of the CURRENT branch's run (matched by git_sha), gathers the
final-step CLOSE prediction (predicted cumulative realized variance over HORIZON) on the
fold TEST slice and compares, on the SAME windows:
  - model:       Spearman(pred_q50_final, realized_RV)
  - persistence: Spearman(prior_window_RV, realized_RV)   (RV of the 15 min just before origin)
plus q90 coverage (fraction realized_RV <= pred_q90_final). Pooled over folds.

The bar (VOL_PIVOT_SPEC.md): model Spearman must beat persistence, with q90 coverage in
[0.85, 0.95]. Level-MSE/pinball are spike-dominated and NOT the bar.

Usage (on the run's branch): uv run python scripts/eval_vol.py
"""
from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import numpy.typing as npt
import torch

from constants import DATA, PREDICTOR
from src.benchmark.engine import _gather_predictions, _load_features, eval_slice_bounds
from src.benchmark.registry import parse_run_tag, read_checkpoint_meta
from src.data.walk_forward import carve_locked_test, make_folds
from src.predictor.model import PatchTST

_CLOSE = DATA.FEATURE_NAMES.index("close_logret")
_Q50 = PREDICTOR.QUANTILES.index(0.50)
_Q90 = PREDICTOR.QUANTILES.index(0.90)
FIN = REPO_ROOT / PREDICTOR.FINISHED_DIR


def _spearman(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    feature_ts, features = _load_features(REPO_ROOT / DATA.KRAKEN_HISTORY_OUT_DIR / DATA.KRAKEN_HISTORY_CSV_NAME)
    folds = make_folds(carve_locked_test(features.shape[0]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H = PREDICTOR.HORIZON

    m_pred, m_real, m_pers = [], [], []
    cover_hits = cover_n = 0
    n_ckpt = 0
    for w in sorted(FIN.glob(f"*{PREDICTOR.CHECKPOINT_WEIGHTS_SUFFIX}")):
        stem = w.name[: -len(PREDICTOR.CHECKPOINT_WEIGHTS_SUFFIX)]
        tag = parse_run_tag(stem)
        if tag is None or tag.git_sha != git_sha:
            continue
        meta = read_checkpoint_meta(w)
        lookback = int(meta["lookback"])  # type: ignore[arg-type]
        model = PatchTST(lookback=lookback)
        model.load_state_dict(torch.load(w, weights_only=True)["state_dict"])
        model.to(device)
        with open(FIN / f"{stem}.scaler.pkl", "rb") as fh:
            scaler = pickle.load(fh)
        fold = folds[tag.fold_index]
        start, end = eval_slice_bounds(fold, lookback=lookback)
        sf = features[start:end]
        pred, _ = _gather_predictions(model, scaler.transform_inference(sf), sf, lookback=lookback, device=device)
        pq50 = pred[:, -1, _CLOSE, _Q50].to(torch.float64).numpy()
        pq90 = pred[:, -1, _CLOSE, _Q90].to(torch.float64).numpy()
        # Realized + persistence RV from raw close returns via a squared-cumsum (O(N)).
        csq = np.concatenate([[0.0], np.cumsum(sf[:, _CLOSE].astype(np.float64) ** 2)])
        n = pred.shape[0]
        i = np.arange(n)
        realized = csq[i + lookback + H] - csq[i + lookback]          # RV of [origin, origin+H)
        persist = csq[i + lookback] - csq[i + lookback - H]           # RV of [origin-H, origin)
        m_pred.append(pq50); m_real.append(realized); m_pers.append(persist)
        cover_hits += int(np.count_nonzero(realized <= pq90)); cover_n += n
        n_ckpt += 1

    if n_ckpt == 0:
        print(f"no finished checkpoints for git_sha={git_sha}")
        return 1
    pred = np.concatenate(m_pred); real = np.concatenate(m_real); pers = np.concatenate(m_pers)
    print(f"git_sha={git_sha}  checkpoints={n_ckpt}  windows={real.size}")
    print(f"MODEL       Spearman(pred_q50, realized_RV) = {_spearman(pred, real):.4f}")
    print(f"PERSISTENCE Spearman(prior_RV, realized_RV) = {_spearman(pers, real):.4f}")
    print(f"q90 coverage = {cover_hits / cover_n:.4f}  (bar: 0.85-0.95)")
    print("BAR: model Spearman must beat persistence AND coverage in band.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
