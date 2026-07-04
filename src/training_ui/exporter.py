"""Fold-completion export to training_metrics.json (training-dashboard.md §"Fold
history export"). The handoff artifact for post-training analysis: a JSON array under
PredictorConfig.CHECKPOINT_DIR / TRAINING_METRICS_FILENAME, one record appended per
completed fold. hyperparams are captured once at training-start time (constants.py
values when the run began) and embedded per-record, so a later constants.py edit
never retroactively relabels historical folds.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NotRequired, TypedDict

from constants import PREDICTOR


class HyperparamsSnapshot(TypedDict):
    lookback: int
    horizon: int
    patch_size: int
    d_model: int
    direction_lambda: float
    max_epochs: int
    early_stopping_patience: int


class FoldRecord(TypedDict):
    fold: int
    # run-tag join key the benchmark analysis endpoint pairs results on. NotRequired so
    # records written before this field existed still validate (they simply don't join).
    stem: NotRequired[str]
    train_loss: float
    val_loss: float
    da: float
    q_coverage: float
    duration_s: float
    hyperparams: HyperparamsSnapshot


def hyperparams_snapshot(lookback: int) -> HyperparamsSnapshot:
    """Frozen-for-the-run hyperparams; `lookback` varies per run (smoke-sweep), the
    rest come straight from constants.PREDICTOR."""
    return {
        "lookback": lookback,
        "horizon": PREDICTOR.HORIZON,
        "patch_size": PREDICTOR.PATCH_SIZE,
        "d_model": PREDICTOR.D_MODEL,
        "direction_lambda": PREDICTOR.DIRECTION_PENALTY_LAMBDA,
        "max_epochs": PREDICTOR.MAX_EPOCHS,
        "early_stopping_patience": PREDICTOR.EARLY_STOPPING_PATIENCE,
    }


def metrics_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / PREDICTOR.TRAINING_METRICS_FILENAME


def append_fold_record(checkpoint_dir: Path, record: FoldRecord) -> Path:
    """Append one fold record to training_metrics.json, creating it (as a JSON array)
    on the first call. Returns the file path."""
    path = metrics_path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    records = read_fold_records(checkpoint_dir)
    records.append(record)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)
    return path


def read_fold_records(checkpoint_dir: Path) -> list[FoldRecord]:
    """All historical fold records — seeds the dashboard's fold table on page load
    (training-dashboard.md: "reads from training_metrics.json on page load")."""
    path = metrics_path(checkpoint_dir)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        records: list[FoldRecord] = json.load(fh)
    return records
