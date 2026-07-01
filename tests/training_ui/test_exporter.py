"""Fold-completion export tests (training-dashboard.md §"Fold history export").
training_metrics.json is a JSON array under PredictorConfig.CHECKPOINT_DIR /
TRAINING_METRICS_FILENAME; one record appended per completed fold. hyperparams are
frozen at training-start time, not re-read per fold.

Committed before src/training_ui/exporter.py, per the test-first discipline.
"""
from __future__ import annotations

import json
from pathlib import Path

from constants import PREDICTOR


def test_hyperparams_snapshot_reflects_predictor_constants() -> None:
    from src.training_ui.exporter import hyperparams_snapshot

    snap = hyperparams_snapshot(lookback=720)
    assert snap == {
        "lookback": 720,
        "horizon": PREDICTOR.HORIZON,
        "patch_size": PREDICTOR.PATCH_SIZE,
        "d_model": PREDICTOR.D_MODEL,
        "direction_lambda": PREDICTOR.DIRECTION_PENALTY_LAMBDA,
        "max_epochs": PREDICTOR.MAX_EPOCHS,
        "early_stopping_patience": PREDICTOR.EARLY_STOPPING_PATIENCE,
    }


def test_append_fold_record_creates_file_and_array(tmp_path: Path) -> None:
    from src.training_ui.exporter import append_fold_record, hyperparams_snapshot

    record = {
        "fold": 0, "train_loss": 0.61, "val_loss": 0.59, "da": 0.512,
        "q_coverage": 0.91, "duration_s": 120.5,
        "hyperparams": hyperparams_snapshot(lookback=1440),
    }
    path = append_fold_record(tmp_path, record)  # type: ignore[arg-type]

    assert path == tmp_path / PREDICTOR.TRAINING_METRICS_FILENAME
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data == [record]


def test_append_fold_record_appends_in_order(tmp_path: Path) -> None:
    from src.training_ui.exporter import append_fold_record, hyperparams_snapshot

    hp = hyperparams_snapshot(lookback=1440)
    rec0 = {
        "fold": 0, "train_loss": 0.61, "val_loss": 0.59, "da": 0.512,
        "q_coverage": 0.91, "duration_s": 120.5, "hyperparams": hp,
    }
    rec1 = {
        "fold": 1, "train_loss": 0.58, "val_loss": 0.55, "da": 0.53,
        "q_coverage": 0.89, "duration_s": 130.0, "hyperparams": hp,
    }
    append_fold_record(tmp_path, rec0)  # type: ignore[arg-type]
    path = append_fold_record(tmp_path, rec1)  # type: ignore[arg-type]

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert [r["fold"] for r in data] == [0, 1]


def test_read_fold_records_empty_when_file_absent(tmp_path: Path) -> None:
    from src.training_ui.exporter import read_fold_records

    assert read_fold_records(tmp_path) == []


def test_read_fold_records_round_trips(tmp_path: Path) -> None:
    from src.training_ui.exporter import (
        append_fold_record,
        hyperparams_snapshot,
        read_fold_records,
    )

    rec = {
        "fold": 2, "train_loss": 0.5, "val_loss": 0.48, "da": 0.54,
        "q_coverage": 0.90, "duration_s": 140.0,
        "hyperparams": hyperparams_snapshot(lookback=1440),
    }
    append_fold_record(tmp_path, rec)  # type: ignore[arg-type]
    assert read_fold_records(tmp_path) == [rec]
