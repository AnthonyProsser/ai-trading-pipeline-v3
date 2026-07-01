"""Training UI startup data-gate tests (training-dashboard.md §"Data gate (startup
check)"). No upload UI (DECISIONS.md training_ui_data_gate) — this is a one-shot
existence check evaluated at process startup, never hot-reloaded.

Committed before src/training_ui/setup_router.py, per the test-first discipline.
"""
from __future__ import annotations

from pathlib import Path

from constants import DATA


def test_check_data_gate_true_when_csv_present(tmp_path: Path) -> None:
    from src.training_ui.setup_router import check_data_gate

    out_dir = tmp_path / DATA.KRAKEN_HISTORY_OUT_DIR
    out_dir.mkdir(parents=True)
    csv_path = out_dir / DATA.KRAKEN_HISTORY_CSV_NAME
    csv_path.write_text("x")

    status = check_data_gate(repo_root=tmp_path)
    assert status.available is True
    assert status.csv_path == csv_path


def test_check_data_gate_false_when_csv_missing(tmp_path: Path) -> None:
    from src.training_ui.setup_router import check_data_gate

    status = check_data_gate(repo_root=tmp_path)
    assert status.available is False
