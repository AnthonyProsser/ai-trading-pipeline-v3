"""Training UI startup data gate (training-dashboard.md §"Data gate (startup check)").

FastAPI checks for the raw Kraken CSV once at process startup. If absent, app.py
disables training controls and the header status line reads
"Data missing — check KRAKEN_DATA_PATH". No upload UI, no hot-reload — data
management happens outside the browser (Google Drive for Desktop sync) and the
server must be restarted after the env var is fixed. See DECISIONS.md
training_ui_data_gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from constants import DATA

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class DataGateStatus:
    available: bool
    csv_path: Path


def check_data_gate(repo_root: Path = REPO_ROOT) -> DataGateStatus:
    """Evaluated once at process startup — never re-checked while the server runs."""
    csv_path = repo_root / DATA.KRAKEN_HISTORY_OUT_DIR / DATA.KRAKEN_HISTORY_CSV_NAME
    return DataGateStatus(available=csv_path.exists(), csv_path=csv_path)
