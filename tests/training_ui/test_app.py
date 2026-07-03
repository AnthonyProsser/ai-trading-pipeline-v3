"""Training UI FastAPI app tests (training-dashboard.md §"Training controls",
§"Live metric stream"). TrainingRunner's `run_training` is injectable so these tests
exercise the state machine / pub-sub / HTTP guard rails without real PyTorch training
or a real dataset — the actual training path (train_all_folds over real/synthetic
data) is covered by tests/predictor/test_training.py.

Committed before src/training_ui/app.py, per the test-first discipline.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from src.training_ui.app import TrainingRunner, create_app


def _runner(
    tmp_path: Path,
    *,
    data_available: bool = True,
    run_training: Callable[[TrainingRunner], None] = lambda r: None,
) -> TrainingRunner:
    return TrainingRunner(
        data_available=data_available, checkpoint_dir=tmp_path, run_training=run_training
    )


def test_runner_starts_in_data_missing_when_unavailable(tmp_path: Path) -> None:
    runner = _runner(tmp_path, data_available=False)
    assert runner.state == "data-missing"
    assert runner.start() is False


def test_runner_start_spawns_thread_and_reaches_stopped(tmp_path: Path) -> None:
    started = threading.Event()

    def fake_train(r: TrainingRunner) -> None:
        started.set()
        r.broadcast({"type": "status", "state": "stopped"})

    runner = _runner(tmp_path, run_training=fake_train)
    assert runner.start() is True
    assert started.wait(timeout=2)

    deadline = time.monotonic() + 2
    while runner.state != "stopped" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert runner.state == "stopped"


def test_runner_start_rejects_while_already_running(tmp_path: Path) -> None:
    gate = threading.Event()

    def fake_train(r: TrainingRunner) -> None:
        gate.wait(timeout=2)

    runner = _runner(tmp_path, run_training=fake_train)
    assert runner.start() is True
    assert runner.start() is False
    gate.set()


def test_runner_start_is_atomic_under_concurrent_calls(tmp_path: Path) -> None:
    """Regression test: two threads racing through start() must not both pass the
    "not already running" check — exactly one may spawn a training thread. Uses a
    barrier to force both calls to arrive at start() simultaneously, widening the
    race window the check-then-act guard must survive."""
    barrier = threading.Barrier(2)
    gate = threading.Event()

    def fake_train(r: TrainingRunner) -> None:
        gate.wait(timeout=2)

    runner = _runner(tmp_path, run_training=fake_train)
    results: list[bool] = []

    def call_start() -> None:
        barrier.wait(timeout=2)
        results.append(runner.start())

    threads = [threading.Thread(target=call_start) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2)

    assert sorted(results) == [False, True]
    gate.set()


def test_runner_stop_and_save_rejected_when_idle(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    assert runner.stop() is False
    assert runner.save() is False


def test_runner_done_state_refuses_start_and_stop(tmp_path: Path) -> None:
    """Natural run completion ("done", broadcast by train_all_folds when every fold
    finishes) is terminal: the whole walk-forward run is over, so start and stop both
    refuse — the UI's grayed-out "Done" button mirrors this server-side guard."""

    def fake_train(r: TrainingRunner) -> None:
        r.broadcast({"type": "status", "state": "done"})

    runner = _runner(tmp_path, run_training=fake_train)
    assert runner.start() is True
    deadline = time.monotonic() + 2
    while runner.state != "done" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert runner.state == "done"
    assert runner.start() is False
    assert runner.stop() is False


def test_http_start_and_stop_return_409_when_done(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    def fake_train(r: TrainingRunner) -> None:
        r.broadcast({"type": "status", "state": "done"})

    runner = _runner(tmp_path, run_training=fake_train)
    client = TestClient(create_app(runner=runner))
    assert client.post("/api/training/start").status_code == 200
    deadline = time.monotonic() + 2
    while runner.state != "done" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert client.post("/api/training/start").status_code == 409
    assert client.post("/api/training/stop").status_code == 409


def test_runner_broadcast_fans_out_to_all_subscribers(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    q1 = runner.subscribe()
    q2 = runner.subscribe()
    runner.broadcast({"type": "alert", "level": "good", "message": "hi"})
    assert q1.get(timeout=1) == {"type": "alert", "level": "good", "message": "hi"}
    assert q2.get(timeout=1) == {"type": "alert", "level": "good", "message": "hi"}


def test_runner_unsubscribe_stops_delivery(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    q = runner.subscribe()
    runner.unsubscribe(q)
    runner.broadcast({"type": "alert", "level": "good", "message": "hi"})
    assert q.empty()


def test_http_start_returns_409_when_data_missing(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create_app(runner=_runner(tmp_path, data_available=False)))
    assert client.post("/api/training/start").status_code == 409


def test_http_stop_and_save_return_409_when_idle(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create_app(runner=_runner(tmp_path)))
    assert client.post("/api/training/stop").status_code == 409
    assert client.post("/api/training/save").status_code == 409


def test_http_config_exposes_thresholds_from_constants(tmp_path: Path) -> None:
    """The frontend must read color/behavior thresholds from constants.py via this
    endpoint rather than hardcoding duplicates — decisions-auditor finding: JS
    literals silently drift from constants.py when the latter changes."""
    from fastapi.testclient import TestClient

    from constants import PREDICTOR, TRAINING_UI

    client = TestClient(create_app(runner=_runner(tmp_path)))
    resp = client.get("/api/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["grad_clip_norm"] == PREDICTOR.GRAD_CLIP_NORM
    assert body["grad_norm_instability_multiplier"] == TRAINING_UI.GRAD_NORM_INSTABILITY_MULTIPLIER
    assert body["deploy_gate_da_threshold"] == PREDICTOR.DEPLOY_GATE_DA_THRESHOLD
    assert body["fold_table_da_amber_floor"] == TRAINING_UI.FOLD_TABLE_DA_AMBER_FLOOR
    assert body["fold_table_qcov_healthy_lower"] == TRAINING_UI.FOLD_TABLE_QCOV_HEALTHY_LOWER
    assert body["fold_table_qcov_healthy_upper"] == TRAINING_UI.FOLD_TABLE_QCOV_HEALTHY_UPPER
    assert body["patience_amber_floor"] == TRAINING_UI.PATIENCE_AMBER_FLOOR
    assert body["patience_orange_floor"] == TRAINING_UI.PATIENCE_ORANGE_FLOOR
    assert body["patience_glow_floor"] == TRAINING_UI.PATIENCE_GLOW_FLOOR
    assert body["early_stopping_patience"] == PREDICTOR.EARLY_STOPPING_PATIENCE
    assert body["max_epochs"] == PREDICTOR.MAX_EPOCHS
    assert body["loss_chart_ema_window"] == TRAINING_UI.LOSS_CHART_EMA_WINDOW
    assert body["loss_chart_visible_epochs"] == TRAINING_UI.LOSS_CHART_VISIBLE_EPOCHS
    assert body["batch_trace_max_points"] == TRAINING_UI.BATCH_TRACE_MAX_POINTS
    assert body["alert_auto_dismiss_seconds"] == TRAINING_UI.ALERT_AUTO_DISMISS_SECONDS
    assert body["alert_max_stack"] == TRAINING_UI.ALERT_MAX_STACK
    assert body["button_saved_flash_seconds"] == TRAINING_UI.BUTTON_SAVED_FLASH_SECONDS


def test_http_history_reads_exporter_records(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from src.training_ui.exporter import append_fold_record, hyperparams_snapshot

    rec = {
        "fold": 0, "train_loss": 0.1, "val_loss": 0.2, "da": 0.5, "q_coverage": 0.9,
        "duration_s": 10.0, "hyperparams": hyperparams_snapshot(lookback=1440),
    }
    append_fold_record(tmp_path, rec)  # type: ignore[arg-type]

    client = TestClient(create_app(runner=_runner(tmp_path)))
    resp = client.get("/api/training/history")
    assert resp.status_code == 200
    assert resp.json() == [rec]


def test_default_run_training_filters_pre_historical_start_candles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test: the dashboard's real-data path must drop pre-HISTORICAL_START
    candles before building folds, exactly like scripts/train_predictor.py::prepare
    does — without the filter the full Kraken CSV (2013+) inflates the fold count."""
    import numpy as np
    import numpy.typing as npt

    import src.training_ui.app as app_module
    from constants import DATA
    from src.data.walk_forward import Fold, carve_locked_test, make_folds

    # n_post yields exactly 1 fold after the locked-test carve; the pre-cutoff candles
    # would add 2 more folds if they leaked through unfiltered.
    n_pre, n_post = 110_000, 331_000
    cutoff = np.datetime64(DATA.HISTORICAL_START, "m")
    timestamps = cutoff + np.arange(-n_pre, n_post) * np.timedelta64(1, "m")
    ohlcv = np.tile(np.array([100.0, 101.0, 99.0, 100.0, 1.0]), (timestamps.shape[0], 1))
    monkeypatch.setattr(app_module, "_load_real_candles", lambda path: (timestamps, ohlcv))

    captured: dict[str, object] = {}

    def fake_train_all_folds(
        features: npt.NDArray[np.float64],
        feature_ts: npt.NDArray[np.datetime64],
        folds: list[Fold],
        **kwargs: object,
    ) -> None:
        captured["feature_ts"] = feature_ts
        captured["folds"] = folds

    monkeypatch.setattr(app_module, "train_all_folds", fake_train_all_folds)

    app_module._default_run_training(_runner(tmp_path))

    feature_ts = captured["feature_ts"]
    folds = captured["folds"]
    assert isinstance(feature_ts, np.ndarray)
    assert isinstance(folds, list)
    assert feature_ts.min() >= cutoff
    assert len(folds) == len(make_folds(carve_locked_test(n_post)))


def test_http_start_success_returns_running_state(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    done = threading.Event()

    def fake_train(r: TrainingRunner) -> None:
        done.wait(timeout=2)
        r.broadcast({"type": "status", "state": "stopped"})

    client = TestClient(create_app(runner=_runner(tmp_path, run_training=fake_train)))
    resp = client.post("/api/training/start")
    assert resp.status_code == 200
    assert resp.json()["state"] == "running"
    done.set()
