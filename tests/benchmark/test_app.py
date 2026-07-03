"""Benchmark app FastAPI tests. BenchmarkRunner's `run_benchmark` is injectable so
these tests exercise the state machine / pub-sub / HTTP guard rails without loading a
real checkpoint or dataset (same pattern as tests/training_ui/test_app.py — mutating
endpoints are only ever smoke-tested against injected fakes, never a live server).

Committed before src/benchmark/app.py, per the test-first discipline.
"""
from __future__ import annotations

import json
import pickle
import threading
import time
from collections.abc import Callable
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from constants import EXECUTION, PREDICTOR
from src.benchmark.app import BenchmarkRunner, create_app
from src.benchmark.registry import write_benchmark_result

_STEM = "4d460ed-s001e4348-c487b9e2e-fold33"
_STEM2 = "4d460ed-s00c53cec-c487b9e2e-fold68"


def _write_fake_checkpoint(
    checkpoint_dir: Path, stem: str, *, semantics: str = PREDICTOR.TARGET_SEMANTICS
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {},
            "lookback": 32,
            "constants_sha256": "a" * 64,
            "trained_through_ts_utc": "2020-01-05T00:00:00Z",
            "train_q90_coverage": 0.9,
            "target_semantics": semantics,
        },
        checkpoint_dir / f"{stem}{PREDICTOR.CHECKPOINT_WEIGHTS_SUFFIX}",
    )
    with open(checkpoint_dir / f"{stem}{PREDICTOR.CHECKPOINT_SCALER_SUFFIX}", "wb") as fh:
        pickle.dump({"fake": True}, fh)


def _runner(
    tmp_path: Path,
    *,
    run_benchmark: Callable[[BenchmarkRunner, str], None] = lambda r, s: None,
) -> BenchmarkRunner:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return BenchmarkRunner(
        checkpoint_dir=checkpoint_dir,
        benchmark_dir=checkpoint_dir / "benchmark",
        run_benchmark=run_benchmark,
    )


def _wait_for_state(runner: BenchmarkRunner, state: str, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while runner.state != state and time.monotonic() < deadline:
        time.sleep(0.02)
    assert runner.state == state


# --- runner state machine ------------------------------------------------------

def test_runner_runs_job_and_returns_to_idle(tmp_path: Path) -> None:
    seen: list[str] = []

    def fake(r: BenchmarkRunner, stem: str) -> None:
        seen.append(stem)

    runner = _runner(tmp_path, run_benchmark=fake)
    assert runner.state == "idle"
    assert runner.start(_STEM) is True
    _wait_for_state(runner, "idle")
    assert seen == [_STEM]


def test_runner_rejects_start_while_running(tmp_path: Path) -> None:
    gate = threading.Event()

    def fake(r: BenchmarkRunner, stem: str) -> None:
        gate.wait(timeout=2)

    runner = _runner(tmp_path, run_benchmark=fake)
    assert runner.start(_STEM) is True
    assert runner.start(_STEM2) is False
    gate.set()


def test_runner_start_is_atomic_under_concurrent_calls(tmp_path: Path) -> None:
    """Two threads racing through start() must not both pass the not-running check
    (same TOCTOU guard the training UI runner carries)."""
    barrier = threading.Barrier(2)
    gate = threading.Event()

    def fake(r: BenchmarkRunner, stem: str) -> None:
        gate.wait(timeout=2)

    runner = _runner(tmp_path, run_benchmark=fake)
    results: list[bool] = []

    def call_start() -> None:
        barrier.wait(timeout=2)
        results.append(runner.start(_STEM))

    threads = [threading.Thread(target=call_start) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2)
    assert sorted(results) == [False, True]
    gate.set()


def test_runner_error_in_job_broadcasts_alert_and_recovers(tmp_path: Path) -> None:
    def fake(r: BenchmarkRunner, stem: str) -> None:
        raise RuntimeError("boom")

    runner = _runner(tmp_path, run_benchmark=fake)
    q = runner.subscribe()
    assert runner.start(_STEM) is True
    _wait_for_state(runner, "idle")
    # The failure surfaced as an alert, and the runner accepts a new job afterwards.
    payloads = []
    while not q.empty():
        payloads.append(q.get_nowait())
    assert any(p.get("type") == "alert" and p.get("level") == "error" for p in payloads)
    assert runner.start(_STEM2) is True
    _wait_for_state(runner, "idle")


def test_runner_broadcast_fans_out_and_unsubscribe_stops(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    q1 = runner.subscribe()
    q2 = runner.subscribe()
    runner.broadcast({"type": "alert", "level": "good", "message": "hi"})
    assert q1.get(timeout=1)["message"] == "hi"
    assert q2.get(timeout=1)["message"] == "hi"
    runner.unsubscribe(q1)
    runner.broadcast({"type": "alert", "level": "good", "message": "again"})
    assert q1.empty()


# --- HTTP endpoints --------------------------------------------------------------

def test_http_models_lists_checkpoints_with_metadata(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    client = TestClient(create_app(runner))

    r = client.get("/api/models")
    assert r.status_code == 200
    models = r.json()["models"]
    assert len(models) == 1
    assert models[0]["stem"] == _STEM
    assert models[0]["display_name"] == _STEM
    assert models[0]["meta"]["lookback"] == 32
    assert models[0]["compatible"] is True
    assert models[0]["has_benchmark"] is False


def test_http_rename_persists_and_unknown_is_404(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    client = TestClient(create_app(runner))

    r = client.post(f"/api/models/{_STEM}/display-name", json={"display_name": "Best"})
    assert r.status_code == 200
    assert client.get("/api/models").json()["models"][0]["display_name"] == "Best"

    r = client.post("/api/models/nope/display-name", json={"display_name": "x"})
    assert r.status_code == 404


def test_http_start_runs_fake_benchmark_and_caches_result(tmp_path: Path) -> None:
    def fake(r: BenchmarkRunner, stem: str) -> None:
        write_benchmark_result(r.benchmark_dir, stem, {"trading": {"net_return": 0.01}})

    runner = _runner(tmp_path, run_benchmark=fake)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    client = TestClient(create_app(runner))

    r = client.post(f"/api/benchmark/{_STEM}/start")
    assert r.status_code == 202
    _wait_for_state(runner, "idle")
    assert client.get("/api/models").json()["models"][0]["has_benchmark"] is True
    # Cached: a second start on the same model is refused (button is gone anyway).
    r = client.post(f"/api/benchmark/{_STEM}/start")
    assert r.status_code == 409


def test_http_start_unknown_model_is_404(tmp_path: Path) -> None:
    client = TestClient(create_app(_runner(tmp_path)))
    assert client.post("/api/benchmark/nope/start").status_code == 404


def test_http_start_incompatible_model_is_409(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM, semantics="per_step_logret")
    client = TestClient(create_app(runner))
    assert client.post(f"/api/benchmark/{_STEM}/start").status_code == 409


def test_http_start_while_running_is_409(tmp_path: Path) -> None:
    gate = threading.Event()

    def fake(r: BenchmarkRunner, stem: str) -> None:
        gate.wait(timeout=2)

    runner = _runner(tmp_path, run_benchmark=fake)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM2)
    client = TestClient(create_app(runner))

    assert client.post(f"/api/benchmark/{_STEM}/start").status_code == 202
    assert client.post(f"/api/benchmark/{_STEM2}/start").status_code == 409
    gate.set()


# --- leaderboard -------------------------------------------------------------------

def test_http_leaderboard_ranks_by_net_return_desc(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM2)
    write_benchmark_result(
        runner.benchmark_dir, _STEM, {"trading": {"net_return": 0.01, "sharpe": 0.2}}
    )
    write_benchmark_result(
        runner.benchmark_dir, _STEM2, {"trading": {"net_return": 0.05, "sharpe": 0.1}}
    )
    client = TestClient(create_app(runner))

    rows = client.get("/api/leaderboard").json()["models"]
    assert [row["stem"] for row in rows] == [_STEM2, _STEM]  # higher net PnL first
    assert rows[0]["rank"] == 1


def test_http_leaderboard_is_strict_json_and_groups_runs(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    write_benchmark_result(
        runner.benchmark_dir, _STEM,
        {"trading": {"net_return": 0.01, "sharpe": float("nan")}},
    )
    client = TestClient(create_app(runner))
    payload = client.get("/api/leaderboard").json()
    json.dumps(payload, allow_nan=False)  # NaN never reaches the browser
    runs = payload["runs"]
    assert len(runs) == 1
    assert runs[0]["git_sha"] == "4d460ed"
    assert runs[0]["n_models"] == 1


# --- config -------------------------------------------------------------------------

def test_http_config_serves_rule_constants(tmp_path: Path) -> None:
    client = TestClient(create_app(_runner(tmp_path)))
    cfg = client.get("/api/config").json()
    assert cfg["fee_threshold"] == EXECUTION.FEE_THRESHOLD
    assert cfg["horizon"] == PREDICTOR.HORIZON
    assert cfg["target_semantics"] == PREDICTOR.TARGET_SEMANTICS
    assert cfg["null_draws"] >= 1
