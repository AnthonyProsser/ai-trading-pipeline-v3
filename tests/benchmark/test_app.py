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

from constants import BENCHMARK, EXECUTION, PREDICTOR, TRAINING_UI
from src.benchmark.app import BenchmarkRunner, create_app
from src.benchmark.registry import write_benchmark_result

_STEM = "4d460ed-s001e4348-c487b9e2e-fold33"
_STEM2 = "4d460ed-s00c53cec-c487b9e2e-fold68"
_STEM3 = "4d460ed-s00d3eea6-c487b9e2e-fold69"
# A second training run = a different (git_sha, constants_sha8) pair. Same-run folds
# share both segments; only the scaler segment (s...) varies per fold.
_RUN_B_STEM = "bbbbbbb-s00000001-c00000002-fold0"
_RUN_B_STEM2 = "bbbbbbb-s00000003-c00000002-fold1"


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
    training_metrics_dir: Path | None = None,
) -> BenchmarkRunner:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return BenchmarkRunner(
        checkpoint_dir=checkpoint_dir,
        benchmark_dir=checkpoint_dir / "benchmark",
        run_benchmark=run_benchmark,
        training_metrics_dir=training_metrics_dir,
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


def test_runner_batch_runs_all_stems_in_order(tmp_path: Path) -> None:
    seen: list[str] = []

    def fake(r: BenchmarkRunner, stem: str) -> None:
        seen.append(stem)

    runner = _runner(tmp_path, run_benchmark=fake)
    assert runner.start_batch([_STEM, _STEM2, _STEM3]) is True
    _wait_for_state(runner, "idle")
    assert seen == [_STEM, _STEM2, _STEM3]


def test_runner_batch_continues_past_a_failing_stem(tmp_path: Path) -> None:
    # One fold raising must not abort the rest of the batch (best-effort per stem).
    seen: list[str] = []

    def fake(r: BenchmarkRunner, stem: str) -> None:
        seen.append(stem)
        if stem == _STEM2:
            raise RuntimeError("boom")

    runner = _runner(tmp_path, run_benchmark=fake)
    q = runner.subscribe()
    assert runner.start_batch([_STEM, _STEM2, _STEM3]) is True
    _wait_for_state(runner, "idle")
    assert seen == [_STEM, _STEM2, _STEM3]
    payloads = []
    while not q.empty():
        payloads.append(q.get_nowait())
    assert any(p.get("type") == "alert" and p.get("level") == "error" for p in payloads)


def test_runner_batch_rejected_while_running(tmp_path: Path) -> None:
    gate = threading.Event()

    def fake(r: BenchmarkRunner, stem: str) -> None:
        gate.wait(timeout=2)

    runner = _runner(tmp_path, run_benchmark=fake)
    assert runner.start_batch([_STEM, _STEM2]) is True
    assert runner.start_batch([_STEM3]) is False
    assert runner.start(_STEM3) is False  # single-stem start shares the same guard
    gate.set()


def test_runner_start_batch_empty_is_false(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    assert runner.start_batch([]) is False
    assert runner.state == "idle"


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


def test_http_results_unknown_stem_is_404_not_path_read(tmp_path: Path) -> None:
    # get_result must validate the stem against known checkpoints before building a
    # path — a bare/unknown stem (or one with traversal segments) yields 404, never a
    # filesystem read outside benchmark_dir.
    runner = _runner(tmp_path)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    client = TestClient(create_app(runner))
    assert client.get("/api/results/nope").status_code == 404
    # No benchmark run yet, so even a KNOWN stem has no result -> 404.
    assert client.get(f"/api/results/{_STEM}").status_code == 404


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


# --- batch "benchmark all" (one run's remaining compatible folds) -------------------

def test_http_run_benchmark_enqueues_pending_compatible_folds(tmp_path: Path) -> None:
    # A run with three folds, one already benchmarked -> "Benchmark all" enqueues the
    # other two (compatible + not-yet-benchmarked), server-derived from the run key.
    started: list[str] = []

    def fake(r: BenchmarkRunner, stem: str) -> None:
        started.append(stem)
        write_benchmark_result(
            r.benchmark_dir, stem, {"trading": {"net_return": 0.01, "trade_count": 30}}
        )

    runner = _runner(tmp_path, run_benchmark=fake)
    for stem in (_STEM, _STEM2, _STEM3):
        _write_fake_checkpoint(runner.checkpoint_dir, stem)
    write_benchmark_result(
        runner.benchmark_dir, _STEM, {"trading": {"net_return": 0.02, "trade_count": 30}}
    )
    client = TestClient(create_app(runner))

    r = client.post(
        "/api/benchmark/run/start",
        json={"git_sha": "4d460ed", "constants_sha8": "487b9e2e"},
    )
    assert r.status_code == 202
    assert sorted(r.json()["stems"]) == sorted([_STEM2, _STEM3])
    _wait_for_state(runner, "idle")
    assert sorted(started) == sorted([_STEM2, _STEM3])


def test_http_run_benchmark_unknown_run_is_404(tmp_path: Path) -> None:
    client = TestClient(create_app(_runner(tmp_path)))
    r = client.post(
        "/api/benchmark/run/start",
        json={"git_sha": "nope", "constants_sha8": "deadbeef"},
    )
    assert r.status_code == 404


def test_http_run_benchmark_nothing_pending_is_409(tmp_path: Path) -> None:
    # Every compatible fold already benchmarked -> nothing to enqueue.
    runner = _runner(tmp_path)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    write_benchmark_result(runner.benchmark_dir, _STEM, _profitable_result())
    client = TestClient(create_app(runner))
    r = client.post(
        "/api/benchmark/run/start",
        json={"git_sha": "4d460ed", "constants_sha8": "487b9e2e"},
    )
    assert r.status_code == 409


def test_http_run_benchmark_while_running_is_409(tmp_path: Path) -> None:
    gate = threading.Event()

    def fake(r: BenchmarkRunner, stem: str) -> None:
        gate.wait(timeout=2)

    runner = _runner(tmp_path, run_benchmark=fake)
    for stem in (_STEM, _STEM2):
        _write_fake_checkpoint(runner.checkpoint_dir, stem)
    client = TestClient(create_app(runner))
    assert client.post(f"/api/benchmark/{_STEM}/start").status_code == 202
    r = client.post(
        "/api/benchmark/run/start",
        json={"git_sha": "4d460ed", "constants_sha8": "487b9e2e"},
    )
    assert r.status_code == 409
    gate.set()


# --- leaderboard (runs-primary: one row per training run, folds nested & ranked) ----
#
# The board groups fold-checkpoints by their training run — (git_sha, constants_sha8)
# parsed from the run tag; the scaler segment varies per fold and is excluded. Each run
# row carries a "70/78 profitable" aggregate; its `models` list is the drill-down,
# ranked within the run by derived per-trade expectancy desc, DA desc tie-break.

def _profitable_result(
    da: float = 0.53, *, net_return: float = 0.05, trade_count: int | None = None
) -> dict[str, object]:
    # Positive per-trade expectancy, above the trade floor, beats the null (p < 0.10).
    tc = BENCHMARK.PROFITABLE_MIN_TRADES if trade_count is None else trade_count
    return {
        "trading": {"net_return": net_return, "trade_count": tc},
        "baselines": {"p_value": 0.02},
        "statistical": {"directional_accuracy": da},
    }


def _not_profitable_result(da: float = 0.52) -> dict[str, object]:
    # Money-positive, above the floor, but indistinguishable from the random-entry null.
    return {
        "trading": {"net_return": 0.05, "trade_count": BENCHMARK.PROFITABLE_MIN_TRADES},
        "baselines": {"p_value": 0.40},
        "statistical": {"directional_accuracy": da},
    }


def _only_run(payload: dict[str, object]) -> dict[str, object]:
    runs = payload["runs"]
    assert isinstance(runs, list) and len(runs) == 1
    run = runs[0]
    assert isinstance(run, dict)
    return run


def test_http_leaderboard_runs_ordered_by_profitable_fraction_then_expectancy(
    tmp_path: Path,
) -> None:
    # Run A: 1 of 2 folds profitable (fraction 0.5). Run B: 2 of 2 (fraction 1.0).
    # B must sort first — the board compares recipes by profitable-fold fraction.
    runner = _runner(tmp_path)
    for stem in (_STEM, _STEM2, _RUN_B_STEM, _RUN_B_STEM2):
        _write_fake_checkpoint(runner.checkpoint_dir, stem)
    write_benchmark_result(runner.benchmark_dir, _STEM, _profitable_result())
    write_benchmark_result(runner.benchmark_dir, _STEM2, _not_profitable_result())
    write_benchmark_result(runner.benchmark_dir, _RUN_B_STEM, _profitable_result())
    write_benchmark_result(runner.benchmark_dir, _RUN_B_STEM2, _profitable_result())
    client = TestClient(create_app(runner))

    runs = client.get("/api/leaderboard").json()["runs"]
    assert [r["git_sha"] for r in runs] == ["bbbbbbb", "4d460ed"]
    assert runs[0]["profitable_fraction"] == 1.0
    assert runs[1]["profitable_fraction"] == 0.5


def test_http_leaderboard_run_fraction_tie_broken_by_mean_expectancy(tmp_path: Path) -> None:
    # Both runs 1/1 profitable (fraction 1.0). Higher mean per-trade expectancy wins.
    # Run B expectancy = 0.10/30; Run A expectancy = 0.05/30 -> B first.
    runner = _runner(tmp_path)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    _write_fake_checkpoint(runner.checkpoint_dir, _RUN_B_STEM)
    write_benchmark_result(runner.benchmark_dir, _STEM, _profitable_result(net_return=0.05))
    write_benchmark_result(
        runner.benchmark_dir, _RUN_B_STEM, _profitable_result(net_return=0.10)
    )
    client = TestClient(create_app(runner))
    runs = client.get("/api/leaderboard").json()["runs"]
    assert [r["git_sha"] for r in runs] == ["bbbbbbb", "4d460ed"]


def test_http_leaderboard_members_ranked_by_expectancy_desc_then_da(tmp_path: Path) -> None:
    # Within one run, folds rank by per-trade expectancy desc. fold33 net 0.09,
    # fold68 net 0.06, fold69 net 0.03 (all 30 trades) -> ranks 1,2,3, per-run.
    runner = _runner(tmp_path)
    for stem in (_STEM, _STEM2, _STEM3):
        _write_fake_checkpoint(runner.checkpoint_dir, stem)
    write_benchmark_result(runner.benchmark_dir, _STEM, _profitable_result(net_return=0.09))
    write_benchmark_result(runner.benchmark_dir, _STEM2, _profitable_result(net_return=0.06))
    write_benchmark_result(runner.benchmark_dir, _STEM3, _profitable_result(net_return=0.03))
    client = TestClient(create_app(runner))
    members = _only_run(client.get("/api/leaderboard").json())["models"]
    assert [m["stem"] for m in members] == [_STEM, _STEM2, _STEM3]
    assert [m["rank"] for m in members] == [1, 2, 3]


def test_http_leaderboard_members_equal_expectancy_tie_broken_by_da(tmp_path: Path) -> None:
    # Equal expectancy (same net/count) -> higher directional accuracy ranks first.
    runner = _runner(tmp_path)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM2)
    write_benchmark_result(
        runner.benchmark_dir, _STEM, _profitable_result(da=0.51, net_return=0.05)
    )
    write_benchmark_result(
        runner.benchmark_dir, _STEM2, _profitable_result(da=0.58, net_return=0.05)
    )
    client = TestClient(create_app(runner))
    members = _only_run(client.get("/api/leaderboard").json())["models"]
    assert [m["stem"] for m in members] == [_STEM2, _STEM]  # higher DA first


def test_http_leaderboard_missing_expectancy_sorts_last_within_run(tmp_path: Path) -> None:
    # A benchmarked fold with no trades (expectancy undefined) ranks below one with a
    # real expectancy, and must not crash the sort.
    runner = _runner(tmp_path)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM2)
    write_benchmark_result(
        runner.benchmark_dir, _STEM, {"trading": {"trade_count": 0}, "statistical": {}}
    )
    write_benchmark_result(runner.benchmark_dir, _STEM2, _profitable_result())
    client = TestClient(create_app(runner))
    members = _only_run(client.get("/api/leaderboard").json())["models"]
    assert members[0]["stem"] == _STEM2 and members[1]["stem"] == _STEM
    assert members[0]["expectancy"] is not None
    assert members[1]["expectancy"] is None


def test_http_leaderboard_insufficient_in_denominator_never_numerator(tmp_path: Path) -> None:
    # A run of three benchmarked folds: profitable, not_profitable, insufficient.
    # fraction = n_profitable / n_benchmarked = 1/3; the insufficient fold counts in the
    # denominator but is surfaced separately and is never graded profitable.
    runner = _runner(tmp_path)
    for stem in (_STEM, _STEM2, _STEM3):
        _write_fake_checkpoint(runner.checkpoint_dir, stem)
    write_benchmark_result(runner.benchmark_dir, _STEM, _profitable_result())
    write_benchmark_result(runner.benchmark_dir, _STEM2, _not_profitable_result())
    write_benchmark_result(
        runner.benchmark_dir, _STEM3,
        {  # below the trade floor -> insufficient
            "trading": {"net_return": 0.9,
                        "trade_count": BENCHMARK.PROFITABLE_MIN_TRADES - 1},
            "baselines": {"p_value": 0.01},
        },
    )
    client = TestClient(create_app(runner))
    run = _only_run(client.get("/api/leaderboard").json())
    assert run["n_benchmarked"] == 3
    assert run["n_profitable"] == 1
    assert run["n_not_profitable"] == 1
    assert run["n_insufficient"] == 1
    assert run["profitable_fraction"] == 1 / 3


# --- completeness gating: only fully-benchmarked runs are scored -------------------
#
# A run is "complete" when every one of its compatible fold-checkpoints has a benchmark
# result. Complete runs are scored and ranked at the top; incomplete runs (partial OR
# zero benchmarked) carry no score, sit at the bottom, and expose "Benchmark all".


def test_http_leaderboard_partial_run_is_incomplete_and_unscored(tmp_path: Path) -> None:
    # 2 compatible folds, only 1 benchmarked -> not complete: unscored, 1 pending.
    runner = _runner(tmp_path)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM2)
    write_benchmark_result(runner.benchmark_dir, _STEM, _profitable_result())
    client = TestClient(create_app(runner))
    run = _only_run(client.get("/api/leaderboard").json())
    assert run["complete"] is False
    assert run["n_checkpoints"] == 2
    assert run["n_benchmarked"] == 1
    assert run["n_pending"] == 1
    assert run["profitable_fraction"] is None
    assert run["mean_expectancy"] is None


def test_http_leaderboard_zero_benchmarked_run_appears_incomplete(tmp_path: Path) -> None:
    # A freshly-finished run (0 benchmarked folds) now APPEARS — incomplete, at the
    # bottom — so its "Benchmark all" is reachable. (Such runs were dropped before.)
    runner = _runner(tmp_path)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM2)
    client = TestClient(create_app(runner))
    run = _only_run(client.get("/api/leaderboard").json())
    assert run["complete"] is False
    assert run["n_benchmarked"] == 0
    assert run["n_pending"] == 2
    assert run["models"] == []


def test_http_leaderboard_complete_run_is_scored_and_marked_complete(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM2)
    write_benchmark_result(runner.benchmark_dir, _STEM, _profitable_result())
    write_benchmark_result(runner.benchmark_dir, _STEM2, _profitable_result())
    client = TestClient(create_app(runner))
    run = _only_run(client.get("/api/leaderboard").json())
    assert run["complete"] is True
    assert run["n_pending"] == 0
    assert run["profitable_fraction"] == 1.0


def test_http_leaderboard_complete_runs_rank_above_incomplete(tmp_path: Path) -> None:
    # Complete run (all folds benchmarked, scored) sorts ABOVE an incomplete run.
    runner = _runner(tmp_path)
    for stem in (_STEM, _STEM2):  # run 4d460ed: both benchmarked -> complete
        _write_fake_checkpoint(runner.checkpoint_dir, stem)
    write_benchmark_result(runner.benchmark_dir, _STEM, _profitable_result())
    write_benchmark_result(runner.benchmark_dir, _STEM2, _profitable_result())
    for stem in (_RUN_B_STEM, _RUN_B_STEM2):  # run bbbbbbb: 1 of 2 -> incomplete
        _write_fake_checkpoint(runner.checkpoint_dir, stem)
    write_benchmark_result(runner.benchmark_dir, _RUN_B_STEM, _profitable_result())
    client = TestClient(create_app(runner))
    runs = client.get("/api/leaderboard").json()["runs"]
    assert [r["git_sha"] for r in runs] == ["4d460ed", "bbbbbbb"]
    assert runs[0]["complete"] is True
    assert runs[1]["complete"] is False


def test_http_leaderboard_incompatible_only_run_absent(tmp_path: Path) -> None:
    # A run whose only checkpoint is incompatible has nothing benchmarkable and never
    # appears on the board (there is no "Benchmark all" that could do anything).
    runner = _runner(tmp_path)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM, semantics="per_step_logret")
    client = TestClient(create_app(runner))
    assert client.get("/api/leaderboard").json()["runs"] == []


def test_http_leaderboard_aggregates_skip_missing_values(tmp_path: Path) -> None:
    # mean_da / mean_expectancy average only present-and-finite members. A single fold
    # with no statistical block and no trades -> both means are null (not a crash).
    runner = _runner(tmp_path)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    write_benchmark_result(
        runner.benchmark_dir, _STEM, {"trading": {"trade_count": 0}, "statistical": {}}
    )
    client = TestClient(create_app(runner))
    run = _only_run(client.get("/api/leaderboard").json())
    assert run["mean_da"] is None
    assert run["mean_expectancy"] is None


def test_http_leaderboard_empty_board(tmp_path: Path) -> None:
    # No finished checkpoints at all -> an empty runs list, still valid strict JSON.
    # (A present-but-unbenchmarked checkpoint now yields an incomplete run, not [] —
    #  see test_http_leaderboard_zero_benchmarked_run_appears_incomplete.)
    runner = _runner(tmp_path)
    client = TestClient(create_app(runner))
    payload = client.get("/api/leaderboard").json()
    assert payload["runs"] == []


def test_http_leaderboard_is_strict_json(tmp_path: Path) -> None:
    # A NaN metric never reaches the browser (json_safe turns it to null); a zero-trade
    # expectancy is null, not NaN.
    runner = _runner(tmp_path)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    write_benchmark_result(
        runner.benchmark_dir, _STEM,
        {"trading": {"net_return": 0.01, "sharpe": float("nan"), "trade_count": 0}},
    )
    client = TestClient(create_app(runner))
    payload = client.get("/api/leaderboard").json()
    json.dumps(payload, allow_nan=False)  # NaN never reaches the browser
    run = _only_run(payload)
    assert run["git_sha"] == "4d460ed"
    assert run["models"][0]["expectancy"] is None


# --- profitability grade (green classification, now on nested member rows) ----------

def test_http_leaderboard_member_grade_profitable(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    write_benchmark_result(runner.benchmark_dir, _STEM, _profitable_result())
    client = TestClient(create_app(runner))
    member = _only_run(client.get("/api/leaderboard").json())["models"][0]
    assert member["profitability"] == "profitable"


def test_http_leaderboard_member_grade_not_profitable_on_weak_p(tmp_path: Path) -> None:
    # Money-positive, above the floor, but indistinguishable from the random-entry null.
    runner = _runner(tmp_path)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    write_benchmark_result(
        runner.benchmark_dir, _STEM,
        {
            "trading": {"net_return": 0.05, "trade_count": BENCHMARK.PROFITABLE_MIN_TRADES},
            "baselines": {"p_value": 0.40},
        },
    )
    client = TestClient(create_app(runner))
    member = _only_run(client.get("/api/leaderboard").json())["models"][0]
    assert member["profitability"] == "not_profitable"


def test_http_leaderboard_member_grade_insufficient_below_floor(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    write_benchmark_result(
        runner.benchmark_dir, _STEM,
        {
            "trading": {"net_return": 0.9,
                        "trade_count": BENCHMARK.PROFITABLE_MIN_TRADES - 1},
            "baselines": {"p_value": 0.01},
        },
    )
    client = TestClient(create_app(runner))
    member = _only_run(client.get("/api/leaderboard").json())["models"][0]
    assert member["profitability"] == "insufficient"


def test_http_leaderboard_member_grade_insufficient_when_blocks_missing(
    tmp_path: Path,
) -> None:
    # A result with no trading/baselines blocks must not crash and grades "insufficient".
    runner = _runner(tmp_path)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    write_benchmark_result(runner.benchmark_dir, _STEM, {"statistical": {}})
    client = TestClient(create_app(runner))
    payload = client.get("/api/leaderboard").json()
    json.dumps(payload, allow_nan=False)
    assert _only_run(payload)["models"][0]["profitability"] == "insufficient"


# --- analysis endpoint (benchmark + training join for AI hypothesis generation) -----

def test_http_analysis_joins_benchmark_and_training(tmp_path: Path) -> None:
    from src.training_ui.exporter import append_fold_record, hyperparams_snapshot

    metrics_dir = tmp_path / "metrics"
    runner = _runner(tmp_path, training_metrics_dir=metrics_dir)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    write_benchmark_result(
        runner.benchmark_dir, _STEM,
        {"trading": {"net_return": 0.01}, "statistical": {"directional_accuracy": 0.53}},
    )
    append_fold_record(
        metrics_dir,
        {
            "fold": 33, "train_loss": 0.6, "val_loss": 0.58, "da": 0.53,
            "q_coverage": 0.9, "duration_s": 12.0, "stem": _STEM,
            "hyperparams": hyperparams_snapshot(lookback=1440),
        },  # type: ignore[arg-type]
    )
    client = TestClient(create_app(runner))

    r = client.get(f"/api/analysis/{_STEM}")
    assert r.status_code == 200
    body = r.json()
    assert body["stem"] == _STEM
    assert body["benchmark"]["trading"]["net_return"] == 0.01
    assert body["training"]["stem"] == _STEM
    assert body["training"]["fold"] == 33
    assert body["provenance"]["fold_index"] == 33


def test_http_analysis_training_null_when_unmatched(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    runner = _runner(tmp_path, training_metrics_dir=metrics_dir)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    write_benchmark_result(runner.benchmark_dir, _STEM, {"trading": {"net_return": 0.01}})
    client = TestClient(create_app(runner))
    body = client.get(f"/api/analysis/{_STEM}").json()
    assert body["training"] is None


def test_http_analysis_unknown_stem_is_404(tmp_path: Path) -> None:
    client = TestClient(create_app(_runner(tmp_path)))
    assert client.get("/api/analysis/nope").status_code == 404


def test_http_analysis_no_benchmark_result_is_404(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    _write_fake_checkpoint(runner.checkpoint_dir, _STEM)
    client = TestClient(create_app(runner))
    assert client.get(f"/api/analysis/{_STEM}").status_code == 404


def test_create_runner_reads_finished_dir() -> None:
    from src.benchmark.app import REPO_ROOT, create_runner

    assert create_runner().checkpoint_dir == REPO_ROOT / PREDICTOR.FINISHED_DIR


# --- config -------------------------------------------------------------------------

def test_http_config_serves_rule_constants(tmp_path: Path) -> None:
    client = TestClient(create_app(_runner(tmp_path)))
    cfg = client.get("/api/config").json()
    assert cfg["fee_threshold"] == EXECUTION.FEE_THRESHOLD
    assert cfg["horizon"] == PREDICTOR.HORIZON
    assert cfg["target_semantics"] == PREDICTOR.TARGET_SEMANTICS
    assert cfg["null_draws"] >= 1
    # UI thresholds sourced from constants so the JS client never hardcodes a copy.
    assert cfg["null_significance_level"] == BENCHMARK.NULL_SIGNIFICANCE_LEVEL
    assert cfg["alert_auto_dismiss_seconds"] == TRAINING_UI.ALERT_AUTO_DISMISS_SECONDS
    # Green-grade thresholds sourced from constants so the JS client never hardcodes them.
    assert cfg["profitable_p_value_max"] == BENCHMARK.PROFITABLE_P_VALUE_MAX
    assert cfg["profitable_min_trades"] == BENCHMARK.PROFITABLE_MIN_TRADES
