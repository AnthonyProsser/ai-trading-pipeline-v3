"""Model benchmark FastAPI app. Owns the one-at-a-time benchmark job lifecycle, the
progress SSE stream, the model registry endpoints, and static file serving. Runs on
its own port (ExecutionConfig.BENCHMARK_UI_BIND_PORT), bound to 127.0.0.1 only —
same stack and process model as the training dashboard (src/training_ui/app.py):
FastAPI + vanilla JS, in-process background thread, queue-per-client SSE.

BenchmarkRunner is the state machine + pub/sub hub; `run_benchmark` is injectable so
it is unit-testable without a checkpoint or dataset (tests/benchmark/test_app.py).
The default production job is src/benchmark/engine.py::run_benchmark_job.
"""
from __future__ import annotations

import asyncio
import json
import math
import queue
import threading
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from constants import BENCHMARK, EXECUTION, PREDICTOR, TRAINING_UI
from src.benchmark.engine import run_benchmark_job
from src.benchmark.registry import (
    json_safe,
    list_models,
    parse_run_tag,
    read_benchmark_result,
    scan_checkpoints,
    set_display_name,
)
from src.benchmark.trading_sim import profitability_grade
from src.training_ui.exporter import read_fold_records

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = REPO_ROOT / "static" / "benchmark"


class BenchmarkRunner:
    """State machine + pub/sub hub for benchmark jobs (one at a time — a job owns the
    GPU and the feature cache; queueing more would just serialize them anyway).
    `run_benchmark(self, stem)` executes on a background thread; unlike the training
    runner, the terminal status broadcast is the runner's own job (in `finally`), so
    an injected job can't leave the state machine stuck in "running"."""

    def __init__(
        self,
        *,
        checkpoint_dir: Path,
        benchmark_dir: Path,
        run_benchmark: Callable[[BenchmarkRunner, str], None],
        training_metrics_dir: Path | None = None,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.benchmark_dir = benchmark_dir
        # Where training_metrics.json lives (the real CHECKPOINT_DIR, not the finished
        # dir checkpoint_dir now points at) — the analysis endpoint joins from here.
        self.training_metrics_dir = training_metrics_dir
        self._run_benchmark = run_benchmark
        self.state: str = "idle"
        self.current_stem: str | None = None
        self._thread: threading.Thread | None = None
        self._subscribers: list[queue.Queue[dict[str, object]]] = []
        self._subs_lock = threading.Lock()
        # Guards the check-then-mutate in start(): FastAPI dispatches `def` handlers
        # on threadpool workers, so two concurrent starts could otherwise both pass
        # the not-running check (same TOCTOU class the training runner fixed).
        self._state_lock = threading.Lock()
        self.last_status: dict[str, object] = {"type": "status", "state": "idle", "stem": None}

    def subscribe(self) -> queue.Queue[dict[str, object]]:
        q: queue.Queue[dict[str, object]] = queue.Queue()
        with self._subs_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue[dict[str, object]]) -> None:
        with self._subs_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def broadcast(self, payload: dict[str, object]) -> None:
        if payload.get("type") == "status":
            self.state = str(payload.get("state", self.state))
            self.last_status = payload
        with self._subs_lock:
            subs = list(self._subscribers)
        for q in subs:
            q.put_nowait(payload)

    def start(self, stem: str) -> bool:
        with self._state_lock:
            if self.state == "running":
                return False
            self.state = "running"
            self.current_stem = stem
        self.broadcast({"type": "status", "state": "running", "stem": stem})
        self._thread = threading.Thread(target=self._run, args=(stem,), daemon=True)
        self._thread.start()
        return True

    def _run(self, stem: str) -> None:
        try:
            self._run_benchmark(self, stem)
        except Exception as exc:  # noqa: BLE001 — job boundary must not crash the server
            self.broadcast(
                {
                    "type": "alert", "level": "error",
                    "message": f"Benchmark failed — {stem}: {exc}",
                }
            )
        finally:
            self.current_stem = None
            self.broadcast({"type": "status", "state": "idle", "stem": None})


def create_runner() -> BenchmarkRunner:
    # The benchmark reads only FINISHED models — a full training run promotes its
    # checkpoints into FINISHED_DIR (scratch/manual/search checkpoints in CHECKPOINT_DIR
    # never appear). training_metrics.json stays in CHECKPOINT_DIR (the analysis join).
    return BenchmarkRunner(
        checkpoint_dir=REPO_ROOT / PREDICTOR.FINISHED_DIR,
        benchmark_dir=REPO_ROOT / BENCHMARK.BENCHMARK_DIR,
        run_benchmark=run_benchmark_job,
        training_metrics_dir=REPO_ROOT / PREDICTOR.CHECKPOINT_DIR,
    )


# --- leaderboard aggregation ------------------------------------------------------
#
# The board is grouped by training RUN: fold-checkpoints that share a (git_sha,
# constants_sha8) — same code + same frozen constants — are siblings of one walk-forward
# run (the run tag's scaler segment varies per fold, so it is excluded from the key).
# Each run row aggregates its folds ("70/78 profitable") and nests them as a drill-down
# ranked by per-trade expectancy. See DECISIONS.md `benchmark_run_grouping`.


def _metric(row: dict[str, object], group: str, key: str) -> float | None:
    """A finite float at row[group][key], else None (missing block / non-finite)."""
    block = row.get(group)
    v = block.get(key) if isinstance(block, dict) else None
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None


def _expectancy_of(row: dict[str, object]) -> float | None:
    """Mean net-of-fee log-return per trade (net_return / trade_count). None when the
    model made no trades (expectancy undefined) or the numbers are missing/non-finite."""
    net = _metric(row, "trading", "net_return")
    count = _metric(row, "trading", "trade_count")
    if net is None or count is None or count == 0:
        return None
    exp = net / count
    return exp if math.isfinite(exp) else None


def _grade_of(row: dict[str, object]) -> str:
    """Task 1's three-state green grade from the persisted result numbers. Missing
    numbers -> NaN/0, which profitability_grade reads as 'insufficient'."""
    net = _metric(row, "trading", "net_return")
    p = _metric(row, "baselines", "p_value")
    count = _metric(row, "trading", "trade_count")
    return profitability_grade(
        float("nan") if net is None else net,
        0 if count is None else int(count),
        float("nan") if p is None else p,
    )


def _member_sort_key(row: dict[str, object]) -> tuple[float, float]:
    """Within-run rank: per-trade expectancy desc, tie-broken by directional accuracy
    desc; missing values sort last on either key (DECISIONS `benchmark_leaderboard_ranking`)."""
    exp = _expectancy_of(row)
    da = _metric(row, "statistical", "directional_accuracy")
    return (
        -exp if exp is not None else float("inf"),
        -da if da is not None else float("inf"),
    )


def _run_sort_key(run: dict[str, object]) -> tuple[float, float]:
    """Run order: profitable-fold fraction desc, tie-broken by mean expectancy desc;
    null aggregates sort last on either key."""
    frac = run["profitable_fraction"]
    mean_exp = run["mean_expectancy"]
    return (
        -frac if isinstance(frac, float) else float("inf"),
        -mean_exp if isinstance(mean_exp, float) else float("inf"),
    )


def build_leaderboard(models: list[dict[str, object]]) -> dict[str, object]:
    """Group list_models() output into per-run rows with nested, ranked fold members.
    Runs with zero benchmarked folds are dropped (no metrics to show). Returns the
    pre-json_safe payload {"runs": [...]}."""
    groups: dict[tuple[object, object], list[dict[str, object]]] = {}
    for m in models:
        groups.setdefault((m["git_sha"], m["constants_sha8"]), []).append(m)

    run_rows: list[dict[str, object]] = []
    for (git_sha, constants_sha8), group in groups.items():
        members: list[dict[str, object]] = []
        for m in group:
            if not m["has_benchmark"]:
                continue
            result = m["result"]
            assert isinstance(result, dict)
            row: dict[str, object] = {
                "stem": m["stem"],
                "display_name": m["display_name"],
                "fold_index": m["fold_index"],
                "trading": result.get("trading", {}),
                "baselines": result.get("baselines", {}),
                "statistical": result.get("statistical", {}),
                "economic": result.get("economic", {}),
                "eval": result.get("eval", {}),
                "benchmarked_at_utc": result.get("benchmarked_at_utc"),
            }
            row["expectancy"] = _expectancy_of(row)
            row["profitability"] = _grade_of(row)
            members.append(row)

        if not members:  # a run with no benchmarked folds has no board presence
            continue

        members.sort(key=_member_sort_key)
        for rank, row in enumerate(members, start=1):
            row["rank"] = rank

        n_benchmarked = len(members)
        n_profitable = sum(r["profitability"] == "profitable" for r in members)
        n_not_profitable = sum(r["profitability"] == "not_profitable" for r in members)
        n_insufficient = sum(r["profitability"] == "insufficient" for r in members)
        exps = [e for e in (r["expectancy"] for r in members) if isinstance(e, float)]
        das = [
            d
            for d in (_metric(r, "statistical", "directional_accuracy") for r in members)
            if d is not None
        ]
        nets = [
            v for v in (_metric(r, "trading", "net_return") for r in members)
            if v is not None
        ]
        run_rows.append(
            {
                "git_sha": git_sha,
                "constants_sha8": constants_sha8,
                "n_checkpoints": len(group),
                "n_benchmarked": n_benchmarked,
                "n_profitable": n_profitable,
                "n_not_profitable": n_not_profitable,
                "n_insufficient": n_insufficient,
                # Denominator = benchmarked folds; 'insufficient' counts here but never in
                # the numerator (a fold that trades too little is not evidence of edge).
                "profitable_fraction": n_profitable / n_benchmarked,
                "mean_expectancy": sum(exps) / len(exps) if exps else None,
                "mean_da": sum(das) / len(das) if das else None,
                "mean_net_return": sum(nets) / len(nets) if nets else None,
                "models": members,
            }
        )

    run_rows.sort(key=_run_sort_key)
    return {"runs": run_rows}


class RenameBody(BaseModel):
    display_name: str


def create_app(runner: BenchmarkRunner | None = None) -> FastAPI:
    runner = runner or create_runner()
    app = FastAPI(title="BTC Predictor Model Benchmark")

    @app.get("/api/models")
    def get_models() -> dict[str, object]:
        return {
            "models": list_models(runner.checkpoint_dir, runner.benchmark_dir),
            "running_stem": runner.current_stem,
        }

    @app.post("/api/models/{stem}/display-name")
    def rename_model(stem: str, body: RenameBody) -> dict[str, object]:
        known = {str(e["stem"]) for e in scan_checkpoints(runner.checkpoint_dir)}
        if stem not in known:
            raise HTTPException(status_code=404, detail=f"unknown model {stem!r}")
        effective = set_display_name(runner.benchmark_dir, stem, body.display_name)
        return {"stem": stem, "display_name": effective}

    @app.post("/api/benchmark/{stem}/start", status_code=202)
    def start_benchmark(stem: str) -> dict[str, object]:
        models = {
            str(m["stem"]): m
            for m in list_models(runner.checkpoint_dir, runner.benchmark_dir)
        }
        model = models.get(stem)
        if model is None:
            raise HTTPException(status_code=404, detail=f"unknown model {stem!r}")
        if not model["compatible"]:
            raise HTTPException(status_code=409, detail=str(model["incompatible_reason"]))
        if model["has_benchmark"]:
            raise HTTPException(
                status_code=409,
                detail="already benchmarked — delete its .benchmark.json to re-run",
            )
        if not runner.start(stem):
            raise HTTPException(
                status_code=409, detail=f"benchmark already running ({runner.current_stem})"
            )
        return {"state": runner.state, "stem": stem}

    @app.get("/api/leaderboard")
    def get_leaderboard() -> dict[str, object]:
        # One row per training run (grouped by git_sha + constants_sha8), each nesting
        # its fold-checkpoints ranked by per-trade expectancy; see build_leaderboard.
        models = list_models(runner.checkpoint_dir, runner.benchmark_dir)
        safe = json_safe(build_leaderboard(models))
        assert isinstance(safe, dict)
        return safe

    @app.get("/api/config")
    def get_config() -> dict[str, object]:
        """The fixed rule's parameters + UI thresholds, sourced from constants.py so
        the JS client never hardcodes a duplicate of a Python value."""
        return {
            "fee_threshold": EXECUTION.FEE_THRESHOLD,
            "horizon": PREDICTOR.HORIZON,
            "target_semantics": PREDICTOR.TARGET_SEMANTICS,
            "quantiles": list(PREDICTOR.QUANTILES),
            "null_draws": BENCHMARK.NULL_DRAWS,
            "null_significance_level": BENCHMARK.NULL_SIGNIFICANCE_LEVEL,
            "profitable_p_value_max": BENCHMARK.PROFITABLE_P_VALUE_MAX,
            "profitable_min_trades": BENCHMARK.PROFITABLE_MIN_TRADES,
            "alert_auto_dismiss_seconds": TRAINING_UI.ALERT_AUTO_DISMISS_SECONDS,
            "deploy_gate_da_threshold": PREDICTOR.DEPLOY_GATE_DA_THRESHOLD,
            "deploy_gate_cal_lower": PREDICTOR.DEPLOY_GATE_CAL_LOWER,
            "deploy_gate_cal_upper": PREDICTOR.DEPLOY_GATE_CAL_UPPER,
        }

    @app.get("/api/results/{stem}")
    def get_result(stem: str) -> dict[str, object]:
        # Validate against the known checkpoint stems before building a path, same as
        # rename_model/start_benchmark — a raw path parameter (e.g. containing `..`)
        # must never reach `benchmark_dir / f"{stem}.benchmark.json"`.
        known = {str(e["stem"]) for e in scan_checkpoints(runner.checkpoint_dir)}
        if stem not in known:
            raise HTTPException(status_code=404, detail=f"no benchmark result for {stem!r}")
        result = read_benchmark_result(runner.benchmark_dir, stem)
        if result is None:
            raise HTTPException(status_code=404, detail=f"no benchmark result for {stem!r}")
        return result

    @app.get("/api/analysis/{stem}")
    def get_analysis(stem: str) -> dict[str, object]:
        """One JSON joining a model's benchmark result with the training fold record that
        produced this exact checkpoint (matched on the run-tag stem) — so an AI can read
        the trading/statistical outcome and the training-session context together and
        hypothesize why it scored as it did. `training` is null for a model whose
        training record predates the stem join key (or was trained outside this UI)."""
        known = {str(e["stem"]) for e in scan_checkpoints(runner.checkpoint_dir)}
        if stem not in known:
            raise HTTPException(status_code=404, detail=f"unknown model {stem!r}")
        result = read_benchmark_result(runner.benchmark_dir, stem)
        if result is None:
            raise HTTPException(status_code=404, detail=f"no benchmark result for {stem!r}")
        training: dict[str, object] | None = None
        if runner.training_metrics_dir is not None:
            for record in read_fold_records(runner.training_metrics_dir):
                if record.get("stem") == stem:
                    training = dict(record)
                    break
        tag = parse_run_tag(stem)
        payload: dict[str, object] = {
            "stem": stem,
            "provenance": {
                "fold_index": tag.fold_index if tag is not None else None,
                "git_sha": tag.git_sha if tag is not None else None,
                "constants_sha8": tag.constants_sha8 if tag is not None else None,
                "trained_through_ts_utc": result.get("trained_through_ts_utc"),
                "lookback": result.get("lookback"),
            },
            "benchmark": result,
            "training": training,
        }
        safe = json_safe(payload)
        assert isinstance(safe, dict)
        return safe

    @app.get("/api/events")
    async def events(request: Request) -> StreamingResponse:
        q = runner.subscribe()

        async def gen() -> AsyncIterator[str]:
            try:
                yield f"data: {json.dumps(json_safe(runner.last_status))}\n\n"
                loop = asyncio.get_event_loop()
                while True:
                    if await request.is_disconnected():
                        break
                    # Bounded get, same rationale as the training UI: an unbounded
                    # q.get() leaks the executor thread + subscriber entry if the
                    # client vanishes without a clean disconnect.
                    try:
                        payload = await loop.run_in_executor(
                            None, q.get, True, TRAINING_UI.SSE_POLL_TIMEOUT_S
                        )
                    except queue.Empty:
                        continue
                    yield f"data: {json.dumps(json_safe(payload))}\n\n"
            finally:
                runner.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app, host=EXECUTION.DASHBOARD_BIND_HOST, port=EXECUTION.BENCHMARK_UI_BIND_PORT
    )
