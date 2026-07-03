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
    read_benchmark_result,
    scan_checkpoints,
    set_display_name,
)

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
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.benchmark_dir = benchmark_dir
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
    return BenchmarkRunner(
        checkpoint_dir=REPO_ROOT / PREDICTOR.CHECKPOINT_DIR,
        benchmark_dir=REPO_ROOT / BENCHMARK.BENCHMARK_DIR,
        run_benchmark=run_benchmark_job,
    )


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
        models = list_models(runner.checkpoint_dir, runner.benchmark_dir)
        rows: list[dict[str, object]] = []
        for m in models:
            if not m["has_benchmark"]:
                continue
            result = m["result"]
            assert isinstance(result, dict)
            rows.append(
                {
                    "stem": m["stem"],
                    "display_name": m["display_name"],
                    "fold_index": m["fold_index"],
                    "git_sha": m["git_sha"],
                    "constants_sha8": m["constants_sha8"],
                    "trading": result.get("trading", {}),
                    "baselines": result.get("baselines", {}),
                    "statistical": result.get("statistical", {}),
                    "economic": result.get("economic", {}),
                    "eval": result.get("eval", {}),
                    "benchmarked_at_utc": result.get("benchmarked_at_utc"),
                }
            )

        def net_of(row: dict[str, object]) -> float:
            trading = row["trading"]
            assert isinstance(trading, dict)
            net = trading.get("net_return")
            return float(net) if isinstance(net, (int, float)) else float("-inf")

        rows.sort(key=net_of, reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank

        groups: dict[tuple[object, object], list[dict[str, object]]] = {}
        for row in rows:
            groups.setdefault((row["git_sha"], row["constants_sha8"]), []).append(row)
        run_rows: list[dict[str, object]] = []
        for (git_sha, constants_sha8), members in groups.items():
            nets = [net_of(r) for r in members if net_of(r) != float("-inf")]
            run_rows.append(
                {
                    "git_sha": git_sha,
                    "constants_sha8": constants_sha8,
                    "n_models": len(members),
                    "mean_net_return": sum(nets) / len(nets) if nets else None,
                }
            )
        run_rows.sort(
            key=lambda g: g["mean_net_return"]
            if isinstance(g["mean_net_return"], float) else float("-inf"),
            reverse=True,
        )

        safe = json_safe({"models": rows, "runs": run_rows})
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
