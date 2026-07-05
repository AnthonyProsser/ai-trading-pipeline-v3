"""Training UI FastAPI app (training-dashboard.md). Owns the training subprocess
lifecycle, the live metric SSE stream, and static file serving. Runs on its own port
(ExecutionConfig.TRAINING_UI_BIND_PORT), separate from the trading dashboard.

Process model: training runs in an in-process background thread (not a subprocess) —
`threading.Event` for stop/save signals, an in-memory `queue.Queue` per connected
browser tab for the metric stream. Chosen over a subprocess because Windows has no
clean signal-delivery story for subprocess control (DECISIONS.md training_ui_controls
names both options; this repo picks threading.Event). Closing the browser tab does not
touch the server process, so training is unaffected either way.

TrainingRunner is the state machine + pub/sub hub; `run_training` is injectable so it
is unit-testable without real PyTorch training (tests/training_ui/test_app.py). The
default production path (`_default_run_training`) loads the real Kraken CSV and drives
`train_all_folds` (src/predictor/training.py) end to end.
"""
from __future__ import annotations

import asyncio
import json
import queue
import subprocess
import threading
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from constants import DATA, PREDICTOR, TRAINING_UI
from src.data.feature_pipeline import compute_features
from src.data.manifest import sha256_file
from src.data.validator import validate_candles
from src.data.walk_forward import carve_locked_test, filter_by_historical_start, make_folds
from src.predictor.training import train_all_folds
from src.training_ui.exporter import (
    FoldRecord,
    append_fold_record,
    hyperparams_snapshot,
    read_fold_records,
)
from src.training_ui.setup_router import check_data_gate

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = REPO_ROOT / "static" / "training_ui"

CandleArrays = tuple[npt.NDArray[np.datetime64], npt.NDArray[np.float64]]
# "done" = natural completion of every walk-forward fold (train_all_folds' final
# status), distinct from a user-initiated "stopped": the run is over, so stop is
# meaningless and a new run requires a server restart (start refuses too).
_TERMINAL_STATES = ("idle", "stopped", "data-missing", "done")


def _git_short_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip() or "nogit"
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "nogit"


def _load_real_candles(path: Path) -> CandleArrays:
    """Same Kraken OHLCVT (no-header) parse as scripts/train_predictor.py's
    load_real_candles — duplicated rather than imported, since src/ never imports
    scripts/ and this web app is a peer entry point, not a script consumer."""
    arr: npt.NDArray[np.float64] = np.loadtxt(
        path, delimiter=",", usecols=(0, 1, 2, 3, 4, 5), dtype=np.float64
    )
    timestamps = arr[:, 0].astype("int64").astype("datetime64[s]")
    ohlcv = arr[:, 1:6].astype(np.float64)
    return timestamps, ohlcv


class TrainingRunner:
    """State machine + pub/sub hub for one training run. `run_training(self)` is
    called on a background thread; it must call `self.broadcast(...)` for every SSE
    message and must eventually broadcast a `type: status` message so the header
    state settles (train_all_folds does this on its own)."""

    def __init__(
        self,
        *,
        data_available: bool,
        checkpoint_dir: Path,
        run_training: Callable[[TrainingRunner], None],
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self._run_training = run_training
        self.state: str = "idle" if data_available else "data-missing"
        self.stop_event = threading.Event()
        self.save_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._subscribers: list[queue.Queue[dict[str, object]]] = []
        self._subs_lock = threading.Lock()
        # Guards the check-then-mutate sequence in start/stop/save: FastAPI dispatches
        # each `def` route handler on its own threadpool worker, so two concurrent
        # requests can otherwise both pass the "not already running" check before
        # either writes state, spawning two training threads for one runner.
        self._state_lock = threading.Lock()
        self.last_status: dict[str, object] = {
            "type": "status", "state": self.state, "wandb_mode": "disabled", "wandb_run_id": None,
        }

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

    def start(self) -> bool:
        with self._state_lock:
            if self.state in ("data-missing", "done") or self.state in ("running", "saving"):
                return False
            # Set state inside the lock so a concurrent start() call sees "running"
            # immediately, before broadcast()/thread-spawn even run.
            self.state = "running"
            self.stop_event.clear()
            self.save_event.clear()
        self.broadcast(
            {"type": "status", "state": "running", "wandb_mode": "disabled", "wandb_run_id": None}
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _run(self) -> None:
        try:
            self._run_training(self)
        except Exception as exc:  # noqa: BLE001 — training boundary must not crash the server
            self.broadcast(
                {
                    "type": "alert", "level": "error",
                    "message": f"Training error — {exc}. Check logs.",
                }
            )
            self.broadcast(
                {"type": "status", "state": "error", "wandb_mode": "disabled", "wandb_run_id": None}
            )

    def stop(self) -> bool:
        with self._state_lock:
            if self.state in _TERMINAL_STATES:
                return False
        self.stop_event.set()
        return True

    def save(self) -> bool:
        with self._state_lock:
            if self.state in _TERMINAL_STATES:
                return False
        self.save_event.set()
        return True


def _default_run_training(runner: TrainingRunner) -> None:
    csv_path = REPO_ROOT / DATA.KRAKEN_HISTORY_OUT_DIR / DATA.KRAKEN_HISTORY_CSV_NAME
    timestamps, ohlcv = _load_real_candles(csv_path)
    validated = validate_candles(timestamps, ohlcv)
    features = compute_features(validated.ohlcv)
    feature_ts = validated.timestamps[1:]
    feature_ts, features = filter_by_historical_start(feature_ts, features)
    folds = make_folds(carve_locked_test(features.shape[0]))
    lookback = DATA.LOOKBACK
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hp = hyperparams_snapshot(lookback)

    def log(payload: dict[str, object]) -> None:
        if payload.get("type") == "fold_complete":
            record: FoldRecord = {
                "fold": int(payload["fold"]),  # type: ignore[call-overload]
                "stem": str(payload["stem"]),
                "train_loss": float(payload["train_loss"]),  # type: ignore[arg-type]
                "val_loss": float(payload["val_loss"]),  # type: ignore[arg-type]
                "da": float(payload["da"]),  # type: ignore[arg-type]
                "q_coverage": float(payload["q_coverage"]),  # type: ignore[arg-type]
                "duration_s": float(payload["duration_s"]),  # type: ignore[arg-type]
                "hyperparams": hp,
            }
            append_fold_record(runner.checkpoint_dir, record)
        runner.broadcast(payload)

    train_all_folds(
        features, feature_ts, folds,
        lookback=lookback, batch_size=PREDICTOR.PROD_BATCH_SIZE, device=device,
        max_epochs=PREDICTOR.MAX_EPOCHS, checkpoint_dir=runner.checkpoint_dir,
        git_sha=_git_short_sha(), constants_sha=sha256_file(REPO_ROOT / "constants.py"),
        log=log, stop_event=runner.stop_event, save_event=runner.save_event,
        finished_dir=REPO_ROOT / PREDICTOR.FINISHED_DIR,
    )


def create_runner() -> TrainingRunner:
    gate = check_data_gate()
    return TrainingRunner(
        data_available=gate.available,
        checkpoint_dir=REPO_ROOT / PREDICTOR.CHECKPOINT_DIR,
        run_training=_default_run_training,
    )


def create_app(runner: TrainingRunner | None = None) -> FastAPI:
    runner = runner or create_runner()
    app = FastAPI(title="BTC Predictor Training Dashboard")

    @app.get("/api/training/history")
    def get_history() -> list[FoldRecord]:
        return read_fold_records(runner.checkpoint_dir)

    @app.get("/api/config")
    def get_config() -> dict[str, object]:
        """Color/behavior thresholds the frontend renders with, sourced from
        constants.py so the JS client never hardcodes a duplicate of a Python value
        (decisions-auditor finding: silent drift when constants.py changes)."""
        return {
            "grad_clip_norm": PREDICTOR.GRAD_CLIP_NORM,
            "grad_norm_instability_multiplier": TRAINING_UI.GRAD_NORM_INSTABILITY_MULTIPLIER,
            "deploy_gate_da_threshold": PREDICTOR.DEPLOY_GATE_DA_THRESHOLD,
            "fold_table_da_amber_floor": TRAINING_UI.FOLD_TABLE_DA_AMBER_FLOOR,
            "fold_table_qcov_healthy_lower": TRAINING_UI.FOLD_TABLE_QCOV_HEALTHY_LOWER,
            "fold_table_qcov_healthy_upper": TRAINING_UI.FOLD_TABLE_QCOV_HEALTHY_UPPER,
            "patience_amber_floor": TRAINING_UI.PATIENCE_AMBER_FLOOR,
            "patience_orange_floor": TRAINING_UI.PATIENCE_ORANGE_FLOOR,
            "patience_glow_floor": TRAINING_UI.PATIENCE_GLOW_FLOOR,
            "early_stopping_patience": PREDICTOR.EARLY_STOPPING_PATIENCE,
            "max_epochs": PREDICTOR.MAX_EPOCHS,
            "loss_chart_ema_window": TRAINING_UI.LOSS_CHART_EMA_WINDOW,
            "loss_chart_visible_epochs": TRAINING_UI.LOSS_CHART_VISIBLE_EPOCHS,
            "batch_trace_max_points": TRAINING_UI.BATCH_TRACE_MAX_POINTS,
            "alert_auto_dismiss_seconds": TRAINING_UI.ALERT_AUTO_DISMISS_SECONDS,
            "alert_max_stack": TRAINING_UI.ALERT_MAX_STACK,
            "button_saved_flash_seconds": TRAINING_UI.BUTTON_SAVED_FLASH_SECONDS,
        }

    @app.post("/api/training/start")
    def start_training() -> dict[str, object]:
        if not runner.start():
            raise HTTPException(status_code=409, detail="cannot start in current state")
        return {"state": runner.state}

    @app.post("/api/training/stop")
    def stop_training() -> dict[str, object]:
        if not runner.stop():
            raise HTTPException(status_code=409, detail="cannot stop in current state")
        return {"state": runner.state}

    @app.post("/api/training/save")
    def save_checkpoint_now() -> dict[str, object]:
        if not runner.save():
            raise HTTPException(status_code=409, detail="cannot save in current state")
        return {"state": runner.state}

    @app.get("/api/events")
    async def events(request: Request) -> StreamingResponse:
        q = runner.subscribe()

        async def gen() -> AsyncIterator[str]:
            try:
                yield f"data: {json.dumps(runner.last_status)}\n\n"
                loop = asyncio.get_event_loop()
                while True:
                    if await request.is_disconnected():
                        break
                    # Bounded get: an unbounded q.get() blocks its executor thread
                    # forever if the client vanishes without a clean disconnect
                    # (network drop, tab crash) — the thread AND this subscriber's
                    # queue would leak until an unrelated broadcast happened to reach
                    # it. Polling with a timeout re-checks is_disconnected() every
                    # SSE_POLL_TIMEOUT_S instead.
                    try:
                        payload = await loop.run_in_executor(
                            None, q.get, True, TRAINING_UI.SSE_POLL_TIMEOUT_S
                        )
                    except queue.Empty:
                        continue
                    yield f"data: {json.dumps(payload)}\n\n"
            finally:
                runner.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


app = create_app()
