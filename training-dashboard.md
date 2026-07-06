# Training Dashboard — Full Spec

Browser-based app (FastAPI + vanilla JS, no React) for managing and monitoring model training runs. Runs on a separate port from the trading dashboard. Both dashboards share the same stack but are independent FastAPI processes.

> **Distinct from the Trading Dashboard.** The trading dashboard (`dashboard.md`) is for live/paper trading telemetry. This app is for training only — it has no kill switch, no position state, and no candle stream.

---

## Data availability

The raw data file (`XBTUSD_1.csv`, Kraken OHLCVT minute data) is synced locally from Google Drive via Google Drive for Desktop. Its location is configured by the `KRAKEN_DATA_PATH` environment variable, which points to the local sync path of `Kraken_OHLCVT.zip`. The server reads this at startup — there is no file upload UI. If the CSV is absent at startup the server logs an error and training controls remain disabled until the env var is corrected and the server is restarted.

---

## Visual aesthetic

Dark-themed, minimal, high information density. Designed to be glanced at across the room during a multi-week background run. Palette:

| Role | Hex |
|---|---|
| Page background | `#0d0d0d` |
| Card / panel background | `#141414` |
| Card border | `#1f1f1f` |
| Primary text | `#e8e8e8` |
| Muted / label text | `#666` |
| Accent (running state) | `#4a9eff` (blue) |
| Success / good | `#2ecc71` (green) |
| Warning | `#f39c12` (amber) |
| Error / alert | `#e74c3c` (red) |
| Chart line — train | `#4a9eff` |
| Chart line — val | `#2ecc71` |

Font: system monospace stack (`ui-monospace, Consolas, "Courier New", monospace`) for all metric values. Sans-serif (`system-ui, -apple-system, sans-serif`) for labels and headings. No external font loads.

---

## Page layout (ASCII wireframe)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BTC Predictor Training                              [Training Dashboard]   │
│  ●  Running — fold 3 / 84,  epoch 12 / 100                          [Stop] │
├───────────────────────┬─────────────────────────────────────────────────────┤
│  A  STATUS BAR        │  B  LOSS CHART (rolling)                            │
│  (fixed-height strip) │                                                     │
│                       │  train ─────    val ─────                           │
├─────────┬─────────────┤                                                     │
│  C      │  D          │                                                     │
│  FOLD   │  EPOCH      │                                                     │
│  GAUGE  │  GAUGE      ├─────────────────────────────────────────────────────┤
│         │             │  E  LOSS COMPONENT STRIP                            │
├─────────┴─────────────┤  pinball ___  direction ___  total ___              │
│  F  ETA STRIP         ├─────────────────────────────────────────────────────┤
│  epoch / fold / total │  G  FOLD HISTORY TABLE                              │
├───────────────────────│                                                     │
│  H  PATIENCE BAR      │  (scrollable, one row per completed fold)           │
├───────────────────────┤                                                     │
│  I  W&B PANEL         │                                                     │
│  (conditional)        │                                                     │
└───────────────────────┴─────────────────────────────────────────────────────┘
│  J  ALERT BANNER  (full width, slides in from bottom, auto-dismisses)       │
└─────────────────────────────────────────────────────────────────────────────┘
```

Column split: left column is 280px fixed; right column fills remaining width. Left column panels stack vertically (no scroll). Right column (B, E, G) stack vertically with the chart and table taking the bulk of the height.

---

## Panel-by-panel specification

### Header bar (always visible, top of page)

Single dark strip (`#0d0d0d`, 48px tall). Three zones:

- **Left:** App title `BTC Predictor Training` in muted small-caps, plus `[Training Dashboard]` subtitle badge in `#222` pill.
- **Center:** Status line: colored dot + plain-English state string (see State machine below) + current fold/epoch summary when running.
- **Right:** ONE stateful control button (outlined style, no fill until hover): `Start` when idle/stopped/errored (disabled and grayed while the data gate has not cleared), `Stop` while running (spinner while a stop is in flight — the server saves the interrupted fold's checkpoint before confirming), a brief `Saved` flash (`TrainingUIConfig.BUTTON_SAVED_FLASH_SECONDS`) once the stop's `stopped` status arrives, then back to `Start`. After natural completion of the whole run the button is a grayed-out, non-interactive `Done`. There is no separate Save button — saving happens automatically on stop and at every fold completion.

**State machine for the center status line:**

| State | Dot color | Text example |
|---|---|---|
| Idle — data ready | Gray | `Idle — ready to start` |
| Idle — data missing | Red | `Data missing — check KRAKEN_DATA_PATH` |
| Running | Blue (pulse) | `Running — fold 3 / 84, epoch 12 / 100` |
| Early stopped | Amber | `Early stopped — fold 3, epoch 47` |
| Error | Red | `Error — see alert below` |
| Stopped (user-initiated, clean) | Green (fade) | `Stopped — checkpoint saved` |
| Saving | Blue | `Saving checkpoint…` |
| Run complete (all folds done) | Green | `Training complete — you may now close this tab.` |

The dot pulses (CSS `@keyframes` opacity 1 → 0.4 → 1 at 1.5s) while running.

---

### A — Status bar (left column, top)

Fixed-height strip (~64px) below the header. Contains only the training subprocess status in large monospace:

```
FOLD  3 / 84
EPOCH 12 / 100
```

Label text in muted gray, values in primary white, bold. This is a quick-glance "where are we" widget that stays visible even when the user has scrolled the right column.

---

### B — Loss chart (right column, dominant panel)

**Purpose:** Rolling line chart of train loss (blue) and val loss (green) across epochs. The chart should be tall enough to see loss curves clearly — minimum 280px, ideally 360px.

**Implementation:** Plain `<canvas>` element with a hand-rolled renderer, or a minimal chart library (Chart.js is acceptable; no D3). No Lightweight Charts — that is the trading dashboard's library.

**What is plotted:**

- X axis: epoch number (global, across folds). Each fold boundary is drawn as a faint vertical dashed line labeled `F0`, `F1`, … in muted text at the top edge.
- Y axis: loss value. Auto-scaled per the visible window. Log scale toggle button in the top-right corner of the panel.
- Two series: `train total` (blue) and `val total` (green). Both are smoothed with a 5-epoch EMA. Raw values shown as faint dots behind the smoothed line.
- The current epoch's raw batch-level train loss is shown as a thin, semi-transparent blue trace below the epoch-averaged series — this gives real-time within-epoch visibility without cluttering the fold-level view.

**Interaction:**

- Hover tooltip: shows `Epoch N | train: X.XXXX | val: X.XXXX` for the nearest X position. 
- No zoom or pan (overkill for a training monitor; the chart auto-scrolls to keep the current epoch at the right edge, and the last 200 epochs are always visible).

**Visual treatment of early-stop events:**

- A vertical amber line is drawn at the epoch where early stopping fired, labeled `ES` in amber. This persists across folds.

---

### C — Fold gauge (left column)

A simple filled progress bar:

```
FOLD
████████░░░░░░░░░░░░  3 / 84
```

Segmented style: 84 thin segments, each filled blue when that fold completes. At 84 segments the segment width makes them very thin (~2px each at 280px width) — acceptable; the numeric `3 / 84` label carries the meaning.

Background: `#141414`. Filled segments: `#4a9eff`. Separator between current-fold and future: `#1f1f1f`.

---

### D — Epoch gauge (left column)

Same segmented style but for epochs within the current fold:

```
EPOCH (fold 3)
██████░░░░░░░░░░░░░░  12 / 100
```

100 segments, each representing one epoch. The best-val-total epoch (the one that will be checkpointed) is marked with a green tick below the bar rather than a different fill color — this avoids confusion with the "progress so far" fill. The patience window (last 10 epochs since the best) is shaded amber to signal how close early stopping is.

Resets on each fold transition.

---

### E — Loss component strip (right column, below chart)

A compact horizontal strip (~72px tall) showing the three live loss components for the current epoch's val split, plus current LR and grad norm from the latest train batch. Loss components update once per epoch; LR and grad norm update once per batch (both change every step under warmup-cosine + AdamW).

```
PINBALL          DIRECTION        TOTAL            LR          GRAD NORM
0.4821           0.1203           0.6024           1.82e-4     1.4203
(–0.0031)        (+0.0012)        (–0.0019)
```

Each loss component has:
- Large monospace value (current epoch)
- Small delta vs. prior epoch in muted text, colored green if improving (decreasing), red if worsening

`LR` and `GRAD NORM` have no delta row (they're per-batch, not per-epoch — a delta vs. the prior batch is too noisy to be useful). `GRAD NORM` is colored amber if it exceeds `PREDICTOR.GRAD_CLIP_NORM` (i.e. clipping is actively engaging that step) and red if it exceeds 10x that threshold (signals a possible instability worth stopping for). `LR` is plain white throughout — it follows the fixed warmup-cosine schedule and isn't a health signal by itself.

Baseline from the real-data smoke (`val q90 coverage = 0.9880`) is not shown here — it lives in the fold table. This strip is raw loss, LR, and grad norm only.

---

### F — ETA strip (left column)

Three ETAs stacked vertically in compact form:

```
EPOCH ETA    0:04:31
FOLD ETA     3:22:10
TOTAL ETA    9d 14:22
```

Values are right-aligned. Labels in muted gray. Updated once per epoch based on elapsed time and remaining epoch/fold count. `TOTAL ETA` uses exponential smoothing over recent fold durations. Shows `—` when fewer than 2 epochs have completed (not enough data to extrapolate).

---

### H — Patience bar (left column)

A horizontal progress bar showing the early-stopping patience counter:

```
PATIENCE  ████████░░  8 / 10
```

Fills left-to-right as patience accumulates. Color transitions:
- 0–4: blue (`#4a9eff`)
- 5–7: amber (`#f39c12`)
- 8–9: orange
- 10: red (triggers early stop)

Resets to 0 at each fold transition. A subtle amber glow appears behind the bar when patience >= 7 to draw the eye.

---

### I — W&B panel (left column, conditional)

Displayed only when W&B mode is `online`. Completely hidden (zero height) when mode is `disabled`. When mode is `offline`, shows a single muted line: `W&B offline — logs saved locally`.

**Online mode content:**

```
W&B RUN
abc1234-s9f3a2b1-c7e4d3f2-fold3

[Open in W&B →]
```

- Run ID in small monospace, truncated to fit the 280px column (full value in `title` attribute for hover).
- Clickable link opens `https://wandb.ai/<user>/btc-bot-v3-predictor/runs/<run_id>` in a new tab.
- Small W&B logo (14×14 inline SVG or emoji `⚡`) before the "W&B RUN" label to make the panel scannable.
- Panel background slightly lighter (`#1a1a1a`) to distinguish it from metric panels.

The run ID format `{git_sha}-s{scaler_sha[:8]}-c{constants_sha[:8]}-fold{fold_id}` comes directly from `train_predictor.py`'s `make_run_tag`. The panel simply displays the current tag; no parsing or prettification.

---

### G — Fold history table (right column, below loss component strip)

Scrollable table, one row per completed fold. Newest fold at the top (insertion order reversed).

| Column | Width | Format |
|---|---|---|
| Fold | 48px | Integer, 0-indexed |
| Train loss | 96px | `0.XXXX` (total epoch-avg) |
| Val loss | 96px | `0.XXXX` (total epoch-avg) |
| DA | 72px | `54.3%` |
| Q-cov | 72px | `0.9880` |
| Duration | 80px | `3h 22m` |

**Column notes:**

- `Val loss`: colored green if it is the best val loss seen so far (column minimum), otherwise white. This lets the user instantly see which fold produced the checkpointed weights.
- `Q-cov`: colored red if below 0.85 or above 0.95 (outside the expected ±5% deploy-gate band around 0.90 baseline), green if 0.85–0.95, white otherwise.
- `DA`: colored green if > 53.5% (above the deploy gate), amber if 50–53.5%, red if < 50%.
- Rows flash briefly (250ms green fade-in) when a new fold row is appended.

Header row is sticky (stays at the top when the table scrolls). Table max-height is set to fill available right-column space below panels B and E, with `overflow-y: auto`.

The table reads from `training_metrics.json` on page load (historical folds) and appends rows via the live metric stream as new folds complete.

---

### J — Alert banner (full-width, bottom of viewport)

A fixed-position strip that slides up from the bottom on an alert event. Auto-dismisses after 8 seconds; also has an `×` close button. Stacks up to 3 simultaneous banners (most recent on top); older banners are dropped once the stack exceeds 3.

Alert events and their colors:

| Event | Color | Example text |
|---|---|---|
| Fold complete | Green | `Fold 3 complete — val loss 0.6024, DA 54.3%, Q-cov 0.9880. Checkpoint saved.` |
| Early stop triggered | Amber | `Early stopping triggered — fold 3, epoch 47 (patience exhausted at 10).` |
| Training error | Red | `Training error — subprocess exited with code 1. Check logs.` |
| Save complete | Blue | `Checkpoint saved — checkpoints/abc1234-...-fold3.pt` |

Each banner uses an icon prefix (plain Unicode, no library): `✓` green, `⚠` amber, `✕` red, `↓` blue. Banners do not replace each other — each event appends a new banner.

Server-side `winsound.Beep` fires on the same events (fold complete, early stop, error). Structured JSON log entry written via structlog.

---

## Data gate (startup check)

On server startup, FastAPI checks for `data/raw/XBTUSD_1.csv` (the path referenced by `DATA.KRAKEN_HISTORY_CSV_NAME` in `constants.py`). If the file is absent:

- The training control button (`Start`) is disabled and grayed.
- The status line reads `Data missing — check KRAKEN_DATA_PATH` in red.
- A yellow info box appears below the header: `Raw data file not found. Set KRAKEN_DATA_PATH to the local sync path of Kraken_OHLCVT.zip and restart the server.`

The dashboard does not offer a download or upload UI. Data management is handled outside the browser (Google Drive for Desktop sync). The server must be restarted after the env var is fixed — no hot-reload of the data path.

---

## Live metric stream

The server exposes a single SSE endpoint (`GET /api/events`) or WebSocket (`WS /api/ws`). Payload is newline-delimited JSON. Each message has a `type` field:

```json
{ "type": "batch", "step": 1042, "fold": 3, "epoch": 12,
  "pinball": 0.4821, "direction": 0.1203, "total": 0.6024,
  "lr": 0.000182, "grad_norm": 1.4203,
  "split": "train" }

{ "type": "epoch", "fold": 3, "epoch": 12, "max_epochs": 100,
  "train_pinball": 0.4821, "train_direction": 0.1203, "train_total": 0.6024,
  "val_pinball": 0.4690, "val_direction": 0.1250, "val_total": 0.5940,
  "patience": 2, "epoch_eta_s": 271, "fold_eta_s": 12130, "total_eta_s": 827640,
  "best_val_total": 0.5880 }

{ "type": "fold_complete", "fold": 3, "stem": "abc1234-s9f3a2b1-c7e4d3f2-fold3",
  "train_loss": 0.6024, "val_loss": 0.5940,
  "da": 0.543, "q_coverage": 0.9880, "duration_s": 12240,
  "checkpoint_path": "checkpoints/abc1234-s9f3a2b1-c7e4d3f2-fold3.pt" }

{ "type": "alert", "level": "error", "message": "subprocess exited with code 1" }

{ "type": "status", "state": "running|idle|stopped|saving|error|done",
  "promoted": 4,
  "wandb_mode": "online|offline|disabled",
  "wandb_run_id": "abc1234-s9f3a2b1-c7e4d3f2-fold3" }
```

The `stem` on `fold_complete` is the run-tag join key the benchmark analysis endpoint
pairs a benchmark result with the training record on. `state:"done"` is emitted on a
run that reaches its natural end (all folds, no user stop); its `promoted` count is how
many finished checkpoints were copied into `PredictorConfig.FINISHED_DIR` for the
benchmark app. A run ended by Stop emits `state:"stopped"` and promotes nothing.

The JS client subscribes on page load and routes each message type to the appropriate panel update function. The chart, gauges, ETA strip, patience bar, W&B panel, and alert banners are all driven by this stream. There is no polling.

---

## Training controls

ONE stateful control button, always visible after the first status message:

| Button face | Fires | Shown when |
|---|---|---|
| `Start` | `POST /api/training/start` | State is `idle`, `stopped` (after the Saved flash), or `error`. Disabled while the data gate has not cleared. |
| `Stop` | `POST /api/training/stop` | State is `running` or `saving`. Spinner from click until the `stopped` status arrives (the server saves the interrupted fold's gate-evaluated checkpoint in between). |
| `Saved` (flash) | — (disabled) | For `TrainingUIConfig.BUTTON_SAVED_FLASH_SECONDS` after a live-observed stop confirms, then reverts to `Start`. |
| `Done` | — (disabled, grayed) | State is `done` — the whole walk-forward run completed. Terminal: the server refuses start/stop (409); a new run requires a server restart. |

There is no Save button. Saving is automatic: every fold completion writes a gate-evaluated checkpoint, and Stop saves the interrupted fold's checkpoint before exiting. `POST /api/training/save` (`save_event` mid-fold snapshot, `train_q90_coverage = NaN`, refused by `deploy_predictor.py`) remains for scripted use only.

Button responses are optimistic: the button state changes immediately on click, with a spinner replacing the label until the server confirms the state change via the SSE stream. If the server returns a 4xx/5xx, the button reverts and an error banner fires.

Closing the browser tab does not kill the training process. The subprocess outlives the browser session.

---

## Fold history export

On each fold completion, the server appends a record to `training_metrics.json` (path defined in `PredictorConfig.CHECKPOINT_DIR` + `training_metrics.json`). This file is the handoff artifact for post-training analysis. Schema:

```json
{
  "fold": 3,
  "train_loss": 0.6024,
  "val_loss": 0.5940,
  "da": 0.543,
  "q_coverage": 0.9880,
  "duration_s": 12240,
  "hyperparams": {
    "lookback": 1440,
    "horizon": 15,
    "patch_size": 16,
    "d_model": 128,
    "direction_lambda": 1.75,
    "max_epochs": 100,
    "early_stopping_patience": 10
  }
}
```

The `hyperparams` snapshot is taken from `constants.py` values at training-start time, not at fold completion — they are frozen for the run.

---

## What the training dashboard does NOT do

- No kill switch (not a trading app).
- No candle stream or position state.
- No model deployment or redeploy trigger — that is a CLI gate (`scripts/deploy_predictor.py`).
- No hyperparameter editing — all config lives in `constants.py` and is frozen at runtime.
- No W&B chart embedding — the W&B panel shows a run link only; charts are viewed at wandb.ai.
- No drag-and-drop data upload — data management is handled outside the browser.

---

## File layout (`src/training_ui/`)

```
src/training_ui/
    app.py              # FastAPI app, SSE/WS endpoint, training subprocess mgmt
    setup_router.py     # startup data-gate check
    exporter.py         # fold completion → training_metrics.json append
static/training_ui/
    index.html
    app.js
    chart.js            # canvas chart renderer (or Chart.js shim)
    style.css
```

No build step. All JS is plain ES6 modules served as static files by FastAPI's `StaticFiles` mount.

---

## Alerts

Browser notification banner (panel J) on fold complete, training error, or early-stop triggered. `winsound.Beep` fired server-side on the same events. Structured JSON log written via the same log sink as the rest of the pipeline. No browser Notification API (requires HTTPS/permission dance; the banner is sufficient for a local-only tool).
