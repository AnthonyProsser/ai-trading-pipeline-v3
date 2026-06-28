# Training Dashboard

Browser-based app (FastAPI + vanilla JS, no React) for managing and monitoring model training runs. Runs on a separate port from the trading dashboard. Both dashboards share the same stack but are independent FastAPI processes.

> **Distinct from the Trading Dashboard.** The trading dashboard (`dashboard.md`) is for live/paper trading telemetry. This app is for training only — it has no kill switch, no position state, and no candle stream.

## First-run: Download Data screen

Shown when `data/raw/BTCUSD_1.csv` is absent. All training controls are disabled until data is present.

- Short instruction paragraph explaining the requirement.
- Google Drive link derived from `DATA.KRAKEN_HISTORY_GDRIVE_ID` (no hardcoded ID in frontend).
- Drag-and-drop zone accepting the Kraken zip or a bare CSV.
  - Zip: server extracts `DATA.KRAKEN_HISTORY_INNER_PATH` member only.
  - CSV: written directly to `data/raw/BTCUSD_1.csv`.
  - Atomic write via temp-file-then-rename.
  - Returns `{"status": "ok", "rows": <row_count>}` on success.
- On success: page transitions to the normal training dashboard without manual refresh.
- Endpoint: `POST /api/setup/upload-data` — shared implementation with the trading dashboard's setup router; disabled (404) once the CSV exists.

## Training controls

Three buttons, always visible after the data gate clears:

| Button | Action |
|---|---|
| Start | `POST /api/training/start` — launches the training subprocess |
| Stop | `POST /api/training/stop` — signals graceful checkpoint save + clean exit |
| Save | `POST /api/training/save` — writes checkpoint now without stopping |

Closing the browser tab does not kill the training process. The subprocess outlives the browser session.

## Live metrics panel

Streamed via WebSocket or SSE from the training subprocess (via `training_metrics.json` tail or an IPC channel):

- Fold index / total folds
- Epoch / max epochs
- Train loss (pinball component, direction component, total)
- Val loss (pinball component, direction component, total)
- Epoch ETA, fold ETA, total run ETA
- Early-stopping patience counter

## Fold history table

One row per completed fold, populated from `training_metrics.json`:

| Column | Value |
|---|---|
| Fold | Index |
| Train loss | Total |
| Val loss | Total |
| DA | Directional accuracy |
| Q-coverage | Quantile coverage |
| Duration | Seconds |

## W&B link

Read-only panel showing the current W&B run ID and a clickable link to the W&B run page. Derived from the run name logged by `train_predictor.py` (`{git_sha}-s{scaler_sha[:8]}-c{constants_sha[:8]}-fold{fold_id}`). Shown only when W&B mode is `online`.

## Alerts

Browser notification banner on fold complete, training error, or early-stop triggered. `winsound.Beep` fired server-side on the same events. Structured JSON log written via the same log sink as the rest of the pipeline.

## What the training dashboard does NOT do

- No kill switch (not a trading app).
- No candle stream or position state.
- No model deployment or redeploy trigger — that is a CLI gate (`scripts/deploy_predictor.py`).
- No hyperparameter editing — all config lives in `constants.py`.
