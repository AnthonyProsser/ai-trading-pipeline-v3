# Trading Dashboard

Read-only telemetry, optimization tooling, and a single capital-safety affordance (the kill button). FastAPI backend + vanilla JS + Lightweight Charts. No React.

> **Distinct from the Training Dashboard.** The training dashboard (`training-dashboard.md`) is a separate browser app for managing training runs. This file covers only the trading dashboard used during live/paper trading.

## Stack

- FastAPI for HTTP + WebSocket
- Vanilla JS for frontend (Lightweight Charts for candle rendering). The v2 `prediction_viewer.jsx` is re-implemented in vanilla JS, not migrated.
- Bound to `127.0.0.1` only (see `execution-engine.md`).
- Kill button writes `KILL_SWITCH.flag` via the same atomic-write path as the command line — no direct API into the inference engine. **Capital safety never depends on the dashboard.**

## Panels

### Core trading panels

1. **Candle stream** — Lightweight Charts; live OHLC + volume.
2. **Predictor accuracy** — quantile band overlay (q10, q50, q90) for the 15-step horizon, vs. realized values once they arrive.
3. **Position state** — current allocation, unrealized PnL (gross + net of fees/slippage), atr_at_entry, time-in-position.
4. **Health** — cycle latency, stale-candle status, predictor SHA256 match, scaler SHA256 match.
5. **Kill button** — single button, double-confirmation, writes the file-flag.
6. **Replay scrubber** — read from in-memory rolling deque for last N minutes; SQLite for older.

### Optimization panels

7. **Performance metrics** — rolling Sharpe, Sortino, win rate, avg win/loss, max drawdown. Three time windows: 7d / 30d / all-time. Sourced from SQLite trade log. Not duplicated from W&B (W&B tracks training loss only).
8. **Regime analysis** — performance breakdown by market regime (trending / ranging). Same trade log source. Required by the pre-paper gate (regime-stratified positive gate).
9. **Fee drag** — cumulative gross PnL vs. net PnL, with the delta labeled as fee + slippage drag. Sanity-checks that the fee model in `ExecutionConfig` matches real outcomes.
10. **Model management** — active checkpoint SHA256 (short), training date, W&B run ID (link if online), and the three deploy gate scores at time of deployment (quantile coverage, DA, calibration rate). Read-only.
11. **Kill criteria status** — live values for all K1–K9 criteria alongside their thresholds. Color-coded: green = safe, yellow = approaching, red = breached. Auto-shutdown criteria (K1, K2, K4, K8) labeled as non-overridable.

### Data collection panel

12. **Data collection** — two sub-sections:
    - **Status**: last candle timestamp, gap count since last ingest, WebSocket connection state, time since last REST gap-fill.
    - **Trigger**: "Collect now" button → `POST /api/data/collect` → runs the Kraken REST gap-fill on demand (same logic as the execution engine's startup gap-fill, not `ingest_kraken_history.py`). Disabled if WebSocket is live and gap count is zero.

### AI analysis panel

13. **AI analysis** — two sub-sections:
    - **LLM report**: "Analyze" button → `POST /api/analysis/run` → server assembles a structured prompt from recent trades, rolling metrics, model drift indicators (NLL trend, calibration coverage), and kill-criteria values, then calls the Claude API and streams the response into a read-only text area. The prompt template lives in `src/dashboard/analysis_prompt.py` (version-controlled, not hardcoded in the handler).
    - **On-chart predictor insights**: additional overlays on the predictor accuracy panel — quantile band width over time (spread = `(q90 - q10) / |q50|`), calibration coverage rolling average, NLL trend. Sourced from the existing WebSocket feed; no additional API call.

## WebSocket payload schema

Versioned. Every payload carries:

```
{
  "schema_version": "1.0",
  "predictor_hash": "<sha256[:8]>",
  "predictor_contract_version": "<sha256[:8]>",
  "ts_utc": "<ISO8601>",
  "type": "candle" | "prediction" | "position" | "health" | "alert" | "metrics" | "criteria",
  "payload": { ... }
}
```

The dashboard refuses to render the prediction panel if `predictor_contract_version` differs from the build version it was compiled against. Prevents silent visualization breakage across predictor versions.

## Color states

- Green: healthy, predictions in calibration, latency < 45s
- Yellow: latency 45–55s, OR calibration drift, OR stale candle 30–90s
- Red: stale candle > 90s, OR K-criteria alert fired, OR predictor SHA mismatch, OR stop-loss confirmation timeout
- Black: kill switch active, OR auto-shutdown fired

## What the trading dashboard does NOT do

- **No drag-and-drop data upload.** Data setup lives in the training dashboard.
- **No control affordances besides kill.** No "increase position size", no "force trade", no "override stop". Anything that touches capital lives in `execution-engine.md` and the file-flag.
- **No retrain trigger button.** Retraining is a manual user decision via CLI scripts, gated by the three deploy gates.
- **No mutable settings.** All config lives in `constants.py` and `agent_config.json`. The dashboard only reads them.

## Calibration panels

Built in week 3–4 of the parallel training run, not week 1. They depend on having real predictor output at calibration time — useless before then.

## Defer list (post-paper-trading)

- A "pretty" version. Functional only is the rule for v3 Phase 4: kill switch button, stale-data banner, candles + orders + health WebSocket types. Polish later.
- AI analysis panel deferred until paper trading produces enough trade history for the LLM report to be meaningful (minimum ~50 trades).
