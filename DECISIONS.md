# DECISIONS.md

Flat key → current value. Source of truth for every coding session. Any change here requires a `CHANGELOG.md` entry in the same commit. Append-only history lives in `CHANGELOG.md`; this file is overwritten in place.

Last consolidated: 2026-06-28

---

## Toolchain

- **package_manager**: UV. `uv sync` installs the locked environment; `uv add` declares new deps; `uv run` executes all scripts and tools. Never use bare `pip install`.
- **build_backend**: hatchling (UV-native; setuptools removed — project ships no importable package, only scripts)
- **lock_file**: `uv.lock` committed to version control for reproducible installs (`.gitignore` pre-configured to track it)
- **uv_package_install_mode**: `[tool.uv] package = false` (temporary while `src/`/package module does not exist; remove once packaging is wired so `uv sync` installs the project package)
- **python_version**: pinned to `3.13` via `.python-version`. `requires-python` stays `>=3.10`, but the interpreter is pinned because PyTorch cu124 wheels ship for cp310–cp313 only (no cp314 build exists yet).
- **ml_framework**: PyTorch `2.6.0+cu124` (CUDA 12.4 build for the RTX 4060 / sm_89). Installed from the explicit `pytorch-cu124` index (`https://download.pytorch.org/whl/cu124`) pinned via `[tool.uv.sources]`. Predictor loss/model code targets this build; CUDA verified available on the 4060.

## Predictor

- **architecture**: PatchTST encoder, patch_size=16, → 90 tokens at lookback=1440
- **output_head**: quantile q10 / q50 / q90 per OHLCV dimension (5 dims × 3 quantiles per future step)
- **target**: log-return per OHLCV dimension (additive across multi-step horizons; symmetric tails)
- **horizon**: 15 steps direct multi-step. **Autoregressive iteration is banned.**
- **lookback**: TBD via smoke-sweep [240, 720, 1440] before the long training run
- **input_features**: 5 per candle — open log-return, high log-return, low log-return, close log-return, log1p(volume change)
- **vol_logret_floor**: clip `log(volume_t / volume_{t-1})` to `-10.0` (≈ 22,000× volume drop) when `volume_t = 0` or the ratio underflows. Prevents `-inf` from entering the per-fold scaler.
- **vol_t_minus_1_zero_handling**: if `volume_{t-1} = 0`, set `vol_change = 0` (treat as "no information") and emit a structured-log warning. The validator separately surfaces extended zero-volume runs.
- **scaling**: per-fold MinMaxScaler with strict fit-window assertion; rolling features computed sequentially before scaler updates
- **loss**: Pinball (quantile) + direction penalty, λ ∈ [1.5, 2.0]; FEE_THRESHOLD subtracted from predicted per-step PnL inside the loss so the model learns net-of-fee profitable moves
- **geometry_enforcement**: O/H/L/C consistency (`H ≥ max(O,C)`, `L ≤ min(O,C)`) enforced at every inference step, not only in the rollout sampler — including in any Trader-side mock harness
- **retrain_trigger_conditional**: 7-day NLL or quantile-coverage > 2.0× baseline → manual retrain
- **retrain_trigger_calendar**: 30-day maximum gap regardless of drift metrics
- **retrain_window**: fine-tune on `[t-21, t-7]`, gate on `[t-7, t]` — strictly non-overlapping
- **retrain_warm_start**: from prior checkpoint
- **deploy_gates** (all three required, simultaneously): (a) quantile coverage on locked test set within ±5% of original training-time coverage; (b) Directional Accuracy > 53.5% computed only over predictions where `|q50| > FEE_THRESHOLD`; (c) Calibration rate 75–85%
- **deploy_gates_output_dim**: `close_logret` — all three gates evaluate the Close dimension only. Rationale: execution fills at close-of-candle; Close is the tradeable signal. Consistent with the close-dim direction penalty in the loss and the trader's close-based confidence gate.
- **deploy_gates_quantile_index_pattern**: `PREDICTOR.QUANTILES.index(value)` called once at module scope into private constants (`_Q10`, `_Q50`, `_Q90`). No alias integer constants (`Q10_IDX = 0` etc.) in `constants.py`. The `.index()` call is self-documenting and robust — if the tuple order ever changes, it auto-corrects; a hardcoded `0` would silently break.
- **bug_regression_tests** (must pass before training loop ships): variance-floor (`assert loss > 0` for first 100 steps), trend-loss synthetic-input (constant candles → known non-zero output), early-stopping patience exposed in `constants.py`

## Trader

- **architecture**: rules-based for v3. RL deferred until rules-based shows stable signal on validated quantile output (≥3 months paper).
- **inputs**: predictor q10/q50/q90, position state (allocation, unrealized PnL, atr_at_entry), context (atr_normalized, regime, UTC hour sin/cos, day-of-week). **No raw 1440-candle history** — predictor consumed it.
- **position_sizing_base**: fixed-fractional 1% per trade
- **position_sizing_cap**: ±4% hard allocation cap
- **uncertainty_input**: `spread = (q90 - q10) / |q50|` is the explicit input to the confidence gate. Formula is named in `constants.py`, not just in prose. Position size scales from 1% base toward 0 as spread widens.
- **confidence_gate**: if `spread > CONFIDENCE_THRESHOLD` → forced allocation = 0
- **exit_priority_stack** (tier number resolves all conflicts): (1) kill switch → (2) hard stop on net PnL → (3) daily-loss circuit breaker → (4) take-profit → (5) signal reversal ≥3 consecutive candles → (6) trailing stop → (7) time-based
- **stop_loss_evaluation**: net PnL after estimated round-trip fees + slippage. **Identical formula in environment, backtester, and live execution** — implemented once, imported everywhere.
- **predictor_staleness_decay**: linear from full at retrain date to floor (50%) at retrain_date + 30 days; multiplies position size

## Execution engine

- **dashboard_stack**: FastAPI + vanilla JS + Lightweight Charts. (No React.)
- **kill_switch**: file-flag `KILL_SWITCH.flag` + dedicated OS-level watchdog process. Polled every 2s by both watchdog and inference engine. Atomic write via `.flag.tmp` + rename. Dashboard kill button is allowed **only if it writes the file-flag** — no direct API path.
- **kill_switch_test_plan_4_cases**: must all pass before paper trading. Case 3 specifically: write flag from command line while dashboard is offline.
- **fee_model**: Kraken base-tier taker 0.26% per side. Slippage floor 0.05% on every market order (not conditional). Round-trip drag `FEE_THRESHOLD = 0.62%`, defined in `ExecutionConfig` and consumed by both training loss and DA evaluation gate.
- **spread_model**: `spread = 0.0005 + 0.0001 × atr_ratio`, where `atr_ratio = current_ATR / rolling_median_ATR` over the last 1440 candles
- **stale_candle_halt**: at >90s, halt new trades
- **stale_candle_auto_close**: at 5 minutes, auto-close all open positions; alert via sound/beep + dashboard banner
- **stop_loss_exchange_native**: mandatory close order at Kraken on every entry. Engine refuses the order entirely if stop-loss cannot be placed or is invalid. **No naked positions on the book under any code path.**
- **stop_loss_confirmation_required**: position not marked open until exchange confirms stop-loss order. If confirmation does not arrive within N seconds, auto-close any partial fill + sound/beep alert.
- **secrets**: Windows Credential Manager via `keyring`. `.env` reserved for non-secret config only.
- **alerts**: sound/beep (winsound.Beep) + structured JSON logs + dashboard color states
- **network_exposure**: FastAPI bound to `127.0.0.1` only. Remote access via SSH tunnel.
- **paper_live_toggle**: `ExecutionBackend` abstract class. `PaperBackend` and `LiveBackend` are sibling implementations. Parity contract test mandatory before live.
- **api_ingest**: Kraken WebSocket v2 OHLC for real-time, REST `GetOHLCData` for gap backfill (≤12h gaps). Gaps >12h trigger `is_interpolated=True` forward-fill.
- **position_reconciliation**: on startup, query `GetOpenOrders` and `GetOpenPositions` from Kraken. Refuse to start if local SQLite mismatch is unexplained. Kraken is the authoritative source of truth.
- **sha256_manifest_scope**: weights + scaler PKL + `constants.py`. Verified at startup AND on every weight reload.
- **persistence**: SQLite acceptable for paper. PostgreSQL+WAL before live capital. Binary BLOBs (`history_blob`, `futures_blob`, `context_blob`) — not `state_json TEXT`.
- **replay_scrubber**: in-memory rolling deque cached for last N minutes; SQLite only for older history
- **cycle_warning**: 45s warning, 55s hard threshold (with 60s loop)
- **backups**: checkpoints + scaler + `agent_config.json` synced to OneDrive/Google Drive after every write (rclone or filewatcher)
- **kill_criteria_K1_session_drawdown**: 3% in 24h rolling — auto-shutdown, no override
- **kill_criteria_K2_total_drawdown**: 10% — auto-shutdown, no override
- **kill_criteria_K3_pnl_anomaly**: 7-day PnL anomaly — alert + manual review
- **kill_criteria_K4_nll_baseline**: 7-day NLL > 2.0× baseline — auto-shutdown, no override
- **kill_criteria_K5_calibration**: < 50% × 3 days — alert
- **kill_criteria_K6_zero_trades**: 4 hours — alert
- **kill_criteria_K7_winrate**: < 40% × 3 days — alert
- **kill_criteria_K8_stale_candle**: 5 min — auto-shutdown, no override
- **kill_criteria_K9_latency**: > 55s × 3 cycles — alert

## Training UI

- **training_ui_stack**: FastAPI + vanilla JS (browser-based, same stack as the trading dashboard). Runs on a separate port. No React. Full spec in `training-dashboard.md`.
- **training_ui_controls**: start (POST to launch training subprocess), stop (graceful checkpoint save + clean exit), save (write checkpoint now without stopping). Signals sent via threading.Event or subprocess signal; training loop catches signal, writes checkpoint, exits cleanly. Closing the browser tab does not kill the training process.
- **training_ui_metrics**: live display via WebSocket or SSE — fold index, epoch, train loss (pinball + direction), val loss, epoch ETA, fold ETA, total run ETA.
- **training_ui_alerts**: browser notification banner + winsound.Beep (server-side) on fold complete, training error, or early-stop trigger; structured JSON log (same log sink as rest of pipeline).
- **training_ui_export**: on each fold completion, append a fold summary record to `training_metrics.json` (path in `PredictorConfig`). This file is the handoff artifact for Claude optimization review. Schema: fold index, train/val loss, DA, quantile coverage, duration seconds, hyperparams snapshot.
- **training_ui_data_gate**: on startup, check for `data/raw/XBTUSD_1.csv` (i.e. `DATA.KRAKEN_HISTORY_CSV_NAME`). If absent, show a "Download Data" screen before any training controls are enabled. The screen has: (a) a short instruction paragraph, (b) the Google Drive URL derived from `DATA.KRAKEN_HISTORY_GDRIVE_ID`, (c) a drag-and-drop zone accepting the Kraken zip or a bare CSV. On successful upload, transitions to the normal training dashboard. Uses `POST /api/setup/upload-data` (shared implementation with the trading dashboard's setup router).

## Cross-cutting

- **kraken_training_pair**: `XBTUSD` — Kraken's original BTC/USD pair. Inner zip path `master_q4/XBTUSD_1.csv`, extracted to `data/raw/XBTUSD_1.csv`. Data spans 2013-10-07 to 2025-12-31 (244.8 MB, no-header format). `BTCUSD` rejected: only starts 2022-01-01 (2 years), below the 2018-01-01 `historical_start` requirement.
- **historical_start**: 2018-01-01 (covers 2018 bear, 2020 DeFi, 2021 institutional, 2022 FTX, 2023+ ETF era; pre-2017 microstructure rejected)
- **walk_forward_train_val_test**: 150,000 / 50,000 / 10,000 candles
- **walk_forward_stride**: 50,000 (= validation block; non-overlapping validation folds)
- **walk_forward_fold_count**: ~84 non-overlapping validation folds against the 2018+ dataset
- **locked_test_size**: 120,960 candles (84 days × 1440 = 12 × 1-week non-overlapping windows)
- **holdout_evaluation**: walk-forward inside the holdout: 12 × 1-week windows; gate requires positive Sortino on **median AND worst-week**, regime-stratified positive in trending AND ranging
- **constants_organization**: single `constants.py` at root, organized into `@dataclass(frozen=True)` groups: `PredictorConfig`, `RLConfig`, `TraderConfig`, `ExecutionConfig`, `DataConfig`
- **test_discipline**: tests written and committed before implementation. Verified by `test-enforcer` subagent + git log order.
- **branch_model**: `phase-XY` off `main`; merged to `main` on phase exit (no intermediate `developer` branch)
- **doc_drift_policy**: no separate "correction" documents allowed. `DECISIONS.md` updated in place; amendments → `CHANGELOG.md`. Pre-merge check: any change to a decision value must touch `CHANGELOG.md` in the same commit.
- **hyperparameter_tuning_during_walk_forward**: forbidden during a fold gate evaluation. If a fold fails Sortino threshold, EITHER training continues unchanged OR the model is rejected. No "tweak and rerun." (Bonferroni defense.)

## Pre-training gate (signal-first sanity check)

- **baselines**: momentum (rolling-return sign), mean-reversion (z-score vs. rolling mean), volatility breakout (ATR-normalized move)
- **gate_threshold**: at least one baseline must show DA > 52% consistently across walk-forward folds
- **on_failure**: revisit feature engineering before any model training
- **baseline_as_minimum_bar**: best baseline Sharpe becomes the minimum bar PatchTST must clear in walk-forward evaluation

## Pre-paper gate

- All 13 critical-path build items green (see `INDEX.md` build-order task)
- Walk-forward 12×1w gate passed
- Stop-loss net PnL evaluated identically in environment, backtester, and live execution (parity test)
- SHA256 manifest verified
- ExecutionBackend parity test green
- UTC + Windows-DST simulation tests green
- Kill switch 4-case test green
- Position reconciliation tested against real Kraken
- Robustness & stress test gate green: feature ablation, noise injection (σ = 0.5× ATR), 2020/2022 sub-period validation, system chaos (kill mid-trade, disconnect internet, restart under active position)

## Pre-live gate

- All paper-trading gates plus:
- ≥1 month minimum paper trading
- Regime-stratified positive
- **Permutation test** on trade PnL distribution vs. random buy/sell signals applied to the same real price series at the bot's actual trade frequency, **p < 0.05** (not t-test; BTC returns have fat tails that violate normality)
- LiveBackend implemented and parity-tested against PaperBackend
- Exchange-native stop-loss confirmed on every entry path
- Kraken API key scope verified programmatically: trade-only, no withdraw, no deposit
- Kill criteria K1–K9 wired in code (auto-shutdown items NOT operator-overridable)
- Disaster recovery runbook written and rehearsed
