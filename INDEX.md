# INDEX.md — task → files

The lookup table for every coding session. Find the row that matches the task, load **only** the files in the right column. Each row caps at 3 files. If a real task needs more, the row is wrong (cards too granular OR task too coarse) — re-split before working around it.

`DECISIONS.md` is loaded in every session as the orientation step (per `CLAUDE.md` §2). It is **not** repeated in the rows below — only files that go beyond the orientation set are listed.

---

## Phase −1 / Setup

| Task | Files |
|---|---|
| Add a new entry to `constants.py` | `constants.py` |
| Stand up the doc set / verify all docs exist | (none — `CLAUDE.md` + `DECISIONS.md` + `INDEX.md` are loaded by default) |
| Lock the predictor I/O contract | `predictor-contract.md` |

## Phase 0 / Data

| Task | Files |
|---|---|
| Add a new candle feature | `feature-pipeline.md`, `constants.py`, `src/data/feature_pipeline.py` |
| Audit for data leakage | `feature-pipeline.md`, `splits-validation.md` |
| Set up a new walk-forward fold | `splits-validation.md`, `constants.py`, `src/data/walk_forward.py` |
| Build per-fold MinMaxScaler | `feature-pipeline.md`, `src/data/scaler.py` |
| Carve out the locked test set | `splits-validation.md`, `src/data/walk_forward.py` |
| Build CandleValidator (corruption + gap rules) | `feature-pipeline.md`, `src/data/validator.py` |
| Run the baseline signal check | `feature-pipeline.md`, `scripts/baseline_signal_check.py` |
| Wire the SHA256 manifest | `agent-config.md`, `src/data/manifest.py` |

## Phase 1 / Predictor

| Task | Files |
|---|---|
| Change the predictor lookback | `predictor-contract.md`, `constants.py` |
| Debug a predictor training-loop issue | `predictor-training.md`, `src/predictor/loss.py` |
| Add or modify the loss function | `predictor-training.md`, `constants.py`, `src/predictor/loss.py` |
| Implement the variance-floor / trend-loss / patience regression tests | `predictor-training.md`, `tests/predictor/test_training_bugs.py` |
| Run the predictor smoke run (1 epoch, 1 fold) | `predictor-training.md`, `scripts/train_predictor.py` |
| Run a lookback sweep | `predictor-training.md`, `scripts/train_predictor.py` |
| Implement geometry enforcement (`H ≥ max(O,C)`, `L ≤ min(O,C)`) at every inference step | `predictor-contract.md`, `src/predictor/rollout.py` |
| Run retrain deploy gates | `predictor-contract.md`, `scripts/deploy_predictor.py` |
| Deploy a new predictor checkpoint | `agent-config.md`, `scripts/deploy_predictor.py`, `agent_config.json` |

## Phase 1.5 / Training Dashboard

Built before the long training run launches, so the first run is observable and diagnosable. See `training-dashboard.md`. (Branch: `phase-15-training-ui` off `main`.)

| Task | Files |
|---|---|
| Training Dashboard: startup data gate (KRAKEN_DATA_PATH check, no upload UI) | `training-dashboard.md`, `src/training_ui/setup_router.py` |
| Wire training controls (start/stop/save) | `training-dashboard.md`, `src/training_ui/app.py` |
| Wire live metrics stream (fold index, epoch, loss, ETAs) | `training-dashboard.md`, `src/training_ui/app.py` |
| Wire fold-completion export to `training_metrics.json` | `training-dashboard.md`, `constants.py`, `src/training_ui/exporter.py` |
| Wire fold history table from `training_metrics.json` | `training-dashboard.md`, `src/training_ui/app.py` |

## Phase 2 / Execution engine + environment

| Task | Files |
|---|---|
| Write the execution engine 60s loop | `execution-engine.md`, `src/execution/loop.py` |
| Wire the kill-switch flag check | `execution-engine.md`, `src/execution/watchdog.py` |
| Run the kill-switch 4-case test | `execution-engine.md`, `tests/execution/test_kill_switch.py` |
| Build PaperBackend / LiveBackend siblings | `execution-engine.md`, `src/execution/backends.py` |
| Implement exchange-native stop-loss path | `execution-engine.md`, `src/execution/orders.py` |
| Implement the fee + slippage model | `execution-engine.md`, `constants.py` |
| Wire Kraken WebSocket ingest + REST gap-fill | `execution-engine.md`, `src/execution/ingest.py` |
| Implement position reconciliation on startup | `execution-engine.md`, `src/execution/reconcile.py` |
| Wire stale-candle halt + auto-close | `execution-engine.md`, `src/execution/staleness.py` |
| Set up secrets via Windows Credential Manager | `execution-engine.md`, `src/execution/secrets.py` |
| Wire sound/beep + log alerter | `execution-engine.md`, `src/execution/alerter.py` |

## Phase 3 / Trader + backtester

| Task | Files |
|---|---|
| Change the stop-loss formula | `trader-rules.md`, `constants.py` |
| Implement a new trader exit rule | `trader-rules.md`, `constants.py`, `src/trader/exit_priority.py` |
| Implement the confidence gate | `trader-rules.md`, `constants.py`, `src/trader/sizing.py` |
| Implement the staleness decay | `trader-rules.md`, `src/trader/sizing.py` |
| Implement quantile-spread → position size formula | `trader-rules.md`, `src/trader/sizing.py` |
| Build the vectorized backtester | `trader-rules.md`, `execution-engine.md`, `scripts/backtest.py` |
| Run a backtest | `trader-rules.md`, `scripts/backtest.py` |
| Run the robustness & stress test gate | `trader-rules.md`, `scripts/robustness_gate.py` |
| Run the chaos test (kill mid-trade, disconnect internet) | `execution-engine.md`, `scripts/chaos_test.py` |

## Phase 4 / Dashboards

The Training Dashboard moved to Phase 1.5. This phase builds the Trading Dashboard only.

### Trading Dashboard (`src/dashboard/`)

| Task | Files |
|---|---|
| Wire core trading panels (candles, predictor accuracy, position, health, kill, replay) | `dashboard.md`, `src/dashboard/main.py` |
| Add predictor accuracy panel with q10/q90 overlay + on-chart insights | `dashboard.md`, `predictor-contract.md`, `src/dashboard/main.py` |
| Wire kill-switch button (writes file-flag only) | `dashboard.md`, `execution-engine.md` |
| Wire WebSocket payload schema with `predictor_hash` + `predictor_contract_version` | `dashboard.md`, `agent-config.md` |
| Wire replay scrubber against in-memory deque | `dashboard.md`, `src/dashboard/replay.py` |
| Build performance metrics + regime analysis panels | `dashboard.md`, `src/dashboard/metrics_router.py` |
| Build fee drag + model management + kill criteria panels | `dashboard.md`, `src/dashboard/metrics_router.py` |
| Build data collection panel (status + collect trigger) | `dashboard.md`, `execution-engine.md`, `src/dashboard/data_router.py` |
| Build AI analysis panel (LLM report + on-chart insights) | `dashboard.md`, `src/dashboard/analysis_router.py`, `src/dashboard/analysis_prompt.py` |

## Cross-phase

| Task | Files |
|---|---|
| Benchmark models by simulated net-of-fee PnL (app) | `DECISIONS.md` (`benchmark_*`), `src/benchmark/app.py`, `src/benchmark/engine.py` |
| Change the benchmark trading rule or null baselines | `DECISIONS.md` (`benchmark_trading_rule`), `constants.py`, `src/benchmark/trading_sim.py` |
| Promote finished models to the benchmark (`FINISHED_DIR`) | `DECISIONS.md` (`benchmark_finished_models_source`), `constants.py`, `src/predictor/training.py` (`train_all_folds`), `src/training_ui/app.py` |
| Change the leaderboard ranking metric | `DECISIONS.md` (`benchmark_leaderboard_ranking`), `src/benchmark/app.py` (`get_leaderboard`), `static/benchmark/` |
| Join benchmark + training data for a model (analysis) | `DECISIONS.md` (`benchmark_analysis_endpoint`), `src/benchmark/app.py` (`/api/analysis`), `src/training_ui/exporter.py` |
| Run the predictor training bake-off (capped seeds) | `scripts/eval_predictor.py`, `src/benchmark/metrics.py` |
| Run the permutation test (pre-live gate) | `splits-validation.md`, `scripts/permutation_test.py` |
| Run the predictor hyperparameter/architecture search loop | `DECISIONS.md` (`search_dev_slice`), `scripts/search_predictor.py` |
| Run the walk-forward 12×1w gate | `splits-validation.md`, `scripts/holdout_evaluator.py` |
| Update `agent_config.json` schema | `agent-config.md`, `agent_config.json` |
| Amend a locked decision | `DECISIONS.md`, `CHANGELOG.md` (and the matching context card) |
