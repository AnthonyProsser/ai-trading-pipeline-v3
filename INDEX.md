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
| Build Kraken OHLCVT ingest | `execution-engine.md`, `scripts/ingest_kraken_history.py` |
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
| Wire Telegram alerter | `execution-engine.md`, `src/execution/alerter.py` |

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

## Phase 4 / Dashboard

| Task | Files |
|---|---|
| Wire dashboard panels | `dashboard.md`, `src/dashboard/main.py` |
| Add the predictor accuracy panel with q10/q90 overlay | `dashboard.md`, `predictor-contract.md`, `src/dashboard/main.py` |
| Wire the kill-switch button (writes file-flag only) | `dashboard.md`, `execution-engine.md` |
| Wire WebSocket payload schema with `predictor_hash` + `predictor_contract_version` | `dashboard.md`, `agent-config.md` |
| Wire the replay scrubber against in-memory deque | `dashboard.md`, `src/dashboard/replay.py` |

## Cross-phase

| Task | Files |
|---|---|
| Run the permutation test (pre-live gate) | `splits-validation.md`, `scripts/permutation_test.py` |
| Run the walk-forward 12×1w gate | `splits-validation.md`, `scripts/holdout_evaluator.py` |
| Update `agent_config.json` schema | `agent-config.md`, `agent_config.json` |
| Amend a locked decision | `DECISIONS.md`, `CHANGELOG.md` (and the matching context card) |
