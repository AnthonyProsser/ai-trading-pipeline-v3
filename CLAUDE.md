# CLAUDE.md

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project-specific instructions

### Identity

Solo paper-trading BTC bot. Three components: **Predictor** (PatchTST encoder, 1,440-candle lookback, multi-step quantile forecast), **Trader** (rules-based, uncertainty-aware sizing, exchange-native stops), **Dashboard** (FastAPI + vanilla JS, read-only telemetry + kill-switch button). Single user, Windows, RTX 4060 8GB. **No real capital until paper-trading gates pass and statistical edge is proven by permutation test.**

### Repo state

**Build phase (pre-training gate).** Building phase by phase toward a green smoke run.

- **Phase 0 (Data)** — COMPLETE, merged to `main`. `src/data/` (feature_pipeline, scaler, walk_forward, validator, manifest) and matching `tests/data/` are fully green.
- **Phase 1 (Predictor)** — modules COMPLETE, merged to `main` (PR #12 merged 2026-06-29). 85 tests green. `early_stopping.py`, `loss.py`, `rollout.py`, `model.py`, `training.py`, `deploy_gates.py`; plus `scripts/train_predictor.py` (`--smoke`/`--synthetic`/`--no-save`) and `scripts/deploy_predictor.py`. Synthetic GPU smoke passed on the 4060. **Train→deploy pipeline now connected** (2026-06-29) — the three checkpoint decisions are recorded in `DECISIONS.md` (`predictor_checkpoint_dir_and_naming`, `_save_format`, `_save_policy`): `train_predictor.py` saves best-by-val-total weights + scaler to `checkpoints/{run_tag}.pt` / `.scaler.pkl` after a real-data run (synthetic runs skip the save); `save_checkpoint` lives in `src/predictor/training.py`; `deploy_predictor.py` reads them under `weights_only=True` + `pickle.load`. Round-trip verified via unit test and a synthetic smoke. **Real-data smoke PASSED** (2026-06-30) — `data/raw/XBTUSD_1.csv`, 4,642 steps, finite loss, no OOM/NaN; first real checkpoint written to `checkpoints/`. val q90 coverage baseline = 0.9880. `deploy_predictor.py` fully wired: reads `trained_through_ts_utc` + `train_q90_coverage` from the checkpoint; `--trained-through`/`--train-coverage` are optional CLI overrides; `--lookback` must match embedded value (else STOP). W&B not installed (offline-capable flag wired; `uv add wandb` to enable). **Merged to `main`** (confirmed via `origin/main`, 2026-06-30).
- **Phase 1.5 (Training Dashboard)** — modules COMPLETE, **merged to `main`** (verified tracked on `main` at 4d460ed, 2026-07-02; the earlier "uncommitted, no PR" note was stale). `src/training_ui/` (`setup_router.py` data gate, `exporter.py` fold-history JSON export, `app.py` FastAPI app) + `static/training_ui/` (`index.html`, `style.css`, `chart.js` hand-rolled canvas renderer, `app.js`) built end-to-end per `training-dashboard.md`, translating the Claude Design mockup (`Training Dashboard.dc.html`) into vanilla JS/CSS (no React/build step/CDN). Process model: in-process background `threading.Thread` (not a subprocess — Windows has no clean subprocess-signal story), `TrainingRunner` state machine + `queue.Queue`-per-client pub/sub bridged into SSE at `GET /api/events`. `GET /api/config` serves color/behavior thresholds from `constants.py` so the JS client never hardcodes a duplicate. `src/predictor/training.py` gained a new `train_all_folds` walk-forward driver (previously single-fold only via `scripts/train_predictor.py`), `stop_event`/`save_event` hooks in `train_one_fold` checked at each batch boundary, enriched log payloads (`lr`/`grad_norm`/`fold`/`epoch`/`patience`/ETAs via true exponential smoothing). `scripts/deploy_predictor.py` gained a `math.isnan` guard refusing to deploy a manually-saved (non-gate-evaluated) checkpoint. 119 tests green, `mypy --strict` clean, `ruff` clean. Reviewed via the mandatory `decisions-auditor` + `python-reviewer` + `test-enforcer` triggers — both found real issues (doc-drift in `DECISIONS.md`, a dead EMA constant, an SSE-endpoint thread/subscriber leak on client disconnect, a `start`/`stop`/`save` TOCTOU race) — all fixed with regression tests, all re-verified green. **Known gap:** no pixel-level screenshot obtained — the preview tool times out on this page's persistent `EventSource` connection (a tooling limitation); network/console-level correctness was confirmed instead (all assets 200, zero console errors, SSE connects). Not yet committed, no PR opened. The long training run launches at Phase 1.5 exit. **Training wall-clock optimization** (2026-07-01, pure engineering): the real run was fed the smoke batch size (32) and used a single-process `DataLoader` with per-step host→device copies + a CUDA sync + JSON SSE broadcast on every one of ~4,642 steps/epoch. Fixed by (a) new `PredictorConfig.PROD_BATCH_SIZE = 256` / `PROD_BATCH_SIZE_FALLBACK = 128` (user-confirmed 256 on 2026-07-01); (b) a device-resident `_WindowLoader` (vectorised gather over an on-GPU fold tensor, identical windowing) replacing the `DataLoader`, `build_fold_loaders` now takes a `device`; (c) on-GPU loss accumulation + SSE `batch`-log throttling to every `TrainingUIConfig.BATCH_LOG_INTERVAL = 50` steps (schema unchanged); (d) TF32 + `cudnn.benchmark` at import. Benchmarked on the 4060 (real data, fold 0, lookback 1440): 116.6 → 14.4 s/epoch (~8×), 474 MB peak VRAM. 123 tests green, `mypy --strict` + `ruff` clean. **Fold-count bug fixed** (2026-07-01): the dashboard's `_default_run_training` built folds over the full CSV (2013+) — it was missing the `filter_by_historical_start` call that `scripts/train_predictor.py` applies, so pre-`HISTORICAL_START` candles inflated the fold count. One-line fix mirroring the script path; regression test in `tests/training_ui/test_app.py`; audited by `decisions-auditor` (all checks PASS) + `python-reviewer` (approve). 131 tests green. **Controls UX fix** (2026-07-02, branch `phase-15-controls-fix` off `main`): (1) natural completion of all folds now broadcasts a distinct SSE status `done` (was `stopped`); `train_all_folds`'s final emit is `done`, `stopped` = user stop only. `TrainingRunner._TERMINAL_STATES` gains `done` (start/stop/save refuse → 409); header reads `Training complete — you may now close this tab.` (2) The three header buttons collapse to ONE stateful control button (`Start`/`Stop`/`Saved`-flash/`Done`) in `static/training_ui/`; the manual "Save checkpoint" UI button is removed (saving is automatic on stop + every fold completion; `POST /api/training/save` + `save_event` kept for scripted use, still NaN-coverage-refused by `deploy_predictor.py`). New `TrainingUIConfig.BUTTON_SAVED_FLASH_SECONDS = 2.0` served via `/api/config`. (3) False "Raw data file not found" banner fixed: author CSS `display` rules overrode the UA `[hidden]` rule, so the banner + buttons rendered unconditionally — added `[hidden]{display:none!important}`; the backend gate was already correct. Tests-first (commit c2068e2 before impl 07b6f8a); 158 tests green, `mypy --strict` + `ruff` clean; browser-verified (banner `display:none` + single enabled `Start` with data present). DECISIONS.md `training_ui_controls` amended + CHANGELOG same-commit. **The mandatory `decisions-auditor` + `python-reviewer` + `test-enforcer` subagents could not complete (hit the account session limit mid-run) — re-run them before merging.** Not yet merged, no PR.
- **Phases 2–4** — not started. `src/execution/`, `src/trader/`, `src/dashboard/` do not exist. (The Training Dashboard, formerly bundled into Phase 4, is now Phase 1.5; Phase 4 builds the Trading Dashboard only.)

`scripts/train_predictor.py` and `scripts/deploy_predictor.py` exist. Other scripts (e.g. `backtest.py`, `holdout_evaluator.py`, `permutation_test.py`) are planned deliverables — they do not exist yet.

**Experimental: predictor search loop** (2026-06-30, cross-phase tooling, not a phase deliverable). `scripts/search_predictor.py` — an autoresearch-style unattended hyperparameter/architecture search loop for the predictor. Propose ONE bold, coherent structural move from an explicit menu (proposer pinned to Opus — `PROPOSER_MODEL = "claude-opus-4-8"`; bolder moves are higher-blast-radius so it uses the strongest model) → train on the new search dev-slice (`--search-slice`, `src/data/walk_forward.py::carve_search_slice`, DECISIONS.md `search_dev_slice`) → repeat-seed noise confirmation (DECISIONS.md `search_confirm_seeds`) → judge via `decisions-auditor` + `leakage-checker` (headless, deliberately kept on Sonnet 5 — cheap independent read-only gate, not co-varied with the proposer) → keep (leaves `constants.py` edited) or revert. Safety net around the unattended `constants.py` edit: pre-iteration on-disk snapshot (`constants.py.searchbak`) + `try/finally` restore (regression guard from commit c2071f7) + `ast.parse` check that reverts a non-parseable patch before any training. Regression tests in `tests/scripts/test_search_predictor.py`. No git auto-commit — review `constants.py`'s diff and `search_log.jsonl`, commit manually. Requires the `claude` CLI on PATH; no `ANTHROPIC_API_KEY` needed (uses the existing Claude Code login). **Verify the CLI accepts `claude-opus-4-8` before a real run** (`claude -p --model claude-opus-4-8 "ping"`). Not yet run end-to-end on GPU.

**Experimental: predictor bake-off harness** (2026-07-01, cross-phase tooling, not a phase deliverable, **MERGED to `main`** via PR #18). `scripts/eval_predictor.py` — a controlled predictor comparison tool to decide `main` (per-step log-return quantiles) vs `fable-5-restructuring` (cumulative-path quantiles) empirically, never on self-report. Branch-agnostic: reads `getattr(PREDICTOR, "TARGET_SEMANTICS", "per_step_logret")` so one file runs unchanged on either branch (merge the commit onto fable-5 to run it there). Subcommands: `train-eval` (trains N capped seeds on one fold in-process — default `--seeds`=`DATA.SEARCH_CONFIRM_SEEDS`, `--max-epochs`=1 or `--max-steps` cap — evaluates each, reports seed-aggregated mean+std; NaN-tolerant so one seed with no tradeable windows can't poison the aggregate), `eval` (evaluate a pre-trained `--checkpoint`), `compare` (JSONs → side-by-side table). Metrics: statistical (coverage/calibration/DA cross-branch-comparable; sharpness/pinball per-branch only) + economic (Krafer-style "fraction of predicted move captured" PAIRED with adverse excursion, sub-fee moves excluded via `EXECUTION.FEE_THRESHOLD` to block shrink-to-win). Decider = calibration→sharpness→pinball now; economic metric promoted to primary once `backtest.py` exists. Reviewed by `decisions-auditor` + `python-reviewer` (a HIGH NaN-poisoning bug in seed aggregation was caught and fixed). Regression tests in `tests/scripts/test_eval_predictor.py`. **First capped bake-off RUN (2026-07-01, fold 0, 3 seeds, 20 epochs, RTX 4060): cumulative model beats old per-step on DA (0.418→0.537) and captured fraction (0.172→0.254) but under-covered (q90 0.866 vs 0.90, calibration 0.747 vs 0.80).** DECISIONS.md entry for the harness deferred until first full run (mirrors how `search_predictor.py` got its entries post-hoc). **Scope reduced (2026-07-03, branch `benchmark-app`):** the `eval` and `compare` subcommands were retired — superseded by the benchmark app below — so `eval_predictor.py` now keeps ONLY `train-eval` (the app never trains). Its reviewed metric layer (`target_to_model_space`/`statistical_metrics`/`excursion_metrics`) moved to `src/benchmark/metrics.py`; the script imports it back (scripts → src). Metric tests moved to `tests/benchmark/test_metrics.py`; `tests/scripts/test_eval_predictor.py` keeps only the `aggregate_seeds` tests.

**Model benchmark app** (2026-07-03, cross-phase tooling, not a phase deliverable, branch `benchmark-app` — **the original app is committed and MERGED via PR #21 (`515d64c`)**; later same-branch amendments below (grade, run-grouping) sit on `benchmark-app` post-merge). `src/benchmark/` + `static/benchmark/` — a FastAPI + vanilla-JS app (same stack/process model as the training dashboard: in-process background thread, one job at a time, queue-per-client SSE, 127.0.0.1-only, own port `ExecutionConfig.BENCHMARK_UI_BIND_PORT = 8002`) that surfaces every checkpoint in `checkpoints/` with its embedded provenance and scores them by **simulated net-of-fee trading PnL**, ranked on a leaderboard. Modules: `metrics.py` (salvaged eval_predictor economic layer), `trading_sim.py` (the fixed measuring instrument + null baselines), `registry.py` (checkpoint discovery, immutable-filename display-name rename via `checkpoints/benchmark/registry.json`, mtime-cached metadata, atomic strict-JSON result cache `{stem}.benchmark.json`), `engine.py` (evaluate one checkpoint on its OWN fold TEST slice — never the locked test set), `app.py` (FastAPI + `BenchmarkRunner`). The trading rule is ONE fixed instrument applied identically to every model (never per-model search — garden-of-forking-paths guard): enter only when the predicted final-horizon cumulative close move clears `FEE_THRESHOLD` AND the [q10,q90] interval doesn't straddle zero (confidence gate), hold to horizon, pay one round trip, non-overlapping. Null baselines every model is read against: buy-and-hold + random-entry at matched trade frequency (`BenchmarkConfig.NULL_DRAWS = 200` permutation-style draws → p-value). Per-fold metrics: net return, per-trade Sharpe, max drawdown, trade count, hit rate, + captured-fraction/adverse-excursion. Constants in new `BenchmarkConfig` group (`BENCHMARK` singleton). DECISIONS.md `benchmark_*` keys + CHANGELOG same-commit; INDEX.md cross-phase rows added. Real-data smoke: 80 checkpoints discovered, 78 compatible, 2 old per-step checkpoints correctly flagged incompatible (semantics mismatch, refused like `deploy_predictor.py`). Tests-first (`tests/benchmark/`); `mypy --strict` + `ruff` clean; audited by `decisions-auditor` (PASS) + `python-reviewer` + `test-enforcer`. Context: last full run's mean DA ≈ 0.506 (coin-flip), so the benchmark is expected to (correctly) show little tradeable edge — it exists to expose that honestly, not flatter it. **Amended 2026-07-03 (same branch, uncommitted):** (1) the benchmark now reads only **finished** models from `PredictorConfig.FINISHED_DIR = "checkpoints/finished"` — `train_all_folds` copies each fold's gate-evaluated checkpoint there on natural full-run completion (new terminal `state:"completed"` + `promoted` count; user-stopped runs promote nothing), the training UI passes the dir; **migration:** the dir starts empty, so the board is blank until a fresh run completes or a completed run's files are copied in by hand (no backfill UI). (2) The leaderboard now ranks by **forecast accuracy** (`directional_accuracy` desc, `pinball` asc tie-break) not total net PnL — PnL ranking mechanically rewarded trading least; DECISIONS.md `benchmark_leaderboard_ranking`. (3) The "<5% hit rate" was the net-of-fee win rate mislabeled: `ledger_stats` now reports gross `directional_hit_rate` (~0.5) alongside net `hit_rate`; both are columns. (4) New `GET /api/analysis/{stem}` joins a model's benchmark result with its training fold record (`FoldRecord` gained `stem`) for AI hypothesis generation. 218 tests green, `mypy --strict` + `ruff` clean; reviewers pending at time of writing. **Amended 2026-07-05 (committed, branch `benchmark-app`, commits fd618c1 tests + 0233e1f impl):** the leaderboard now carries an expectancy-based green **"profitable" grade** (three-state `profitability` per row: `profitable` / `not_profitable` / `insufficient`) — green iff net-of-fee expectancy per trade > 0 AND net PnL beats the random-entry null at `p < BenchmarkConfig.PROFITABLE_P_VALUE_MAX` (= 0.10) AND `trade_count >= PROFITABLE_MIN_TRADES` (= 30); below the floor / non-finite → `insufficient` (neither green nor red). New pure `trading_sim.py::profitability_grade`, computed at leaderboard read time (NOT persisted, so threshold changes don't invalidate cached results); run groups gain `n_profitable`; `/api/config` serves both thresholds; frontend tints rows. Rank basis UNCHANGED (DA desc / pinball asc — the grade is orthogonal). `NULL_SIGNIFICANCE_LEVEL = 0.05` kept as the separate p-cell display-emphasis threshold. Both thresholds user-chosen. DECISIONS.md `benchmark_leaderboard_ranking` revised + CHANGELOG same-commit. Verified against the 12 real cached results (all `not_profitable`/`insufficient` — coin-flip DA, so the board honestly shows zero profitable). 231 tests green; `decisions-auditor` (PASS all checks) + `python-reviewer` (approve) + `test-enforcer` (PASS) all clean. **Amended 2026-07-05 (committed, branch `benchmark-app`, commits 5de33b7 tests + this impl):** the leaderboard is now **runs-primary** — one row per training run instead of a flat per-checkpoint ranking, so recipe versions compare at a glance ("70/78 folds profitable"). Grouping key = `(git_sha, constants_sha8)` from the run tag (fold-siblings share both; scaler segment varies per fold, excluded — DECISIONS.md `benchmark_run_grouping`); no new run-id. Run rows rank by profitable-fold fraction desc (tie: mean expectancy), and expand into a drill-down that ranks the run's fold-checkpoints by **per-trade expectancy desc / DA desc** (supersedes the old DA/pinball member sort — `benchmark_leaderboard_ranking` revised in place). `profitable_fraction = n_profitable / n_benchmarked`; `insufficient` counts in the denominator only. `GET /api/leaderboard` reshaped to `{"runs": [...]}` (flat `"models"` + run-card summary removed); all grouping in the new pure `app.py::build_leaderboard` — `registry.py`/`engine.py` untouched (join fields already existed). `static/benchmark/` gains run rows + expandable member sub-table (expansion state preserved across SSE refreshes). 236 tests green, `mypy --strict` + `ruff` clean. CHANGELOG same-commit; `decisions-auditor` after. **Amended 2026-07-05 (branch `benchmark-all-button` off `benchmark-app`, uncommitted): completeness-gated scoring + per-run "Benchmark all".** (1) A run is now SCORED/ranked only when **complete** — every *compatible* fold-checkpoint benchmarked (`complete = n_pending == 0`; incompatible folds don't block it). Complete runs rank at top by `profitable_fraction`; **incomplete** runs (partial OR zero benchmarked — including a run fresh out of training) now appear at the **bottom, unscored** (score fields null), fewest-`n_pending` first — this SUPERSEDES the old "zero-benchmarked run is dropped" rule (only a run with no compatible fold at all is still dropped). New `/api/leaderboard` per-run fields `complete`/`n_compatible`/`n_pending`. (2) A **"Benchmark all"** button on each unscored run enqueues its remaining compatible-unbenchmarked folds back-to-back via new `BenchmarkRunner.start_batch` (single `start` delegates to it; best-effort per fold — one failure alerts but the batch continues) and endpoint `POST /api/benchmark/run/start` (`{git_sha, constants_sha8}` → server-derived pending stems; declared before `/{stem}/start` to avoid capturing `run` as a stem). Tests-first; 250 tests green, `mypy --strict` + `ruff` clean; browser-verified via DOM inspection (complete run scored w/ drill-down; two incomplete runs at bottom with enabled `Benchmark all (N)`; button POSTs the run key, does not toggle the drill-down, disables while a job runs — screenshot times out on the SSE page as always, a known tooling limit). DECISIONS.md `benchmark_run_grouping` revised in place + new `benchmark_batch_benchmark` key; CHANGELOG same-commit. Mandatory `decisions-auditor` + `python-reviewer` + `test-enforcer` pending. Not yet committed, no PR.

**Coverage-penalty loss amendment** (2026-07-01, **MERGED to `main`** via PR #19, after PR #18 landed the bake-off harness; human signed off, waiving the full-length re-verify — gaps to be addressed post-hoc during the full run. `training.py` SSE payloads now also log the `coverage`/`train_coverage`/`val_coverage` component so its trajectory is visible during the run). Fixes the under-coverage above: third loss term in `src/predictor/loss.py::coverage_penalty` — squared gap between smooth (sigmoid) batch coverage and the nominal 0.10/0.90 tail levels, close dim only, per-step-std-relative temperature, **width-only** (tails enter as `q_τ + (q50.detach() − q50)`, cancelling the median-anchor gradient exactly), fp32 under bf16 autocast. Constants: `COVERAGE_PENALTY_WEIGHT = 1.0`, `COVERAGE_PENALTY_TEMPERATURE_FRAC = 0.02`, `COVERAGE_PENALTY_STD_FLOOR = 1e-8`. DECISIONS.md `loss_coverage_penalty` + CHANGELOG same-commit. Three disclosed bake-off iterations on identical fold/seeds: anchor-coupled w=1.0 and w=0.1 both collapsed DA to ~0.49 (median drag — rejected); width-only at temp 0.1 overshot coverage to 0.964 (sigmoid kernel bias ∝ temp² — rejected); final width-only temp 0.02 landed q90_coverage 0.902±0.023, calibration 0.826±0.031 (inside deploy-gate band), DA 0.529±0.010 / captured 0.256 / adverse 0.229 all within seed noise of baseline, n_used 10,377→14,836. Cost: honestly wider intervals (sharpness 0.0065→0.084), ~55% slower epochs (14.5→22.5 s). Audited: `test-enforcer` PASS, `decisions-auditor` PASS (after fixing a quantile-index-pattern violation it caught), `python-reviewer` (fp32-AMP finding fixed). Values validated at the capped budget only — re-verify calibration on the first full run.

**Horizon sweep** (2026-07-06, branch `horizon-sweep` off `main`, cross-phase experiment — committed, not merged, no PR). Motivated by the benchmark verdict on the 78-fold run: **0/78 profitable**, mean loss = exactly the 0.62% round-trip fee per trade, because at `HORIZON=15` typical BTC moves can't clear `FEE_THRESHOLD`. Landed (tests-first, auditor trio green, 228 tests / `mypy --strict` / `ruff` clean): (1) `tests/predictor/test_training.py` fold literals derived from `_LOOKBACK + _H` — green at any horizon; (2) `scripts/eval_predictor.py::fixed_instrument_summary` — `train-eval` now scores each seed with the benchmark app's exact fixed trading instrument (net PnL / trades / p-value vs random-entry null) on the fold VAL gather, emitted as a `"trading"` result group; hold length from `pred.shape[1]`, a codified exception in DECISIONS `benchmark_trading_rule` (sweep candidates train under edited horizon constants); (3) **deploy gate (b) DA is now final-horizon-step only** (the traded quantity) — final-step slices in `statistical_metrics` / `evaluate_directional_accuracy`, `da_pred`/`da_target` kwargs on `evaluate_deploy_gates`, used by `deploy_predictor.py`; DECISIONS `deploy_gates` amended + CHANGELOG same-commit. **Stage A bake-off (fold 77, 3 seeds, 20-epoch cap, 4060, ~3.7h): ALL FOUR candidates net-negative** — {H60,H240}×{λ1.5,1.75}: H60 ≈ −3.7 over ~610 trades, H240 ≈ −1.0 over ~165 trades, per-trade loss ≈ the fee at every horizon; net hit rate improved 3.5%→14% at H240 but DA stayed 0.48–0.50 (coin flip), p vs null 0.38–0.53. **Per the pre-registered decision rule the full 78-fold run was NOT launched**; `constants.py` untouched (HORIZON stays 15 pending a signal-side fix). Conclusion: fixing fee economics was necessary but not sufficient — the binding constraint is directional signal, not horizon. Unexplored levers: feature engineering (only 5 raw OHLCV log-return features today — no time-of-day / multi-scale / higher-timeframe features), the specced-but-never-run lookback sweep [240/720/1440], capacity/factorized head. Bake-off JSONs in the 2026-07-06 session scratchpad.

**horizon-sweep merged to `main`** (2026-07-07) — the infra (fixed-instrument PnL hook, DA final-step gate, horizon-independent fold literals) was solid even though the horizon experiment itself was net-negative; branch deleted post-merge. Stale merged branches (`phase-15-controls-fix`, `benchmark-app`, `benchmark-all-button`) also deleted — `main` is now the sole branch.

**Idea-search loop** (2026-07-07, cross-phase, unattended multi-day GPU run). Started to find a directionally-profitable recipe — every prior benchmark run showed DA ≈ coin-flip. Each idea gets its own `idea-*` branch off `main`; a fast capped bake-off (1 fold, few seeds, epoch-capped) screens each before any full run. The mandatory per-change subagent review triggers are suspended for this loop (see ECC integrations note below) to prioritize trying many ideas over polishing each one; re-invoke the auditor trio by hand once an idea is chosen for refinement.

Read first, every coding session, in this order: `DECISIONS.md` → `INDEX.md` → `constants.py` → the relevant context card named by the matching INDEX row. They are the source of truth for architectural decisions, task → file mapping, and frozen magic numbers respectively.

### Tech stack

- **UV** (package manager — `uv sync` / `uv add` / `uv run`; never bare `pip install`)
- Python 3.x with full type hints; `mypy --strict` on PR
- PyTorch (PatchTST encoder, patch_size=16)
- pandas, numpy (data pipeline)
- FastAPI + vanilla JS + Lightweight Charts (dashboard, bound to `127.0.0.1` only)
- pytest (`tests/` mirrors `src/` exactly)
- structlog (structured JSON logs)
- Weights & Biases (training tracking)
- SQLite (paper) / PostgreSQL+WAL (live)
- Kraken WebSocket v2 + REST (data + trading)
- Windows Credential Manager via `keyring` (secrets — never `.env`)

### Project-specific rules

- **Tests first** for any code under `src/`. Failing tests encoding the exit criteria are committed before implementation. 
- **Magic numbers live in `constants.py` only**, inside frozen dataclasses. No bare module-level constants. No hardcoded thresholds like `if x > 0.62:`.
- **`src/` never imports from `scripts/`.** Scripts call into `src/`, never the inverse.
- **`data/test_locked/` never referenced from `src/`.** Enforced by `grep -r 'test_locked' src/` plus `sys.settrace` runtime guard during training.
- **All timestamps UTC.** `datetime.now()` without `tz=timezone.utc` is a CI failure in `src/`.
- **Exchange-native stop-loss is mandatory.** Execution refuses any order whose stop cannot be placed at Kraken. No naked positions under any code path.
- **SHA256 manifest verified on every weight read.** Covers weights + scaler + `constants.py`.
- **`asyncio` only in `src/execution/`.** Predictor and trader code stays synchronous.
- **One branch per phase:** `phase-XY` off `main`; merged to `main` at phase exit. Never commit to `main` directly.
- **Any `DECISIONS.md` change requires a `CHANGELOG.md` entry in the same commit.**
- **Flag every unspecced decision.** If a task requires a value not in `DECISIONS.md` or `constants.py`, stop and ask. Do not "use a reasonable default" — this aligns with Karpathy §1.
- **Update `CLAUDE.md` (Repo state section) after every completed feature or phase.** It is the first file read every session — stale repo-state causes wasted work at session start.

Three project-specific subagents live in `.claude/agents/` (`decisions-auditor`, `leakage-checker`, `test-enforcer`). They are read-only auditors, still available via the Agent tool, but there is currently no mandatory trigger requiring them on every `src/`/`scripts/` change — see the idea-search loop note in Repo state for why, and invoke them by hand once an idea is picked for refinement.

### ECC integrations

The `everything-claude-code` plugin is installed. Use these capabilities where they apply:

**Agents (invoke via Agent tool):**
| Agent | When to use |
|---|---|
| `everything-claude-code:python-reviewer` | After writing or editing any Python source under `src/` — catches type-hint gaps, Pythonic issues, security, and PEP 8 drift. Run after `decisions-auditor`. |
| `everything-claude-code:pytorch-build-resolver` | When a PyTorch training run or inference crashes — shape mismatches, device errors, gradient issues, DataLoader failures. |
| `everything-claude-code:security-reviewer` | When touching `src/execution/` (API keys, WebSocket, order submission) or any code that reads external input. |
| `everything-claude-code:performance-optimizer` | When training is slow or a DataLoader is bottlenecked — before concluding hardware is the constraint. |
| `everything-claude-code:tdd-guide` | When writing new `src/` modules — enforces the tests-first mandate and 80%+ coverage. |
| `everything-claude-code:docs-lookup` | When needing current API docs for PyTorch, pandas, FastAPI, or Kraken SDK — fetches live docs rather than relying on stale training knowledge. |

**Skills (invoke via Skill tool):**
| Skill | When to use |
|---|---|
| `everything-claude-code:pytorch-patterns` | When designing or debugging the PatchTST encoder, quantile loss, training loop, or rollout. |
| `everything-claude-code:python-review` | Quick per-file Python review (lighter-weight alternative to the full agent). |
| `everything-claude-code:tdd` | TDD workflow guidance when implementing a new phase module. |
| `everything-claude-code:security-review` | Security scan on `src/execution/` changes. |
| `code-review` (`/code-review`) | After any implementation diff — reviews the current branch changes for correctness bugs and simplification opportunities. |

### Commands (when toolchain lands)

`scripts/train_predictor.py` and `scripts/deploy_predictor.py` are currently present; other script/module commands below are planned phase deliverables tracked in `INDEX.md` and become runnable as those files land.

| Task | Command |
|---|---|
| Install / sync deps | `uv sync` |
| Add a dependency | `uv add <package>` |
| Test suite | `uv run pytest` |
| Single test | `uv run pytest tests/path/test_file.py::test_name` |
| Type check | `uv run mypy` |
| Leakage audit | `grep -r 'test_locked' src/` (must return nothing) |
| Predictor smoke (1 epoch, 1 fold) | `uv run python scripts/train_predictor.py --smoke` |
| Backtest | `uv run python scripts/backtest.py` |
| Walk-forward holdout (12×1w gate) | `uv run python scripts/holdout_evaluator.py` |
| Permutation test (pre-live gate) | `uv run python scripts/permutation_test.py` |
| Dashboard | `uv run python -m src.dashboard.main` |
| Execution loop | `uv run python -m src.execution.loop` |