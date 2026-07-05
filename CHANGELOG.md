# CHANGELOG

Append-only history of amendments to `DECISIONS.md`. Every change to a decision value must add an entry here in the same commit (pre-merge check enforces).

Format:

```
## YYYY-MM-DD — short title
- decision_key: old_value → new_value
- Reason: …
- Source: audit / conversation / phase exit
```

---

## 2026-07-05 — Benchmark: expectancy-based green "profitable" grade vs the random-entry null
- Amended `benchmark_leaderboard_ranking`: added a **green-grade** definition. Rank basis is UNCHANGED (`directional_accuracy` desc / `pinball` asc). New orthogonal three-state per-row classification `profitability` ∈ {`profitable`, `not_profitable`, `insufficient`}: green iff net-of-fee expectancy per trade > 0 (`trading.net_return / trading.trade_count > 0`) AND `baselines.p_value < PROFITABLE_P_VALUE_MAX` (beats the random-entry null) AND `trading.trade_count >= PROFITABLE_MIN_TRADES`; `insufficient` below the trade floor or on non-finite inputs; else `not_profitable`. Run-group summary gains `n_profitable`.
- New `constants.py` values: `BenchmarkConfig.PROFITABLE_P_VALUE_MAX = 0.10`, `BenchmarkConfig.PROFITABLE_MIN_TRADES = 30` (served via `/api/config` so `static/benchmark/app.js` never hardcodes them). `NULL_SIGNIFICANCE_LEVEL = 0.05` unchanged (stays the p-cell display-emphasis level / pre-live-gate mirror — deliberately two thresholds).
- New pure function `src/benchmark/trading_sim.py::profitability_grade(net_return, trade_count, p_value)`; grade computed at leaderboard read time (not persisted into `{stem}.benchmark.json`, so threshold changes don't invalidate cached results). Frontend tints profitable rows green, dims insufficient rows.
- Reason: user wants the board to headline how many models clear a real profitability bar, not just the single best forecaster. Expectancy > $0/trade is the money question; the p-value gate blocks luck; the trade floor blocks a 2-trade fluke grading the same as a 50-trade edge. p < 0.10 and ≥ 30 trades chosen by the user (looser than the 0.05 capital gate — this is a dashboard, not an allocation decision).
- Source: conversation (user request + delegated threshold choices).

## 2026-07-03 — Benchmark: finished-models source, accuracy ranking, hit-rate fix, analysis endpoint
- New decision keys: `benchmark_finished_models_source`, `benchmark_leaderboard_ranking`, `benchmark_analysis_endpoint`. Amended `benchmark_app` (ranking key: `trading.net_return` desc → `statistical.directional_accuracy` desc, `pinball` asc tie-break) and `benchmark_trading_rule` (hit rate now split: `directional_hit_rate` gross + `hit_rate` net-of-fee).
- New `constants.py` values: `PredictorConfig.FINISHED_DIR = "checkpoints/finished"`; `BenchmarkConfig.NULL_SIGNIFICANCE_LEVEL = 0.05` (leaderboard p-value emphasis, served via `/api/config` alongside `alert_auto_dismiss_seconds` so `static/benchmark/app.js` stops hardcoding `0.05`/`8000` — decisions-auditor finding).
- `train_all_folds` gained `finished_dir` param; on natural full-run completion it copies each fold's gate-evaluated checkpoint there and emits terminal `state:"completed"` (+`promoted` count) instead of `"stopped"`; `fold_complete` payload gained `stem`. `FoldRecord` gained `stem: NotRequired[str]`. Benchmark `create_runner` reads `FINISHED_DIR`; added `GET /api/analysis/{stem}` (benchmark⋈training join) and `BenchmarkRunner.training_metrics_dir`.
- Reason: (a) show only finished models; (b) leaderboard was ranking by total net PnL, which rewarded trading least (negative-EV trades) — the model forecasts price, so rank by accuracy; (c) "<5% hit rate" was the net-of-fee win rate mislabeled — added gross directional hit rate; (d) expose combined data for AI hypothesis generation.
- Source: conversation (user report + delegated ranking choice).

## 2026-07-03 — Model benchmark app (`src/benchmark/`) + eval_predictor scope reduction
- New decision keys: `benchmark_app`, `benchmark_trading_rule`, `benchmark_eval_slice`, `benchmark_null_baselines`, `benchmark_registry`, `benchmark_metric_salvage` (new "Model benchmark app" section in DECISIONS.md Cross-cutting).
- New `constants.py` group `BenchmarkConfig` (`BENCHMARK` singleton): `BENCHMARK_DIR = "checkpoints/benchmark"`, `REGISTRY_FILENAME = "registry.json"`, `RESULT_SUFFIX = ".benchmark.json"`, `NULL_DRAWS = 200`, `NULL_SEED = 0`. Added `ExecutionConfig.BENCHMARK_UI_BIND_PORT = 8002`.
- eval_predictor.py: `eval` and `compare` subcommands **removed** (superseded by the app's per-model evaluation + leaderboard); only `train-eval` remains. Its reviewed metric layer (`target_to_model_space`/`statistical_metrics`/`excursion_metrics`) moved to `src/benchmark/metrics.py`; eval_predictor imports it back (scripts → src). Metric tests moved to `tests/benchmark/test_metrics.py`; `aggregate_seeds` tests stay in `tests/scripts/test_eval_predictor.py`; the `format_comparison` test retired with the `compare` subcommand.
- Flagged unspecced (creative-latitude task, no prior spec): the app itself, port 8002, leaderboard ranking key (`trading.net_return` desc), the fixed trading rule's shape (fee gate + no-straddle confidence gate, hold-to-horizon, non-overlapping), evaluation on each model's own fold TEST slice (locked test set untouched), and `NULL_DRAWS`/`NULL_SEED`. Trading rule and null baselines were described by the user; the numeric/structural choices above were made this session.
- Reason: user asked for a model benchmarking app that scores checkpoints by honest simulated net-of-fee PnL on out-of-sample walk-forward folds (mean DA ≈ 0.506 last run — the benchmark is meant to expose weak edge, not flatter it), reusing eval_predictor's reviewed economic-metric code, and to retire eval_predictor once covered.
- Source: conversation 2026-07-03 (benchmark-app branch)

## 2026-07-01 — Loss amendment: width-only coverage penalty (calibration-to-nominal)
- loss: `pinball + λ×direction` → `pinball + λ×direction + COVERAGE_PENALTY_WEIGHT×coverage_penalty`; new decision key `loss_coverage_penalty`. New `PredictorConfig` constants: `COVERAGE_PENALTY_WEIGHT = 1.0`, `COVERAGE_PENALTY_TEMPERATURE_FRAC = 0.02`, `COVERAGE_PENALTY_STD_FLOOR = 1e-8`. `LossComponents` gains a `coverage` field; all existing consumers read fields by name.
- Reason: the capped bake-off (`scripts/eval_predictor.py`, fold 0, 3 seeds, 20 epochs) measured under-coverage on the cumulative model — q90_coverage 0.866 vs 0.90 nominal, calibration_rate 0.747 vs 0.80, both tails too narrow. Pinball's marginal calibration pressure is ∝ the coverage gap spread over all 45 (dim × quantile) coordinates — too weak at capped budgets. The new term optimizes the two calibration deploy-gate metrics directly on the close dim and is self-limiting (gradient vanishes at nominal coverage).
- Design guards forced by measured failures during the same bake-off (three disclosed iterations on identical fold/seeds): (1) width-only anchor-gradient cancellation (`q_τ + (q50.detach() − q50)`) — without it, tail-coverage force drags the shared median anchor and collapses DA 0.537→~0.49 at weights 1.0 and 0.1 alike; (2) `temperature_frac` 0.1→0.02 — the smooth indicator's kernel bias (∝ temperature²) mislocates the coverage fixed point, overshooting hard coverage to 0.964/0.930; (3) fp32 sigmoid path under bf16 autocast (python-reviewer finding).
- Final measured result (same fold/seeds, vs pre-amendment baseline): q90_coverage 0.866→0.902±0.023, calibration_rate 0.747→0.826±0.031 (inside deploy-gate band), DA 0.537→0.529±0.010 (within seed noise), captured 0.254→0.256, adverse 0.230→0.229, n_used 10,377→14,836. Cost: sharpness 0.0065→0.084 (honestly wider intervals), ~55% slower epochs.
- Source: conversation 2026-07-01 (fable5-calibration-improve branch; empirical bake-off, three iterations reported including the two negative results)

## 2026-07-01 — Predictor rework (fable-5-restructuring): cumulative targets, monotone quantiles, RevIN, horizon-level direction penalty, plateau LR
- target: per-step log-return per OHLCV dim → quantiles of the **cumulative** log-return path from the forecast origin (`PredictorConfig.TARGET_SEMANTICS = "cumulative_logret"`). Quantiles are not additive; per-step quantiles cannot give a calibrated interval for the 15-minute move the trader trades. Loaders/WindowDataset unchanged (raw per-step targets); `predictor_loss` and the gate collectors cumsum.
- output_head: independent per-quantile outputs → monotone head (median anchor ± cumulative softplus offsets); q10 ≤ q50 ≤ q90 by construction.
- architecture: adds model-internal RevIN-style per-window instance normalization (learnable affine; `REVIN_EPS`) + volatility-conditioned output scaling. Data pipeline (`src/data/`) untouched — instance norm is invariant to the fold MinMax scaler's fixed affine, and the fold-constant scale factor folds into learned head weights.
- loss: direction penalty moved from every (step) to the **final-horizon cumulative close** only. Old form compared the per-trade round-trip fee (0.62%) against single 1-minute moves — ~12× the true per-minute median scale, with a per-element gradient ~2–3 orders above pinball's — turning every q50 into a fee-scaled sign flag and destroying median calibration. λ unchanged (1.75 ∈ [1.5, 2.0]); flat-market floor at FEE_THRESHOLD preserved (trend-loss regression test unchanged).
- predictor_lr_schedule: new decision. Warmup-cosine → linear warmup (`WARMUP_FRAC` 0.05 → 0.01) then constant LR with plateau decay (`LR_PLATEAU_FACTOR = 0.5`, `LR_PLATEAU_PATIENCE = 3`). The cosine annealed over the MAX_EPOCHS=100 horizon that early stopping (patience 10) never let it reach, so real runs trained at near-peak LR throughout.
- predictor_checkpoint_save_format: adds `"target_semantics"` provenance key; `deploy_predictor.py` refuses absent/mismatched tags (retires the bare-state_dict acceptance path — a pre-rework per-step model gate-evaluated against cumulative targets would yield silently meaningless metrics).
- deploy_gates: gate functions unchanged (pure same-space comparisons) but evaluated in cumulative space; gate (b)'s `|q50| > FEE_THRESHOLD` filter now gates on the horizon move a trade actually spans.
- Also fixes a latent device bug in `_collect_predictions`: predictions were moved to CPU while `_WindowLoader` targets stayed device-resident — a CPU-vs-CUDA tensor comparison on any CUDA gate-metric pass.
- Reason: DECISIONS.md's predictor entries were authored by an earlier model (ChatGPT 5.5) and re-examined decision-by-decision on this branch. The PatchTST encoder core, patch geometry, capacity constants, walk-forward/leakage machinery, early stopping, AMP, grad clipping, and checkpoint/run-tag plumbing were judged sound and kept. The changes above target the empirical comparison gates: quantile calibration (monotone head, honest medians, vol-scaled intervals), DA/sharpness at the trader's horizons (cumulative targets, horizon-level fee logic), and training stability/convergence (plateau LR). To be settled by training both models on identical folds — this rework must win on held-out metrics, not by construction.
- Source: conversation 2026-07-01 (fable-5-restructuring branch; user-directed re-examination of ChatGPT 5.5-era decisions)

## 2026-07-01 — Search loop: Opus proposer + structural bold-move policy + constants.py safety net
- search_model_routing: new decision. `scripts/search_predictor.py` proposer pinned to `PROPOSER_MODEL = "claude-opus-4-8"` (was `claude-sonnet-5`); judges (`decisions-auditor` + `leakage-checker`) deliberately kept on `JUDGE_MODEL = "claude-sonnet-5"` so the compliance gate stays an independent, cheap read-only check rather than co-varying with the proposer.
- search_proposal_policy: new decision. Each iteration proposes exactly ONE bold, coherent structural move from a fixed `MOVE_MENU` (CAPACITY / TOKENIZATION / RECEPTIVE_FIELD / LOSS_SHAPE / SCHEDULE / REGULARIZATION) instead of small multi-field nudges; search space expanded to include `WARMUP_FRAC`, `PATCH_SIZE`, `LOOKBACK` with a `PATCH_SIZE | LOOKBACK` divisibility guard in `validate_candidate`.
- Safety net (code-only, no new constant): per-iteration on-disk snapshot (`constants.py.searchbak`) + retained `try/finally` restore (regression guard from c2071f7) + `ast.parse` check that reverts a non-parseable patch before any training runs. Regression tests in `tests/scripts/test_search_predictor.py`.
- Reason: the prior loop was too conservative (small per-iteration changes) and used Sonnet for proposals; the user asked for Opus and bolder, structurally-attributable moves. Bolder proposals raise `constants.py` corruption risk, so the safety net was hardened before loosening the proposer.
- Source: conversation 2026-07-01 (search_predictor.py rework task)

## 2026-07-01 — Training wall-clock speedup (~8×): prod batch size + GPU-resident loader + log throttle
- predictor_prod_batch_size: absent (real run used `SMOKE_BATCH_SIZE = 32`, leaked in via `src/training_ui/app.py`) → new `PredictorConfig.PROD_BATCH_SIZE = 256`; `app.py` and `scripts/train_predictor.py` (real path) now feed the production run at 256. `PROD_BATCH_SIZE_FALLBACK = 128` is added as a documented **manual** OOM step-down only — not auto-wired into an OOM-retry loop (batch 256 peaks at 474 MB on the 4060's 8 GB, so a retry path is unbuilt for now). 256/128 chosen from an RTX 4060 benchmark; **user signed off on 256 on 2026-07-01.**
- predictor_training_perf: new decision. `build_fold_loaders` gains a `device` arg and returns a device-resident `_WindowLoader` (vectorised gather over an on-GPU fold tensor) instead of a `DataLoader`, eliminating per-step host→device copies; `train_one_fold` accumulates loss on-GPU and syncs only at log boundaries; `torch.set_float32_matmul_precision("high")` + `torch.backends.cudnn.benchmark = True` set at module import. Windowing semantics + all safety invariants (stop/save events, best-by-val-total, early stopping, pre-clip grad-norm, non-finite raise) unchanged.
- training_ui_batch_log_interval: new `TrainingUIConfig.BATCH_LOG_INTERVAL = 50` — SSE `"batch"` cadence throttled from every step to every 50th (plus first + epoch-final flush); payload schema unchanged.
- Reason: real training was ~56–117 s/epoch on the 4060 with ~2/3 of per-step time being fixed overhead (single-process DataLoader host→device copy + forced CUDA syncs + a JSON SSE broadcast every one of ~4,642 steps/epoch) and the smoke batch size (32) leaking into production. Benchmarked before/after on real data (fold 0, lookback 1440, bf16 AMP): 116.6 → 14.4 s/epoch at 474 MB peak VRAM (~8× faster). Pure engineering — no change to loss, model, algorithm, or geometry enforcement.
- Source: conversation 2026-07-01 (training-speed optimization task)

## 2026-06-30 — Lock training_ui_controls process model + NaN-coverage deploy guard
- training_ui_controls: "threading.Event or subprocess signal" (open either/or) → locked as in-process `threading.Thread` running `train_all_folds`, `stop_event`/`save_event` checked at each batch boundary in `train_one_fold`. Also documents that manual-save checkpoints embed `train_q90_coverage` as NaN.
- `scripts/deploy_predictor.py`: added a `math.isnan(train_coverage)` guard immediately after resolving the checkpoint's embedded/CLI coverage value — refuses to deploy (STOP) a NaN-coverage checkpoint rather than silently comparing NaN against `DEPLOY_GATE_COVERAGE_TOLERANCE`.
- Reason: implementing Phase 1.5, `decisions-auditor` (mandatory trigger per `CLAUDE.md`) flagged two gaps: (1) `training_ui_controls` was still phrased as an open decision after the code had already committed to `threading.Event` (`src/training_ui/app.py`'s docstring stated the rationale, but a decision documented only in a code comment isn't a substitute for `DECISIONS.md` per `doc_drift_policy`); (2) a manually-saved mid-training checkpoint (via the dashboard's "Save" button) has no fresh q90-coverage measurement available without an expensive extra eval pass inside the hot training loop, so `train_all_folds`'s `on_save_request` embeds NaN — but `deploy_predictor.py` had no guard against silently gate-evaluating that NaN. `python-reviewer` (also mandatory) separately confirmed no other correctness issues in the same code path.
- Source: conversation 2026-06-30 (Training Dashboard implementation, post-review fixes)

## 2026-06-30 — Lock search-loop noise defense (search_confirm_seeds)
- search_confirm_seeds: absent (bare `default=3` in scripts/search_predictor.py) → locked as `DataConfig.SEARCH_CONFIRM_SEEDS = 3` + a DECISIONS.md entry documenting the strict-majority repeat-seed rule.
- Reason: decisions-auditor review of the search-loop diff flagged that the loop's actual Bonferroni-style safeguard (repeat-seed confirmation before a "kept" verdict) was an undocumented Python default rather than a recorded decision, despite being the mechanism that makes "kept" statistically meaningful. Per CLAUDE.md's "flag every unspecced decision" rule, surfaced to the user rather than assumed; user chose to lock it in.
- Source: conversation 2026-06-30 (decisions-auditor finding, user confirmed)

## 2026-06-30 — Add search-loop dev slice (search_dev_slice)
- search_dev_slice: absent → added. New `DataConfig.SEARCH_SLICE_TRAIN` (20,000) / `SEARCH_SLICE_VAL` (8,000) constants and `src/data/walk_forward.py::carve_search_slice()`, which carves the most recent 28,000 candles strictly before `HISTORICAL_START` for a fast, repeatable dev slice with zero Bonferroni exposure to the real walk-forward gate. Wired into `scripts/train_predictor.py::prepare()` via a new `--search-slice` CLI flag; checkpoint save is skipped in this mode (same as `--synthetic` — not a deployable artifact).
- Reason: prerequisite for an autoresearch-style unattended hyperparameter/architecture search loop (`scripts/search_predictor.py`, this session). The loop needs to iterate fast on real market structure without ever touching a production fold or the locked test set. The now-enforced `HISTORICAL_START` filter (previous entry, same day) already excludes 2,227,825 pre-2018 candles (verified: span 2013-10-06T21:35–2017-12-31T23:59) from every real fold — this decision claims a small, fixed slice of that already-excluded range rather than inventing a new exclusion. Test-first: `tests/data/test_walk_forward.py::test_carve_search_slice_*` and `tests/predictor/test_train_predictor.py::test_prepare_search_slice_uses_only_pre_historical_start_candles` (all committed failing before the implementation). 100 tests green.
- Source: conversation 2026-06-30 (autoresearch-adaptation design discussion)

## 2026-06-30 — Enforce HISTORICAL_START in the real-data walk-forward path
- walk_forward_fold_count: "~84 non-overlapping validation folds against the 2018+ dataset" → 78, recomputed against the real XBTUSD CSV's actual 2018-01-01–2025-12-31 span (`n_total`=4,207,680 → `n_usable`=4,086,720 → 78 folds; see `DECISIONS.md` for the full derivation).
- Reason: `DataConfig.HISTORICAL_START` ("2018-01-01") was documented in `historical_start` as excluding pre-2018 BTC data ("pre-2017 microstructure rejected") but had exactly one reference in the whole codebase — its own declaration. `scripts/train_predictor.py::prepare()`'s real-data branch passed the full validated/feature array (spanning the CSV's true start, 2013-10-06) straight into `src/data/walk_forward.py::carve_locked_test`/`make_folds`, neither of which filter by date. Fold 0 — the default fold used by `--smoke` — was silently training on 2013 data, contradicting the documented decision. Fixed by adding `filter_by_historical_start(timestamps, features)` to `src/data/walk_forward.py` (enforced for any caller building folds from timestamped features, not just this script) and calling it in `prepare()` before `carve_locked_test`. The previous "~84" fold estimate was never verified against the real CSV's actual date range — the corrected figure (78) reflects the true post-filter candle count. Test-first: `tests/data/test_walk_forward.py::test_filter_by_historical_start_*` and `tests/predictor/test_train_predictor.py::test_prepare_real_data_drops_pre_historical_start_candles` (both committed failing before the fix). 95 tests green, mypy clean on changed files. `splits-validation.md` (the active `INDEX.md`-referenced context card) corrected to match — "Roughly 84" → "78". `docs/post-mortem.md`'s "~84" references left untouched per its own stated policy (frozen historical record, never a live source of truth).
- Source: conversation 2026-06-30 (gap identified via repo-wide `HISTORICAL_START` grep audit; doc-drift in `splits-validation.md` caught by decisions-auditor)

## 2026-06-30 — Training UI constants group + data-gate doc-drift fix
- constants_organization: groups `PredictorConfig, RLConfig, TraderConfig, ExecutionConfig, DataConfig` → adds `TrainingUIConfig` (new group in `constants.py`: loss-chart EMA/window, alert timing, grad-norm/DA/Q-cov/patience color thresholds, ETA display gate — see inline comments for which values are cosmetic defaults vs. spec-derived). Also added `PredictorConfig.TRAINING_METRICS_FILENAME` and `ExecutionConfig.TRAINING_UI_BIND_PORT` (8001).
- training_ui_data_gate: "Download Data" drag-and-drop upload screen (`POST /api/setup/upload-data`) → **no upload UI**. Data missing at startup just disables controls + shows the `KRAKEN_DATA_PATH` instruction banner; restart required after fixing the env var.
- Reason: implementing Phase 1.5 (Training Dashboard) surfaced that `training_ui_data_gate` was stale — it described an upload/drag-and-drop flow that `training-dashboard.md`'s full spec (drafted 2026-06-30, uncommitted until this session) explicitly supersedes ("No drag-and-drop data upload — data management is handled outside the browser"). Fixed before building `src/training_ui/setup_router.py` so the wrong feature isn't built. `TrainingUIConfig` added rather than folding UI-only cosmetic thresholds into `PredictorConfig`, to keep the locked group list meaningful (training UI is a distinct subsystem, Phase 1.5).
- Source: conversation 2026-06-30 (Training Dashboard implementation)

## 2026-06-30 — Pull Training Dashboard ahead to new Phase 1.5
- phase_structure: 6 phases (−1, 0, 1, 2, 3, 4) → 7 phases (adds **Phase 1.5: Training Dashboard**). The Training Dashboard (`src/training_ui/`) moves out of Phase 4 (which now builds the Trading Dashboard only) into its own phase between Phase 1 and Phase 2.
- long_training_run_timing: launches within Phase 1 → launches at **Phase 1.5 exit**, so the first long run is monitored and diagnosable from the start (live fold/epoch/loss/ETA + `training_metrics.json` export).
- branch_model: adds branch `phase-15-training-ui` off `main` (consistent with the per-phase `phase-XY` rule), cut after PR #13 merges Phase 1.
- Reason: user directive — the training dashboard's entire purpose is to observe and diagnose the long training run, but it was slated for Phase 4, which (per the parallel-work plan) runs *during* that run. Building the instrument during the run it observes is chicken-and-egg and leaves the first run unwatched. Pulling it to Phase 1.5 (full scope) fixes the sequencing. Authoritative phase list lives in `post-mortem.md` §3.0 and `INDEX.md`; both updated alongside `CLAUDE.md` repo-state.
- Source: conversation 2026-06-30 (decision questions answered: full dashboard scope, new branch off main)

## 2026-06-29 — Deploy reads gate inputs from the checkpoint (val-split coverage baseline)
- deploy_gate_train_coverage_baseline_split: absent → `validation`. Training-time q90 coverage (deploy gate (a) baseline) is computed on the held-out VAL split with the best-by-val-total weights via new `evaluate_q90_coverage` in `src/predictor/training.py` (reuses `deploy_gates.q90_coverage` + `rollout.enforce_geometry` for an identical computation to deploy's locked-test path).
- predictor_checkpoint_save_format: added `train_q90_coverage` to the embedded wrapped dict (now `{state_dict, lookback, constants_sha256, trained_through_ts_utc, train_q90_coverage}`).
- `scripts/deploy_predictor.py`: `--train-coverage` and `--trained-through` are now optional — they default to the values embedded in the checkpoint (CLI overrides). Added an architecture-mismatch guard: `--lookback` must equal the embedded lookback or deploy STOPs. `scripts/train_predictor.py` computes + embeds the val-split coverage after a successful real-data run.
- Reason: user-approved follow-up to the checkpoint-persistence change — removes hand-entered gate inputs (`--train-coverage`/`--trained-through`) as an error source now that the checkpoint carries provenance. Verified end-to-end: `deploy_predictor.py` ran for the first time on synthetic fixtures, reading coverage/timestamp from the checkpoint (gates correctly fail on a random model; lookback guard fires). 85 tests green, mypy clean. The VAL split (vs in-sample train) was chosen so gate (a) compares held-out coverage to held-out coverage.
- Source: conversation 2026-06-29 (decision question answered: validation split)

## 2026-06-29 — Wire train→deploy checkpoint persistence (3 decisions)
- predictor_checkpoint_dir_and_naming: absent → weights `checkpoints/{run_tag}.pt` + scaler `checkpoints/{run_tag}.scaler.pkl` (run_tag-based; run-history-preserving; binds weights↔scaler↔constants by hash). Added `PredictorConfig.CHECKPOINT_DIR` / `CHECKPOINT_WEIGHTS_SUFFIX` / `CHECKPOINT_SCALER_SUFFIX` to `constants.py`.
- predictor_checkpoint_save_format: absent → wrapped dict `{state_dict (CPU), lookback, constants_sha256, trained_through_ts_utc}` via `torch.save` (loads under `weights_only=True`) + pickled scaler; matches what `deploy_predictor.py` already reads.
- predictor_checkpoint_save_policy: absent → best-by-`val_total`. `train_one_fold` now retains the lowest-`val_total` weights and restores them into the model before returning; `scripts/train_predictor.py` saves the checkpoint after a successful run (`--no-save` opts out).
- Reason: Phase 1 had modules complete but the train→deploy persistence layer was unbuilt (`train_predictor.py` never saved weights; `deploy_predictor.py` required `--checkpoint/--scaler`). Three save conventions were unspecced; user chose run_tag-based naming, wrapped-dict format, and best-by-val-total policy. Added `save_checkpoint` to `src/predictor/training.py` (tests-first: `test_save_checkpoint_round_trips`, `test_train_one_fold_restores_best_val_weights`). 84 tests green, mypy clean; synthetic smoke produces a checkpoint that loads under deploy's `weights_only=True` path.
- Post-review fixes (same change set): (1) decisions-auditor — `scripts/deploy_predictor.py` manifest default now uses `PREDICTOR.CHECKPOINT_DIR` instead of a bare `"checkpoints"` literal (single-source rule); (2) python-reviewer — `scripts/train_predictor.py --synthetic` now skips the checkpoint save (a synthetic run's fabricated 2020-origin timestamps + random-data weights are never deployable); (3) documented the pickle/Python-version coupling caveat for the scaler artifact in `predictor_checkpoint_save_format`.
- Source: conversation 2026-06-29 (decision questions answered)

## 2026-06-29 — Fix stale BTCUSD references after pair switch
- training_ui_data_gate: `data/raw/BTCUSD_1.csv` → `data/raw/XBTUSD_1.csv` (i.e. `DATA.KRAKEN_HISTORY_CSV_NAME`)
- Reason: decisions-auditor flagged internal inconsistency — kraken_training_pair was updated to XBTUSD but training_ui_data_gate still referenced BTCUSD_1.csv. Also updated training-dashboard.md and CLAUDE.md repo-state to match.
- Source: decisions-auditor finding (2026-06-29)

## 2026-06-29 — Switch training data to XBTUSD; record Kraken pair decision
- kraken_training_pair: absent → `XBTUSD` (`master_q4/XBTUSD_1.csv`, 2013-10-07 to 2025-12-31)
- DataConfig.KRAKEN_HISTORY_INNER_PATH: `"master_q4/BTCUSD_1.csv"` → `"master_q4/XBTUSD_1.csv"`
- DataConfig.KRAKEN_HISTORY_CSV_NAME: absent → `"XBTUSD_1.csv"`
- Reason: BTCUSD_1.csv only covers 2022-01-01 to 2024-01-01 (2 years), below the 2018-01-01 historical_start requirement. XBTUSD is Kraken's original/primary pair with 12+ years of data. XBTUSD also uses no-header format consistent with np.loadtxt in load_real_candles; BTCUSD's header row would break the loader.
- Source: user directive (conversation 2026-06-29)

## 2026-06-29 — Record deploy-gates output dimension and quantile-index pattern
- deploy_gates_output_dim: absent → `close_logret` (Close dimension only; execution fills at close-of-candle; consistent with close-dim direction penalty in loss and trader confidence gate)
- deploy_gates_quantile_index_pattern: absent → `PREDICTOR.QUANTILES.index(value)` at module scope into private constants; no alias integer constants in `constants.py` (self-documenting and tuple-order-robust)
- Reason: code in `deploy_gates.py` made both choices without an explicit DECISIONS.md entry. User confirmed both during Phase 1 review.
- Source: user directive (conversation 2026-06-29)

## 2026-06-28 — Move agent_config schema version into constants.py
- ExecutionConfig.AGENT_CONFIG_SCHEMA_VERSION: absent → `"1.0"`
- Reason: code-review audit flagged a bare module-level `SCHEMA_VERSION = "1.0"` in `scripts/deploy_predictor.py`, violating the "no bare module-level constants / magic numbers live in constants.py" rule. Value unchanged; deploy now reads `EXECUTION.AGENT_CONFIG_SCHEMA_VERSION`. Same audit pass (no other decision values changed): validator rejects NaN/inf candles; `train_one_fold` guards empty `train_loader` + non-finite val loss and reports epoch-averaged train loss; `verify_manifest` rejects uncovered artifacts; scaler gains `transform_inference` so deploy no longer re-implements the scaling formula.
- Source: audit

## 2026-06-28 — Retire gdown ingest; manual CSV placement
- Removed `scripts/ingest_kraken_history.py` (Google Drive / gdown bootstrap ingest). The raw `BTCUSD_1.csv` is now placed into `data/raw/` manually (drag-and-drop) instead of being downloaded.
- Dropped dead dependencies `browser-cookie3` and `gdown` from `pyproject.toml` (regenerated `uv.lock`); removed the script's per-file ruff ignore.
- Cleaned remaining references in `CLAUDE.md`, `INDEX.md`, `dashboard.md`, `consolidated-consolidated-plan.md`, `.gitignore`, and the `train_predictor.py` "csv not found" message.
- Reason: user directive — the gdown/Google Drive ingest path is retired in favor of manual CSV placement.
- Source: user directive (2026-06-28)

## 2026-06-28 — Drop `developer` branch; phase branches merge to `main`
- branch_model: "`phase-XY` off `developer`; `developer` → `main` on phase exit" → "`phase-XY` off `main`; merged to `main` on phase exit (no intermediate `developer` branch)"
- Reason: user directive — the `developer` integration branch is no longer used; phase branches merge straight to `main` (matching the open `phase-1-predictor` → `main` PR). Same doc-consolidation pass also updated `CLAUDE.md` §"Project-specific rules" and the consolidated plan (§1.4, §3.0) to match.
- Source: user directive (2026-06-28) — doc-consolidation sweep

## 2026-06-27 — Bind encoder activation + norm_first to constants
- predictor_config.ACTIVATION: absent → `"gelu"`
- predictor_config.NORM_FIRST: absent → `True`
- Reason: decisions-auditor flagged `activation="gelu"` / `norm_first=True` as bare literals in `src/predictor/model.py`. Because the SHA256 manifest hashes `constants.py` but not `model.py`, behaviour-defining architecture choices must live in `constants.py` to be manifest-bound (else a post-training change silently alters the architecture without invalidating the manifest). Same commit also adds a final `LayerNorm` to the pre-LN encoder (python-reviewer: the last block's output was otherwise unnormalised before the head).
- Source: Phase 1 build — decisions-auditor + python-reviewer review of model.py

## 2026-06-27 — Predictor architecture + training hyperparameters
- predictor_config PatchTST architecture: absent → `PATCH_EMBED_MODE="channel_mixing"`, `D_MODEL=128`, `N_HEADS=8`, `N_LAYERS=3`, `D_FF=256`, `DROPOUT=0.1`
- predictor_config training loop: absent → `LEARNING_RATE=3e-4`, `WEIGHT_DECAY=1e-2`, `WARMUP_FRAC=0.05`, `GRAD_CLIP_NORM=1.0`, `MAX_EPOCHS=100`, `USE_AMP=True`, `SEED=0`, `SMOKE_BATCH_SIZE=32`, `SMOKE_BATCH_SIZE_FALLBACK=16`, `WANDB_PROJECT="btc-bot-v3-predictor"`
- Reason: `predictor-training.md` §"Architecture"/"Smoke run" left every PatchTST hyperparameter and optimizer setting unspecced (flagged in the Phase 1 brief as STOP-and-ask). User approved the recommended config: an ~3M-param channel-mixing PatchTST sized for the RTX 4060 8GB at batch 32 / lookback 1440, AdamW + cosine-with-warmup, AMP (bf16). Channel-mixing chosen over PatchTST's channel-independent default because OHLCV are facets of one instrument (intra-candle cross-feature interaction matters) and it matches the card's "90 tokens" framing.
- Source: Phase 1 build — user directive (AskUserQuestion approval, "whatever you recommend")

## 2026-06-27 — Predictor geometry-enforcement constant
- predictor_config.GEOMETRY_RESAMPLE_CAP: absent → `5` (max resample attempts before a deterministic clamp when a sampled candle violates `H >= max(O,C)` / `L <= min(O,C)`)
- Reason: relocate the "resampled up to 5×" literal from `predictor-contract.md` §"Geometry enforcement" into `constants.py` so `src/predictor/rollout.py` carries no bare magic number (single-source-of-truth policy). No decision value changed; the 5× cap was already specified in the contract card.
- Source: Phase 1 build — rollout geometry enforcement (deliverable #1)

## 2026-06-27 — Remove Telegram; add Training TUI
- alerts: "Telegram bot (push to phone) + structured JSON logs + dashboard color states" → "sound/beep (winsound.Beep) + structured JSON logs + dashboard color states"
- stale_candle_auto_close: "alert via Telegram + dashboard banner" → "alert via sound/beep + dashboard banner"
- stop_loss_confirmation_required: "auto-close any partial fill + Telegram alert" → "auto-close any partial fill + sound/beep alert"
- training_ui_stack: absent → Textual (Python TUI)
- training_ui_controls: absent → start / stop (graceful checkpoint save) / save (checkpoint now)
- training_ui_stop_behavior: absent → signal-based; closing TUI does not kill training process
- training_ui_metrics: absent → fold index, epoch, train/val loss, epoch/fold/total run ETA
- training_ui_alerts: absent → in-app Textual banner + winsound.Beep + structured JSON log
- training_ui_export: absent → per-fold JSON record appended to `training_metrics.json` (path in `PredictorConfig`); schema: fold index, losses, DA, quantile coverage, duration, hyperparams snapshot
- Reason: user directive — no Telegram; replace with a lightweight Textual TUI for training management and a JSON export handoff for Claude optimization review.
- Source: user directive (Phase 1 training tooling)

## 2026-06-28 — Both dashboards are browser apps; trading dashboard gains optimization panels
- training_ui_stack: Textual TUI → FastAPI + vanilla JS (browser). Separate port from trading dashboard. Full spec in `training-dashboard.md`.
- training_ui_data_gate: file-path `[Input]` widget → drag-and-drop zone (browser natively supports it).
- training_ui_controls/metrics/alerts: unchanged in function; delivery mechanism changes from Textual widgets to browser buttons + WebSocket/SSE + browser notification banner.
- dashboard (trading): drag-and-drop removed (data setup belongs in training dashboard). Added panels: performance metrics (Sharpe/Sortino/win rate/drawdown 7d/30d/all-time), regime analysis, fee drag, model management (checkpoint SHA + deploy gate scores + W&B link), kill criteria status (K1–K9 live vs. threshold), data collection (status + on-demand gap-fill trigger), AI analysis (Claude API LLM report + on-chart predictor insights). WebSocket type enum extended with `"metrics"` and `"criteria"`.
- Reason: user directive — both dashboards are browser-based. Training dashboard owns drag-and-drop data upload; trading dashboard owns live optimization tooling that W&B does not cover (trade-level performance, regime breakdown, fee drag, model drift, LLM analysis).
- Source: user directive (2026-06-28)

## 2026-06-28 — Training TUI data gate + trading dashboard drag-and-drop
- training_ui_data_gate: absent → on startup check for `data/raw/BTCUSD_1.csv`; if absent, block training and show a "Download Data" screen with instruction text, Google Drive URL (from `DATA.KRAKEN_HISTORY_GDRIVE_ID`), and a file-path `[Input]` widget; atomic copy/extract on submission
- dashboard (trading): first-run data panel added → drag-and-drop area accepting zip or CSV, Google Drive link, `POST /api/setup/upload-data` endpoint; panel shown only when CSV absent, disabled (404) once present
- Reason: user directive — two distinct apps: (1) Training TUI (Textual, terminal) for model training; (2) Trading Dashboard (FastAPI + vanilla JS) for live/paper trading. Each independently detects missing raw data and guides the user to supply it. The TUI uses a file-path input (idiomatic for Textual); the dashboard uses drag-and-drop (native to the browser). `dashboard.md` retitled "Trading Dashboard" and cross-reference added.
- Source: user directive (2026-06-28)

## 2026-06-27 — PyTorch toolchain (Phase 1 predictor)
- Toolchain.python_version: absent → `3.13` (pinned via `.python-version`)
- Toolchain.ml_framework: absent → PyTorch `2.6.0+cu124` (CUDA 12.4, RTX 4060 / sm_89), from the explicit `pytorch-cu124` index pinned in `[tool.uv.sources]`
- Reason: Phase 1 predictor loss requires PyTorch (autograd). The active venv was Python 3.14, for which no torch cu124 wheel exists (wheels ship cp310–cp313 only), so the interpreter is pinned to 3.13. User selected the CUDA-now build to match the 4060 for the eventual smoke run + training. `requires-python` unchanged (`>=3.10`); numpy resync'd to 2.2.6 under torch's constraint (still satisfies `>=2.2.6`).
- Source: Phase 1 build / user directive

## 2026-05-29 — Phase 0 data pipeline constants
- data_config.FEATURE_NAMES: absent → ("open_logret", "high_logret", "low_logret", "close_logret", "vol_change")
- data_config.VOL_CHANGE_DEGENERATE_FILL: absent → 0.0
- Reason: relocate the feature-name schema into `constants.py` (single source of truth; removes a bare module-level constant from `src/` per CLAUDE.md), and record the previously-unspecced degenerate-volume fill value (vol_change when prior volume is 0 → neutral 0.0). Both were flagged by the decisions-auditor during the Phase 0 review. Also clarified `feature-pipeline.md` "Strict fit-window assertion": the scaler's allowed transform window is the whole fold `[fold_start, fold_end]` with min/max fit on the train slice only — resolving an internal contradiction in the card.
- Source: Phase 0 build / decisions-auditor review

## 2026-05-11 — Feature pipeline volume edge cases
- data_config.vol_logret_floor: absent → `-10.0` (clip for `log(volume_t / volume_{t-1})` when `volume_t = 0` or the ratio underflows)
- feature_pipeline.vol_t_minus_1_zero_handling: absent → impute `vol_change = 0` and emit a structured-log warning
- Reason: `feature-pipeline.md` flagged `volume_t = 0 → log1p(-1) = -inf` as failure mode #2 without prescribing a clip value. Both decisions are required before the feature pipeline module (Phase 0 task 1) can be written without inserting magic numbers in `src/`.
- Source: Phase 0 task 1 plan — feature pipeline module

## 2026-05-09 — UV as primary package manager
- package_manager: absent → UV (`uv sync` / `uv add` / `uv run`)
- build_backend: absent → hatchling
- lock_file: absent → `uv.lock` committed
- uv_package_install_mode: absent → `[tool.uv] package = false` (temporary)
- Reason: user directive to standardize on UV as the #1 toolchain. This introduced Toolchain decision keys in `DECISIONS.md`; `pyproject.toml` previously used setuptools before this decision was formalized. `uv.lock` is committed per `.gitignore` pre-configuration.
- Source: user directive

## 2026-05-09 — Bootstrap ingest constants
- data_config.kraken_history_bootstrap_defaults: absent → `KRAKEN_HISTORY_GDRIVE_ID`, `KRAKEN_HISTORY_INNER_PATH`, `KRAKEN_HISTORY_ZIP_STEM`, `KRAKEN_HISTORY_OUT_DIR`, `KRAKEN_HISTORY_CACHE_DIR`
- Reason: move ingest bootstrap literals into `constants.py` so the bootstrap script follows the repository constants policy.
- Source: PR #2 review follow-up

## 2026-05-08 — Initial v3 lockdown

Initial population from `consolidated-consolidated-plan.md`. All entries in `DECISIONS.md` as of this date are considered "locked from v3 master plan §1." Subsequent amendments enumerate specific deltas only.
