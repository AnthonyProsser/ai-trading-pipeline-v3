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
