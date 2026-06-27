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
