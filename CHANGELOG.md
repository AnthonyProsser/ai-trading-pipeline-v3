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

## 2026-05-29 — Phase 0 data pipeline constants
- data_config.FEATURE_NAMES: absent → ("open_logret", "high_logret", "low_logret", "close_logret", "vol_change")
- data_config.VOL_CHANGE_DEGENERATE_FILL: absent → 0.0
- Reason: relocate the feature-name schema into `constants.py` (single source of truth; removes a bare module-level constant from `src/` per CLAUDE.md), and record the previously-unspecced degenerate-volume fill value (vol_change when current or prior volume is 0 → neutral 0.0). Both were flagged by the decisions-auditor during the Phase 0 review. Also clarified `feature-pipeline.md` "Strict fit-window assertion": the scaler's allowed transform window is the whole fold `[fold_start, fold_end]` with min/max fit on the train slice only — resolving an internal contradiction in the card.
- Source: Phase 0 build / decisions-auditor review

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
