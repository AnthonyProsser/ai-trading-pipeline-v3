# agent_config.json schema

Runtime configuration that varies between deployments / checkpoints / data refreshes. Distinct from `constants.py`, which holds locked architectural values that should not vary at runtime.

`agent_config.json` is **machine-written** by Phase 0 (Data) and Phase 2 (Environment) scripts, **not human-edited**. Manual edits are a code smell.

## Schema

```json
{
  "schema_version": "1.0",
  "predictor": {
    "checkpoint_path": "checkpoints/predictor_<wandb_run_id>_<sha[:8]>.pt",
    "checkpoint_sha256": "<full sha256>",
    "scaler_path": "data/processed/<YYYY-MM-DD>/scaler.pkl",
    "scaler_sha256": "<full sha256>",
    "constants_sha256": "<full sha256>",
    "contract_version": "<sha[:8] of predictor-contract.md content>",
    "lookback": 1440,
    "trained_through_ts_utc": "<ISO8601>",
    "deploy_gate_results": {
      "coverage_delta": 0.012,
      "directional_accuracy": 0.547,
      "calibration_rate": 0.81
    }
  },
  "data": {
    "raw_dir": "data/raw/",
    "processed_dir": "data/processed/<YYYY-MM-DD>/",
    "test_locked_dir": "data/test_locked/",
    "atr_median_1440": 0.0,
    "last_ingested_ts_utc": "<ISO8601>"
  },
  "execution": {
    "backend": "PaperBackend" | "LiveBackend",
    "kill_flag_path": "KILL_SWITCH.flag",
    "kraken_api_key_keyring_service": "btc-bot-v3",
    "watchdog_pid_file": "watchdog.pid"
  },
  "manifest": {
    "manifest_path": "checkpoints/manifest.json",
    "verified_at_startup_ts_utc": "<ISO8601>",
    "last_weight_reload_ts_utc": "<ISO8601>"
  }
}
```

## SHA256 manifest

A separate file (`checkpoints/manifest.json`) lists every artifact and its sha256. The execution engine verifies at startup AND on every weight reload that the on-disk hashes match the manifest, AND that the manifest references match `agent_config.json`. Mismatch = refuse to start (or refuse the reload), sound/beep + log alert.

Manifest scope:
- Predictor weights (`.pt`)
- Scaler PKL (`.pkl`)
- `constants.py` content hash

## Population

- `predictor.*` — written by `scripts/deploy_predictor.py` after all three deploy gates pass
- `data.atr_median_1440` — written by Phase 0 ingest+feature pipeline once per data refresh
- `data.last_ingested_ts_utc` — updated each cycle by the ingest module
- `execution.*` — written once at environment setup; `backend` toggled at paper→live transition
- `manifest.*` — `verified_at_startup_ts_utc` and `last_weight_reload_ts_utc` written by execution engine

## Versioning

`schema_version` bumps on any structural change. `deploy_predictor.py` refuses to write a config whose `schema_version` doesn't match its expected version — surfaces silent schema drift instead of writing nonsense.

## Backup

Synced to OneDrive/Google Drive on every write (same mechanism as checkpoints + scaler). Single-machine deployment otherwise = single point of failure.
