"""deploy_predictor.py — evaluate the three deploy gates and, only on a simultaneous
pass, write the SHA256 manifest + the predictor section of agent_config.json.

Flow (predictor-contract.md §"Deploy gates", agent-config.md):
  1. Load checkpoint + per-fold scaler + an evaluation candle set (the locked test set,
     supplied via --eval-data so this script never hardcodes data/test_locked/).
  2. Run inference (single forward pass) with geometry enforcement, gather (pred, target).
  3. evaluate_deploy_gates -> all three must pass simultaneously, else STOP (no deploy).
  4. On pass: build the manifest over weights + scaler + constants.py (reusing
     src/data/manifest.py) and write the machine-generated agent_config.json predictor
     section. agent_config.json is never hand-edited.

NOTE: this script is exercised once a trained checkpoint + locked test set exist; it is
structurally complete and type-checked but not run end-to-end in the Phase 1 build (no
checkpoint yet). All thresholds/shapes come from constants.py and the deploy_gates module.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from constants import DATA, EXECUTION, PREDICTOR
from src.data.feature_pipeline import compute_features
from src.data.manifest import build_manifest, sha256_file, write_manifest
from src.data.scaler import PerFoldMinMaxScaler
from src.data.validator import validate_candles
from src.predictor.deploy_gates import DeployGateResult, evaluate_deploy_gates
from src.predictor.model import PatchTST
from src.predictor.rollout import enforce_geometry
from src.predictor.training import WindowDataset

def load_eval_candles(path: Path) -> tuple[npt.NDArray[np.datetime64], npt.NDArray[np.float64]]:
    arr: npt.NDArray[np.float64] = np.loadtxt(
        path, delimiter=",", usecols=(0, 1, 2, 3, 4, 5), dtype=np.float64
    )
    timestamps = arr[:, 0].astype("int64").astype("datetime64[s]")
    return timestamps, arr[:, 1:6].astype(np.float64)


def gather_predictions(
    model: PatchTST,
    scaler: PerFoldMinMaxScaler,
    features: npt.NDArray[np.float64],
    *,
    lookback: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model over sliding eval windows and return (pred, target).

    Scaling is applied manually from the fitted stats: PerFoldMinMaxScaler.transform carries
    a train-time fold-window assertion that (correctly) rejects out-of-fold timestamps, but
    inference on the locked test set legitimately uses the trained min/max on new windows.
    """
    if scaler.data_min_ is None or scaler.data_max_ is None:
        raise ValueError("deploy requires a fitted scaler")
    x_scaled = scaler.transform_inference(features).astype(np.float32)

    dataset = WindowDataset(
        torch.from_numpy(x_scaled),
        torch.from_numpy(features.astype(np.float32)),
        lookback,
        PREDICTOR.HORIZON,
    )
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            preds.append(enforce_geometry(model(x_batch)))
            # Model output = cumulative log-return path (TARGET_SEMANTICS); convert the
            # raw per-step targets to the same space so the gates compare like with like.
            targets.append(y_batch.cumsum(dim=1))
    return torch.cat(preds), torch.cat(targets)


def write_agent_config(
    config_path: Path,
    *,
    checkpoint: Path,
    scaler_path: Path,
    manifest: dict[str, str],
    contract_version: str,
    lookback: int,
    trained_through_ts_utc: str,
    result: DeployGateResult,
) -> None:
    """Merge the predictor section into agent_config.json (preserving other sections).

    Refuses to overwrite a config whose schema_version differs from this script's."""
    schema_version = EXECUTION.AGENT_CONFIG_SCHEMA_VERSION
    if config_path.exists():
        config: dict[str, object] = json.loads(config_path.read_text(encoding="utf-8"))
        existing = config.get("schema_version")
        if existing != schema_version:
            raise SystemExit(
                f"[stop] {config_path} schema_version {existing!r} != expected "
                f"{schema_version!r}; refusing to write (schema drift)."
            )
    else:
        config = {"schema_version": schema_version}

    config["predictor"] = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": manifest["weights"],
        "scaler_path": str(scaler_path),
        "scaler_sha256": manifest["scaler"],
        "constants_sha256": manifest["constants"],
        "contract_version": contract_version,
        "lookback": lookback,
        "trained_through_ts_utc": trained_through_ts_utc,
        "deploy_gate_results": {
            "coverage_delta": round(result.coverage_delta, 4),
            "directional_accuracy": round(result.directional_accuracy, 4),
            "calibration_rate": round(result.calibration_rate, 4),
        },
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def _from_cli_or_checkpoint(cli_value: object, embedded: object, flag: str) -> object:
    """Resolve a deploy parameter: an explicit CLI flag wins, else the value embedded by
    save_checkpoint, else STOP (refuse to guess gate inputs)."""
    value = cli_value if cli_value is not None else embedded
    if value is None:
        raise SystemExit(
            f"[stop] --{flag} not provided and not embedded in the checkpoint; cannot proceed."
        )
    return value


def run(args: argparse.Namespace) -> int:
    # weights_only=True avoids arbitrary-code execution from a crafted .pt. Only the
    # wrapper {"state_dict": ..., <provenance>} written by save_checkpoint is deployable:
    # provenance (lookback, train-coverage, trained-through, target-semantics) is read
    # here, and the semantics guard below rejects bare state_dicts by construction.
    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    meta: dict[str, object] = loaded if isinstance(loaded, dict) else {}
    state = meta.get("state_dict", loaded)
    if not isinstance(state, dict):
        raise SystemExit("[stop] checkpoint has no usable state_dict; corrupt or wrong file.")

    # Target-semantics guard: a checkpoint trained under a different prediction
    # convention (e.g. the pre-rework per-step log-return models) must never be gate-
    # evaluated against cumulative targets — the metrics would be silently meaningless.
    semantics = meta.get("target_semantics")
    if semantics != PREDICTOR.TARGET_SEMANTICS:
        raise SystemExit(
            f"[stop] checkpoint target_semantics {semantics!r} != expected "
            f"{PREDICTOR.TARGET_SEMANTICS!r} — trained under an incompatible target "
            f"convention (or predates the semantics tag); refusing to deploy."
        )

    embedded_lookback = meta.get("lookback")
    if embedded_lookback is not None:
        if not isinstance(embedded_lookback, int):
            raise SystemExit(f"[stop] checkpoint lookback is not an int: {embedded_lookback!r}")
        if embedded_lookback != args.lookback:
            raise SystemExit(
                f"[stop] --lookback {args.lookback} != checkpoint lookback {embedded_lookback}; "
                f"refusing (architecture mismatch)."
            )
    cov = _from_cli_or_checkpoint(
        args.train_coverage, meta.get("train_q90_coverage"), "train-coverage"
    )
    tt = _from_cli_or_checkpoint(
        args.trained_through, meta.get("trained_through_ts_utc"), "trained-through"
    )
    if not isinstance(cov, (int, float)):
        raise SystemExit(f"[stop] checkpoint train_q90_coverage is not numeric: {cov!r}")
    if not isinstance(tt, str):
        raise SystemExit(f"[stop] checkpoint trained_through_ts_utc is not a string: {tt!r}")
    train_coverage = float(cov)
    if math.isnan(train_coverage):
        raise SystemExit(
            "[stop] checkpoint train_q90_coverage is NaN — this is a manual mid-training "
            "snapshot (training UI 'Save' button), not a gate-evaluated checkpoint. "
            "Refusing to deploy; use a natural fold-completion checkpoint, or pass "
            "--train-coverage explicitly if you are certain this snapshot is deployable."
        )
    trained_through = tt
    print(
        f"[provenance] lookback={args.lookback} train_coverage={train_coverage:.4f} "
        f"trained_through={trained_through}"
    )

    model = PatchTST(lookback=args.lookback)
    model.load_state_dict(state)
    with open(args.scaler, "rb") as fh:
        scaler: PerFoldMinMaxScaler = pickle.load(fh)

    timestamps, ohlcv = load_eval_candles(args.eval_data)
    validated = validate_candles(timestamps, ohlcv)
    features = compute_features(validated.ohlcv)
    pred, target = gather_predictions(
        model, scaler, features, lookback=args.lookback, batch_size=PREDICTOR.SMOKE_BATCH_SIZE
    )

    # DA gate on the final horizon step only (the traded quantity); coverage/calibration
    # keep scoring the whole forecast path.
    result = evaluate_deploy_gates(
        pred, target, train_time_q90_coverage=train_coverage,
        da_pred=pred[:, -1], da_target=target[:, -1],
    )
    print(
        f"[gates] coverage={result.q90_coverage:.4f} (delta {result.coverage_delta:+.4f}) "
        f"DA={result.directional_accuracy:.4f} calibration={result.calibration_rate:.4f}"
    )
    print(
        f"[gates] coverage_pass={result.coverage_pass} da_pass={result.da_pass} "
        f"calibration_pass={result.calibration_pass}"
    )
    if not result.all_passed:
        print("[stop] deploy gates failed — checkpoint NOT deployed (all three must pass).")
        return 1

    manifest = build_manifest(
        {"weights": args.checkpoint, "scaler": args.scaler, "constants": REPO_ROOT / "constants.py"}
    )
    write_manifest(manifest, args.manifest)
    contract_version = sha256_file(REPO_ROOT / "predictor-contract.md")[:8]
    write_agent_config(
        args.config,
        checkpoint=args.checkpoint,
        scaler_path=args.scaler,
        manifest=manifest,
        contract_version=contract_version,
        lookback=args.lookback,
        trained_through_ts_utc=trained_through,
        result=result,
    )
    print(f"[deploy] gates passed — manifest -> {args.manifest}, agent_config -> {args.config}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True, help="predictor weights (.pt)")
    ap.add_argument("--scaler", type=Path, required=True, help="per-fold scaler (.pkl)")
    ap.add_argument(
        "--eval-data", type=Path, required=True, dest="eval_data",
        help="locked-test candle CSV for gate evaluation",
    )
    ap.add_argument(
        "--train-coverage", type=float, default=None, dest="train_coverage",
        help="training-time q90 coverage (gate (a) baseline); "
        "defaults to the value embedded in the checkpoint",
    )
    ap.add_argument(
        "--trained-through", default=None, dest="trained_through",
        help="ISO8601 UTC timestamp the checkpoint was trained through; "
        "defaults to the value embedded in the checkpoint",
    )
    ap.add_argument("--lookback", type=int, default=DATA.LOOKBACK)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "agent_config.json")
    ap.add_argument(
        "--manifest", type=Path, default=REPO_ROOT / PREDICTOR.CHECKPOINT_DIR / "manifest.json"
    )
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
