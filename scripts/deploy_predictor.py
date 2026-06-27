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

from constants import DATA, PREDICTOR
from src.data.feature_pipeline import compute_features
from src.data.manifest import build_manifest, sha256_file, write_manifest
from src.data.scaler import PerFoldMinMaxScaler
from src.data.validator import validate_candles
from src.predictor.deploy_gates import DeployGateResult, evaluate_deploy_gates
from src.predictor.model import PatchTST
from src.predictor.rollout import enforce_geometry
from src.predictor.training import WindowDataset

# Bumps on any structural change to agent_config.json; deploy refuses to overwrite a
# config whose schema_version differs (surfaces silent schema drift). See agent-config.md.
SCHEMA_VERSION = "1.0"


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
    span = scaler.data_max_ - scaler.data_min_
    span[span == 0.0] = 1.0
    x_scaled = ((features - scaler.data_min_) / span).astype(np.float32)

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
            targets.append(y_batch)
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
    if config_path.exists():
        config: dict[str, object] = json.loads(config_path.read_text(encoding="utf-8"))
        existing = config.get("schema_version")
        if existing != SCHEMA_VERSION:
            raise SystemExit(
                f"[stop] {config_path} schema_version {existing!r} != expected "
                f"{SCHEMA_VERSION!r}; refusing to write (schema drift)."
            )
    else:
        config = {"schema_version": SCHEMA_VERSION}

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


def run(args: argparse.Namespace) -> int:
    model = PatchTST(lookback=args.lookback)
    # weights_only=True avoids arbitrary-code execution from a crafted .pt. Accept either a
    # bare state_dict or a wrapper {"state_dict": ...}; the save convention is pinned once
    # the training run gains checkpointing.
    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded
    model.load_state_dict(state)
    with open(args.scaler, "rb") as fh:
        scaler: PerFoldMinMaxScaler = pickle.load(fh)

    timestamps, ohlcv = load_eval_candles(args.eval_data)
    validated = validate_candles(timestamps, ohlcv)
    features = compute_features(validated.ohlcv)
    pred, target = gather_predictions(
        model, scaler, features, lookback=args.lookback, batch_size=PREDICTOR.SMOKE_BATCH_SIZE
    )

    result = evaluate_deploy_gates(pred, target, train_time_q90_coverage=args.train_coverage)
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
        trained_through_ts_utc=args.trained_through,
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
        "--train-coverage", type=float, required=True, dest="train_coverage",
        help="training-time q90 coverage (gate (a) baseline)",
    )
    ap.add_argument(
        "--trained-through", required=True, dest="trained_through",
        help="ISO8601 UTC timestamp the checkpoint was trained through",
    )
    ap.add_argument("--lookback", type=int, default=DATA.LOOKBACK)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "agent_config.json")
    ap.add_argument("--manifest", type=Path, default=REPO_ROOT / "checkpoints" / "manifest.json")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
