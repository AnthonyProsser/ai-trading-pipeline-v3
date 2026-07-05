"""Checkpoint registry for the benchmark app: discovery, display names, result cache.

The registry NEVER renames, moves, or rewrites checkpoint files — the run-tag
filename ({git_sha}-s{scaler8}-c{constants8}-fold{id}) binds weights <-> scaler <->
constants by hash and the SHA256 manifest covers those exact paths, so the files are
immutable. A user-facing "rename" is a display name stored in registry.json, keyed by
the immutable stem.

On-disk layout (BenchmarkConfig, all under BENCHMARK_DIR):
    registry.json             {"display_names": {stem: name},
                               "meta_cache": {stem: {"mtime_ns": int, "meta": {...}}}}
    {stem}.benchmark.json     one cached benchmark result per model; its existence is
                              what makes the UI's "Start benchmark" button disappear.

Checkpoint metadata (lookback, trained_through_ts_utc, train_q90_coverage,
target_semantics, constants_sha256) is embedded in the .pt wrapped dict
(predictor_checkpoint_save_format) and read under weights_only=True; reads are
mtime-cached in registry.json so listing 78 checkpoints doesn't re-deserialize
~1 GB of tensors on every page load.

All JSON writes are atomic (tmp + os.replace) and strict (NaN -> null via json_safe;
browsers cannot parse bare NaN literals).
"""
from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, cast

import torch

from constants import BENCHMARK, PREDICTOR
from src.data.manifest import sha256_file

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Run tag = {git_sha}-s{scaler8}-c{constants8}-fold{id} (make_run_tag in
# src/predictor/training.py). The git segment is [^-]+ rather than hex because the
# training UI falls back to the literal "nogit" when git is unavailable.
_RUN_TAG_RE = re.compile(r"^([^-\s]+)-s([0-9a-f]{8})-c([0-9a-f]{8})-fold(\d+)$")

_META_KEYS = (
    "lookback",
    "constants_sha256",
    "trained_through_ts_utc",
    "train_q90_coverage",
    "target_semantics",
)


class RunTag(NamedTuple):
    git_sha: str
    scaler_sha8: str
    constants_sha8: str
    fold_index: int


def parse_run_tag(stem: str) -> RunTag | None:
    """Parse a checkpoint filename stem into its provenance parts (None if not a tag)."""
    match = _RUN_TAG_RE.match(stem)
    if match is None:
        return None
    return RunTag(match.group(1), match.group(2), match.group(3), int(match.group(4)))


def json_safe(value: object) -> object:
    """Recursively replace non-finite floats with None — strict-JSON for the browser."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def _json_safe_dict(value: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], json_safe(value))


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


# --------------------------------------------------------------------------- scanning

def scan_checkpoints(checkpoint_dir: Path) -> list[dict[str, object]]:
    """All *.pt checkpoints in the dir (non-recursive): stem, path, scaler pairing, tag."""
    if not checkpoint_dir.is_dir():
        return []
    entries: list[dict[str, object]] = []
    for weights in sorted(checkpoint_dir.glob(f"*{PREDICTOR.CHECKPOINT_WEIGHTS_SUFFIX}")):
        stem = weights.name[: -len(PREDICTOR.CHECKPOINT_WEIGHTS_SUFFIX)]
        scaler = checkpoint_dir / f"{stem}{PREDICTOR.CHECKPOINT_SCALER_SUFFIX}"
        tag = parse_run_tag(stem)
        entries.append(
            {
                "stem": stem,
                "weights_path": str(weights),
                "scaler_path": str(scaler),
                "scaler_present": scaler.exists(),
                "tag": tag,
            }
        )
    return entries


def read_checkpoint_meta(weights_path: Path) -> dict[str, object]:
    """Embedded provenance from the wrapped checkpoint dict (weights_only-safe).

    Missing keys come back as None (pre-format or foreign checkpoints) — the
    compatibility check downstream turns that into an explicit incompatible reason
    rather than a crash.
    """
    loaded = torch.load(weights_path, map_location="cpu", weights_only=True)
    raw: dict[str, object] = loaded if isinstance(loaded, dict) else {}
    return {key: raw.get(key) for key in _META_KEYS}


# --------------------------------------------------------------------------- registry file

def _registry_path(benchmark_dir: Path) -> Path:
    return benchmark_dir / BENCHMARK.REGISTRY_FILENAME


def load_registry(benchmark_dir: Path) -> dict[str, dict[str, object]]:
    path = _registry_path(benchmark_dir)
    if not path.exists():
        return {"display_names": {}, "meta_cache": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "display_names": dict(data.get("display_names", {})),
        "meta_cache": dict(data.get("meta_cache", {})),
    }


def save_registry(benchmark_dir: Path, registry: dict[str, dict[str, object]]) -> None:
    _atomic_write_json(_registry_path(benchmark_dir), registry)


def set_display_name(benchmark_dir: Path, stem: str, display_name: str) -> str:
    """Store (or clear, when blank) the user-facing name for a checkpoint stem.

    Returns the effective display name (the stem itself when cleared). Only the
    registry JSON changes — checkpoint files are never touched.
    """
    registry = load_registry(benchmark_dir)
    name = display_name.strip()
    if name:
        registry["display_names"][stem] = name
    else:
        registry["display_names"].pop(stem, None)
    save_registry(benchmark_dir, registry)
    return name or stem


# --------------------------------------------------------------------------- results

def benchmark_result_path(benchmark_dir: Path, stem: str) -> Path:
    return benchmark_dir / f"{stem}{BENCHMARK.RESULT_SUFFIX}"


def read_benchmark_result(benchmark_dir: Path, stem: str) -> dict[str, object] | None:
    path = benchmark_result_path(benchmark_dir, stem)
    if not path.exists():
        return None
    loaded: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def write_benchmark_result(
    benchmark_dir: Path, stem: str, record: dict[str, object]
) -> Path:
    """Persist one model's benchmark record (adds the UTC stamp; atomic; strict JSON)."""
    stamped = dict(record)
    stamped["benchmarked_at_utc"] = (
        datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    path = benchmark_result_path(benchmark_dir, stem)
    _atomic_write_json(path, stamped)
    return path


# --------------------------------------------------------------------------- joined view

def _compatibility(
    entry: dict[str, object], meta: dict[str, object]
) -> tuple[bool, str | None]:
    if not entry["scaler_present"]:
        return False, "paired .scaler.pkl missing — weights unusable without their scaler"
    if entry["tag"] is None:
        return False, "filename is not a run tag — cannot locate its walk-forward fold"
    if meta.get("error") is not None:
        return False, f"checkpoint unreadable: {meta['error']}"
    semantics = meta.get("target_semantics")
    if semantics != PREDICTOR.TARGET_SEMANTICS:
        return False, (
            f"target_semantics {semantics!r} != current {PREDICTOR.TARGET_SEMANTICS!r} — "
            f"the straddle-gate rule is undefined outside cumulative-path quantiles"
        )
    if not isinstance(meta.get("lookback"), int):
        return False, "checkpoint has no embedded lookback"
    return True, None


def _current_constants_sha8() -> str | None:
    """First 8 hex of the running constants.py SHA256 (the run-tag's `c` segment), or
    None if it can't be read. Compared against each checkpoint's embedded constants hash
    to flag — informationally, never as a hard gate — models trained under a different
    constants.py than the one now defining the trading rule's fee/horizon/quantiles."""
    try:
        return sha256_file(REPO_ROOT / "constants.py")[:8]
    except OSError:
        return None


def list_models(checkpoint_dir: Path, benchmark_dir: Path) -> list[dict[str, object]]:
    """The joined model view the API serves: scan + cached meta + names + result state."""
    registry = load_registry(benchmark_dir)
    meta_cache = registry["meta_cache"]
    current_c8 = _current_constants_sha8()
    cache_dirty = False
    models: list[dict[str, object]] = []

    for entry in scan_checkpoints(checkpoint_dir):
        stem = str(entry["stem"])
        weights_path = Path(str(entry["weights_path"]))
        mtime_ns = weights_path.stat().st_mtime_ns
        cached = meta_cache.get(stem)
        if isinstance(cached, dict) and cached.get("mtime_ns") == mtime_ns:
            meta = dict(cast(dict[str, object], cached["meta"]))
        else:
            try:
                meta = read_checkpoint_meta(weights_path)
            except Exception as exc:  # unreadable/corrupt file must not kill the listing
                meta = {"error": str(exc)}
            meta = _json_safe_dict(meta)
            meta_cache[stem] = {"mtime_ns": mtime_ns, "meta": meta}
            cache_dirty = True
        tag = entry["tag"]
        compatible, reason = _compatibility(entry, meta)
        result = read_benchmark_result(benchmark_dir, stem)
        models.append(
            {
                "stem": stem,
                "display_name": registry["display_names"].get(stem, stem),
                "fold_index": tag.fold_index if isinstance(tag, RunTag) else None,
                "git_sha": tag.git_sha if isinstance(tag, RunTag) else None,
                "constants_sha8": tag.constants_sha8 if isinstance(tag, RunTag) else None,
                "constants_match": (
                    tag.constants_sha8 == current_c8
                    if isinstance(tag, RunTag) and current_c8 is not None
                    else None
                ),
                "scaler_present": entry["scaler_present"],
                "meta": meta,
                "compatible": compatible,
                "incompatible_reason": reason,
                "has_benchmark": result is not None,
                "result": result,
            }
        )

    if cache_dirty:
        save_registry(benchmark_dir, registry)
    return [_json_safe_dict(m) for m in models]
