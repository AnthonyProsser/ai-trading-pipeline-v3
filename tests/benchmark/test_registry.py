"""Tests for src/benchmark/registry.py — checkpoint discovery, display names, results.

Invariants pinned here:
  * .pt/.scaler.pkl files are NEVER renamed or touched — a user-facing rename writes a
    display name into registry.json keyed by the immutable run-tag stem (renaming the
    files would break the weights<->scaler pairing and the SHA256 manifest guarantee).
  * Checkpoints with a target_semantics different from this branch's
    PREDICTOR.TARGET_SEMANTICS are surfaced as incompatible, never silently scored
    (mirrors deploy_predictor.py's refusal).
  * A model "has a benchmark" exactly when its result JSON exists on disk — that file
    is the cache that makes the Start-benchmark button disappear.

Committed before src/benchmark/registry.py, per the test-first discipline.
"""
from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path

import torch

from constants import PREDICTOR
from src.benchmark.registry import (
    list_models,
    parse_run_tag,
    read_benchmark_result,
    read_checkpoint_meta,
    scan_checkpoints,
    set_display_name,
    write_benchmark_result,
)

_STEM = "4d460ed-s001e4348-c487b9e2e-fold33"


def _write_fake_checkpoint(
    checkpoint_dir: Path,
    stem: str = _STEM,
    *,
    semantics: str = PREDICTOR.TARGET_SEMANTICS,
    with_scaler: bool = True,
    coverage: float = 0.9,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights = checkpoint_dir / f"{stem}{PREDICTOR.CHECKPOINT_WEIGHTS_SUFFIX}"
    torch.save(
        {
            "state_dict": {},
            "lookback": 32,
            "constants_sha256": "a" * 64,
            "trained_through_ts_utc": "2020-01-05T00:00:00Z",
            "train_q90_coverage": coverage,
            "target_semantics": semantics,
        },
        weights,
    )
    if with_scaler:
        scaler = checkpoint_dir / f"{stem}{PREDICTOR.CHECKPOINT_SCALER_SUFFIX}"
        with open(scaler, "wb") as fh:
            pickle.dump({"fake": True}, fh)
    return weights


# --- parse_run_tag -----------------------------------------------------------

def test_parse_run_tag_valid() -> None:
    tag = parse_run_tag(_STEM)
    assert tag is not None
    assert tag.git_sha == "4d460ed"
    assert tag.scaler_sha8 == "001e4348"
    assert tag.constants_sha8 == "487b9e2e"
    assert tag.fold_index == 33


def test_parse_run_tag_rejects_non_tag_names() -> None:
    assert parse_run_tag("training_metrics") is None
    assert parse_run_tag("manifest") is None
    assert parse_run_tag("model-fold3") is None


# --- scan_checkpoints ----------------------------------------------------------

def test_scan_finds_pairs_and_flags_missing_scaler(tmp_path: Path) -> None:
    _write_fake_checkpoint(tmp_path)
    _write_fake_checkpoint(
        tmp_path, "4d460ed-s00c53cec-c487b9e2e-fold68", with_scaler=False
    )
    (tmp_path / "training_metrics.json").write_text("{}", encoding="utf-8")  # ignored

    entries = {e["stem"]: e for e in scan_checkpoints(tmp_path)}
    assert set(entries) == {_STEM, "4d460ed-s00c53cec-c487b9e2e-fold68"}
    assert entries[_STEM]["scaler_present"] is True
    assert entries["4d460ed-s00c53cec-c487b9e2e-fold68"]["scaler_present"] is False


def test_scan_of_missing_dir_is_empty(tmp_path: Path) -> None:
    assert scan_checkpoints(tmp_path / "nope") == []


# --- read_checkpoint_meta -------------------------------------------------------

def test_read_checkpoint_meta_roundtrip(tmp_path: Path) -> None:
    weights = _write_fake_checkpoint(tmp_path)
    meta = read_checkpoint_meta(weights)
    assert meta["lookback"] == 32
    assert meta["trained_through_ts_utc"] == "2020-01-05T00:00:00Z"
    assert meta["train_q90_coverage"] == 0.9
    assert meta["target_semantics"] == PREDICTOR.TARGET_SEMANTICS


# --- display names ---------------------------------------------------------------

def test_display_name_roundtrip_never_touches_checkpoint_files(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    benchmark_dir = checkpoint_dir / "benchmark"
    weights = _write_fake_checkpoint(checkpoint_dir)
    before = sorted(p.name for p in checkpoint_dir.iterdir())

    set_display_name(benchmark_dir, _STEM, "coverage-fix fold33")
    models = list_models(checkpoint_dir, benchmark_dir)
    assert models[0]["display_name"] == "coverage-fix fold33"
    # The immutable files were not renamed, moved, or rewritten.
    assert weights.exists()
    after = sorted(
        p.name for p in checkpoint_dir.iterdir() if p.name != "benchmark"
    )
    assert after == before


def test_display_name_defaults_to_stem_and_clears_on_empty(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    benchmark_dir = checkpoint_dir / "benchmark"
    _write_fake_checkpoint(checkpoint_dir)

    assert list_models(checkpoint_dir, benchmark_dir)[0]["display_name"] == _STEM
    set_display_name(benchmark_dir, _STEM, "My Best Model")
    set_display_name(benchmark_dir, _STEM, "")  # empty -> revert to stem
    assert list_models(checkpoint_dir, benchmark_dir)[0]["display_name"] == _STEM


def test_registry_write_is_atomic_no_tmp_left_behind(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmark"
    set_display_name(benchmark_dir, _STEM, "x")
    leftovers = [p for p in benchmark_dir.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []
    payload = json.loads((benchmark_dir / "registry.json").read_text(encoding="utf-8"))
    assert payload["display_names"][_STEM] == "x"


# --- benchmark results ------------------------------------------------------------

def test_benchmark_result_roundtrip_and_nan_sanitization(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmark"
    record = {
        "stem": _STEM,
        "trading": {"net_return": 0.01, "sharpe": float("nan")},
    }
    write_benchmark_result(benchmark_dir, _STEM, record)
    loaded = read_benchmark_result(benchmark_dir, _STEM)
    assert loaded is not None
    trading = loaded["trading"]
    assert isinstance(trading, dict)
    assert trading["net_return"] == 0.01
    # NaN must be serialized as null (strict JSON) — browsers cannot parse NaN.
    assert trading["sharpe"] is None
    raw = (benchmark_dir / f"{_STEM}.benchmark.json").read_text(encoding="utf-8")
    json.loads(raw)  # strict-parseable
    assert "NaN" not in raw
    # A UTC stamp is added; 'Z' suffix per the repo's ISO8601-UTC convention.
    assert str(loaded["benchmarked_at_utc"]).endswith("Z")


def test_read_benchmark_result_missing_is_none(tmp_path: Path) -> None:
    assert read_benchmark_result(tmp_path / "benchmark", _STEM) is None


# --- list_models (the joined view the API serves) -----------------------------------

def test_list_models_compatibility_and_has_benchmark(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    benchmark_dir = checkpoint_dir / "benchmark"
    _write_fake_checkpoint(checkpoint_dir)  # compatible
    _write_fake_checkpoint(
        checkpoint_dir, "c631b22-s0bed48d6-c683600d1-fold0", semantics="per_step_logret"
    )
    _write_fake_checkpoint(
        checkpoint_dir, "4d460ed-s00c53cec-c487b9e2e-fold68", with_scaler=False
    )

    models = {m["stem"]: m for m in list_models(checkpoint_dir, benchmark_dir)}
    assert models[_STEM]["compatible"] is True
    assert models[_STEM]["has_benchmark"] is False
    assert models["c631b22-s0bed48d6-c683600d1-fold0"]["compatible"] is False
    assert "semantics" in str(models["c631b22-s0bed48d6-c683600d1-fold0"]["incompatible_reason"])
    assert models["4d460ed-s00c53cec-c487b9e2e-fold68"]["compatible"] is False

    write_benchmark_result(benchmark_dir, _STEM, {"trading": {"net_return": 0.02}})
    models = {m["stem"]: m for m in list_models(checkpoint_dir, benchmark_dir)}
    assert models[_STEM]["has_benchmark"] is True


def test_list_models_meta_is_cached_after_first_scan(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    benchmark_dir = checkpoint_dir / "benchmark"
    weights = _write_fake_checkpoint(checkpoint_dir)
    list_models(checkpoint_dir, benchmark_dir)  # first scan populates the cache

    registry = json.loads((benchmark_dir / "registry.json").read_text(encoding="utf-8"))
    assert _STEM in registry["meta_cache"]
    # Corrupt the weights file: a cached second listing must NOT re-read it.
    weights.write_bytes(b"not a checkpoint")
    # Restore the recorded mtime so the cache stays valid despite the rewrite.
    cached_mtime = int(registry["meta_cache"][_STEM]["mtime_ns"])
    os.utime(weights, ns=(cached_mtime, cached_mtime))
    models = list_models(checkpoint_dir, benchmark_dir)
    meta = models[0]["meta"]
    assert isinstance(meta, dict)
    assert meta["lookback"] == 32


def test_list_models_nan_coverage_serializes_to_null(tmp_path: Path) -> None:
    # Manual (mid-fold "Save") checkpoints embed train_q90_coverage = NaN; the joined
    # view must be strict-JSON-safe for the browser.
    checkpoint_dir = tmp_path / "checkpoints"
    benchmark_dir = checkpoint_dir / "benchmark"
    _write_fake_checkpoint(checkpoint_dir, coverage=float("nan"))
    models = list_models(checkpoint_dir, benchmark_dir)
    meta = models[0]["meta"]
    assert isinstance(meta, dict)
    coverage = meta["train_q90_coverage"]
    assert coverage is None or (isinstance(coverage, float) and not math.isnan(coverage))
    json.dumps(models, allow_nan=False)  # must not raise
