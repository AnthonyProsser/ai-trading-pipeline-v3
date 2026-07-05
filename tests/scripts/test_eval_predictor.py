"""Tests for scripts/eval_predictor.py — the predictor training bake-off harness.

After the benchmark app landed, this script keeps only the ``train-eval`` role, so
only its seed-aggregation logic (``aggregate_seeds``) is unit-tested here. The reviewed
metric layer (``target_to_model_space`` / ``statistical_metrics`` / ``excursion_metrics``)
moved to ``src/benchmark/metrics.py`` and is tested in ``tests/benchmark/test_metrics.py``;
the ``format_comparison`` table went with the retired ``compare`` subcommand.

Torch orchestration (training against a real fold) is exercised end-to-end on GPU by the
operator; ``aggregate_seeds`` is pinned here with hand-built records so a regression in the
NaN-tolerant aggregation is caught without a training run.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_PATH = REPO_ROOT / "scripts" / "eval_predictor.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("eval_predictor_under_test", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def ep() -> Any:
    return _load_module()


# --- aggregate_seeds -------------------------------------------------------

def test_aggregate_seeds_mean_and_std(ep: Any) -> None:
    seed_results = [
        {"statistical": {"q90_coverage": 0.90, "pinball": 0.02},
         "economic": {"mean_captured_fraction": 0.4, "n_used": 100}},
        {"statistical": {"q90_coverage": 0.92, "pinball": 0.04},
         "economic": {"mean_captured_fraction": 0.6, "n_used": 100}},
    ]
    agg = ep.aggregate_seeds("main", "per_step_logret", seed_results)
    assert agg["label"] == "main"
    assert agg["semantics"] == "per_step_logret"
    assert agg["seeds"] == 2
    # Means over the two seeds.
    assert agg["statistical"]["q90_coverage"] == pytest.approx(0.91)
    assert agg["economic"]["mean_captured_fraction"] == pytest.approx(0.5)
    # Population std (ddof=0): std([0.90, 0.92]) = 0.01.
    assert agg["statistical_std"]["q90_coverage"] == pytest.approx(0.01, abs=1e-9)
    assert agg["economic_std"]["mean_captured_fraction"] == pytest.approx(0.1, abs=1e-9)


def test_aggregate_seeds_skips_nan_seed(ep: Any) -> None:
    # A seed with zero tradeable windows yields NaN economic metrics; it must be skipped,
    # not allowed to poison the mean across the other seeds.
    seed_results = [
        {"statistical": {"q90_coverage": 0.90},
         "economic": {"mean_captured_fraction": float("nan"), "n_used": 0}},
        {"statistical": {"q90_coverage": 0.90},
         "economic": {"mean_captured_fraction": 0.5, "n_used": 100}},
    ]
    agg = ep.aggregate_seeds("main", "per_step_logret", seed_results)
    assert agg["economic"]["mean_captured_fraction"] == pytest.approx(0.5)  # NaN seed skipped
    # n_used is a count (fold-fixed), reported once — never averaged or std'd.
    assert agg["economic"]["n_used"] == 100
    assert "n_used" not in agg["economic_std"]


def test_aggregate_seeds_includes_timing_group(ep: Any) -> None:
    # A `timing` group (train speed) is aggregated like any other group when present, and
    # its absence must not break aggregation.
    seed_results = [
        {"statistical": {"q90_coverage": 0.90}, "economic": {"mean_captured_fraction": 0.4},
         "timing": {"wall_seconds": 10.0, "seconds_per_epoch": 1.0, "epochs": 10.0}},
        {"statistical": {"q90_coverage": 0.90}, "economic": {"mean_captured_fraction": 0.6},
         "timing": {"wall_seconds": 12.0, "seconds_per_epoch": 1.2, "epochs": 10.0}},
    ]
    agg = ep.aggregate_seeds("new", "cumulative_logret", seed_results)
    assert agg["timing"]["wall_seconds"] == pytest.approx(11.0)
    assert agg["timing"]["seconds_per_epoch"] == pytest.approx(1.1)
    assert agg["timing_std"]["wall_seconds"] == pytest.approx(1.0, abs=1e-9)


def test_aggregate_seeds_single_seed_zero_std(ep: Any) -> None:
    seed_results = [
        {"statistical": {"q90_coverage": 0.9}, "economic": {"mean_captured_fraction": 0.5}},
    ]
    agg = ep.aggregate_seeds("fable-5", "cumulative_logret", seed_results)
    assert agg["seeds"] == 1
    assert agg["statistical_std"]["q90_coverage"] == pytest.approx(0.0)
