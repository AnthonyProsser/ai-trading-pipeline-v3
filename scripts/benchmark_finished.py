"""Headless benchmark scorer for the current branch's finished checkpoints.

Scores every checkpoint in FINISHED_DIR whose embedded ``constants_sha256`` matches the
running constants.py, writing ``{stem}.benchmark.json`` (what the benchmark app reads).
The constants-hash gate is REQUIRED, not optional: clock- and multi-scale-feature
checkpoints share tensor shapes (both 9 input features), so a state_dict shape check
would silently load one under the other's ``compute_features`` and score garbage. Only
same-recipe checkpoints are scored; the rest are skipped (they belong to other branches).

Idempotent: a stem whose result JSON already exists is skipped unless --force.

Usage: checkout the branch a run was trained on, then
    uv run python scripts/benchmark_finished.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from constants import BENCHMARK, PREDICTOR
from src.benchmark.engine import run_benchmark_job
from src.benchmark.registry import parse_run_tag, read_checkpoint_meta
from src.data.manifest import sha256_file


class _Runner:
    """Minimal RunnerLike: run_benchmark_job only touches these three members."""

    def __init__(self, checkpoint_dir: Path, benchmark_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.benchmark_dir = benchmark_dir

    def broadcast(self, payload: dict[str, object]) -> None:
        if payload.get("type") == "benchmark_complete":
            print(f"  scored {payload['stem']} net={payload['net_return']} "
                  f"trades={payload['trade_count']} p={payload['p_value']}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-score even if result JSON exists")
    ap.add_argument("--ignore-constants", action="store_true",
                    help="skip constants-sha gate; safe only when the head/arch shape is "
                         "unique (e.g. a distinct HORIZON) so load_state_dict rejects other recipes")
    args = ap.parse_args()

    finished = REPO_ROOT / PREDICTOR.FINISHED_DIR
    bench_dir = REPO_ROOT / BENCHMARK.BENCHMARK_DIR
    bench_dir.mkdir(parents=True, exist_ok=True)
    cur_sha = sha256_file(REPO_ROOT / "constants.py")
    runner = _Runner(finished, bench_dir)

    weights = sorted(finished.glob(f"*{PREDICTOR.CHECKPOINT_WEIGHTS_SUFFIX}"))
    scored = skipped_recipe = skipped_done = failed = 0
    for w in weights:
        stem = w.name[: -len(PREDICTOR.CHECKPOINT_WEIGHTS_SUFFIX)]
        if parse_run_tag(stem) is None:
            continue
        if (bench_dir / f"{stem}{BENCHMARK.RESULT_SUFFIX}").exists() and not args.force:
            skipped_done += 1
            continue
        meta = read_checkpoint_meta(w)
        if not args.ignore_constants and meta.get("constants_sha256") != cur_sha:
            skipped_recipe += 1  # different recipe (branch) — not scoreable under current code
            continue
        try:
            run_benchmark_job(runner, stem)
            scored += 1
        except Exception as exc:  # noqa: BLE001 — one bad checkpoint must not stop the batch
            failed += 1
            print(f"  [skip] {stem}: {exc}", flush=True)
    print(f"[benchmark_finished] scored={scored} already_done={skipped_done} "
          f"other_recipe={skipped_recipe} failed={failed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
