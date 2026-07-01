"""search_predictor.py — autoresearch-style hyperparameter/architecture search loop
for the predictor (see DECISIONS.md 'search_dev_slice').

Each iteration: propose ONE bold, coherent structural move (see MOVE_MENU) -> train it
on the search dev-slice (scripts/train_predictor.py --search-slice) -> require the
improvement to hold across repeat seeds (Bonferroni-style noise skepticism, analogous
to 'hyperparameter_tuning_during_walk_forward' in DECISIONS.md; strict majority of
DataConfig.SEARCH_CONFIRM_SEEDS must improve) -> judge compliance with decisions-auditor
+ leakage-checker -> keep (leave constants.py edited) or revert (restore the
pre-iteration snapshot).

Model routing:
  - PROPOSER: PROPOSER_MODEL ('claude-opus-4-8'). Bolder proposals are higher-blast-radius,
    so the loop uses the strongest reasoning model. To change it, edit the PROPOSER_MODEL
    constant below. NOTE: verify the local `claude` CLI actually accepts this id —
    `claude -p --model claude-opus-4-8 "ping"` — before a real run; an unrecognized id may
    error or silently route to a default.
  - JUDGES (decisions-auditor + leakage-checker): JUDGE_MODEL ('claude-sonnet-5').
    Deliberately NOT moved with the proposer — the judges are cheap, read-only spec
    checkers, and a bold proposer needs a fast independent gate, not a co-varying one.
    Change JUDGE_MODEL only with an explicit reason.

Every LLM call pins --model explicitly — never the CLI default, which may resolve to an
older model.

Safety around constants.py (this loop mutates it directly, unattended — a corrupt write
would silently change the trained model's hyperparameters and still pass the SHA256
manifest, which just hashes whatever is on disk). Every iteration:
  1. snapshots constants.py to constants.py.searchbak on disk before any patch,
  2. wraps the whole train->judge window in try/finally so any crash/timeout/Ctrl-C
     restores the snapshot (regression guard from commit c2071f7, kept intact),
  3. re-parses the patched constants.py with ast.parse; a non-parseable edit is reverted
     immediately, before any training runs against a broken file.

State lives in search_log.jsonl (append-only) and constants.py itself, not in this
process's memory, so the loop can be killed (Ctrl-C, or create SEARCH_KILL_SWITCH.flag)
and resumed later without losing history. Nothing is committed to git automatically —
review the resulting constants.py diff and commit manually when satisfied.

Usage:
  uv run python scripts/search_predictor.py [--max-iterations N] [--confirm-seeds N]
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load-time snapshot, used only in main() before any patch. Do NOT read this binding
# mid-loop for a live value — after a patch+reload it is stale; use current_values().
from constants import DATA  # noqa: E402 -- needs REPO_ROOT on sys.path first

CONSTANTS_PATH = REPO_ROOT / "constants.py"
CONSTANTS_BAK_PATH = REPO_ROOT / "constants.py.searchbak"
LOG_PATH = REPO_ROOT / "search_log.jsonl"
KILL_FLAG_PATH = REPO_ROOT / "SEARCH_KILL_SWITCH.flag"

# Proposer model. Bolder moves are higher-blast-radius, so use the strongest reasoning
# model. Hardcoded here (change this one line to switch); judges stay on Sonnet below.
PROPOSER_MODEL = "claude-opus-4-8"
# Judges stay on Sonnet on purpose (see module docstring). Do not move with the proposer.
JUDGE_MODEL = "claude-sonnet-5"
JUDGE_ALLOWED_TOOLS = "Read Grep Glob"  # read-only; no headless call may ever mutate files

# Search space (hard bounds). Not a DECISIONS.md-locked choice (adjust freely) except
# DIRECTION_PENALTY_LAMBDA ([1.5, 2.0]) and LOOKBACK ({240,720,1440}), whose bounds are
# fixed elsewhere. Each field is tagged with its owning config singleton so the patcher
# and the current-value reader touch the right dataclass (LOOKBACK lives on DataConfig,
# the rest on PredictorConfig).
SEARCH_SPACE: dict[str, dict[str, Any]] = {
    # --- optimizer / schedule ---
    "LEARNING_RATE": {"kind": "float", "low": 1e-5, "high": 3e-3, "config": "PREDICTOR"},
    "WEIGHT_DECAY": {"kind": "float", "low": 0.0, "high": 0.1, "config": "PREDICTOR"},
    "WARMUP_FRAC": {"kind": "float", "low": 0.0, "high": 0.2, "config": "PREDICTOR"},
    "DROPOUT": {"kind": "float", "low": 0.0, "high": 0.5, "config": "PREDICTOR"},
    # --- loss shape ---
    "DIRECTION_PENALTY_LAMBDA": {"kind": "float", "low": 1.5, "high": 2.0, "config": "PREDICTOR"},
    # --- capacity (moved together, see CAPACITY move) ---
    "D_MODEL": {"kind": "choice", "choices": [64, 128, 192, 256], "config": "PREDICTOR"},
    "N_HEADS": {"kind": "choice", "choices": [2, 4, 8, 16], "config": "PREDICTOR"},
    "N_LAYERS": {"kind": "choice", "choices": [2, 3, 4, 5, 6], "config": "PREDICTOR"},
    "D_FF": {"kind": "choice", "choices": [128, 256, 384, 512, 768, 1024], "config": "PREDICTOR"},
    # --- tokenization / receptive field ---
    # PATCH_SIZE must divide LOOKBACK (PatchTST tokenizes lookback/patch tokens).
    "PATCH_SIZE": {"kind": "choice", "choices": [8, 12, 16, 24, 32, 48], "config": "PREDICTOR"},
    "LOOKBACK": {"kind": "choice", "choices": [240, 720, 1440], "config": "DATA"},
}

# Boldness is STRUCTURAL, not "randomize more numbers". Each iteration commits to exactly
# ONE named move from this menu — a single attributable lever that is nonetheless a large,
# coherent step (e.g. resize the whole transformer at once rather than nudging D_FF alone).
# Single-lever moves keep the confirm-seed noise defense interpretable: when a move
# survives repeat seeds you know *which* structural change earned it.
MOVE_MENU: dict[str, str] = {
    "CAPACITY": (
        "Resize the transformer as one coherent block: pick a new (D_MODEL, N_HEADS, "
        "N_LAYERS, D_FF) together (D_MODEL divisible by N_HEADS; D_FF ~2-4x D_MODEL). "
        "Take a BIG step in capacity (e.g. 128->256 or 128->64), not a one-notch nudge."
    ),
    "TOKENIZATION": (
        "Change PATCH_SIZE to reshape how the lookback window is tokenized (larger patch "
        "= fewer, coarser tokens; smaller = more, finer). PATCH_SIZE must divide LOOKBACK."
    ),
    "RECEPTIVE_FIELD": (
        "Sweep LOOKBACK to {240, 720, 1440}. This changes how much history the model "
        "sees. Keep PATCH_SIZE a divisor of the new LOOKBACK (adjust PATCH_SIZE in the "
        "same move if needed to preserve divisibility)."
    ),
    "LOSS_SHAPE": (
        "Move DIRECTION_PENALTY_LAMBDA decisively within [1.5, 2.0] to re-weight the "
        "directional term against the pinball term."
    ),
    "SCHEDULE": (
        "Reshape the optimizer schedule: take a bold combined step on LEARNING_RATE and "
        "WARMUP_FRAC (e.g. an order-of-magnitude LR change with a matching warmup change) "
        "to escape a schedule-induced local minimum."
    ),
    "REGULARIZATION": (
        "Take a decisive combined step on DROPOUT and WEIGHT_DECAY to move the "
        "over/under-fitting regime, not a small nudge."
    ),
}


def read_constants_text() -> str:
    return CONSTANTS_PATH.read_text(encoding="utf-8")


def patch_constants(text: str, fields: dict[str, Any]) -> str:
    """Regex-substitute config field values in constants.py source text.

    The value plus any pre-existing inline `# comment` on the line is consumed and
    replaced by the new value + a single `# search_predictor.py` marker. Consuming the old
    comment matters: leaving it would produce doubled `# ` markers and, worse, a stale
    self-contradicting annotation (e.g. `N_HEADS: int = 4  # head_dim = ... = 16`) in the
    diff the human reviews before the keep decision.
    """
    for name, value in fields.items():
        pattern = re.compile(
            rf"(\b{re.escape(name)}\s*:\s*[\w.\[\], ]+?\s*=\s*)[^\n#]+(?:#[^\n]*)?"
        )
        if not pattern.search(text):
            raise ValueError(f"field {name} not found in constants.py")
        replacement = repr(value)

        def _replace(m: re.Match[str], r: str = replacement) -> str:
            return m.group(1) + r + "  # search_predictor.py"

        text = pattern.sub(_replace, text, count=1)
    return text


def current_values() -> dict[str, Any]:
    import constants
    importlib.reload(constants)
    singletons = {"PREDICTOR": constants.PREDICTOR, "DATA": constants.DATA}
    return {
        name: getattr(singletons[spec["config"]], name)
        for name, spec in SEARCH_SPACE.items()
    }


def append_log(entry: dict[str, Any]) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def read_log() -> list[dict[str, Any]]:
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def current_best(log: list[dict[str, Any]]) -> float | None:
    kept = [e["val_total"] for e in log if e.get("kept") and e.get("val_total") is not None]
    return min(kept) if kept else None


def claude_call(prompt: str, *, model: str, agent: str | None = None) -> str:
    # --allowedTools is a variadic flag: it greedily consumes every following
    # non-flag argument, so the prompt must come before it or it gets swallowed.
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "json"]
    if agent:
        cmd += ["--agent", agent]
    cmd += ["--allowedTools", JUDGE_ALLOWED_TOOLS]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"claude -p failed (exit {result.returncode}): {result.stderr[-2000:]}")
    payload = json.loads(result.stdout)
    return str(payload["result"])


def extract_json_object(text: str) -> dict[str, Any]:
    """The model sometimes wraps JSON in prose/fences despite instructions; grab the
    outermost {...} block."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in response: {text[:500]}")
    return dict(json.loads(match.group(0)))


def propose(
    history: list[dict[str, Any]],
    current: dict[str, Any],
    best: float | None,
) -> dict[str, Any]:
    prompt = f"""You are proposing ONE candidate change for a PatchTST BTC predictor.
Candidates are trained on a small dev slice and compared by val_total (pinball +
direction-penalty loss; lower is better). Your job is to ESCAPE LOCAL MINIMA, so make a
BOLD, COHERENT structural move — not a timid nudge of a single number.

Pick EXACTLY ONE move from this menu and execute it decisively:
{json.dumps(MOVE_MENU, indent=2)}

Rules for a good proposal:
- Commit to ONE move type. Only change the fields that move touches (a CAPACITY move may
  set all of D_MODEL/N_HEADS/N_LAYERS/D_FF; a LOSS_SHAPE move changes only
  DIRECTION_PENALTY_LAMBDA). This keeps the result attributable to one lever.
- Make it BIG. Take a real step, not one notch. A change that only survives noise if it
  is tiny is not worth an iteration.
- Every value must stay within the search-space hard bounds below.
- Do not repeat a combination already present in the history.

Search space (hard bounds; each field notes which config owns it):
{json.dumps(SEARCH_SPACE, indent=2)}

Hard constraints (a violating proposal is auto-rejected before training):
- D_MODEL must be evenly divisible by N_HEADS.
- PATCH_SIZE must evenly divide LOOKBACK.

Current values: {json.dumps(current)}
Current best val_total so far: {best}
Recent iteration history, most recent last (kept=false means reverted):
{json.dumps(history[-8:], indent=2)}

Respond with ONLY a raw JSON object, no markdown fences, no prose. Schema:
{{"move": "ONE_MENU_KEY", "changes": {{"FIELD": value, ...}}, "hypothesis": "one sentence"}}"""
    return extract_json_object(claude_call(prompt, model=PROPOSER_MODEL))


def validate_candidate(changes: dict[str, Any], current: dict[str, Any]) -> str | None:
    if not changes:
        return "empty candidate"
    merged = {**current, **changes}
    for name, value in changes.items():
        if name not in SEARCH_SPACE:
            return f"{name} is not in the search space"
        spec = SEARCH_SPACE[name]
        if spec["kind"] == "choice" and value not in spec["choices"]:
            return f"{name}={value} not in allowed choices {spec['choices']}"
        if spec["kind"] == "float" and not (spec["low"] <= value <= spec["high"]):
            return f"{name}={value} out of bounds [{spec['low']}, {spec['high']}]"
    if merged["D_MODEL"] % merged["N_HEADS"] != 0:
        return f"D_MODEL={merged['D_MODEL']} not divisible by N_HEADS={merged['N_HEADS']}"
    if merged["LOOKBACK"] % merged["PATCH_SIZE"] != 0:
        return f"LOOKBACK={merged['LOOKBACK']} not divisible by PATCH_SIZE={merged['PATCH_SIZE']}"
    return None


def run_training(seed: int) -> float | None:
    cmd = [
        "uv", "run", "python", "scripts/train_predictor.py",
        "--search-slice", "--wandb", "disabled", "--no-save", "--seed", str(seed),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800)
    val_total = None
    for line in result.stdout.splitlines():
        if line.startswith("[done]"):
            m = re.search(r"val_total=([\d.]+)", line)
            if m:
                val_total = float(m.group(1))
    if val_total is None:
        print(f"[iter] training run failed or produced no val_total (seed={seed})")
        print(f"       stderr tail: {result.stderr[-1000:]}")
    return val_total


def judge(agent_name: str, changes: dict[str, Any], move: str) -> tuple[bool, str]:
    prompt = f"""constants.py was just changed by an automated search loop.
Move type: {move} (the fields below are one coherent structural move, not independent edits).
Changes: {json.dumps(changes)}

Review the current state of constants.py against DECISIONS.md and the relevant context
cards. Does this change violate any locked decision, introduce a value outside a
documented bound, or otherwise break project spec? Respond with ONLY a raw JSON object,
no markdown fences, no prose: {{"ok": true|false, "reason": "one sentence"}}"""
    verdict = extract_json_object(claude_call(prompt, model=JUDGE_MODEL, agent=agent_name))
    return bool(verdict.get("ok")), str(verdict.get("reason", ""))


def run_iteration(iteration: int, confirm_seeds: int) -> None:
    log = read_log()
    current = current_values()
    best = current_best(log)

    proposal = propose(log, current, best)
    changes = proposal.get("changes", {})
    move = proposal.get("move", "?")
    print(f"[propose] move={move} {changes} -- {proposal.get('hypothesis', '')}")

    reason = validate_candidate(changes, current)
    if reason is not None:
        print(f"[reject] invalid candidate: {reason}")
        append_log({"iteration": iteration, "move": move, "changes": changes,
                    "kept": False, "reason": reason})
        return

    # Snapshot to disk BEFORE any mutation. This is the deterministic restore source; the
    # in-memory original_text is a second belt-and-braces copy.
    original_text = read_constants_text()
    CONSTANTS_BAK_PATH.write_text(original_text, encoding="utf-8")
    patched_text = patch_constants(original_text, changes)
    CONSTANTS_PATH.write_text(patched_text, encoding="utf-8")
    resolved = False  # set True once the candidate is explicitly kept or reverted

    def restore() -> None:
        # Prefer the on-disk snapshot; fall back to the in-memory copy if it vanished.
        source = (
            CONSTANTS_BAK_PATH.read_text(encoding="utf-8")
            if CONSTANTS_BAK_PATH.exists()
            else original_text
        )
        CONSTANTS_PATH.write_text(source, encoding="utf-8")

    def revert(reason: str, **extra: Any) -> None:
        nonlocal resolved
        restore()
        resolved = True
        print(f"[reject] {reason}")
        append_log(
            {"iteration": iteration, "move": move, "changes": changes,
             "kept": False, "reason": reason, **extra}
        )

    try:
        # Guard against a malformed edit reaching training: a bold proposal raises the odds
        # of a non-parseable constants.py. Revert immediately if it will not even parse.
        try:
            ast.parse(patched_text)
        except SyntaxError as exc:
            revert(f"patched constants.py did not parse: {exc}")
            return

        first_val = run_training(seed=0)
        if first_val is None or (best is not None and first_val >= best):
            revert("no improvement on first seed", val_total=first_val)
            return

        seed_vals = [first_val]
        for seed in range(1, confirm_seeds):
            v = run_training(seed=seed)
            if v is not None:
                seed_vals.append(v)
        improved = sum(1 for v in seed_vals if best is None or v < best)
        majority = (len(seed_vals) + 1) // 2
        if improved < majority:
            revert("improvement did not hold across repeat seeds", seed_vals=seed_vals)
            return
        median_val = statistics.median(seed_vals)

        ok_da, reason_da = judge("decisions-auditor", changes, move)
        ok_lc, reason_lc = judge("leakage-checker", changes, move)
        if not (ok_da and ok_lc):
            revert(
                f"judge veto -- decisions-auditor={ok_da} ({reason_da}); "
                f"leakage-checker={ok_lc} ({reason_lc})",
                seed_vals=seed_vals,
            )
            return

        print(f"[keep] {changes} -- median val_total {median_val:.6f} (previous best: {best})")
        append_log({
            "iteration": iteration, "move": move, "changes": changes, "kept": True,
            "val_total": median_val, "seed_vals": seed_vals,
            "hypothesis": proposal.get("hypothesis", ""),
        })
        resolved = True
    finally:
        # Guards against a crash/timeout/Ctrl-C between the write above and a keep/
        # revert decision, which would otherwise leave constants.py silently patched
        # with no matching log entry. (Regression guard from commit c2071f7.)
        if not resolved:
            restore()
            print(
                "[reject] iteration interrupted before a keep/revert decision -- "
                "restored constants.py"
            )
            # Leave an audit-trail entry so the durable history (search_log.jsonl) records
            # the attempt; propose() feeds this history back so it won't blindly retry the
            # interrupted combination. Best-effort: a logging failure must not mask the
            # restore above.
            with contextlib.suppress(Exception):
                append_log({
                    "iteration": iteration, "move": move, "changes": changes,
                    "kept": False, "reason": "interrupted before keep/revert decision",
                })
        # Snapshot is transient scratch; clear it so a stale .searchbak can never be
        # mistaken for a live restore source on a later run.
        CONSTANTS_BAK_PATH.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-iterations", type=int, default=None, dest="max_iterations")
    ap.add_argument(
        "--confirm-seeds", type=int, default=DATA.SEARCH_CONFIRM_SEEDS, dest="confirm_seeds",
        help="repeat-seed runs required before a first-seed improvement is trusted "
        "(DECISIONS.md 'search_confirm_seeds'; default = DataConfig.SEARCH_CONFIRM_SEEDS)",
    )
    args = ap.parse_args()
    print(f"[config] proposer={PROPOSER_MODEL} judges={JUDGE_MODEL} "
          f"confirm_seeds={args.confirm_seeds}")

    iteration = 0
    while args.max_iterations is None or iteration < args.max_iterations:
        if KILL_FLAG_PATH.exists():
            print(f"[stop] {KILL_FLAG_PATH.name} present -- stopping.")
            break
        iteration += 1
        print(f"\n=== iteration {iteration} ===")
        try:
            run_iteration(iteration, args.confirm_seeds)
        except Exception as exc:  # noqa: BLE001 -- keep the loop alive; log and move on
            print(f"[error] iteration {iteration} failed: {exc}")
            append_log({"iteration": iteration, "kept": False, "reason": f"exception: {exc}"})

    print(
        f"\n[done] {iteration} iteration(s). constants.py reflects the current best "
        f"(uncommitted -- review the diff and commit manually when satisfied)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
