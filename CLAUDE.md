# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions below.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project-specific instructions

### Identity

Solo paper-trading BTC bot. Three components: **Predictor** (PatchTST encoder, 1,440-candle lookback, multi-step quantile forecast), **Trader** (rules-based, uncertainty-aware sizing, exchange-native stops), **Dashboard** (FastAPI + vanilla JS, read-only telemetry + kill-switch button). Single user, Windows, RTX 4060 8GB. **No real capital until paper-trading gates pass and statistical edge is proven by permutation test.**

### Repo state

**Build phase (pre-training gate).** Building phase by phase toward a green smoke run.

- **Phase 0 (Data)** — COMPLETE, merged to `main`. `src/data/` (feature_pipeline, scaler, walk_forward, validator, manifest) and matching `tests/data/` are fully green.
- **Phase 1 (Predictor)** — modules COMPLETE on `phase-1-predictor` (PR pending). All built tests-first and green (68 tests): `early_stopping.py`, `loss.py`, `rollout.py` (geometry enforcement), `model.py` (channel-mixing PatchTST), `training.py` (`build_fold_loaders` + `train_one_fold`), `deploy_gates.py`; plus `scripts/train_predictor.py` (`--smoke`/`--synthetic`) and `scripts/deploy_predictor.py`. The **synthetic** GPU smoke passed on the 4060 (lookback 1440, batch 32, finite loss, no OOM/NaN). The **real-data** smoke is still pending: `data/raw/BTCUSD_1.csv` is absent (Google Drive ingest rate-limited; re-run `scripts/ingest_kraken_history.py` after cooldown, then `uv run python scripts/train_predictor.py --smoke`). `deploy_predictor.py` runs end-to-end only once a trained checkpoint + locked test set exist. W&B is not installed (offline-capable flag wired; `uv add wandb` to enable).
- **Phases 2–4** — not started. `src/execution/`, `src/trader/`, `src/dashboard/` do not exist.

`scripts/ingest_kraken_history.py`, `scripts/train_predictor.py`, and `scripts/deploy_predictor.py` exist. Other scripts (e.g. `backtest.py`, `holdout_evaluator.py`, `permutation_test.py`) are planned deliverables — they do not exist yet.

Read first, every coding session, in this order: `DECISIONS.md` → `INDEX.md` → `constants.py` → the relevant context card named by the matching INDEX row. They are the source of truth for architectural decisions, task → file mapping, and frozen magic numbers respectively.

### Tech stack

- **UV** (package manager — `uv sync` / `uv add` / `uv run`; never bare `pip install`)
- Python 3.x with full type hints; `mypy --strict` on PR
- PyTorch (PatchTST encoder, patch_size=16)
- pandas, numpy (data pipeline)
- FastAPI + vanilla JS + Lightweight Charts (dashboard, bound to `127.0.0.1` only)
- pytest (`tests/` mirrors `src/` exactly)
- structlog (structured JSON logs)
- Weights & Biases (training tracking)
- SQLite (paper) / PostgreSQL+WAL (live)
- Kraken WebSocket v2 + REST (data + trading)
- Windows Credential Manager via `keyring` (secrets — never `.env`)

### Project-specific rules (override Karpathy defaults where they conflict)

- **Tests first** for any code under `src/`. Failing tests encoding the exit criteria are committed before implementation. This overrides Karpathy §4's "verify after"; verification is built in from the start.
- **Magic numbers live in `constants.py` only**, inside frozen dataclasses. No bare module-level constants. No hardcoded thresholds like `if x > 0.62:`.
- **`src/` never imports from `scripts/`.** Scripts call into `src/`, never the inverse.
- **`data/test_locked/` never referenced from `src/`.** Enforced by `grep -r 'test_locked' src/` plus `sys.settrace` runtime guard during training.
- **All timestamps UTC.** `datetime.now()` without `tz=timezone.utc` is a CI failure in `src/`.
- **Exchange-native stop-loss is mandatory.** Execution refuses any order whose stop cannot be placed at Kraken. No naked positions under any code path.
- **SHA256 manifest verified on every weight read.** Covers weights + scaler + `constants.py`.
- **`asyncio` only in `src/execution/`.** Predictor and trader code stays synchronous.
- **One branch per phase:** `phase-XY` off `developer`; `developer` → `main` at phase exit. Never commit to `main` directly.
- **Any `DECISIONS.md` change requires a `CHANGELOG.md` entry in the same commit.**
- **Flag every unspecced decision.** If a task requires a value not in `DECISIONS.md` or `constants.py`, stop and ask. Do not "use a reasonable default" — this aligns with Karpathy §1.

### Custom agents — mandatory trigger rules

Three project-specific subagents live in `.claude/agents/`. They are read-only auditors; invoke them via the Agent tool at the triggers below. Do not skip them.

| Agent | When to invoke |
|---|---|
| `decisions-auditor` | After implementing or editing **any** module under `src/` or `scripts/`. Also whenever a constant, formula, threshold, schema, or I/O shape appears in a diff. |
| `leakage-checker` | Before any Phase 0 exit. On **any** change to `src/data/` (feature pipeline, scaler, walk-forward, validator). Pre-merge on PRs touching `src/data/`. |
| `test-enforcer` | At the **start** of every implementation task (confirms test-first ordering) and at **phase completion** (confirms mirror completeness). |

### ECC integrations

The `everything-claude-code` plugin is installed. Use these capabilities where they apply:

**Agents (invoke via Agent tool):**
| Agent | When to use |
|---|---|
| `everything-claude-code:python-reviewer` | After writing or editing any Python source under `src/` — catches type-hint gaps, Pythonic issues, security, and PEP 8 drift. Run after `decisions-auditor`. |
| `everything-claude-code:pytorch-build-resolver` | When a PyTorch training run or inference crashes — shape mismatches, device errors, gradient issues, DataLoader failures. |
| `everything-claude-code:security-reviewer` | When touching `src/execution/` (API keys, WebSocket, order submission) or any code that reads external input. |
| `everything-claude-code:performance-optimizer` | When training is slow or a DataLoader is bottlenecked — before concluding hardware is the constraint. |
| `everything-claude-code:tdd-guide` | When writing new `src/` modules — enforces the tests-first mandate and 80%+ coverage. |
| `everything-claude-code:docs-lookup` | When needing current API docs for PyTorch, pandas, FastAPI, or Kraken SDK — fetches live docs rather than relying on stale training knowledge. |

**Skills (invoke via Skill tool):**
| Skill | When to use |
|---|---|
| `everything-claude-code:pytorch-patterns` | When designing or debugging the PatchTST encoder, quantile loss, training loop, or rollout. |
| `everything-claude-code:python-review` | Quick per-file Python review (lighter-weight alternative to the full agent). |
| `everything-claude-code:tdd` | TDD workflow guidance when implementing a new phase module. |
| `everything-claude-code:security-review` | Security scan on `src/execution/` changes. |
| `code-review` (`/code-review`) | After any implementation diff — reviews the current branch changes for correctness bugs and simplification opportunities. |

### Commands (when toolchain lands)

`scripts/ingest_kraken_history.py` is currently present; other script/module commands below are planned phase deliverables tracked in `INDEX.md` and become runnable as those files land.

| Task | Command |
|---|---|
| Install / sync deps | `uv sync` |
| Add a dependency | `uv add <package>` |
| Test suite | `uv run pytest` |
| Single test | `uv run pytest tests/path/test_file.py::test_name` |
| Type check | `uv run mypy` |
| Leakage audit | `grep -r 'test_locked' src/` (must return nothing) |
| Predictor smoke (1 epoch, 1 fold) | `uv run python scripts/train_predictor.py --smoke` |
| Backtest | `uv run python scripts/backtest.py` |
| Walk-forward holdout (12×1w gate) | `uv run python scripts/holdout_evaluator.py` |
| Permutation test (pre-live gate) | `uv run python scripts/permutation_test.py` |
| Dashboard | `uv run python -m src.dashboard.main` |
| Execution loop | `uv run python -m src.execution.loop` |
| Bootstrap ingest | `uv run python scripts/ingest_kraken_history.py` |
