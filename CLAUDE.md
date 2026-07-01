# CLAUDE.md

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
- **Phase 1 (Predictor)** — modules COMPLETE, merged to `main` (PR #12 merged 2026-06-29). 85 tests green. `early_stopping.py`, `loss.py`, `rollout.py`, `model.py`, `training.py`, `deploy_gates.py`; plus `scripts/train_predictor.py` (`--smoke`/`--synthetic`/`--no-save`) and `scripts/deploy_predictor.py`. Synthetic GPU smoke passed on the 4060. **Train→deploy pipeline now connected** (2026-06-29) — the three checkpoint decisions are recorded in `DECISIONS.md` (`predictor_checkpoint_dir_and_naming`, `_save_format`, `_save_policy`): `train_predictor.py` saves best-by-val-total weights + scaler to `checkpoints/{run_tag}.pt` / `.scaler.pkl` after a real-data run (synthetic runs skip the save); `save_checkpoint` lives in `src/predictor/training.py`; `deploy_predictor.py` reads them under `weights_only=True` + `pickle.load`. Round-trip verified via unit test and a synthetic smoke. **Real-data smoke PASSED** (2026-06-30) — `data/raw/XBTUSD_1.csv`, 4,642 steps, finite loss, no OOM/NaN; first real checkpoint written to `checkpoints/`. val q90 coverage baseline = 0.9880. `deploy_predictor.py` fully wired: reads `trained_through_ts_utc` + `train_q90_coverage` from the checkpoint; `--trained-through`/`--train-coverage` are optional CLI overrides; `--lookback` must match embedded value (else STOP). W&B not installed (offline-capable flag wired; `uv add wandb` to enable). **Merged to `main`** (confirmed via `origin/main`, 2026-06-30).
- **Phase 1.5 (Training Dashboard)** — modules COMPLETE (uncommitted, branch `phase-15-training-ui` off `main`). `src/training_ui/` (`setup_router.py` data gate, `exporter.py` fold-history JSON export, `app.py` FastAPI app) + `static/training_ui/` (`index.html`, `style.css`, `chart.js` hand-rolled canvas renderer, `app.js`) built end-to-end per `training-dashboard.md`, translating the Claude Design mockup (`Training Dashboard.dc.html`) into vanilla JS/CSS (no React/build step/CDN). Process model: in-process background `threading.Thread` (not a subprocess — Windows has no clean subprocess-signal story), `TrainingRunner` state machine + `queue.Queue`-per-client pub/sub bridged into SSE at `GET /api/events`. `GET /api/config` serves color/behavior thresholds from `constants.py` so the JS client never hardcodes a duplicate. `src/predictor/training.py` gained a new `train_all_folds` walk-forward driver (previously single-fold only via `scripts/train_predictor.py`), `stop_event`/`save_event` hooks in `train_one_fold` checked at each batch boundary, enriched log payloads (`lr`/`grad_norm`/`fold`/`epoch`/`patience`/ETAs via true exponential smoothing). `scripts/deploy_predictor.py` gained a `math.isnan` guard refusing to deploy a manually-saved (non-gate-evaluated) checkpoint. 119 tests green, `mypy --strict` clean, `ruff` clean. Reviewed via the mandatory `decisions-auditor` + `python-reviewer` + `test-enforcer` triggers — both found real issues (doc-drift in `DECISIONS.md`, a dead EMA constant, an SSE-endpoint thread/subscriber leak on client disconnect, a `start`/`stop`/`save` TOCTOU race) — all fixed with regression tests, all re-verified green. **Known gap:** no pixel-level screenshot obtained — the preview tool times out on this page's persistent `EventSource` connection (a tooling limitation); network/console-level correctness was confirmed instead (all assets 200, zero console errors, SSE connects). Not yet committed, no PR opened. The long training run launches at Phase 1.5 exit. **Training wall-clock optimization** (2026-07-01, pure engineering): the real run was fed the smoke batch size (32) and used a single-process `DataLoader` with per-step host→device copies + a CUDA sync + JSON SSE broadcast on every one of ~4,642 steps/epoch. Fixed by (a) new `PredictorConfig.PROD_BATCH_SIZE = 256` / `PROD_BATCH_SIZE_FALLBACK = 128` (user-confirmed 256 on 2026-07-01); (b) a device-resident `_WindowLoader` (vectorised gather over an on-GPU fold tensor, identical windowing) replacing the `DataLoader`, `build_fold_loaders` now takes a `device`; (c) on-GPU loss accumulation + SSE `batch`-log throttling to every `TrainingUIConfig.BATCH_LOG_INTERVAL = 50` steps (schema unchanged); (d) TF32 + `cudnn.benchmark` at import. Benchmarked on the 4060 (real data, fold 0, lookback 1440): 116.6 → 14.4 s/epoch (~8×), 474 MB peak VRAM. 123 tests green, `mypy --strict` + `ruff` clean. **Fold-count bug fixed** (2026-07-01): the dashboard's `_default_run_training` built folds over the full CSV (2013+) — it was missing the `filter_by_historical_start` call that `scripts/train_predictor.py` applies, so pre-`HISTORICAL_START` candles inflated the fold count. One-line fix mirroring the script path; regression test in `tests/training_ui/test_app.py`; audited by `decisions-auditor` (all checks PASS) + `python-reviewer` (approve). 131 tests green.
- **Phases 2–4** — not started. `src/execution/`, `src/trader/`, `src/dashboard/` do not exist. (The Training Dashboard, formerly bundled into Phase 4, is now Phase 1.5; Phase 4 builds the Trading Dashboard only.)

`scripts/train_predictor.py` and `scripts/deploy_predictor.py` exist. Other scripts (e.g. `backtest.py`, `holdout_evaluator.py`, `permutation_test.py`) are planned deliverables — they do not exist yet.

**Experimental: predictor search loop** (2026-06-30, cross-phase tooling, not a phase deliverable). `scripts/search_predictor.py` — an autoresearch-style unattended hyperparameter/architecture search loop for the predictor. Propose ONE bold, coherent structural move from an explicit menu (proposer pinned to Opus — `PROPOSER_MODEL = "claude-opus-4-8"`; bolder moves are higher-blast-radius so it uses the strongest model) → train on the new search dev-slice (`--search-slice`, `src/data/walk_forward.py::carve_search_slice`, DECISIONS.md `search_dev_slice`) → repeat-seed noise confirmation (DECISIONS.md `search_confirm_seeds`) → judge via `decisions-auditor` + `leakage-checker` (headless, deliberately kept on Sonnet 5 — cheap independent read-only gate, not co-varied with the proposer) → keep (leaves `constants.py` edited) or revert. Safety net around the unattended `constants.py` edit: pre-iteration on-disk snapshot (`constants.py.searchbak`) + `try/finally` restore (regression guard from commit c2071f7) + `ast.parse` check that reverts a non-parseable patch before any training. Regression tests in `tests/scripts/test_search_predictor.py`. No git auto-commit — review `constants.py`'s diff and `search_log.jsonl`, commit manually. Requires the `claude` CLI on PATH; no `ANTHROPIC_API_KEY` needed (uses the existing Claude Code login). **Verify the CLI accepts `claude-opus-4-8` before a real run** (`claude -p --model claude-opus-4-8 "ping"`). Not yet run end-to-end on GPU.

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

### Project-specific rules

- **Tests first** for any code under `src/`. Failing tests encoding the exit criteria are committed before implementation. 
- **Magic numbers live in `constants.py` only**, inside frozen dataclasses. No bare module-level constants. No hardcoded thresholds like `if x > 0.62:`.
- **`src/` never imports from `scripts/`.** Scripts call into `src/`, never the inverse.
- **`data/test_locked/` never referenced from `src/`.** Enforced by `grep -r 'test_locked' src/` plus `sys.settrace` runtime guard during training.
- **All timestamps UTC.** `datetime.now()` without `tz=timezone.utc` is a CI failure in `src/`.
- **Exchange-native stop-loss is mandatory.** Execution refuses any order whose stop cannot be placed at Kraken. No naked positions under any code path.
- **SHA256 manifest verified on every weight read.** Covers weights + scaler + `constants.py`.
- **`asyncio` only in `src/execution/`.** Predictor and trader code stays synchronous.
- **One branch per phase:** `phase-XY` off `main`; merged to `main` at phase exit. Never commit to `main` directly.
- **Any `DECISIONS.md` change requires a `CHANGELOG.md` entry in the same commit.**
- **Flag every unspecced decision.** If a task requires a value not in `DECISIONS.md` or `constants.py`, stop and ask. Do not "use a reasonable default" — this aligns with Karpathy §1.
- **Update `CLAUDE.md` (Repo state section) after every completed feature or phase.** It is the first file read every session — stale repo-state causes wasted work at session start.

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

`scripts/train_predictor.py` and `scripts/deploy_predictor.py` are currently present; other script/module commands below are planned phase deliverables tracked in `INDEX.md` and become runnable as those files land.

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