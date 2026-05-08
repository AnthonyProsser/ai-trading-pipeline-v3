# CLAUDE.md — Claude Code orientation

## 1. Project identity

Solo paper-trading BTC bot. Three components: **Predictor** (PatchTST on 1,440 1-min candles → quantile multi-step forecast), **Trader** (rules-based, uncertainty-aware sizing, exchange-native stops), **Dashboard** (FastAPI + vanilla JS, read-only telemetry plus kill-switch button). Single RTX 4060 8GB. Single user, Windows. **No real capital until paper-trading gates pass and statistical edge is proven by permutation test.** v2 stalled in predictor training because three AIs coordinated badly across drifting docs; v3's defense is single AI (this one) + adversarial human review + filesystem-as-state. The five v3 directives below exist because v2 had five fatal flaws that the constitution forgot.

## 2. What to read first (every session, in this order)

1. `DECISIONS.md` — locked architectural decisions; key→value
2. `INDEX.md` — task→files lookup; never speculatively load context cards
3. The 1–3 context cards `INDEX.md` names for the current task

**Do not load `docs/archive/old_project.md` in coding sessions.** It is v2 history and contains decisions superseded by `DECISIONS.md`. Loading it costs ~6,500 tokens and creates drift.

Per-session orientation budget: 8–11KB before touching code. If you find yourself loading >3 files, stop and re-read the task definition — either the cards are too granular or the task is too coarse.

## 3. Non-negotiable rules

- **Tests first.** Write failing tests encoding the exit criteria, commit them, then implement. The `test-enforcer` subagent verifies git log order at phase exit.
- **Magic numbers live in `constants.py` only.** Inside their frozen dataclass. No bare module-level constants in `src/`. No hardcoded thresholds in `if x > 0.62:` form — that is a bug.
- **`src/` never imports from `scripts/`.** Scripts call into `src/`, never the inverse. Enforced by CI.
- **`data/test_locked/` never referenced from `src/`.** Enforced by `grep -r 'test_locked' src/` in CI plus a `sys.settrace` runtime guard during training.
- **One branch per phase.** `phase-XY` off `developer`; `developer` → `main` at phase exit. Never commit to `main` directly.
- **Flag every unspecced decision.** If a task requires a value not in `DECISIONS.md` or `constants.py`, stop and ask. Do not guess, do not "use a reasonable default."
- **`asyncio` only in execution.** Predictor and trader code stays synchronous; the event loop lives in `src/execution/`.
- **SHA256 anchor verified on every weight read.** Manifest covers weights + scaler PKL + `constants.py`. A one-line constants change otherwise silently rewrites reward/risk shape post-training.
- **Exchange-native stop-loss is mandatory.** The execution engine refuses any order whose attached stop-loss cannot be placed at Kraken. No naked positions on the book under any code path.
- **All timestamps are UTC.** `datetime.now()` without `tz=timezone.utc` is a CI failure in `src/`.
- **Any `DECISIONS.md` change requires a `CHANGELOG.md` entry in the same commit.** Pre-merge check enforces.

## 4. The five v3 directives

1. **Direct multi-step quantile forecasting.** Autoregressive iteration is banned; PatchTST encoder outputs q10/q50/q90 per OHLCV dim for 15 steps directly.
2. **Uncertainty-aware trading.** Quantile spread `(q90-q10)/|q50|` continuously scales position size — widening uncertainty automatically reduces exposure.
3. **Hardware-level OS isolation.** File-flag kill switch + dedicated watchdog process + exchange-native stops. Capital safety never depends on the dashboard.
4. **Forward-only processing.** Per-fold `MinMaxScaler` with strict fit-window assertion. Rolling features computed sequentially before the scaler updates. Leakage tests written before pipeline code.
5. **Rigorous statistical validation.** Null-hypothesis permutation test (p < 0.05) against random buy/sell signals at the bot's actual trade frequency, before any live capital. Sharpe and win rate are monitoring, not the gate.

## 5. Workflow per phase

1. **Plan.** Re-read `DECISIONS.md` plus the relevant context card(s). Write the phase exit criteria into a markdown plan file.
2. **Branch + tests-first.** `git checkout -b phase-XY developer`. Create failing tests under `tests/{mirror}/` that encode the exit criteria. Commit them.
3. **Implement + summarize.** Make tests pass. End the session with a 5-bullet summary that explicitly flags any unspecced decision encountered.
4. **Review diff + approve.** User reviews the diff against `DECISIONS.md`; the `decisions-auditor` subagent validates constants/formulas first.
5. **Log.** Append a session-log entry under `docs/sessions/`. Merge phase branch → `developer` only after tests green and audit passes.

The `phase-workflow` skill produces (1)–(2)–(5) automatically.

## 6. Skills

- `phase-workflow` — branch + failing-tests + session-log scaffolding
- `add-predictor-feature` — feature pipeline + scaler + leakage test + observation-space note
- `run-backtest` — vectorized backtest, timestamped result file, W&B run, Sortino gate pass/fail
- `check-leakage` — per-fold scaler audit + `grep -r test_locked src/` + leakage regression
- `deploy-predictor` — SHA256 verify, walk-forward 12×1w gate, three retrain gates (coverage ±5%, DA > 53.5%, Cal 75–85%); HARD STOP on any failure
- `write-test-first` — failing tests file mirroring `src/` path, encoding exit criteria
- `run-baseline-check` — momentum/mean-reversion/breakout baselines vs. 52% DA threshold (build-order item 0)
- `run-chaos-test` — kill-mid-trade, internet-disconnect, watchdog catch verification

## 7. Subagents

- `leakage-checker` (Haiku, read-only) — pre-merge on any `src/data/` change; before Phase 0 exit
- `backtest-runner` (Haiku, writes only `scripts/backtest_results/`) — before walk-forward gate; before deploy
- `test-enforcer` (Haiku, read-only) — start of every implementation task; phase exit
- `decisions-auditor` (Sonnet, read-only) — end of every implementation task before user diff review

## 8. File structure

```
btc-bot-v3/
├── CLAUDE.md, DECISIONS.md, CHANGELOG.md, INDEX.md, NOTES.md
├── constants.py             # Frozen dataclasses
├── agent_config.json        # Runtime: SHA256, atr_median, paths
├── pyproject.toml
├── docs/
│   ├── context/             # 8 cards, ≤1.5KB each
│   ├── archive/             # old_project.md + Phase A audits — never load in sessions
│   └── sessions/            # one log per phase exit
├── src/
│   ├── data/ predictor/ trader/ execution/ dashboard/
├── tests/                   # mirrors src/ exactly
├── scripts/                 # one-time runners; src/ never imports from here
├── data/
│   ├── raw/                 # write-once Kraken OHLCVT
│   ├── processed/2026-MM-DD/
│   └── test_locked/         # 120,960 candles; src/ never references this
├── checkpoints/             # {component}_{wandb_run}_{sha[:8]}.pt
└── logs/                    # rotating JSON, 10MB × 30
```

## 9. Currently open questions (live; pinned to a resolution gate)

- **Lookback** (240 / 720 / 1440) — resolved by smoke-sweep before long training run
- **Confidence threshold** for the binary gate — calibrated empirically against feature-pipeline output before paper trading
- **Starting live capital** — user input post-paper-gate (must be ≥ Kraken's ~$10 minimum order at 1% sizing)
- **Whether to add RSI/MACD/Bollinger** — defer to post-paper-trading; do NOT tune feature set in-sample now
- **Move to RL trader** — re-evaluate after ≥3 months stable paper trading on rules; rules remain risk-gate even if RL is added

If a session needs a value that has no resolution gate above, stop and ask.

## 10. Conventions

- Naming: `snake_case` modules, `PascalCase` classes, `UPPER_SNAKE` constants, `phase-XY` branches, `NN-component-topic.md` for atomic docs.
- Types: full type hints in `src/`. `mypy --strict` on PR.
- Versioning: every checkpoint stamps `{component}_{wandb_run_id}_{sha256[:8]}.pt`. WebSocket payloads carry `predictor_hash` + `predictor_contract_version`.
- Logging: structured JSON via `structlog`. Run tag = git SHA + scaler hash + `constants.py` hash + fold ID. W&B starts on the first training step.
- Time: `datetime.now(timezone.utc)` always. Kraken candle close time is the canonical timestamp.
