# v3 Post-Mortem, Rationale & Planning Record

> **Frozen historical record — not a live source of truth.** This document holds the v2 failure analysis, the v2→v3 decision rationale, the build-order/parallel-work plan, and the source-attribution audit that originally lived in `consolidated-consolidated-plan.md` (synthesized from multiple AI sources, dated 2026-05-07). It was distributed here so no information is lost while the per-phase context cards remain the place coding sessions actually load.
>
> **For current, authoritative values, read `DECISIONS.md` and the context cards — never this file.** The "v3" columns in the delta tables below record what was decided *at the v2→v3 transition*; they are not updated when a decision changes. Per `CLAUDE.md`, this file is in `docs/` and is **never loaded in coding sessions**.

---

## 0. Why v3 Exists

v2 stalled in predictor training with three unresolved bugs (negative NLL, zero trend loss, premature early-stopping) on a 10-day Junie/Copilot sprint. The diagnosis (`old_project.md §6`) was a **coordination failure across multiple AIs without shared context** — not a capability failure. v3's structural defense is single-AI (Claude Code) + adversarial human review + filesystem-as-state. That defense fails the same way if v3 launches without:

1. A canonical decision document, atomic enough that a coding session loads ≤3 files.
2. The locked I/O contract between Predictor and Trader, set before either is built.
3. Safety scaffolding (kill switch, fee model, stale-candle policy) before any backtest is trusted.

The v3 doc set (`DECISIONS.md`, `INDEX.md`, `CLAUDE.md`, the context cards) locks all three.

---

## 0.1 Post-Mortem: The Five Fatal Flaws of v2

The transition to v3 explicitly resolves five critical failures identified in historical audits. Each flaw is named so it can be referenced unambiguously in future coding sessions.

### Structural Flaw 1: The Autoregressive Illusion (Red Team C1)
The v2 plan assumed a transformer could iteratively feed its own point-estimates forward 100 steps. Reality: this violates the physical bounds of financial time-series modeling. Error compounds super-linearly. By step 30, the model reacts to a "phantom distribution" disconnected from the Kraken order book. **v3 resolution: Autoregressive iteration is banned. Direct multi-step forecasting only.**

### Structural Flaw 2: The Scaler Time Machine / Lookahead Leakage (Predictor C1, Red Team C2)
The v2 plan assumed a global MinMaxScaler applied offline before training was safe. Reality: rolling features were computed before the scaler was fit, silently absorbing future variance into past data points. This lookahead leakage ensured live performance would instantly collapse. **v3 resolution: Per-fold MinMaxScaler with strict fit-window assertion. Rolling features are computed sequentially before the scaler updates; the scaler never sees data outside its explicit fit window. Forward-only processing is enforced by leakage tests written before the pipeline code, not by convention.**

### Structural Flaw 3: The UI-Bound Kill Switch (Ops C1)
System safety was coupled to a FastAPI web dashboard. Reality: UI layers are fragile. Tying capital preservation to a browser tab or WebSocket connection is fundamentally negligent. **v3 resolution: File-flag (`KILL_SWITCH.flag`) + dedicated OS-level watchdog process. Capital safety never depends on dashboard availability.**

### Structural Flaw 4: Predictor Uncertainty Was Unmanaged (Trader C1)
The v2 trader consumed naked point estimates with no measure of predictor confidence. A volatile regime where the model's effective error widens looked identical to a calm regime where it narrowed, so the trader had no way to throttle exposure when the predictor became less reliable. **v3 resolution: Predictor outputs q10/q50/q90 quantiles per dimension. The quantile spread `(q90-q10)/|q50|` is a first-class input to the trader; widening uncertainty automatically reduces position size.**

### Structural Flaw 5: False-Positive Backtesting (Red Team C3)
Statistical edge was measured by arbitrary timeframes (e.g., "3 positive months") rather than rigorous mathematical distinction from random market drift. **v3 resolution: Null-hypothesis permutation testing (p < 0.05) before live capital. PnL distribution must be statistically distinct from shuffled-return baseline.**

---

## 0.2 The v3 Core Directives (Architectural Mandates)

These five mandates are non-negotiable. Any proposed change that violates a directive requires explicit user escalation before implementation. They are listed here because prose rationale tables in §1 can be skimmed; directives cannot. (Each directive resolves the correspondingly-numbered flaw in §0.1.)

**Directive 1: Direct Multi-Step Quantile Forecasting (Kill Autoregression)**
Autoregressive iteration on financial time-series is **banned**. The system uses sequence-to-sequence direct forecasting, outputting q10/q50/q90 per OHLCV dimension for 15 steps ahead using Pinball loss.

**Directive 2: Uncertainty-Aware Trading**
The trader consumes predictor uncertainty as a first-class input, not an afterthought. Quantile spread `(q90-q10)/|q50|` continuously scales position size — widening uncertainty automatically reduces exposure, regardless of whether the trader is rules-based or RL. v3 ships rules-based as the auditable floor; RL is the planned next iteration once rules-based demonstrates stable signal on validated quantile output.

**Directive 3: Hardware-Level OS Isolation (The Dead Man's Switch)**
Capital safety is managed by an independent OS-level Watchdog and Exchange-Native stop losses. The dashboard is strictly read-only for telemetry. The execution engine **refuses any order where the attached stop-loss cannot be placed** — naked positions cannot be stranded on the book.

**Directive 4: Forward-Only Processing (Kill Scaler Leakage)**
All rolling statistics are calculated sequentially before scalers update. The per-fold MinMaxScaler never sees data outside its explicit fit window. This is enforced by leakage tests written before the pipeline code, not by convention.

**Directive 5: Rigorous Statistical Validation (Kill False Confidence)**
Edge is measured by null-hypothesis permutation testing (p < 0.05). The null is **random buy/sell signals applied to the same real price series at the bot's actual trade frequency**. The bot's PnL distribution must be statistically distinct from this null before any live capital is deployed. Sharpe ratio and win rate are monitoring metrics, not the gate.

---

## 0.3 The Signal-First Sanity Check — Rationale

> The operational spec (the three baselines, the >52% DA gate, best-baseline-Sharpe as the minimum bar PatchTST must clear) is in `feature-pipeline.md` §"Baseline signal check" and `DECISIONS.md` §"Pre-training gate." Only the *why* is preserved here.

**The core principle:** *If a simple model cannot make money, a complex one will not fix it.*

The v2 audits established that PatchTST is the right architecture for the implementation. They did not establish that the *feature pipeline* is producing a learnable signal. Before committing to the 4-week PatchTST training run, a 2-day baseline signal check is mandatory. The check evaluates rule-based strategies on **walk-forward validation folds (not the holdout)**, and is read as a three-way gate:

- If any baseline shows directional accuracy > 52% consistently across folds → feature pipeline has learnable signal → proceed to PatchTST.
- If no baseline clears 52% on any fold → the feature set, target definition, or horizon is broken. Revisit feature engineering (Phase 0) before any model training.
- The best baseline Sharpe becomes the minimum bar PatchTST must clear. A transformer that barely beats momentum is not production-ready.

This gate costs 2 days maximum. Skipping it and discovering a broken feature pipeline after a 4-week training run costs 4 weeks.

---

## 1. The v2→v3 Deltas (Frozen Decision Ledger)

> This is the diff against `old_project.md §2` "Decisions Made (Locked)." Everything else in §2 carries forward. **The "v3" column is a transition-time snapshot — current values live in `DECISIONS.md` and the context cards.** The unique value of these tables is the v2 baseline and the source/rationale columns.

### 1.1 Predictor

| Decision | v2 | v3 | Source / rationale |
| --- | --- | --- | --- |
| Architecture | Decoder-only Transformer, `d_model=128`, full O(L²) attention | **PatchTST** (encoder, patch=16 → 90 tokens at L=1440) | Predictor M2 / Decision 2. Quadratic attention at 1440 is the binding VRAM constraint on 8GB; PatchTST drops it ~256× while retaining Transformer expressiveness. Also resolves Red Team "RTX 4060 capacity." |
| Output head | Gaussian (μ, log σ²) per feature | **Quantile q10/q50/q90 per feature** | Predictor C2 / Decision 5. Gaussian is wrong distributionally for crypto (fat tails); quantile gives the same calibration signal without the variance-collapse failure mode that produced the v2 negative-NLL bug. |
| Input features | Unspecified (5 features mentioned, never enumerated) | **5 features per candle: open log-return, high log-return, low log-return, close log-return, log1p volume change** | User decision (this conversation). OHLC log-returns are signed unbounded reals; volume change uses log1p to compress heavy tails. All 5 require per-fold MinMaxScaler. |
| Target | Raw relative change | **Log-return per OHLCV dimension** | Predictor M3 / Decision 1. Additive across multi-step horizons; symmetric tails. Zero implementation cost. |
| Horizon / chain | 100-step iterative autoregressive | **15-step direct multi-step** | Predictor C3 / Decision 3 + Red Team C1. Iterative compounding on a misspecified Markovian model is a structural risk, not a tunable. Direct kills exposure bias and aligns training to inference. 15 steps matches the trader's actionable horizon. |
| Lookback | 1,440 (locked by v2 Amendment 8 "intuitively") | **Sweep [240, 720, 1440] before the long training run** | Predictor M1 / Decision 4. v2 lookback was justified by intuition only ("missed Asian session"). One sweep round costs <1 week of GPU; chasing it after a 4-week training run costs 4 weeks. |
| Loss | Gaussian NLL + dense slope composite | **Pinball (quantile) + direction penalty (`λ`=1.5–2.0); FEE_THRESHOLD subtracted from predicted per-step PnL inside the loss so the model learns net-of-fee profitable moves** | Predictor Decision 6 + user decision (fee handling). Natural fit for quantile head. Direction penalty preserves v2 slope-composite intent without coupling to NLL. Loss-side fee accounting means the trader has no hard predicted-move threshold for entry; the model's gradient already pushes toward predictions that survive fees. |
| Geometry enforcement | Resample-up-to-5x in rollout sampler | **Same rule, applied at every inference step**, not just sampler | Predictor H3. Independent log-returns on O/H/L/C can violate `H≥max(O,C)`; this must be enforced wherever predictions are consumed, including Trader-side mock harnesses. |
| Retrain policy | "Predictor never retrained; only RL agent" | **Dual trigger: (1) conditional — 7-day NLL or quantile coverage > 2.0× baseline → manual retrain; (2) calendar — 30-day maximum gap regardless of drift metrics** | Red Team C4 / Decision 1 + Predictor H2 / Ops C4 + Gemini §6. Conditional catches sudden regime shifts; calendar bound prevents silent staleness. Warm-start from prior checkpoint when triggered. |
| Retrain deploy gates | None | **All three gates must pass before deployment:** (a) quantile coverage on locked test set within ±5% of original training-time coverage; (b) Directional Accuracy > 53.5% computed only over predictions where `\|q50\| > FEE_THRESHOLD`; (c) Calibration rate between 75–85% | Claude §2 + Gemini §6 + user decision. No single metric captures all failure modes. The `\|q50\| > FEE_THRESHOLD` filter on DA prevents the gate from being satisfied by accuracy on sub-fee moves that aren't tradeable. |
| Holdout size | 50,000 candles ("the last 3 months") | **120,960 candles (84 days × 1440 = 12 × 1-week)** | Predictor C4 + user decision. Pure arithmetic correction; 50k = ~35 days, not 3 months. Aligned exactly to the 12 × 1-week walk-forward gate so each gate window is a unique, non-overlapping holdout slice. |
| Holdout evaluation | Single terminal block | **Walk-forward inside the holdout: 12 × 1-week non-overlapping windows; gate requires positive Sortino on median AND worst-week** | Predictor Decision 7 + Red Team Decision 2. Stronger than a single-block holdout without sacrificing the "evaluated once" guarantee. Regime-stratified — must be positive in trending AND ranging sub-periods. |
| Retraining holdout overlap | "7-day overlap, fix in v2" | **Strictly non-overlapping: fine-tune on `[t-21, t-7]`, gate on `[t-7, t]`** | Red Team H1 / Decision 3 + Predictor M5. Non-negotiable; the v2 carry-forward language is self-deception. |
| Predictor-training bug-prevention | Fixes documented after the fact | **Three regression tests written before the training loop:** variance-floor on output (`assert loss > 0` for first 100 steps), trend-loss synthetic input (constant candles → known non-zero output), early-stopping patience parameter exposed in `constants.py` | Red Team `predictor-training bugs all three are v3 risks` + Build-Order item 12. All three v2 bugs were in predictor training. Re-encountering them mid-training costs days; preventing them costs hours. |

### 1.2 Trader

| Decision | v2 | v3 | Source / rationale |
|---|---|---|---|
| Architecture | PPO with custom feature extractor on Dict obs space | **Rules-based** | Trader Decision 1 / OQ1 + Build-Order R-list. The brief said "could be ML, rules, or hybrid"; v2 stalled in predictor training before RL was built, so its viability is untested. Rules-based is auditable, debuggable at 3am, and ships in days — a clean baseline RL must beat. RL becomes the next iteration once rules-based demonstrates a stable signal on validated quantile output. |
| Position sizing | `Box(-0.04, 0.04)` learned by PPO | **Fixed-fractional 1% per trade with hard ±4% allocation cap** | Trader Decision 2 + H1. Simple, auditable, no learned reward shape; ±4% retained as an outer guardrail, not the operating range. |
| Uncertainty-driven sizing | Implicit in PPO's σ² consumption | **The quantile spread `(q90-q10)/\|q50\|` is the explicit mathematical input to the confidence gate.** Position size scales from 1% base toward 0 as spread widens. Formula must appear in code and in `constants.py`, not just in prose. | Trader C1 + Gemini Directive 2. Required to throttle trading during distribution shift; naming the formula prevents drift between docs and implementation. |
| Confidence gate | Implicit in PPO's σ² consumption | **Explicit gate: if `(q90-q10)/\|q50\|` > `CONFIDENCE_THRESHOLD` (defined in `constants.py`), force allocation to zero** | Trader C1. Binary floor when uncertainty is extreme; continuous sizing formula handles the range between. |
| Exit priority | Single dynamic stop in RL reward | **7-tier priority stack: kill-switch > hard stop (net PnL) > daily-loss circuit breaker > take-profit > signal reversal (≥3 consecutive candles) > trailing stop > time-based** | Trader Decision 3 Option A. Each tier has explicit precedence; conflicts resolved by tier number, not interaction logic. |
| Stop-loss evaluation | "NET PnL" flagged HIGH in Master_Gaps 3B but not enforced | **Net PnL after estimated round-trip fees + slippage; identical formula in environment, backtester, and live execution** | Red Team C5 + Trader M2. Any divergence between the three breaks training validity. Implement once, import everywhere. |
| History in inputs | `(1440, 5)` raw OHLCV in agent observation | **Removed from Trader inputs.** Trader sees: predictor quantile output, position state (allocation, unrealized PnL, atr_at_entry), context (atr_normalized, regime, UTC hour sin/cos, day-of-week) | Trader H3 + M4. Predictor already consumed 1440 history; rules-based trader has no need to re-learn it. UTC encoding is cheap and well-supported. |
| Spread model | `SPREAD_PCT = 0.0005` hardcoded half-spread | **`spread = 0.0005 + 0.0001 × atr_ratio`** (ATR-scaled), where `atr_ratio = current_ATR / rolling_median_ATR` over the last 1440 candles | Trader M5 + Red Team M1. v2's hardcode was 5–20× too small in low-liquidity hours; ATR scaling captures this without needing live bid/ask (which Kraken OHLCV doesn't provide). |
| Predictor staleness handling | None | **Linear confidence decay from full at retrain date to floor (e.g., 50%) at retrain_date + 30 days; multiplies position size** | Trader OQ4. Caps blast radius from a stale predictor; bridges the gap between conditional retrain triggers. |

### 1.3 Execution engine + dashboard

| Decision | v2 | v3 | Source / rationale |
|---|---|---|---|
| Dashboard stack | FastAPI + vanilla JS + Lightweight Charts | **Same — kept** | Ops Decision 1A. v2 choice was correct for a control interface where kill-switch click latency matters. `prediction_viewer.jsx` re-implemented in vanilla JS, not migrated as React. |
| Kill switch | Implicit dashboard control | **File-flag (`KILL_SWITCH.flag`) + dedicated watchdog process** | Ops C1 / Decision 2B + Red Team K-set + Trader C3. Survives dashboard crash, inference engine crash, browser tab throttling. Polled every 2s by both watchdog and inference engine. Atomic write via `.flag.tmp` then rename. Mandatory test plan: 4 cases must pass before any paper trading. Dashboard kill button is allowed **only if it writes the file-flag** (not a direct API call). Capital safety never depends on dashboard availability. |
| Fee model | `FEE_RATE = 0.002` (between maker/taker) | **0.26% taker per side + 0.05% slippage floor on every market order. `FEE_THRESHOLD = 2 × 0.26% + 2 × 0.05% = 0.62%` round-trip drag, defined in `constants.py` ExecutionConfig and consumed by both the training loss and the DA evaluation gate** | Trader C2 + Red Team H2 + Build-Order item-#8 + user decision. Kraken base-tier taker is 0.26%; 0.2% is optimistic. Slippage floor on every order, not just `atr_normalized > 1.5 AND atr_ratio < 2.0` (where `atr_ratio = current_ATR / rolling_median_ATR` over the last 1440 candles) — the v2 condition excludes the worst events. |
| Stale candle | Halt at >90s, no specified action for open positions | **Halt at 90s; auto-close all positions at 5 minutes; alert via sound/beep and dashboard banner** | Ops H2 + Red Team K8 + Trader Decision 4. Unmanaged open position during outage was the largest unspecified failure mode. |
| Secrets | Plaintext `.env` (not stated explicitly but implied) | **Windows Credential Manager via `keyring`** | Ops Decision 5B. Encrypted at rest; non-portable to Linux but v3 is single-user Windows. `.env` reserved for non-secret config. |
| Alerts | Dashboard-only RED indicator | **Sound/beep (winsound.Beep) + structured JSON logs + dashboard color states** | Ops Decision 4B + user directive (2026-06-27): Telegram removed. Local audible alert; no external dependency or network egress. Threshold table locked in `execution-engine.md`. |
| Network exposure | Unspecified | **Bind FastAPI to `127.0.0.1` only; remote access via SSH tunnel if needed** | Ops H6. Threat model: another device on LAN can trigger live trade or kill switch. |
| Paper/live toggle | Same code path with conditional branching | **`ExecutionBackend` abstract class; `PaperBackend` and `LiveBackend` are sibling implementations; parity contract test mandatory before live** | Ops H1 / Decision 7 + Trader C4. Kills the conditional-branch divergence failure mode by construction. |
| API ingest | Unspecified for live operation | **Kraken WebSocket v2 OHLC channel for real-time, REST `GetOHLCData` for gap backfill** (≤12h gaps) | Ops Decision 8 + H4. WebSocket on reconnect doesn't backfill; explicit gap detection + REST fill required. Gaps >12h trigger `is_interpolated=True` forward-fill (same rule as historical). |
| Position reconciliation | Unspecified | **On startup, query `GetOpenOrders` and `GetOpenPositions` from Kraken; refuse to start if local SQLite mismatch is unexplained** | Ops C3 + Trader "Exchange outage" failure mode. Kraken treated as authoritative source of truth. |
| Exchange-native stop-loss | None | **Mandatory close order placed at Kraken at every entry. Execution engine refuses the order entirely if stop-loss cannot be placed or is invalid. No naked positions.** | Ops Decision 3 / OQ3 + Red Team `unmanaged position` + Gemini Directive 3. Software stop fails if local machine crashes; exchange-native stop survives any local failure. Refusal on missing stop-loss prevents positions being stranded on the book by construction. |
| SHA256 integrity | Predictor weights only | **Manifest covers weights + scaler PKL + `constants.py`; verified at startup AND on every weight reload** | Red Team C6 / Decision 5 + Build-Order item 10. A one-line `constants.py` change otherwise silently changes reward/risk shape post-training. |
| Persistence | SQLite, `experience_buffer.state_json TEXT` | **SQLite acceptable for paper trading; binary BLOBs (`history_blob`, `futures_blob`, `context_blob`) per Master_Gaps 1A correction; PostgreSQL+WAL before live capital** | Ops H3 + `old_project.md §7` doc-drift correction not pulled back into Phase4Master. Pulled forward here. |
| Replay scrubber | 250ms debounce reading SQLite | **In-memory rolling deque cached for last N minutes; SQLite only for older history** | Ops M1. Live trading writes contend with scrubber reads at the same SQLite. |
| Cycle warning | 55s | **45s warning, 55s hard threshold** | Ops M2. 55s leaves no time for alert delivery before the next cycle starts. |
| Backups | None described | **Checkpoints, scaler, `agent_config.json` synced to OneDrive/Google Drive via rclone or filewatcher after every write** | Ops M5. Single-machine deployment otherwise = single point of failure for months of training. |

### 1.4 Cross-cutting

| Decision | v2 | v3 | Source / rationale |
|---|---|---|---|
| Historical start date | Open since session_01 | **2018-01-01** | Predictor OQ5 + Red Team `Historical start date`. Covers 2018 bear, 2020 DeFi, 2021 institutional, 2022 FTX contagion, 2023+ ETF era. Pre-2017 microstructure is too different (low liquidity, wide spreads). |
| Walk-forward folds | 150k/50k/10k → 396 folds (stride ~10k, heavy overlap) | **150k/50k/10k with stride = 50k → ~84 non-overlapping validation folds** | User decision. Stride = validation block size ensures each validation slice is unique. Cleaner statistical independence for the permutation test (no Bonferroni inflation from overlapping validation sets). ~84 folds still covers all major regime transitions in the 2018-onward dataset. |
| `constants.py` scope | "At root, all magic numbers" | **Single file at root, organized into `@dataclass(frozen=True)` groups: `PredictorConfig`, `RLConfig`, `TraderConfig`, `ExecutionConfig`, `DataConfig`** | Structure Decision 2C. Single point of change preserved; namespacing prevents 300-line monolith and groups blast radius. |
| Test discipline | Tests written before implementation | **Same — kept; enforced by `test-enforcer` subagent and git log order check** | Structure M4 + Red Team `coordination failure`. Without enforcement this becomes aspirational. |
| Branch model | One per phase | **`phase-XY` branches off `main`; merged to `main` on phase exit (no intermediate `developer` branch)** | `old_project.md §2` Workflow; `developer` dropped per 2026-06-28 user directive (see CHANGELOG). |
| Doc drift | Master_Gaps corrections applied to original docs (was not done in v2) | **No correction documents allowed. Decisions update `DECISIONS.md` in place; amendment history goes to `CHANGELOG.md`. Pre-merge check: any change to a decision value must touch CHANGELOG in the same commit** | Red Team `Doc-drift recurrence` + Structure C1 + Build-Order R8. v2's `Master_Gaps.md` flagged HIGH-priority corrections that were never folded back into Phase Masters. Structurally prevent this in v3. |
| Hyperparameter tuning during walk-forward | Implicit | **Forbidden during a fold gate evaluation. If a fold's Sortino is below threshold, EITHER training continues unchanged OR the model is rejected. No "tweak and rerun"** | Build-Order R10 + Red Team H7 (Bonferroni). Procedural defense; locked as a non-negotiable rule in CLAUDE.md. |

---

## 2. Decisions Deferred (and to Which Gate)

These are NOT carried open into coding sessions — they have a defined resolution point.

| Decision | Resolved at | How |
|---|---|---|
| Lookback (240 / 720 / 1440) | Before the long training run | 3-variant smoke sweep with early-stopping; pick by validation MAE + direction accuracy. |
| Starting live capital amount | Before live (post-paper-gate) | User input. Affects whether 1% per trade ≥ Kraken's ~$10 minimum order size. |
| Predictor retrain decision when triggered | Per occurrence | Manual approval; gate = all three retrain deploy gates (coverage ±5%, DA > 53.5%, Cal 75–85%). |
| Whether to add RSI/MACD/Bollinger features | After paper trading, only if rules-based shows untapped signal value | Defer; do NOT tune feature set in-sample now (Bonferroni). |
| Eventual move to RL trader | After ≥3 months stable paper trading on rules | Re-evaluate; rules remain risk-gate even if RL is added (Trader Decision 1 Option C). |
| PostgreSQL migration | Before live capital | Mechanical migration script written in week 4 of training, run at paper→live transition. |
| Azure A100 for training | If RTX 4060 smoke run shows OOM at batch=16 on chosen architecture | Decision point on the day of the smoke run. Not pre-emptive. |

Compare against `old_project.md §3` "Open Questions": every Q1–Q13 listed there is now either locked above or has a defined resolution point above. The v2 "open vs locked" contradiction does not survive.

---

## 3. Build Order & Parallel-Work Plan

### 3.0 Phase Structure

The 13-item build order maps onto a 7-phase structure that drives the git branch model and INDEX.md task→file mapping (Phase 1.5 added 2026-06-30 — pulls the Training Dashboard ahead of the long run so the first run is observable; see CHANGELOG):

- **Phase −1: Setup** — build items 0–3 (baseline check, doc set, `constants.py`, predictor I/O contract)
- **Phase 0: Data** — build items 4–9 (ingest, validator, feature pipeline, per-fold scaler, walk-forward splitter, leakage suite, SHA256 manifest)
- **Phase 1: Prediction Model** — build items 10–13 + retrain scripts. The long training run is deferred to **Phase 1.5 exit** so it can be monitored from the start.
- **Phase 1.5: Training Dashboard** — the browser app (`src/training_ui/`; FastAPI + vanilla JS, separate port) that makes the long run observable: first-run drag-and-drop data gate, start/stop/save controls, live fold/epoch/loss/ETA metrics, and per-fold export to `training_metrics.json` (the handoff artifact for optimization review). Built **before** the long training run launches, so the first run is watched and diagnosable. See `training-dashboard.md`.
- **Phase 2: Environment** — paper backend, live backend, fee/slippage model, kill switch + watchdog, exchange-native stop-loss, Kraken WebSocket/REST ingest, position reconciliation
- **Phase 3: Trading Model** — rules-based trader (sizing, confidence gate, 7-tier exit stack), backtester, robustness gate (§3.4)
- **Phase 4: Dashboards** — the **Trading Dashboard** (`src/dashboard/`; FastAPI + vanilla JS + Lightweight Charts telemetry, predictor accuracy + optimization panels, kill button that writes the file-flag). The **Training Dashboard** moved to Phase 1.5. See `dashboard.md`.

Branches: `phase-0-data`, `phase-1-predictor`, `phase-15-training-ui`, `phase-2-environment`, `phase-3-trader`, `phase-4-dashboard`. Each merges directly to `main` at phase exit (the `developer` integration branch was dropped per the 2026-06-28 user directive).

No time estimates attached to phases. Done is when the phase exit gates pass.

### 3.1 Critical Path to "Training Is Running"

13 items, in build order, from Build-Order audit (table preserved). All gates green before the 4-week training launch. Prepend the 2-day baseline sanity check (§0.3) as item 0.

**Item 0 (new): 2-day baseline signal check** — Run momentum/mean-reversion/breakout baselines against the feature pipeline. Gate: at least one baseline shows DA > 52% consistently across walk-forward folds. If gate fails: revisit feature engineering before any model training.

1. **Doc set:** `CLAUDE.md`, `DECISIONS.md`, `INDEX.md`, context cards (`*.md` at repo root) — single source of truth, prevents Junie-style drift.
2. **`constants.py`** with frozen dataclasses — single-point-of-change for every magic number.
3. **Predictor I/O contract (DECISIONS.md entry)** — locked: target form, horizon, output head, retrain policy. Trader, environment, and execution all consume it.
4. Kraken OHLCVT ingest + write-once `data/raw/`.
5. CandleValidator with corruption + gap rules.
6. Feature pipeline (5 features, log1p on `rel_vol`).
7. Per-fold MinMaxScaler with explicit fit-window assertion.
8. Walk-forward splitter (150k/50k/10k) + locked test set carve-out at **120,960** candles.
9. **Leakage + integrity test suite** (Red Team Tests 1–10 + pipeline parity test 7) — written first as failing tests, then code makes them pass.
10. **SHA256 manifest** covering weights + scaler + `constants.py`.
11. Predictor architecture + loss + training loop with W&B from step 1.
12. Variance-floor guard + trend-loss synthetic test + patience in `constants.py` (the three predictor-training bug regression tests).
13. **Smoke run** (1 epoch, 1 fold) — catches OOM, NaN losses, dataloader issues before committing 4 weeks.

After 13 passes: launch the long training run.

### 3.2 Risk-Ordered Reorderings (Where This Differs from the Dependency Graph)

Build-Order audit pulled four items earlier than their natural technical dependencies:

- **Tests written first as failing tests** (item 9 written before items 4–8 are "complete"). The tests are the executable specification. v2's predictor-training bugs were precisely from skipping this.
- **SHA256 manifest** (item 10 pulled earlier than its natural position). Without it, no training output is reproducibly verifiable.
- **Kill switch + watchdog** (built in week 1 of parallel work, before any paper position is held). Kill switch needs operational time to discover bugs; bad habits in paper carry into live.
- **Fee model** (built before any backtest). Wrong fee model = wrong number = false confidence — fee model is part of safety scaffolding, not part of the backtester.

Justification thread for all four: things that fail silently must be caught by tests written before the code they test.

### 3.3 Parallel Work During Training Run

| Track | In flight | Exit criteria |
|---|---|---|
| **1** | Trader rules module against MockPredictor; ExecutionBackend + PaperBackend; SQLite schema (binary BLOBs from day 1); Kraken WebSocket+REST ingest; UTC discipline CI test; file-flag kill switch + watchdog + **4-case test plan** (case 3: write flag from command line while dashboard offline). | Trader passes against mock contract; PaperBackend logs identical state to LiveBackend mock; kill switch passes all four cases; ingest survives 10-min simulated outage. |
| **2** | 60s asyncio loop wiring Trader+PaperBackend+ingest against live Kraken (mock predictor); fee-drag sensitivity sweep against last 30 days; sound/beep + log alerter; Windows Credential Manager + key-scope startup assertion; position reconciliation logic; stale-candle 90s halt + 5min auto-close; exchange-native stop-loss in every order path. | Fully functional paper trading loop with mock predictor running continuously against Kraken live data; all safety scaffolding in place; fees and slippage characterized empirically at the system's actual trade frequency. |
| **3** | Vectorized backtester against mock; lookahead-bias regression suite as CI on every PR touching `src/data/`; walk-forward holdout evaluator (12 × 1-week); regime-stratified P&L; FastAPI + vanilla JS + Lightweight Charts dashboard scaffold; predictor accuracy panel with q10/q90 percentile overlay; checkpoint backup to OneDrive. | Dashboard renders mock data end-to-end with quantile bands visible; backtester ready to consume real weights; checkpoint backup automated. |
| **4** | `retrain_predictor.py` + `deploy_predictor.py` (built, NOT run); hot-reload weight swap with SHA256 verify on every reload; **Chaos testing protocol** (kill engine mid-trade, disconnect internet, verify watchdog catches it, verify graceful failure + position auto-close); final docs pass; full Tests 1–10 re-run; pre-launch checklist (kill criteria K1–K9 wired, exchange-native stop on every entry). | Training completes (or: see R3). System ready to consume new weights and start paper dry-run within hours. Chaos tests green. |

> **Pre-paper robustness/stress gate (former §3.4) and the inter-phase gate table (former §3.5)** were dropped from this record because they are fully captured, and authoritative, in `DECISIONS.md` (§"Pre-paper gate", §"Pre-live gate", §"Pre-training gate"), `splits-validation.md` (§"Robustness gate", §"Walk-forward 12×1w gate", §"Permutation test"), and `trader-rules.md` (§"Robustness gate"). Their source attributions survive in Appendix A below.

---

## 5. Token-Efficiency Strategy

The user's explicit priority is reducing token usage with one task per chat. The structure of this project is designed around it, not bolted on.

### 5.1 Per-Session Orientation Budget

Target per session: **8–11KB loaded before touching code** (~2,000–2,800 tokens at ~4 tokens/byte). Structure:

| Always loaded | Sometimes loaded | Never loaded in coding sessions |
|---|---|---|
| `CLAUDE.md` (~3KB) | 1–2 context cards (`*.md` at repo root, ~1.5KB each) | `old_project.md` (26KB / ~6,500 tokens) |
| `DECISIONS.md` (~4KB) | The specific `src/` file being modified | The full `src/` directory tree |
| `INDEX.md` (~2KB, only to find which cards to load) | Its corresponding test file | Multiple context cards at once unless task spans two domains |

`INDEX.md` is the lookup table — Claude Code never speculatively loads files. Every entry in INDEX names ≤3 files; if a task needs more, the cards are too granular OR the task is too coarse. Re-split.

> The sample task→files mapping that was former §5.2 lives, authoritatively, in `INDEX.md` and was dropped from this record to avoid duplication.

### 5.3 Failure Modes and Structural Defenses

From Structure audit "How this could fail":

| Failure | Defense |
|---|---|
| Context drift (DECISIONS.md updates without context-card update) | Pre-merge check: any commit to `DECISIONS.md` must also touch the matching context card. |
| Skill rot (skills assume Phase 0 conventions, codebase moves on) | "Skill review" step in phase-completion workflow. |
| INDEX staleness (rows reference files that got renamed) | `check-index-integrity` step in `phase-workflow` skill — asserts every filename in INDEX exists. |
| Subagent scope creep (read-only subagent gains writes) | Tool allowlist defined explicitly per subagent. |
| `DECISIONS.md` becomes new `old_project.md` | Strict structure: flat decision name → current value + date. Append-only history goes in `CHANGELOG.md`. |
| Token-budget blowout (compaction truncates decisions) | Hard size limits: DECISIONS ≤4KB, each card ≤1.5KB. CLAUDE.md defines load order so smallest useful set loads first. |

---

## 6. Skills and Subagents

### 6.1 Skills (Workflows Always Present, Stateless)

| Skill | Trigger | Produces |
|---|---|---|
| `phase-workflow` | Start of any new phase or sub-phase | Branch `phase-XY`, failing tests file with exit-criteria, implementation summary flagging unspecced decisions, session log entry |
| `add-predictor-feature` | User wants to experiment with a new candle-derived feature | Updated `feature_pipeline.py`, updated `constants.py`, leakage test, observation-space note |
| `run-backtest` | Before any walk-forward gate; after any agent checkpoint change | Timestamped result file, W&B run, pass/fail vs Sortino gate |
| `check-leakage` | Before every Phase 0 (Data) exit-check; on any `feature_pipeline.py` change | Pass/fail report with failing positions; `grep -r test_locked src/` result |
| `deploy-predictor` | User wants to deploy a retrained predictor | Verified SHA256, walk-forward 12×1w gate evaluated, **all three retrain gates checked** (coverage ±5%, DA > 53.5%, Cal 75–85%), archived old weights with timestamp, `agent_config.json` updated, hard STOP if any gate fails |
| `write-test-first` | Start of every implementation task | `tests/{mirror-path}/test_{module}.py` with failing tests encoding exit criteria |
| `run-baseline-check` | Before long training run (item 0 in build order) | Baseline strategy results vs. 52% DA threshold; pass/fail gate |
| `run-chaos-test` | Before paper trading launch | Kill-mid-trade test, internet-disconnect test, watchdog catch verification; pass/fail gate |

### 6.2 Subagents (Specialist Roles with Tool Restrictions)

| Subagent | Role | Read paths | Write paths | Model | Invoke when |
|---|---|---|---|---|---|
| `leakage-checker` | Detect feature leakage, test-set contamination, scaler-on-wrong-window | `src/data/`, `tests/data/`, `data/processed/`, `constants.py` | None (read-only) | Haiku | Before any 0B/0C exit; on any `feature_pipeline.py` or scaler change; pre-merge on PRs touching `src/data/` |
| `backtest-runner` | Execute vectorized backtests, produce standardized reports | `checkpoints/`, `data/processed/`, `agent_config.json`, `constants.py` | `scripts/backtest_results/` (timestamped) | Haiku | Before walk-forward gate; before deployment; on user "how is current agent doing?" |
| `test-enforcer` | Verify no `src/` module exists without mirrored test; verify test commit predates implementation | `src/`, `tests/`, git log | None (read-only) | Haiku | Start of every implementation task; phase completion |
| `decisions-auditor` | Check every constant/formula/schema in new code matches `DECISIONS.md` and context cards | `DECISIONS.md`, context cards (`*.md`), `constants.py`, `agent_config.json`, the file being audited | None (read-only) | Sonnet | End of every implementation task before user diff review; whenever Claude Code flags an unspecced decision |

**Model rationale:** Haiku for read-only, grep-like tasks (token efficiency); Sonnet for decisions-auditor because semantic matching of formulas to prose requires stronger reasoning.

Add new subagents only when a pattern repeats 3+ times (Structure Decision 3B). Avoid premature abstraction.

### 6.3 ECC Selective Install

From `old_project.md §4` "Considered, no firm decision": ECC (everything-claude-code) at github.com/affaan-m/everything-claude-code. Install only the **PyTorch skill** and **TDD agent**; defer everything else. If ECC's TDD agent duplicates `test-enforcer` defined above, keep one — verify scope before installing (Structure OQ3).

---

## 7. CLAUDE.md Structure (Original Outline)

The `CLAUDE.md` written into the repo before the first coding session contains, in this order:

1. **Project identity** (50–100 words). Paper trading only; no real capital; single RTX 4060; Claude Code builds; user specs and adversarially reviews. Reference the five structural flaws (§0.1) as the motivation for all constraints.
2. **What to read first** (ordered). `DECISIONS.md` → `INDEX.md` → relevant context cards. **Do not load `old_project.md` in coding sessions.**
3. **Non-negotiable rules** (bulleted). Tests-first; magic numbers in `constants.py` only; `src/` ↛ `scripts/`; `test_locked` rule; one branch per phase; flag every unspecced decision; `asyncio` only in execution; SHA256 anchor verified on every weight read. **Exchange-native stop-loss must be present or order is refused.**
4. **The five v3 directives** (one line each, bolded). Direct multi-step quantile; deterministic execution; OS-level kill switch; forward-only scaler; permutation test before live.
5. **Workflow per phase** (5 steps). plan → branch+exit-criteria-first → implement+summarize → review-diff+approve → log. References the `phase-workflow` skill.
6. **How to invoke skills** (one line per skill).
7. **How to invoke subagents** (one line per subagent, with model).
8. **File structure** (annotated tree from §4 above).
9. **Currently open questions** (live list — those in §2 above with their resolution gates).
10. **Conventions** (naming, types, versioning).

Total target: ~3KB.

---

## 8. Top Risks and Mitigations

Distilled from Build-Order risk register R1–R10, augmented by Red Team K-set criteria.

| # | Risk | Mitigation |
|---|---|---|
| R1 | 4-week training fails halfway, logging too sparse to diagnose. | W&B run starts on FIRST training step. Run tag includes git SHA + scaler hash + `constants.py` hash + fold ID. Loss components logged separately. Synthetic-input regression test on every batch for first 100 steps. |
| R2 | Predictor I/O contract wrong, Trader needs rewrite when real weights arrive. | Lock contract before Trader build. Trader built against `MockPredictor` emitting exact contract. Any contract change = `CHANGELOG.md` entry triggering Trader regression. |
| R3 | Architecture (PatchTST) doesn't fit / OOM at long-run config. | Smoke run on chosen architecture before committing 4-week run. OOM at batch=32 → drop to 16 → if still OOM, this is the Azure A100 decision point. Do NOT start 4-week run on borderline-OOM config. |
| R4 | A predictor-training bug recurs. | Three named regression tests must pass before training loop ships. Variance-floor (`assert loss > 0` first 100 steps). Trend-loss synthetic-input test. Patience parameter exposed in `constants.py`. |
| R5 | Dashboard panels break across predictor versions. | WebSocket payload schema versioned; `health` payload includes `predictor_hash` + `predictor_contract_version`. Dashboard refuses to render prediction panels if contract version differs from build version. |
| R6 | Fee drag exceeds gross PnL silently. | Fee-drag sensitivity sweep against mock predictor in week 2 (parallel work plan), against last 30 days of live candles. If unsupportable at trade frequency, revise Trader exit priority before real weights arrive. |
| R7 | UTC/DST clock bug on Windows. | CI test asserts no `datetime.now()` without `tz=timezone.utc` in `src/`. Windows DST-simulation test (clock 30min before transition, run candle reconciliation, assert no false stale-candle alerts). |
| R8 | Doc set drifts; `DECISIONS.md` becomes the new `old_project.md`. | DECISIONS is flat key→value. Every amendment = `CHANGELOG.md` entry + `DECISIONS.md` update in same commit. Pre-merge check enforces. |
| R9 | Kill switch passes unit tests but fails real outage. | Manual run of 4-step kill-switch test plan against the live system every Friday during paper trading. Specifically test case 3: write flag from command line while dashboard offline. |
| R10 | Hyperparameter tuning during walk-forward silently turns it in-sample. | Forbidden during fold gate. Documented as non-negotiable in CLAUDE.md. Bonferroni problem (Red Team H7) compounds invisibly — the structural defense is procedural. |
| R11 | Retrain deploy gates (DA, calibration) pass on lucky holdout week. | Gates require 3-week minimum fine-tune window; deploy gate evaluated on a fresh holdout week not seen in fine-tuning. All three gates (coverage ±5%, DA > 53.5%, Cal 75–85%) must pass simultaneously. |
| R12 | Exchange-native stop-loss silently fails to place. | Execution engine asserts stop-loss order confirmation before marking position open. If confirmation doesn't arrive in N seconds, auto-close any partial fill and alert via sound/beep. |

> The **Kill Criteria Summary (former §8.1, K1–K9)** was dropped from this record: the full K-set with auto-shutdown-vs-alert classification is authoritative in `DECISIONS.md` (§"Execution engine", keys `kill_criteria_K1`…`K9`) and `execution-engine.md` (§"Alert thresholds"). All K-criteria are wired in code in the execution engine, encoded in `constants.py` `ExecutionConfig`; auto-shutdown criteria (K1, K2, K4, K8) have no operator-override path.

---

## 9. What Is NOT on the Critical Path (Defer List)

From Build-Order "What is NOT on the critical path." Do not let any of these block training start.

- **A pretty dashboard.** Functional only before paper trading: kill switch button (file-flag writer only), stale-data banner, candles + orders + health WebSocket types. Calibration panels in week 3–4 of training.
- **Inference-path optimization** (`torch.compile`, KV cache, branch attribution speedup). 60s loop has 60s of headroom; defer until paper trading shows actual latency numbers.
- **RL agent design.** Trader v3 default is rules-based. RL is a 4–6 week build deferred until rules-based demonstrates a stable signal on validated predictor output.
- **Predictor warm-start / retraining cadence experiments.** First training takes ~4 weeks. Retraining cadence cannot be answered until first paper trading data exists. Build the scripts in week 4 (ready); run them only after 2+ months of paper data.
- **Cloud training.** RTX 4060 first to prove pipeline end-to-end. Revisit only if smoke run forces it.
- **PostgreSQL migration.** SQLite acceptable for paper. Migrate before live, not before.
- **Hyperparameter sweeps.** Lookback sweep is the one exception (it gates the long run). Everything else is post-first-paper-validation.
- **Docker.** Rejected in v2 for good reason (single-user local). Revisit only on VPS move.
- **PyTorch Lightning, Hugging Face Transformers, Ray/RLlib, Optuna.** All explicitly rejected in v2; rejections still hold (`old_project.md §4`).
- **General retraining / deployment UI.** Retrain and deploy stay CLI-gated (`scripts/deploy_predictor.py` + the three deploy gates); no UI button triggers them. Distinct from the browser **Training Dashboard** (start/stop/save + live metrics + first-run data gate; a **Phase 1.5** deliverable) and the **Trading Dashboard** (Phase 4) — see `training-dashboard.md` / `dashboard.md`.
- **DiscoRL / meta-learned RL update rules.** DeepMind-scale infra; deferred post-August.

---

## 10. Loose Ends from `old_project.md §7` to Close Before Coding Starts

| Loose end | Action |
|---|---|
| `session_01_march20.md` is 0 bytes | Archive as-is into `docs/archive/`; note in CHANGELOG that the file was empty in source. |
| `Project Overview.md` has unfilled architecture-diagram placeholder | Discard; superseded by this document + DECISIONS.md. |
| `MasterArchitecture.md` `state_json TEXT` not corrected | Folded into v3 spec as binary BLOBs (Master_Gaps 1A correction applied here, §1.3). |
| `Phase4Master.md` MAE/Directional metrics not corrected | Folded into v3 as NLL/Calibration/Trend Direction (Master_Gaps 1B correction applied here). For v3: with quantile head, "Calibration Rate" becomes "quantile coverage rate at q90" — same monitoring purpose. |
| `PROJECT_MASTER.md` and `project_master (2).md` near-duplicate | Both archived; this document is canonical. |
| `prediction_viewer.jsx` outside project | Discard React; re-implement in vanilla JS + Lightweight Charts during week 3 dashboard work. |
| `agent_config.json` placeholders (`atr_median: 0.0000`) | Populated automatically by Phase 0 (Data) and Phase 2 (Environment) scripts; not human-edited. |
| `developer` branch only mentioned in session_03 | Locked here in §1.4 and §4. |
| Phase structure redesign for Claude Code workflow | This document IS that redesign. |

---

## 11. Open Questions — Resolved During Planning

All items below were open at plan-authoring time and have since been resolved (confirmed by the user during the planning conversation and folded into §1 / `DECISIONS.md`). Retained for the record:

1. **Rules-based Trader for v3 Phase 1** — confirmed. Deviation from the brief's "Trader ML model" framing accepted; RL deferred until rules-based shows stable signal (Trader Decision 1A; DECISIONS `architecture`).
2. **2018-01-01 historical start date** — confirmed (DECISIONS `historical_start`).
3. **Quantile output (q10/q50/q90) vs. point estimate** — locked; quantile is the v3 default (Predictor C2, §1.1 Output head, DECISIONS `output_head`).
4. **15-step direct horizon** — confirmed; expands the brief's "next 1-min candle" to 15 steps direct, autoregression banned (DECISIONS `horizon`).
5. **Kraken account fee tier** — locked as base-tier (taker 0.26%); fee model adjusts if the user moves to a higher volume tier (DECISIONS `fee_model`; `ExecutionConfig.FEE_RATE_TAKER`).

No open user-input items remain at this layer; remaining decisions are gated to pre-paper / pre-live and tracked in §2.

### 11.1 Recently Locked During Planning

Decisions confirmed during the planning conversation, now folded into §1.1 / §1.4 / §3:

- **5 input features:** open log-return, high log-return, low log-return, close log-return, volume change (`log1p`). Captured in §1.1 Input features row.
- **Walk-forward stride = 50k** → ~84 non-overlapping validation folds. Captured in §1.4 Walk-forward folds row.
- **Permutation test null hypothesis:** random buy/sell signals applied to the same real price series at the bot's actual trade frequency. Captured in §0.2 Directive 5 and §3.5 pre-live row.
- **Fee handling:** FEE_THRESHOLD = 0.62% subtracted inside the training loss; no hard predicted-move threshold for trader entry; FEE_THRESHOLD applied to the DA evaluation gate only. Captured in §1.1 Loss row, §1.1 Retrain deploy gates row, and §1.3 Fee model row.
- **Phase structure** (no time estimates): Phase −1 Setup, Phase 0 Data, Phase 1 Prediction Model, Phase 2 Environment, Phase 3 Trading Model, Phase 4 Dashboard. Captured in §3.0.
- **Holdout size:** 120,960 candles (84 days = 12 × 1-week), aligned to the walk-forward gate. Captured in §1.1 Holdout size row.
- **Scaler restored:** per-fold MinMaxScaler with strict fit-window assertion. The earlier "no scaler" framing was based on `body_pct`, which is no longer in the feature set; with OHLC log-returns + log1p volume change, scaling is needed. Captured in §0.1 Flaw 2.

---

## 12. Educational Resources (Non-Blocking Prerequisites)

The following resources are recommended before building and reviewing the code. They are not on the critical path but their absence is part of what caused v2's coordination failure — the ability to adversarially review the code depends on understanding what the code is doing.

| Resource | Why it matters for this project |
|---|---|
| Karpathy "Let's Build GPT" (must build alongside, not just watch) | PatchTST is an encoder-only transformer. Building GPT from scratch gives the intuition needed to audit the attention mechanism and patch embedding. |
| CS50 AI: Week 6 (Transformers) | Architectural context for PatchTST; helps recognize when the model is doing something implausible. |
| PyTorch Official 60-min blitz + autograd tutorial | Required to audit gradient flow, loss backprop, and the variance-floor regression test. |
| Stable Baselines 3 (SB3) Docs: Custom Environments | Understanding the RL framework even though v3 doesn't use it yet; needed to evaluate future RL transition. |
| Gymnasium Docs: `env_checker` | Same rationale; the rules-based Trader uses a similar observation-action structure. |
| Kraken REST API: `GetOHLCData` docs | Required to audit the ingest module and gap-fill logic. |

These resources are not a prerequisite for starting — they are parallel work for the user while Claude Code builds.

---

## 13. What This Plan Did NOT Cover

For traceability (as of plan authoring; several have since been delivered):

- The actual content of `DECISIONS.md` (next deliverable; this document justifies it).
- The actual content of each context card (`*.md` at repo root; one per coding-task domain; ≤1.5KB each).
- The text of `CLAUDE.md` (outline in §7; full text is the next-next deliverable).
- The full backtest harness specification (parallel work week 3; references `execution-engine.md` for fee/slippage rules).
- The retrain workflow scripts beyond their existence in `/scripts/` and their gate criteria.
- Anything explicitly deferred in §9.

---

## 4. Reference — Locked Directory Structure

Locked from `old_project.md §2` "File structure" + Structure audit recommendation. No deviation without `CHANGELOG.md` entry. (Retained here as the original architectural reference; the live tree is whatever is on disk.)

```
btc-bot-v3/
├── CLAUDE.md                          # Claude Code orientation, ~3KB
├── DECISIONS.md                       # Flat key→value locked decisions, ~4KB
├── CHANGELOG.md                       # Append-only amendments to DECISIONS
├── INDEX.md                           # Task → ≤3 files mapping, ~2KB
├── NOTES.md                           # Scratch notes; not a decision record
├── constants.py                       # Frozen dataclasses (PredictorConfig etc.)
├── agent_config.json                  # Runtime: SHA256, atr_median, paths
├── pyproject.toml                     # Deps, lint config
│
├── feature-pipeline.md                # context card: 5 features, scaler contract, log1p
├── predictor-contract.md              # context card: I/O shapes, quantile head, SHA256
├── predictor-training.md              # context card: PatchTST, pinball+direction loss
├── trader-rules.md                    # context card: sizing, confidence gate, exit stack
├── execution-engine.md                # context card: 60s loop, fees, kill switch, stops
├── dashboard.md                       # context card: Trading Dashboard (telemetry + kill)
├── training-dashboard.md              # context card: Training Dashboard (controls + metrics)
├── agent-config.md                    # context card: agent_config.json schema
├── splits-validation.md               # context card: WF folds, locked test set, gates
│
├── docs/
│   └── archive/                        # v2 history + original audits (never loaded in sessions)
│
├── src/
│   ├── data/                          # Ingest, validator, feature pipeline, scaler, splitter
│   ├── predictor/                     # PatchTST, loss, training loop, rollout
│   ├── trader/                        # Rules, sizing, exit stack, confidence gate
│   ├── execution/                     # asyncio loop, ExecutionBackend, watchdog, alerter
│   ├── dashboard/                     # Trading dashboard: FastAPI, WebSocket, static frontend
│   └── training_ui/                   # Training dashboard: controls, live metrics, data gate
│
├── tests/                             # Mirrors src/ exactly; one test file per module
│   ├── data/
│   ├── predictor/
│   ├── trader/
│   ├── execution/
│   ├── dashboard/
│   └── training_ui/
│
├── scripts/                           # One-time runners; src/ NEVER imports from here
│   ├── train_predictor.py
│   ├── retrain_predictor.py
│   ├── deploy_predictor.py
│   └── backtest_results/
│
├── data/
│   ├── raw/                           # Write-once Kraken OHLCVT
│   ├── processed/2026-MM-DD/          # Date-stamped subdirs; scalers stored alongside
│   └── test_locked/                   # 120,960 candles; src/ NEVER references this
│
├── checkpoints/
│   └── {component}_{wandb_run_id}_{sha256[:8]}.pt
│
└── logs/                              # Rotating JSON logs, 10MB × 30 files
```

**Hard rules** (encoded in CLAUDE.md and as merge checks):
- `src/` never imports from `scripts/`. Justification: scripts are one-time runners that call deep into `src/`; the inverse path creates circular maintenance.
- `data/test_locked/` never referenced in `src/`. Enforced by both `grep -r 'test_locked' src/` CI step AND `sys.settrace` runtime guard during training (Red Team Test 10).
- All magic numbers go in `constants.py` in their named frozen dataclass — no bare module-level constants.
- All tests written and committed before implementation; git log order verified by `test-enforcer` subagent.
- All timestamps use `datetime.timezone.utc` explicitly. CI test asserts no `datetime.now()` without `tz=` anywhere in `src/`.

---

## Appendix A — Source Attribution Map

For every claim above, the originating audit (so a future reader can verify):

| Section | Primary sources |
|---|---|
| §0 motivation | `old_project.md §6`, Red Team `Junie/Copilot` (Claude plan §0) |
| §0.1 five structural flaws | Gemini §2 (named and framed); Claude §1 (rationale tables); Red Team C1–C3 |
| §0.2 five directives | Gemini §3.1–3.5 (mandate framing adopted); Claude §1 (rationale detail) |
| §0.3 signal-first sanity check | ChatGPT Phase 2 + Phase 4 (idea adopted); adapted for v3 context |
| §1.1 Predictor | Claude Predictor audit Decisions 1–7, C1–C4, H1–H5, M1–M5; Red Team C1, C4; Gemini §6 (retrain gates) |
| §1.2 Trader | Claude Trader audit Decisions 1–3, C1–C4, H1–H5, M1–M5; Red Team C5; Gemini Directive 2 (quantile-spread sizing formula) |
| §1.3 Execution | Claude Ops audit Decisions 1–8, C1–C4, H1–H6, M1–M5; Red Team K-set, C6; Gemini Directive 3 (mandatory stop-loss refusal) |
| §1.4 Cross-cutting | Claude Structure C1–C3, H1–H5, Decisions 1–3; Red Team `Doc-drift`, `Bonferroni` |
| §2 deferred decisions | Claude Build-Order "Open questions before each phase" |
| §3.1 critical path | Claude Build-Order audit (entire) + §0.3 baseline check (ChatGPT) |
| §3.2 risk-ordered reorderings | Claude Build-Order audit risk register R1–R10 |
| §3.3 parallel work | Claude parallel work table; Gemini Week 15 chaos testing (added to week 4) |
| §3.4 robustness gate | ChatGPT Phase 5 (noise injection, feature ablation, different periods); Gemini Week 15 (chaos protocol) |
| §3.5 gates | Claude pre-training/paper/live gates; permutation test (Gemini Directive 5) replaces t-test |
| §4 directory | Claude Structure audit + `old_project.md §2` |
| §5 token efficiency | Claude Structure audit "Token budget" + Decision 1C |
| §6.1 skills | Claude Structure audit "Skills to create"; `run-baseline-check` and `run-chaos-test` added from new gates |
| §6.2 subagents | Claude Structure audit "Subagents to define"; model column added (Haiku/Sonnet rationale) |
| §7 CLAUDE.md | Claude Structure audit "CLAUDE.md outline"; five directives added |
| §8 risks | Claude Build-Order R1–R10; R11 (retrain gates) and R12 (stop-loss confirmation) added |
| §9 defer list | Claude Build-Order "What is NOT on the critical path" |
| §10 loose ends | Claude `old_project.md §7` |
| §11 open questions | Claude §11 |
| §12 educational resources | Gemini §4b (adopted as non-blocking appendix) |

---

## Appendix B — Conflict Resolution Audit Trail

For every concrete disagreement between the three source plans, this table records what was decided and why. Tiebreaker priority: (1) token efficiency, (2) robustness to unreliable file system dates, (3) clarity of write-path restrictions.

| Conflict | ChatGPT says | Claude says | Gemini says | Resolution | Reason |
|---|---|---|---|---|---|
| **C1** Model choice | Start simple (logistic→boosting→MLP) | Go directly to PatchTST (audits decided) | Go directly to PatchTST | **PatchTST + 2-day baseline sanity check gate (§0.3)** | Audits support the architecture; the gate catches if the feature pipeline is broken before 4 weeks are wasted |
| **C2** Retraining trigger | Vague | Conditional: 7-day NLL > 2.0× baseline | Calendar: 30 days fixed cadence | **Both: drift trigger as primary + 30-day maximum cap** | Conditional catches sudden shifts; calendar prevents silent staleness |
| **C3** Statistical gate | Sharpe + IC | t-test p<0.05 | Permutation test p<0.05 | **Permutation test (Gemini)** | Distribution-free; BTC returns have fat tails that violate t-test's normality assumption |
| **C4** Baseline no-ML gate | Required (Phase 2 + Phase 4 hard stop) | Absent | Absent | **Added as item 0 in build order (§3.1)** | 2-day cost; prevents 4-week wasted training run |
| **C5** Robustness testing | Phase 5 explicit (noise, ablation, different periods) | Regime-stratified only | Absent | **Added as §3.4 pre-paper gate** | Signal that only survives in-sample in one asset is not a signal |
| **C6** Exchange stop mandatory | Absent | Conditional close order | Refuse order without stop-loss | **Mandatory refusal (Gemini)** | Code that refuses is better than policy that fails |
| **C7** Dashboard role | Absent | Control interface (kill button OK) | Read-only telemetry only | **Merge: dashboard kill button allowed if it writes the file-flag only** | Agrees with Gemini's principle (capital safety never depends on dashboard) while preserving Claude's convenience |
| **C8** Position sizing formula | Unspecified | Binary confidence gate (formula implicit) | Quantile spread controls sizing continuously | **Fixed-fractional 1% base + binary gate + explicit formula in constants.py (merge)** | Naming the formula prevents doc/code drift |
| **C9** Scaler approach | None | Per-fold MinMaxScaler + leakage tests | Unified Stream Processor (forward-only) | **Both: per-fold MinMaxScaler + forward-only principle as the justification** | Architecture (Gemini) + tests (Claude) both needed |
| **C10** Retrain quality gates | None | ±5% coverage | DA > 53.5%, Cal 75–85% | **All three gates required simultaneously** | No single metric captures all failure modes |
| **C11** Chaos testing | None | 4-case kill switch test | Week 15 protocol (kill mid-trade, disconnect internet) | **Both: 4-case test (week 1) + full chaos protocol (week 4 pre-launch)** | Earlier failure is cheaper |
| **C12** Educational resources | None | None | Mandatory list | **Added as §12 non-blocking appendix** | Understanding beats documentation alone; non-blocking means it doesn't delay coding |
| **C13** Directory layout | None | Full annotated tree | None | **Claude's tree (unchanged)** | Only plan with a spec |
| **C14** File set | One DECISIONS.md | Three files (DECISIONS + CHANGELOG + INDEX) | DECISIONS.md | **Claude's three-file system (unchanged)** | INDEX.md is the most valuable addition for token efficiency |

---

*Frozen historical record, distributed from `consolidated-consolidated-plan.md` (originally dated 2026-05-07). The live source of truth is `DECISIONS.md` + the context cards + `INDEX.md`.*
