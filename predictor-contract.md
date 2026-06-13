# Predictor I/O contract

The locked interface between Predictor and everything downstream (Trader, backtester, dashboard, training-side mock harnesses). This contract is referenced by SHA in `agent_config.json` as `predictor_contract_version`. **Any change to the shapes or semantics below requires a CHANGELOG.md entry and a Trader regression run.**

## Input

```
x: float32 tensor, shape (batch, lookback, 5)
```

- `lookback` ∈ {240, 720, 1440} — locked per checkpoint, recorded in `agent_config.json`. Determined by smoke-sweep before the long training run.
- The 5 features are exactly those defined in `feature-pipeline.md`, in that order, scaled by the per-fold MinMaxScaler whose PKL hash matches the manifest.

## Output

```
y: float32 tensor, shape (batch, horizon=15, num_dims=5, num_quantiles=3)
```

- `horizon=15`: direct multi-step. **Autoregressive iteration is banned.** The model emits all 15 future steps in a single forward pass.
- `num_dims=5`: O, H, L, C, V — same order as input.
- `num_quantiles=3`: q10, q50, q90 — in that order. Index 0 is q10, index 1 is q50, index 2 is q90.
- All values are **log-returns**, not prices. Downstream consumers reconstruct prices by exponentiating against the last known close.

## Geometry enforcement

For every emitted step `s ∈ [0, 14]` and every quantile `q`:

- `q90.high ≥ max(q90.open, q90.close)`
- `q10.low ≤ min(q10.open, q10.close)`

Violations are resampled (up to 5×) at the rollout sampler. The same enforcement applies at every inference step in production AND in any Trader-side mock harness — not only during training rollouts. Inconsistent O/H/L/C produced by independent quantile heads otherwise feeds physically impossible candles to the trader.

## SHA256 anchor

The predictor binds to a manifest containing:

- Weights file (`.pt`)
- Scaler file (`.pkl`)
- `constants.py` content hash

Verified at startup AND on every weight reload. A one-line `constants.py` change otherwise silently changes reward/risk shape post-training.

## Confidence gate consumer interface

The Trader consumes predictor output and computes:

```
spread = (q90 - q10) / |q50|        # per-dimension; trader uses close dim by default
```

This is the explicit input to the confidence gate (`trader-rules.md`). `constants.py` must define `CONFIDENCE_THRESHOLD`, and when the Trader module lands this formula must be implemented in code (planned path: `src/trader/sizing.py`) rather than living only in prose.

## Patch size and token count

PatchTST processes the input sequence in non-overlapping patches.

- `patch_size = 16` → `lookback / patch_size = 1440 / 16 = 90 tokens` at the default lookback.
- Source: `PredictorConfig.PATCH_SIZE = 16` in `constants.py`; DECISIONS `architecture`.

This is an implementation detail of the encoder, not a downstream I/O shape. It is listed here because a lookback change must be divisible by `PATCH_SIZE` to keep the token count integer.

## Training loss

The predictor is trained with a composite loss:

```
L = pinball_loss(q10, q50, q90) + λ × direction_penalty
```

- `λ` is bounded to `[1.5, 2.0]`; the current frozen value is `PredictorConfig.DIRECTION_PENALTY_LAMBDA = 1.75` (`constants.py`).
- `FEE_THRESHOLD` (round-trip drag = 0.62%; `ExecutionConfig.FEE_THRESHOLD = 0.0062`) is subtracted from predicted per-step PnL **inside** the loss computation, so the model learns net-of-fee profitable moves rather than gross moves.
- Source: DECISIONS `loss`.

## Retrain triggers

Two independent triggers; either one fires a manual retrain:

| Trigger | Condition | Constant |
|---|---|---|
| Conditional (drift) | 7-day NLL or quantile-coverage > 2.0× baseline | `PredictorConfig.RETRAIN_NLL_TRIGGER_MULT = 2.0` |
| Calendar (time-based) | 30-day maximum gap regardless of drift metrics | `PredictorConfig.RETRAIN_CALENDAR_DAYS = 30` |

Source: DECISIONS `retrain_trigger_conditional`, `retrain_trigger_calendar`.

## Retrain windows

When a retrain is triggered:

- **Fine-tune window**: `[t-21, t-7]` — 14 days (`PredictorConfig.RETRAIN_FINETUNE_WINDOW_DAYS = 14`)
- **Gate window**: `[t-7, t]` — 7 days (`PredictorConfig.RETRAIN_GATE_WINDOW_DAYS = 7`)
- The two windows are strictly non-overlapping.
- Training warm-starts from the prior checkpoint (no cold init on retrain).

Source: DECISIONS `retrain_window`, `retrain_warm_start`.

## Deploy gates

All three gates are required simultaneously before a retrained checkpoint replaces the active one. A partial pass does not deploy.

| Gate | Criterion | Constant |
|---|---|---|
| (a) Coverage | q90 coverage on locked test set within ±5% of training-time coverage | `PredictorConfig.DEPLOY_GATE_COVERAGE_TOLERANCE = 0.05` |
| (b) Directional Accuracy | DA > 53.5%, computed only over predictions where `\|q50\| > FEE_THRESHOLD` | `PredictorConfig.DEPLOY_GATE_DA_THRESHOLD = 0.535` |
| (c) Calibration rate | Between 75% and 85% | `PredictorConfig.DEPLOY_GATE_CAL_LOWER = 0.75`, `DEPLOY_GATE_CAL_UPPER = 0.85` |

`FEE_THRESHOLD` in gate (b) is `ExecutionConfig.FEE_THRESHOLD = 0.0062`.

Source: DECISIONS `deploy_gates`.

## What is NOT in the contract

- Any internal architecture detail. PatchTST is an implementation choice; downstream consumers see only the input/output tensors and the manifest hash.
- Sampling temperature or any stochastic hyperparameter. The predictor is deterministic given a fixed input.
- Confidence/probability vectors. Quantiles are the only uncertainty signal.
