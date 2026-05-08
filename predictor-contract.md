# Predictor I/O contract

The locked interface between Predictor and everything downstream (Trader, backtester, dashboard, training-side mock harnesses). This contract is referenced by SHA in `agent_config.json` as `predictor_contract_version`. **Any change to the shapes or semantics below requires a CHANGELOG.md entry and a Trader regression run.**

## Input

```
x: float32 tensor, shape (batch, lookback, 5)
```

- `lookback` ∈ {240, 720, 1440} — locked per checkpoint, recorded in `agent_config.json`. Determined by smoke-sweep before the long training run.
- The 5 features are exactly those defined in `docs/context/feature-pipeline.md`, in that order, scaled by the per-fold MinMaxScaler whose PKL hash matches the manifest.

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

This is the explicit input to the confidence gate (`docs/context/trader-rules.md`). The formula must appear in code (`src/trader/sizing.py`) and in `constants.py` as `CONFIDENCE_THRESHOLD`, never only in prose.

## What is NOT in the contract

- Any internal architecture detail. PatchTST is an implementation choice; downstream consumers see only the input/output tensors and the manifest hash.
- Sampling temperature or any stochastic hyperparameter. The predictor is deterministic given a fixed input.
- Confidence/probability vectors. Quantiles are the only uncertainty signal.
