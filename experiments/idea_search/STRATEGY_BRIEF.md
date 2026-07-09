# Strategy brief — predictor-side search concluded (2026-07-09)

## Bottom line
Directional prediction of BTC from 1-min OHLCV does **not** work — at any horizon,
feature set, or loss we tried. Volatility, by contrast, is **highly predictable**.
Recommendation: pivot away from directional price prediction.

## Evidence (all on the real 78-fold walk-forward, all benchmarked net-of-fee)
| recipe | mean DA | net/trade | break-even fee | folds net+ |
|---|---|---|---|---|
| baseline (H=15) | 0.5155 | -0.0062 | ~0.00% | 0/78 |
| clock features | 0.5158 | -0.0062 | ~0% | 0/78 |
| multi-scale returns | 0.5201 | -0.0062 | ~0% | 0/78 |
| swing hi/lo levels | 0.5190 | -0.0062 | 0.002% | 0/78 |
| aux direction head | 0.483* | (broke calibration) | — | — |
| **H=240 (4h, fee-neutralized)** | **0.491** | **-0.0063** | **<0%** | 5/70 (noise) |

- **Confidence-threshold sweep** (150k windows): NO profitable trade subset at any |q50|
  threshold. Even |q50|>2.5% predictions realize ~0.45% moves (< 0.62% fee) at 52% winrate.
- **H=240** removed the fee excuse (4h moves ~1% > fee) and direction was STILL ~0.49.
- **Vol vs direction predictability** (4h windows): realized-vol autocorr **0.833 / R² 0.66**;
  return-direction autocorr **0.005 / sign-persistence 0.47**. Vol is forecastable; direction is not.

## Options (ranked, with my recommendation)
1. **Vol-forecasting strategy — keeps your ML investment. [RECOMMENDED primary]**
   Retarget the PatchTST from price quantiles to *realized-volatility* forecasting (proven 66% R²).
   Monetize via (a) volatility-breakout / straddle-style entries — capture large moves WITHOUT
   predicting direction; and/or (b) vol-targeting position sizing. Catch: cleanest monetization
   (options) needs crypto-options access; a spot breakout variant is simpler but weaker.
   Effort: medium (new target + loss + benchmark meaning; infra mostly reused).
2. **Funding-rate / basis carry — drop prediction entirely. [RECOMMENDED parallel]**
   Cash-and-carry on Kraken (long spot, short perp, collect funding). Robust structural crypto
   edge, no forecasting. Catch: different bot; needs funding-rate data + basis/liquidation risk
   rails; modest returns (~5-15%/yr, variable). Effort: medium-high (new data + execution).
3. **New data for direction** (order book / on-chain / cross-asset). Uncertain payoff, high effort.
4. **Drop directional trading** / shelve the project.

## My recommendation
Pivot to (1) vol-forecasting as the primary path — it reuses the predictor and rests on the
strongest evidence in this whole search. Evaluate (2) carry in parallel as the lowest-signal-risk
income path. Do NOT resume directional feature grinding (VAP/absorption/wicks will be null).

## Status
Nothing is running. 6 recipes trained + benchmarked (in the benchmark app). Tooling built:
run_full_training.py, benchmark_finished.py. Awaiting your pick (1-4) — then I spec + build it.
