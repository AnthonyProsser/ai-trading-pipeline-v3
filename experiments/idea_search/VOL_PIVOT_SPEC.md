# Vol-pivot spec (2026-07-09, user-approved direction)

## Roadmap (user, 2026-07-09)
1. THIS: realized-volatility quantile forecasting.
2. If (1) fails its bar -> pre-test Polymarket 5-min up/down (train H=5, measure binary DA on
   our data; only build a bot if DA - spread-implied-fee clears ~55/45 economics).
3. If (2) fails -> user supplies Krafer video transcripts; extract + screen testable claims.

## Target reformulation (branch pivot-vol-01, NOT merged until auditors run)
- New TARGET_SEMANTICS = "cumulative_sqret" : the model predicts quantiles of the
  CUMULATIVE SQUARED 1-min close log-return path (realized variance path) over HORIZON=15.
  Final step = total realized variance of the next 15 min; sqrt = realized vol (RV).
  Reuses the existing cumulative-path machinery (cumsum boundary, monotone quantile head,
  coverage penalty) with y_t = close_logret_t^2 instead of close_logret_t.
- Loss: pinball on the cumulative variance path + coverage penalty. DIRECTION PENALTY REMOVED
  (direction is falsified; the term is meaningless for vol). Close dim only.
- Model/features unchanged (baseline 5 features, lookback 1440) for v1 — one change at a time.

## Pre-registered success bar (locked BEFORE training)
The known-free baseline is PERSISTENCE: predict next-window RV = current-window RV
(measured autocorr 0.833 / R^2 0.66 at 4h; compute the 15-min equivalent on the same folds).
The model is interesting ONLY if, on the 78-fold walk-forward VAL/TEST:
  (a) pinball loss on the RV target BEATS a persistence-quantile baseline (empirical quantiles
      of RV_next | RV_now from train split), AND
  (b) q90 coverage in [0.85, 0.95] and calibration 0.75-0.85 (existing gate bands), AND
  (c) rank correlation (Spearman) of predicted-vs-realized final-step RV > persistence's.
If the model cannot beat persistence, the pivot FAILS -> step 2 (Polymarket pre-test).
No trading-PnL bar in v1: monetization (breakout/vol-targeting/options) is a SEPARATE later
decision; first prove forecast skill above the free baseline.

## Out of scope v1
Strategy layer, benchmark-app vol scoring, options access, order-book data.
