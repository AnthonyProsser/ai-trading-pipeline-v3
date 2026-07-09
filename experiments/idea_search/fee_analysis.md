# Fee-sensitivity analysis (2026-07-09)

Aggregated benchmark results (test-slice, fixed trading instrument) across every
78-fold run. Break-even fee = mean gross return per trade = net_return/trade_count + 0.0062.

| recipe | folds | net/trade | mean DA | folds net+ | break-even fee |
|---|---|---|---|---|---|
| all recipes (baseline / clock / multiscale / prior) | ~75 ea | ~-0.0062 | 0.508-0.516 | 0/75 | 0.00-0.006% |

Current round-trip fee = 0.62%.

## Verdict
Break-even fee is ~0% for EVERY recipe: mean GROSS return per trade is ~0. Even at a
zero-fee venue none of these models is profitable. DA 0.51-0.52 does NOT translate to
positive gross expectancy — the model is right on direction slightly more than half the
time, but wins and losses are equal-sized (right on small moves, wrong on the big ones).

## Implication for the search
- Beating coin-flip DA is necessary but ~100-200x short of sufficient.
- The target metric should probably be MAGNITUDE-WEIGHTED directional capture (being
  right specifically on the large moves), not sign-frequency DA.
- Feature ideas that move DA by ~0.005 cannot close this gap alone.
- Candidate reframes (for user): (a) train/select on captured-fraction or per-trade
  expectancy, not DA; (b) a magnitude-aware loss that penalizes being wrong on big moves
  more; (c) accept that 1-min OHLCV -> 15-min move may be near-efficient and the edge
  must come from different data (order book, cross-asset) or a longer horizon where moves
  dwarf fees. This echoes the horizon-sweep conclusion.
