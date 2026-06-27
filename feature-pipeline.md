# Feature pipeline

## Inputs (5 features per candle)

Computed sequentially, candle by candle, **before** the per-fold scaler updates. Forward-only — there is no global pre-fit step.

| # | Feature | Formula | Notes |
|---|---|---|---|
| 1 | `open_logret` | `log(open_t / close_{t-1})` | Signed; unbounded |
| 2 | `high_logret` | `log(high_t / close_{t-1})` | Signed; almost always ≥ 0 in practice |
| 3 | `low_logret` | `log(low_t / close_{t-1})` | Signed; almost always ≤ 0 |
| 4 | `close_logret` | `log(close_t / close_{t-1})` | Signed; the headline target component |
| 5 | `vol_change` | `log1p(volume_t / volume_{t-1} - 1)` | `log1p` compresses heavy tails; defined for volume_t = 0 only if volume_{t-1} > 0 |

All five must pass through the scaler. The earlier "no scaler" framing referenced a `body_pct` feature that no longer exists; with the current feature set scaling is required.

## Scaler contract — non-negotiable

- **Per-fold MinMaxScaler.** Never global. The scaler is fit on the train slice of one walk-forward fold and used unchanged for that fold's val + test slices.
- **Strict fit-window assertion.** Min/max statistics are computed on the **train slice only**, but the scaler stores the **whole fold's** timestamp bounds `[fold_start, fold_end]` (train through test) as its allowed transform window — so the same scaler can transform that fold's val + test slices, while a `transform()` call passing any timestamp outside `[fold_start, fold_end]` raises immediately. This catches "next-fold leakage" by construction (a later fold's data can never be scaled with this fold's scaler), not by convention.
- **Forward-only ordering.** Inside any one fold: rolling features (if added later) computed on candle `t` must use only `[..., t-1]`. The scaler then updates. Then the model sees the scaled value for `t`. Never reverse this order.
- **Scaler PKL goes into the SHA256 manifest.** Scaler drift between training and inference silently rewrites the model's input distribution.

## Validator (CandleValidator)

Runs before feature computation. Produces `is_interpolated: bool` per candle.

- **Corruption rules:** `high < max(open, close)` → reject. `low > min(open, close)` → reject. `volume < 0` → reject. `close <= 0` → reject.
- **Gap rules:** missing minute → forward-fill if gap ≤ 12 hours, mark `is_interpolated=True`. Gaps > 12h forward-fill but log a structured warning AND require user acknowledgement before training proceeds.
- **Duplicate timestamps:** keep last; log.

## Baseline signal check (pre-training gate)

Run before any model training. Three rule-based strategies against the feature pipeline + walk-forward splitter:

1. Momentum — go long/short on n-period rolling return sign
2. Mean reversion — long/short on z-score of price vs. rolling mean
3. Volatility breakout — trade in direction of ATR-normalized price move

Gate: at least one baseline shows directional accuracy > 52% **consistently across walk-forward folds** (not just one lucky fold). Failure → revisit feature engineering before any model training. The best baseline Sharpe becomes the minimum bar PatchTST must clear.

## Failure modes to test against

- Per-fold scaler accidentally fits on val data (catch: fit-window assertion + Test 4 in the leakage suite)
- Volume zero for legitimate low-liquidity minute → `log1p(0/X - 1) = log1p(-1) = -inf` → must be clipped or handled
- Forward-fill across a >12h gap silently absorbed into training data — Validator must surface this for user acknowledgement
