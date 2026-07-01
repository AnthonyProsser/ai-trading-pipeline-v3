# Splits, validation, and gates

## Walk-forward splitter

Per fold:

- Train: 150,000 candles
- Validation: 50,000 candles
- Test: 10,000 candles
- Stride: **50,000** candles

Stride = validation block size ⇒ each validation slice is **non-overlapping**. This matters for the permutation test (no Bonferroni inflation from overlapping validation sets) and for honest fold-by-fold reporting. 78 non-overlapping validation folds against the 2018-onward dataset (covers all major regime transitions) — see `DECISIONS.md::walk_forward_fold_count` for the derivation against the real XBTUSD candle count.

The earlier 396-fold count came from a stride of ~10k with heavy overlap; rejected for v3.

## Locked test set

**120,960 candles = 84 days × 1440 = 12 × 1-week non-overlapping windows.** The 2018-onward dataset's terminal block, carved out before any walk-forward fold is created.

- Stored in `data/test_locked/`
- **Never referenced from `src/`.** Enforced by `grep -r 'test_locked' src/` in CI plus a `sys.settrace` runtime guard during training.
- Only `scripts/holdout_evaluator.py` reads from this directory, and only at gate evaluation time.

The 50,000-candle "3-month holdout" framing in v2 was an arithmetic error (50,000 / 1440 ≈ 35 days). Corrected here.

## Search dev-slice (hyperparameter/architecture search loop)

`src/data/walk_forward.py::carve_search_slice()` carves 28,000 candles (20,000 train / 8,000 val, no test split) from the most recent portion of the pre-`HISTORICAL_START` range for `scripts/search_predictor.py`. Structurally disjoint from every walk-forward fold and the locked test set — those only ever draw from `>= HISTORICAL_START`; the search slice only ever draws from `< HISTORICAL_START`. See `DECISIONS.md::search_dev_slice` and `search_confirm_seeds`.

## Walk-forward 12×1w gate (pre-paper-trading)

Inside the locked test set, evaluate the predictor on 12 non-overlapping 1-week windows.

**Pass condition:**

- **Positive Sortino on the median window AND on the worst window.** Single-block evaluation is too forgiving for a model that needs to survive across regimes.
- **Regime-stratified positive.** Compute Sortino separately on trending sub-periods and ranging sub-periods; both must be positive.

The gate is evaluated **once.** Re-running with hyperparameter changes turns walk-forward into in-sample fitting (Bonferroni). If the gate fails, the model is rejected — not tweaked and re-gated.

## Three retrain deploy gates

All three required, simultaneously, every retrain:

1. **Coverage** on locked test set within ±5% of original training-time coverage at q90
2. **Directional Accuracy** > 53.5%, computed only over predictions where `|q50| > FEE_THRESHOLD`. The fee filter prevents the gate from being satisfied by accuracy on sub-fee moves that aren't tradeable.
3. **Calibration rate** between 75% and 85%

A single failing gate blocks deployment. The fine-tune window is `[t-21, t-7]`, the gate window is `[t-7, t]` — strictly non-overlapping. Warm-start from the prior checkpoint.

## Permutation test (pre-live gate)

Distribution-free test on **trade PnL** distribution.

- **Null hypothesis:** PnL distribution from random buy/sell signals applied to the same real price series at the bot's actual trade frequency. Random-signal generator must match the bot's trade rate per hour to keep the comparison apples-to-apples.
- **Alternative:** the bot's actual trade PnL distribution.
- **Test:** permutation, p < 0.05.
- **Why not t-test:** BTC returns have fat tails that violate the t-test's normality assumption. Permutation tests don't require it.

Gate fails ⇒ no live capital. The bot may have a positive Sharpe by luck on the paper period; only the permutation test against shuffled-return baseline distinguishes signal from drift at the bot's actual trade frequency.

## Hyperparameter tuning forbidden during walk-forward

Locked as a non-negotiable rule. If a fold gate fails, EITHER training continues unchanged OR the model is rejected. Tweaking and resuming a fold is in-sample fitting in disguise and silently invalidates the entire walk-forward exercise.

Encoded in CLAUDE.md §3.

## Robustness gate (pre-paper-trading)

Adapted from the signal-validation framework, applied at the system level. All four must pass:

1. Feature ablation — remove each feature one at a time; trader degrades gracefully (flat) rather than producing nonsense.
2. Noise injection — Gaussian noise (σ = 0.5× ATR) on predictor output; kill criteria trigger before catastrophic drawdown.
3. Sub-period validation — backtester run on 2020 (COVID) and 2022 (FTX); no blow-up in regime transitions.
4. System chaos — kill engine mid-trade, disconnect internet, restart under active position; watchdog + exchange-native stop catches each case.

Failures here are cheaper to fix than failures in paper trading with a real predictor.
