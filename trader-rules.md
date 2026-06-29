# Trader rules

Rules-based for v3. Auditable, debuggable at 3am, ships in days. RL becomes the next iteration once rules-based demonstrates a stable signal on validated quantile output (≥3 months stable paper trading).

## Inputs

The trader sees:

- Predictor output: `(horizon=15, num_dims=5, num_quantiles=3)` log-return tensor
- Position state: current allocation, unrealized PnL, ATR-at-entry, time-in-position
- Context: `atr_normalized`, regime label, UTC hour `sin/cos`, day-of-week one-hot

It does **not** see raw 1440-candle history. The predictor consumed it; re-feeding it to a rules-based trader is wasted complexity.

## Position sizing — fixed-fractional + uncertainty scaling

Base size: **1% per trade**. Hard cap: **±4% allocation**.

Confidence is computed from the predictor output:

```
spread = (q90 - q10) / |q50|        # close-dim, summed or averaged across horizon as defined in src/trader/sizing.py
```

Continuous scaling: position size scales from `POSITION_SIZE_BASE` toward 0 as `spread` widens. The exact functional form (linear, sigmoid, etc.) lives in `src/trader/sizing.py` and is referenced by name in `constants.py`. The formula must appear in code, not only in prose.

Binary gate (floor): if `spread > CONFIDENCE_THRESHOLD` → forced allocation = 0. Catches the regime where uncertainty is so high the continuous formula would still allow a small, foolish position.

`CONFIDENCE_THRESHOLD` is calibrated empirically against feature-pipeline output before paper trading.

## Predictor staleness decay

Linear from 1.0 at retrain date to `STALENESS_DECAY_FLOOR` (0.50) at retrain_date + 30 days. Multiplies position size. Bridges the gap between conditional retrain triggers — caps blast radius from a stale predictor.

## Exit priority stack (7 tiers)

Conflicts resolved by tier number, not by interaction logic. Lower tier wins.

| Tier | Trigger | Notes |
|---|---|---|
| 1 | Kill switch | `KILL_SWITCH.flag` present → close all, reject new |
| 2 | Hard stop on net PnL | Stop-loss in net-of-fees+slippage terms |
| 3 | Daily-loss circuit breaker | Triggers K1 / K2 thresholds; auto-shutdown, no override |
| 4 | Take-profit | Net of fees + slippage |
| 5 | Signal reversal | ≥3 consecutive opposite-direction predictions (`SIGNAL_REVERSAL_CANDLES`) |
| 6 | Trailing stop | Net PnL terms |
| 7 | Time-based | If used at all; lowest priority |

## Stop-loss net-PnL formula — implemented once, imported everywhere

Identical formula in environment, backtester, and live execution. Any divergence breaks training validity. The function lives in `src/trader/exit_priority.py` and is imported by `src/execution/`, `scripts/backtest.py`, and any environment used during training. Parity is tested by a contract test that runs the same scenario through all three paths and asserts byte-identical decisions.

## Exit priority interaction with exchange-native stop

The exchange-native stop-loss order at Kraken (`execution-engine.md`) is **redundant with** the local tier-2 hard stop, not a replacement. The local stop fires first under normal operation; the exchange-native stop is the failsafe for local-machine outage.

## Robustness gate (pre-paper-trading)

Before moving from mock-predictor loop to live paper, all four checks must pass:

1. **Feature ablation** — remove each input feature one at a time; trader degrades gracefully (falls back to flat) rather than producing nonsense allocations.
2. **Noise injection** — Gaussian noise (σ = 0.5× typical ATR) on predictor output; kill criteria trigger before catastrophic drawdown.
3. **Different time period validation** — backtester run on 2020 (COVID) and 2022 (FTX) sub-periods; system does not blow up in regime transitions.
4. **System chaos** — kill engine mid-trade; disconnect internet; restart under active position; watchdog + exchange-native stop catches each case.

## What the trader does NOT decide

- Whether the predictor is stale enough to force exit. That's the staleness decay and the K-set criteria.
- Whether to retrain. That's a manual user decision triggered by drift or calendar.
- Order routing or fee-tier selection. Execution layer's job.
- Position reconciliation against Kraken. Execution layer's job.
