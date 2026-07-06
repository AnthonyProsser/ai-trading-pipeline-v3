"""Fixed-instrument trading simulation for the model benchmark.

ONE trading rule, applied identically to every model — the rule is the measuring
stick, never a tunable (searching over rules and reporting each model's best would
overfit the eval slice; the project's permutation test exists to guard against exactly
that garden-of-forking-paths). The rule uses the quantile forecast's uncertainty, not
just the median direction:

    ENTER long/short at origin t iff, at the FINAL horizon step of the cumulative
    close forecast:
      * |q50| > EXECUTION.FEE_THRESHOLD   (predicted move clears the round-trip fee)
      * q10 and q90 are strictly the same sign (the 80% interval does not straddle
        zero — the model is confidently one-sided)
    HOLD exactly HORIZON steps (exit at the forecast's own endpoint), pay one
    round-trip fee, never overlap positions (in-position signals are skipped).

This tests whether the calibration/coverage work bought anything *tradeable*.

Null baselines every model must beat:
  * buy-and-hold over the same evaluation span (one round-trip fee), and
  * random-entry at matched trade frequency: NULL_DRAWS resamples of the same number
    of non-overlapping entries with random direction — the permutation-style null PnL
    distribution; p_value = (1 + #{null >= model}) / (draws + 1).

The rule intentionally defines no constants of its own — fee and hold length come
from ExecutionConfig/PredictorConfig, the straddle gate compares against zero, and
the null parameters live in BenchmarkConfig.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch

from constants import BENCHMARK, DATA, EXECUTION, PREDICTOR

_CLOSE = DATA.FEATURE_NAMES.index("close_logret")
_Q10 = PREDICTOR.QUANTILES.index(0.10)
_Q50 = PREDICTOR.QUANTILES.index(0.50)
_Q90 = PREDICTOR.QUANTILES.index(0.90)

FloatArray = npt.NDArray[np.float64]
SignalArray = npt.NDArray[np.int8]


class TradeLedger(NamedTuple):
    """Executed trades: entry origin indices, directions (+1/-1), and returns.

    ``gross_returns`` is the pre-fee outcome (direction x realised move); ``net_returns``
    subtracts one round-trip fee. The two split the "was the trade on the right side"
    question (gross > 0) from the "did it also clear the fee" question (net > 0).
    """

    entries: npt.NDArray[np.int64]
    directions: SignalArray
    net_returns: FloatArray
    gross_returns: FloatArray


def extract_signals(
    pred: torch.Tensor, *, fee_threshold: float = EXECUTION.FEE_THRESHOLD
) -> SignalArray:
    """Per-origin trade direction (+1 long, -1 short, 0 no trade) from the fixed rule.

    ``pred`` is (N, H, DIM, Q) in CUMULATIVE-close space (PredictorConfig
    TARGET_SEMANTICS); only the final horizon step is read — the move a
    hold-to-horizon trade actually spans. An interval endpoint exactly at zero counts
    as straddling (``q10 * q90 > 0`` is strict): the model is not confidently
    one-sided, so the gate stays closed.
    """
    q10 = pred[:, -1, _CLOSE, _Q10]
    q50 = pred[:, -1, _CLOSE, _Q50]
    q90 = pred[:, -1, _CLOSE, _Q90]
    tradeable = (q50.abs() > fee_threshold) & ((q10 * q90) > 0)
    direction = torch.sign(q50) * tradeable
    signals: SignalArray = direction.to(torch.int8).cpu().numpy()
    return signals


def simulate_trades(
    directions: SignalArray, realized_final: FloatArray, *, horizon: int, fee: float
) -> TradeLedger:
    """Sequential non-overlapping execution: enter on a signal, hold ``horizon`` steps.

    ``realized_final[i]`` is the realised cumulative close log-return over the
    ``horizon`` steps after origin ``i`` — exactly the move the trade holds through.
    While a position is open (the next ``horizon`` origins), signals are skipped, so
    trades never overlap. Net per trade = direction x realised move − one round-trip fee.
    """
    if directions.shape != realized_final.shape:
        raise ValueError(
            f"directions {directions.shape} and realized_final {realized_final.shape} "
            f"must align"
        )
    entries: list[int] = []
    dirs: list[int] = []
    nets: list[float] = []
    grosses: list[float] = []
    i = 0
    n = int(directions.shape[0])
    while i < n:
        d = int(directions[i])
        if d == 0:
            i += 1
            continue
        gross = d * float(realized_final[i])
        entries.append(i)
        dirs.append(d)
        grosses.append(gross)
        nets.append(gross - fee)
        i += horizon
    return TradeLedger(
        np.asarray(entries, dtype=np.int64),
        np.asarray(dirs, dtype=np.int8),
        np.asarray(nets, dtype=np.float64),
        np.asarray(grosses, dtype=np.float64),
    )


def ledger_stats(
    net_returns: FloatArray, gross_returns: FloatArray | None = None
) -> dict[str, float]:
    """Net return, per-trade Sharpe, max drawdown, trade count, and two hit rates.

    All in net-of-fee log-return units. ``hit_rate`` is the NET win rate (share of
    trades profitable AFTER the round-trip fee). ``directional_hit_rate`` is the gross
    win rate from ``gross_returns`` (share of trades on the right SIDE, pre-fee, ~0.5
    expected) — the number a reader intuitively means by "hit rate"; NaN when
    ``gross_returns`` is absent/empty. Sharpe is per-trade (mean/sample-std of trade
    returns, no annualization — trade spacing is signal-dependent, so a time
    annualization would fabricate precision). NaN where undefined (no trades, or a
    single trade with no return dispersion). Max drawdown is peak-to-trough of the
    cumulative equity curve, measured from a starting equity of 0.
    """
    directional_hit_rate = (
        float(np.mean(gross_returns > 0.0))
        if gross_returns is not None and gross_returns.size > 0
        else float("nan")
    )
    n = int(net_returns.size)
    if n == 0:
        return {
            "net_return": 0.0, "sharpe": float("nan"), "max_drawdown": 0.0,
            "trade_count": 0, "hit_rate": float("nan"),
            "directional_hit_rate": directional_hit_rate,
        }
    equity = np.cumsum(net_returns)
    peaks = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    max_drawdown = float(np.max(peaks - equity))
    if n >= 2:
        std = float(np.std(net_returns, ddof=1))
        sharpe = float(np.mean(net_returns)) / std if std > 0.0 else float("nan")
    else:
        sharpe = float("nan")
    return {
        "net_return": float(np.sum(net_returns)),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "trade_count": n,
        "hit_rate": float(np.mean(net_returns > 0.0)),
        "directional_hit_rate": directional_hit_rate,
    }


def buy_and_hold(total_logret: float, *, fee: float) -> float:
    """Net log-return of holding across the whole evaluation span (one round trip)."""
    return total_logret - fee


def _sample_non_overlapping(
    rng: np.random.Generator, n_origins: int, count: int, horizon: int
) -> npt.NDArray[np.int64]:
    """Uniformly sample EXACTLY ``count`` origins pairwise >= ``horizon`` apart.

    Stars-and-bars placement, not greedy: ``count`` trades reserve ``(count-1)*horizon``
    origin-slots of mandatory spacing, leaving ``slack = n_origins - 1 - (count-1)*
    horizon`` free slots to distribute as gaps. Drawing ``count`` distinct cut positions
    from ``slack + count`` and mapping ``entry_j = cut_j + j*(horizon-1)`` gives a
    uniform valid placement that ALWAYS holds ``count`` entries. (The greedy predecessor
    silently fell short under tight packing — it rarely reached ``count`` near the
    packing limit — so each null draw traded fewer times than the model, shrinking the
    null distribution and biasing the p-value toward looking significant.) Feasibility
    (``slack >= 0``) is guaranteed whenever the model itself placed ``count`` such trades
    on the same origins (an existence proof); a genuinely infeasible request raises
    rather than quietly returning fewer.
    """
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    slack = n_origins - 1 - (count - 1) * horizon
    if slack < 0:
        raise ValueError(
            f"cannot place {count} non-overlapping trades of hold {horizon} "
            f"in {n_origins} origins"
        )
    cuts = np.sort(rng.choice(slack + count, size=count, replace=False))
    return cuts + np.arange(count, dtype=np.int64) * (horizon - 1)


def random_entry_null(
    *,
    n_origins: int,
    trade_count: int,
    realized_final: FloatArray,
    horizon: int,
    fee: float,
    draws: int,
    seed: int,
    model_net: float,
) -> dict[str, float]:
    """Random-entry null at matched trade frequency (the permutation-test idea).

    Each draw places ``trade_count`` non-overlapping random entries with random
    direction on the SAME realised returns and pays the same fees; the model's net PnL
    is then read against that null distribution. p_value uses the add-one estimator
    (1 + #{null >= model}) / (draws + 1) — never exactly 0, floor 1/(draws+1).
    NaN p-value when the model made no trades (nothing to test).
    """
    if trade_count <= 0 or n_origins <= 0:
        return {
            "null_mean": float("nan"), "null_std": float("nan"),
            "p_value": float("nan"), "null_draws": draws,
        }
    rng = np.random.default_rng(seed)
    nulls = np.empty(draws, dtype=np.float64)
    for d in range(draws):
        entries = _sample_non_overlapping(rng, n_origins, trade_count, horizon)
        directions = rng.choice(np.array([-1.0, 1.0]), size=entries.shape[0])
        nulls[d] = float(np.sum(directions * realized_final[entries])) - fee * entries.shape[0]
    p_value = (1.0 + float(np.count_nonzero(nulls >= model_net))) / (draws + 1.0)
    return {
        "null_mean": float(np.mean(nulls)),
        "null_std": float(np.std(nulls)),
        "p_value": p_value,
        "null_draws": draws,
    }


def profitability_grade(net_return: float, trade_count: int, p_value: float) -> str:
    """Leaderboard green-grade for one model: 'profitable' | 'not_profitable' |
    'insufficient'.

    A model is 'profitable' (green) iff it clears all three gates:
      * mean net-of-fee log-return per trade > 0 (positive expectancy — the > $0/trade
        the user asked for; at trade_count > 0 this is just sign(net_return)),
      * its net PnL beats the random-entry null at ``p_value <
        BENCHMARK.PROFITABLE_P_VALUE_MAX`` (statistically distinct from luck), and
      * ``trade_count >= BENCHMARK.PROFITABLE_MIN_TRADES``.

    Below the trade floor, or with a non-finite ``net_return``/``p_value`` (a model that
    made no trades has a NaN p-value), the expectancy is luck-dominated / undefined and
    the model is 'insufficient' — deliberately neither green nor red. Otherwise
    'not_profitable' (red). Pure function of the three persisted result numbers, so
    thresholds can change without re-running any benchmark.
    """
    if (
        trade_count < BENCHMARK.PROFITABLE_MIN_TRADES
        or not math.isfinite(net_return)
        or not math.isfinite(p_value)
    ):
        return "insufficient"
    expectancy = net_return / trade_count
    if expectancy > 0.0 and p_value < BENCHMARK.PROFITABLE_P_VALUE_MAX:
        return "profitable"
    return "not_profitable"
