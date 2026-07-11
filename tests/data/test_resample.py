"""Exit criteria for src/data/resample.py (1-minute -> 1-hour OHLCV aggregation,
idea-07-daily-hourly)."""
from __future__ import annotations

import numpy as np
import pytest

from src.data.resample import resample_to_hourly
from src.data.validator import OhlcvCol


def _minute_ohlcv(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    open_ = 100.0 + rng.standard_normal(n).cumsum() * 0.01
    close = open_ + rng.standard_normal(n) * 0.01
    high = np.maximum(open_, close) + np.abs(rng.standard_normal(n)) * 0.01
    low = np.minimum(open_, close) - np.abs(rng.standard_normal(n)) * 0.01
    volume = np.abs(rng.standard_normal(n)) + 1.0
    return np.column_stack([open_, high, low, close, volume]).astype(np.float64)


def _minute_timestamps(start: str, n: int) -> np.ndarray:
    base = np.datetime64(start, "m")
    return base + np.arange(n).astype("timedelta64[m]")


def test_hourly_aggregation_matches_manual_reduction() -> None:
    # Two complete hours: 00:00-00:59 and 01:00-01:59.
    n = 120
    ts = _minute_timestamps("2020-01-01T00:00", n)
    ohlcv = _minute_ohlcv(n, seed=0)

    hourly_ts, hourly_ohlcv = resample_to_hourly(ts, ohlcv)

    assert hourly_ts.shape[0] == 2
    np.testing.assert_array_equal(
        hourly_ts, np.array(["2020-01-01T00", "2020-01-01T01"], dtype="datetime64[h]")
    )
    for i, (lo, hi) in enumerate([(0, 60), (60, 120)]):
        assert hourly_ohlcv[i, OhlcvCol.OPEN] == ohlcv[lo, OhlcvCol.OPEN]
        assert hourly_ohlcv[i, OhlcvCol.CLOSE] == ohlcv[hi - 1, OhlcvCol.CLOSE]
        assert hourly_ohlcv[i, OhlcvCol.HIGH] == ohlcv[lo:hi, OhlcvCol.HIGH].max()
        assert hourly_ohlcv[i, OhlcvCol.LOW] == ohlcv[lo:hi, OhlcvCol.LOW].min()
        assert hourly_ohlcv[i, OhlcvCol.VOLUME] == pytest.approx(
            ohlcv[lo:hi, OhlcvCol.VOLUME].sum()
        )


def test_trailing_partial_hour_dropped() -> None:
    n = 90  # one full hour + 30 trailing minutes
    ts = _minute_timestamps("2020-01-01T00:00", n)
    ohlcv = _minute_ohlcv(n, seed=1)

    hourly_ts, hourly_ohlcv = resample_to_hourly(ts, ohlcv)

    assert hourly_ts.shape[0] == 1
    assert hourly_ohlcv.shape[0] == 1
    np.testing.assert_array_equal(hourly_ts, np.array(["2020-01-01T00"], dtype="datetime64[h]"))


def test_forward_only_no_leakage() -> None:
    n = 120
    ts = _minute_timestamps("2020-01-01T00:00", n)
    ohlcv = _minute_ohlcv(n, seed=2)

    _, before = resample_to_hourly(ts, ohlcv)

    mutated = ohlcv.copy()
    mutated[100, OhlcvCol.HIGH] = 1e9  # a minute inside the SECOND hour
    mutated[100, OhlcvCol.CLOSE] = 1e9
    _, after = resample_to_hourly(ts, mutated)

    # The first hour's bar must be untouched by a mutation confined to the second hour.
    np.testing.assert_array_equal(before[0], after[0])
    assert not np.array_equal(before[1], after[1])
