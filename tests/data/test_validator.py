"""Exit criteria for the CandleValidator (feature-pipeline.md "Validator")."""
from __future__ import annotations

import numpy as np

from constants import DATA
from src.data.validator import validate_candles

MIN = np.timedelta64(1, "m")
T0 = np.datetime64("2018-01-01T00:00", "m")


def _grid(n: int) -> np.ndarray:
    return T0 + MIN * np.arange(n)


def _ohlcv(rows: list[list[float]]) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


def test_clean_series_unchanged() -> None:
    ts = _grid(3)
    ohlcv = _ohlcv([[10, 11, 9, 10, 100], [10, 12, 10, 11, 120], [11, 11, 10, 10, 80]])
    res = validate_candles(ts, ohlcv)
    assert res.report.n_corrupt_dropped == 0
    assert res.report.n_interpolated == 0
    assert not res.report.requires_acknowledgement
    assert res.timestamps.shape == (3,)
    assert not res.is_interpolated.any()


def test_rejects_high_below_body() -> None:
    ts = _grid(3)
    # middle candle corrupt: high=10 < max(open=10.5, close=11)
    ohlcv = _ohlcv([[10, 11, 9, 10, 100], [10.5, 10, 9, 11, 120], [11, 12, 10, 11, 80]])
    res = validate_candles(ts, ohlcv)
    assert res.report.n_corrupt_dropped == 1
    assert res.timestamps.shape == (3,)  # corrupt minute re-gridded + interpolated
    assert res.is_interpolated[1]


def test_rejects_low_volume_and_close_rules() -> None:
    ts = _grid(4)
    ohlcv = _ohlcv([
        [10, 11, 9, 10, 100],
        [10, 12, 10.5, 9, 120],  # low 10.5 > min(open 10, close 9) -> corrupt
        [10, 12, 9, 11, -5],     # volume < 0 -> corrupt
        [10, 12, 9, 0, 80],      # close <= 0 -> corrupt
    ])
    res = validate_candles(ts, ohlcv)
    assert res.report.n_corrupt_dropped == 3


def test_duplicate_timestamp_keeps_last() -> None:
    ts = np.array([T0, T0 + MIN, T0 + MIN], dtype="datetime64[m]")
    ohlcv = _ohlcv([[10, 11, 9, 10, 100], [10, 12, 10, 11, 120], [10, 13, 10, 12, 130]])
    res = validate_candles(ts, ohlcv)
    assert res.report.n_duplicates_dropped == 1
    assert res.ohlcv[1, 3] == 12.0  # last duplicate's close kept


def test_short_gap_forward_filled() -> None:
    ts = np.array([T0, T0 + 3 * MIN], dtype="datetime64[m]")  # minutes 1,2 missing
    ohlcv = _ohlcv([[10, 11, 9, 10, 100], [10, 12, 10, 11, 120]])
    res = validate_candles(ts, ohlcv)
    assert res.timestamps.shape == (4,)
    assert res.report.n_interpolated == 2
    assert res.is_interpolated[1] and res.is_interpolated[2]
    assert res.ohlcv[1, 3] == 10.0 and res.ohlcv[1, 4] == 0.0  # flat candle from prior close, vol 0
    assert not res.report.requires_acknowledgement


def test_long_gap_requires_acknowledgement() -> None:
    ts = np.array([T0, T0 + np.timedelta64(13, "h")], dtype="datetime64[m]")
    ohlcv = _ohlcv([[10, 11, 9, 10, 100], [10, 12, 10, 11, 120]])
    res = validate_candles(ts, ohlcv)
    assert res.report.requires_acknowledgement
    assert any(h > DATA.GAP_INTERPOLATE_MAX_HOURS for h in res.report.long_gap_hours)


def test_output_grid_is_contiguous_and_sorted() -> None:
    ts = np.array([T0 + 2 * MIN, T0, T0 + 4 * MIN], dtype="datetime64[m]")  # unsorted + gaps
    ohlcv = _ohlcv([[10, 11, 9, 10, 100], [10, 12, 10, 11, 120], [11, 12, 10, 11, 80]])
    res = validate_candles(ts, ohlcv)
    diffs = np.diff(res.timestamps)
    assert (diffs == MIN).all()
