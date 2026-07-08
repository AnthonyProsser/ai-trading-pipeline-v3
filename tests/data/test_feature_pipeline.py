"""Exit criteria for the feature pipeline (feature-pipeline.md "Inputs").

idea-01-timefeatures: compute_features now also takes the aligned timestamps array and
appends 4 clock features (tod_sin, tod_cos, dow_sin, dow_cos) after the 5 OHLCV
log-return/volume-change columns, so NUM_INPUT_FEATURES goes 5 -> 9. The 5 OHLCV
columns and their row alignment (row t uses close_{t-1}) are unchanged.
"""
from __future__ import annotations

import numpy as np

from constants import DATA
from src.data.feature_pipeline import compute_features


def _ohlcv(rows: list[list[float]]) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


def _ts(n: int, start: str = "2024-01-01T00:00:00") -> np.ndarray:
    """n minute-spaced timestamps, aligned 1:1 with an n-row ohlcv array."""
    return np.datetime64(start) + np.arange(n).astype("timedelta64[m]")


def test_feature_count_matches_constant() -> None:
    assert len(DATA.FEATURE_NAMES) == DATA.NUM_INPUT_FEATURES


def test_num_input_features_is_nine() -> None:
    assert DATA.NUM_INPUT_FEATURES == 9
    assert DATA.FEATURE_NAMES[:5] == (
        "open_logret", "high_logret", "low_logret", "close_logret", "vol_change",
    )
    assert DATA.FEATURE_NAMES[5:] == ("tod_sin", "tod_cos", "dow_sin", "dow_cos")


def test_output_shape_is_n_minus_one_by_nine() -> None:
    ohlcv = _ohlcv([[10, 11, 9, 10, 100]] * 6)
    feats = compute_features(ohlcv, _ts(6))
    assert feats.shape == (5, DATA.NUM_INPUT_FEATURES)


def test_constant_close_gives_zero_close_logret() -> None:
    ohlcv = _ohlcv([[10, 11, 9, 10, 100], [10, 11, 9, 10, 100], [10, 11, 9, 10, 100]])
    feats = compute_features(ohlcv, _ts(3))
    close_idx = DATA.FEATURE_NAMES.index("close_logret")
    assert np.allclose(feats[:, close_idx], 0.0)


def test_close_logret_matches_log_ratio() -> None:
    # close path 10 -> 20 -> 10
    ohlcv = _ohlcv([[10, 21, 9, 10, 100], [10, 21, 9, 20, 100], [10, 21, 9, 10, 100]])
    feats = compute_features(ohlcv, _ts(3))
    close_idx = DATA.FEATURE_NAMES.index("close_logret")
    assert np.isclose(feats[0, close_idx], np.log(20 / 10))
    assert np.isclose(feats[1, close_idx], np.log(10 / 20))


def test_open_logret_uses_previous_close() -> None:
    # row 1: open 15 vs prev close 10
    ohlcv = _ohlcv([[10, 21, 9, 10, 100], [15, 21, 9, 12, 100]])
    feats = compute_features(ohlcv, _ts(2))
    open_idx = DATA.FEATURE_NAMES.index("open_logret")
    assert np.isclose(feats[0, open_idx], np.log(15 / 10))


def test_vol_change_doubling_and_constant() -> None:
    vol_idx = DATA.FEATURE_NAMES.index("vol_change")
    ohlcv = _ohlcv([[10, 11, 9, 10, 100], [10, 11, 9, 10, 200], [10, 11, 9, 10, 200]])
    feats = compute_features(ohlcv, _ts(3))
    assert np.isclose(feats[0, vol_idx], np.log1p(200 / 100 - 1))
    assert np.isclose(feats[1, vol_idx], 0.0)


def test_degenerate_volume_is_finite() -> None:
    # volume_t = 0 -> log -inf -> clipped to floor ; volume_{t-1} = 0 -> "no information"
    vol_idx = DATA.FEATURE_NAMES.index("vol_change")
    ohlcv = _ohlcv([[10, 11, 9, 10, 100], [10, 11, 9, 10, 0], [10, 11, 9, 10, 50]])
    feats = compute_features(ohlcv, _ts(3))
    assert np.isfinite(feats[:, vol_idx]).all()
    assert feats[0, vol_idx] == DATA.VOL_LOGRET_FLOOR
    assert feats[1, vol_idx] == DATA.VOL_CHANGE_DEGENERATE_FILL


def test_no_nan_or_inf_in_output() -> None:
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 1, 50))
    ohlcv = np.column_stack(
        [closes, closes + 1, closes - 1, closes, rng.uniform(1, 100, 50)]
    ).astype(np.float64)
    feats = compute_features(ohlcv, _ts(50))
    assert np.isfinite(feats).all()


def test_first_five_columns_match_ohlcv_logret_computation() -> None:
    """Guard against accidental column reordering: appending the 4 clock features must
    not disturb the pre-existing OHLCV log-return/vol-change computation or order."""
    ohlcv = _ohlcv(
        [[10, 21, 9, 10, 100], [12, 22, 8, 20, 150], [11, 23, 7, 15, 90], [9, 20, 6, 11, 40]]
    )
    feats = compute_features(ohlcv, _ts(4))

    prev_close = ohlcv[:-1, 3]
    prev_volume = ohlcv[:-1, 4]
    cur = ohlcv[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_open = np.log(cur[:, 0] / prev_close)
        expected_high = np.log(cur[:, 1] / prev_close)
        expected_low = np.log(cur[:, 2] / prev_close)
        expected_close = np.log(cur[:, 3] / prev_close)
        expected_vol = np.log1p(cur[:, 4] / prev_volume - 1.0)
    expected_vol[prev_volume == 0.0] = DATA.VOL_CHANGE_DEGENERATE_FILL
    expected_vol = np.maximum(expected_vol, DATA.VOL_LOGRET_FLOOR)

    assert np.allclose(feats[:, 0], expected_open)
    assert np.allclose(feats[:, 1], expected_high)
    assert np.allclose(feats[:, 2], expected_low)
    assert np.allclose(feats[:, 3], expected_close)
    assert np.allclose(feats[:, 4], expected_vol)


def test_time_of_day_known_values() -> None:
    tod_sin_idx = DATA.FEATURE_NAMES.index("tod_sin")
    tod_cos_idx = DATA.FEATURE_NAMES.index("tod_cos")
    ohlcv = _ohlcv([[10, 11, 9, 10, 100]] * 3)
    ts = np.array(
        ["2024-01-01T23:00:00", "2024-01-02T00:00:00", "2024-01-02T06:00:00"],
        dtype="datetime64[s]",
    )
    feats = compute_features(ohlcv, ts)
    # feats[i] is derived from ts[i + 1] (row 0 dropped, forward-only alignment).
    # ts[1] = 00:00:00Z -> tod_frac=0 -> sin=0, cos=1
    assert np.isclose(feats[0, tod_sin_idx], 0.0, atol=1e-9)
    assert np.isclose(feats[0, tod_cos_idx], 1.0, atol=1e-9)
    # ts[2] = 06:00:00Z -> tod_frac=0.25 -> sin=1, cos=0
    assert np.isclose(feats[1, tod_sin_idx], 1.0, atol=1e-9)
    assert np.isclose(feats[1, tod_cos_idx], 0.0, atol=1e-9)


def test_day_of_week_matches_independent_weekday_oracle() -> None:
    """dow_sin/dow_cos use a fixed Monday=0 phase (days-since-epoch + 3, mod
    DAYS_PER_WEEK -- 1970-01-01 was a Thursday). Cross-checked here against Python's
    stdlib date.weekday() (Monday=0), an oracle independent of the implementation."""
    import datetime as dt

    dow_sin_idx = DATA.FEATURE_NAMES.index("dow_sin")
    dow_cos_idx = DATA.FEATURE_NAMES.index("dow_cos")
    ohlcv = _ohlcv([[10, 11, 9, 10, 100]] * 2)

    for date_str in ("2024-03-04", "2024-03-05", "2024-03-10"):  # Mon, Tue, Sun
        expected_dow = dt.date.fromisoformat(date_str).weekday()
        ts = np.array(
            [np.datetime64(date_str) - np.timedelta64(1, "D"), np.datetime64(date_str)],
            dtype="datetime64[s]",
        )
        feats = compute_features(ohlcv, ts)
        expected_sin = np.sin(2 * np.pi * expected_dow / DATA.DAYS_PER_WEEK)
        expected_cos = np.cos(2 * np.pi * expected_dow / DATA.DAYS_PER_WEEK)
        assert np.isclose(feats[0, dow_sin_idx], expected_sin, atol=1e-9)
        assert np.isclose(feats[0, dow_cos_idx], expected_cos, atol=1e-9)


def test_day_of_week_seven_days_apart_matches_but_one_day_apart_differs() -> None:
    dow_sin_idx = DATA.FEATURE_NAMES.index("dow_sin")
    dow_cos_idx = DATA.FEATURE_NAMES.index("dow_cos")
    ohlcv = _ohlcv([[10, 11, 9, 10, 100]] * 2)

    def _dow_at(date_str: str) -> tuple[float, float]:
        ts = np.array(
            [np.datetime64(date_str) - np.timedelta64(1, "D"), np.datetime64(date_str)],
            dtype="datetime64[s]",
        )
        feats = compute_features(ohlcv, ts)
        return float(feats[0, dow_sin_idx]), float(feats[0, dow_cos_idx])

    same_dow = _dow_at("2024-01-08")  # 7 days after 2024-01-01, same weekday
    ref = _dow_at("2024-01-01")
    next_day = _dow_at("2024-01-02")  # 1 day later, different weekday

    assert np.isclose(same_dow[0], ref[0], atol=1e-9)
    assert np.isclose(same_dow[1], ref[1], atol=1e-9)
    assert not np.isclose(next_day[0], ref[0], atol=1e-6) or not np.isclose(
        next_day[1], ref[1], atol=1e-6
    )
