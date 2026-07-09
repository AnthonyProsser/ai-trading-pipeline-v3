"""Exit criteria for the feature pipeline (feature-pipeline.md "Inputs")."""
from __future__ import annotations

import numpy as np

from constants import DATA
from src.data.feature_pipeline import compute_features


def _ohlcv(rows: list[list[float]]) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


def test_feature_count_matches_constant() -> None:
    assert len(DATA.FEATURE_NAMES) == DATA.NUM_INPUT_FEATURES


def test_output_shape_is_n_minus_one_by_seven() -> None:
    ohlcv = _ohlcv([[10, 11, 9, 10, 100]] * 6)
    feats = compute_features(ohlcv)
    assert feats.shape == (5, DATA.NUM_INPUT_FEATURES)


def test_num_input_features_is_seven() -> None:
    # idea-05-swinglevels: 5 OHLCV + 2 swing-high/low distance features.
    assert DATA.NUM_INPUT_FEATURES == 7
    assert DATA.FEATURE_NAMES[:5] == (
        "open_logret", "high_logret", "low_logret", "close_logret", "vol_change",
    )
    assert DATA.FEATURE_NAMES[5:] == ("dist_swing_high", "dist_swing_low")


def test_constant_close_gives_zero_close_logret() -> None:
    ohlcv = _ohlcv([[10, 11, 9, 10, 100], [10, 11, 9, 10, 100], [10, 11, 9, 10, 100]])
    feats = compute_features(ohlcv)
    close_idx = DATA.FEATURE_NAMES.index("close_logret")
    assert np.allclose(feats[:, close_idx], 0.0)


def test_close_logret_matches_log_ratio() -> None:
    # close path 10 -> 20 -> 10
    ohlcv = _ohlcv([[10, 21, 9, 10, 100], [10, 21, 9, 20, 100], [10, 21, 9, 10, 100]])
    feats = compute_features(ohlcv)
    close_idx = DATA.FEATURE_NAMES.index("close_logret")
    assert np.isclose(feats[0, close_idx], np.log(20 / 10))
    assert np.isclose(feats[1, close_idx], np.log(10 / 20))


def test_open_logret_uses_previous_close() -> None:
    # row 1: open 15 vs prev close 10
    ohlcv = _ohlcv([[10, 21, 9, 10, 100], [15, 21, 9, 12, 100]])
    feats = compute_features(ohlcv)
    open_idx = DATA.FEATURE_NAMES.index("open_logret")
    assert np.isclose(feats[0, open_idx], np.log(15 / 10))


def test_vol_change_doubling_and_constant() -> None:
    vol_idx = DATA.FEATURE_NAMES.index("vol_change")
    ohlcv = _ohlcv([[10, 11, 9, 10, 100], [10, 11, 9, 10, 200], [10, 11, 9, 10, 200]])
    feats = compute_features(ohlcv)
    assert np.isclose(feats[0, vol_idx], np.log1p(200 / 100 - 1))
    assert np.isclose(feats[1, vol_idx], 0.0)


def test_degenerate_volume_is_finite() -> None:
    # volume_t = 0 -> log -inf -> clipped to floor ; volume_{t-1} = 0 -> "no information"
    vol_idx = DATA.FEATURE_NAMES.index("vol_change")
    ohlcv = _ohlcv([[10, 11, 9, 10, 100], [10, 11, 9, 10, 0], [10, 11, 9, 10, 50]])
    feats = compute_features(ohlcv)
    assert np.isfinite(feats[:, vol_idx]).all()
    assert feats[0, vol_idx] == DATA.VOL_LOGRET_FLOOR
    assert feats[1, vol_idx] == DATA.VOL_CHANGE_DEGENERATE_FILL


def test_no_nan_or_inf_in_output() -> None:
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 1, 50))
    ohlcv = np.column_stack(
        [closes, closes + 1, closes - 1, closes, rng.uniform(1, 100, 50)]
    ).astype(np.float64)
    feats = compute_features(ohlcv)
    assert np.isfinite(feats).all()


def test_first_five_columns_match_pre_swing_ohlcv_computation() -> None:
    # Guards against the swing columns being inserted before / mixed into the original
    # 5 OHLCV columns (reorder regression) -- output dims [0:5] must stay OHLCV.
    rng = np.random.default_rng(2)
    n = 40
    closes = 100 + np.cumsum(rng.normal(0, 1, n))
    ohlcv = np.column_stack(
        [closes, closes + 1, closes - 1, closes, rng.uniform(1, 100, n)]
    ).astype(np.float64)

    prev_close = ohlcv[:-1, 3]
    prev_volume = ohlcv[:-1, 4]
    cur = ohlcv[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        open_lr = np.log(cur[:, 0] / prev_close)
        high_lr = np.log(cur[:, 1] / prev_close)
        low_lr = np.log(cur[:, 2] / prev_close)
        close_lr = np.log(cur[:, 3] / prev_close)
        vol_change = np.log1p(cur[:, 4] / prev_volume - 1.0)
    vol_change[prev_volume == 0.0] = DATA.VOL_CHANGE_DEGENERATE_FILL
    vol_change = np.maximum(vol_change, DATA.VOL_LOGRET_FLOOR)
    expected = np.column_stack([open_lr, high_lr, low_lr, close_lr, vol_change])

    feats = compute_features(ohlcv)
    assert np.array_equal(feats[:, :5], expected)


def test_swing_distance_known_value_on_monotonic_series() -> None:
    # idea-05-swinglevels: a strictly monotonically-rising series with high == low ==
    # close means the trailing rolling high/low over any window ending at row i is
    # always high[i]/low[i] itself (nothing later in the window beats the newest bar)
    # -- so dist_swing_high is 0 everywhere and dist_swing_low is > 0 everywhere except
    # the very first output row is guaranteed > 0 too once there's a strictly-smaller
    # earlier bar in the window.
    n = DATA.SWING_WINDOW + 10
    closes = 100.0 + np.arange(n, dtype=np.float64)
    ohlcv = np.column_stack([closes, closes, closes, closes, np.full(n, 100.0)])

    feats = compute_features(ohlcv)
    high_idx = DATA.FEATURE_NAMES.index("dist_swing_high")
    low_idx = DATA.FEATURE_NAMES.index("dist_swing_low")

    assert np.allclose(feats[:, high_idx], 0.0)
    assert feats[-1, low_idx] > 0.0  # "at the top": far above the trailing window low
    assert np.all(feats[:, low_idx] > 0.0)


def test_swing_distance_is_forward_only() -> None:
    # Mutating a FUTURE ohlcv row must not change an earlier feature row's swing (or
    # any) columns -- no leakage.
    rng = np.random.default_rng(1)
    n = DATA.SWING_WINDOW + 50
    closes = 100 + np.cumsum(rng.normal(0, 0.5, n))
    ohlcv = np.column_stack(
        [closes, closes + 1, closes - 1, closes, rng.uniform(1, 100, n)]
    ).astype(np.float64)
    feats_before = compute_features(ohlcv)

    mutated = ohlcv.copy()
    mutated[-1, :] *= 2.0  # mutate the FUTURE-most row only
    feats_after = compute_features(mutated)

    # Every row except the very last output row (which depends on the final ohlcv row)
    # must be byte-identical.
    assert np.array_equal(feats_before[:-1], feats_after[:-1])
