"""Exit criteria for the feature pipeline (feature-pipeline.md "Inputs")."""
from __future__ import annotations

import numpy as np

from constants import DATA
from src.data.feature_pipeline import compute_features


def _ohlcv(rows: list[list[float]]) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


def test_feature_count_matches_constant() -> None:
    assert len(DATA.FEATURE_NAMES) == DATA.NUM_INPUT_FEATURES


def test_output_shape_is_n_minus_one_by_nine() -> None:
    ohlcv = _ohlcv([[10, 11, 9, 10, 100]] * 6)
    feats = compute_features(ohlcv)
    assert feats.shape == (5, DATA.NUM_INPUT_FEATURES)


def test_num_input_features_is_nine() -> None:
    # idea-02-multiscale: 5 OHLCV + 4 multi-scale trailing momentum returns.
    assert DATA.NUM_INPUT_FEATURES == 9
    assert DATA.FEATURE_NAMES[:5] == (
        "open_logret", "high_logret", "low_logret", "close_logret", "vol_change",
    )
    assert DATA.FEATURE_NAMES[5:] == ("ret_5m", "ret_15m", "ret_60m", "ret_240m")


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


def test_first_five_columns_match_pre_multiscale_ohlcv_computation() -> None:
    # Guards against the multi-scale columns being inserted before / mixed into the
    # original 5 OHLCV columns (reorder regression) -- output dims [0:5] must stay OHLCV.
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


def test_multiscale_return_known_value_and_incomplete_window_fill() -> None:
    # idea-02-multiscale: constant per-step close log-return r -> ret_Wm == W*r for a
    # fully-covered row, and == VOL_CHANGE_DEGENERATE_FILL for an incomplete-window row.
    r = 0.001
    max_w = max(DATA.MULTISCALE_RETURN_WINDOWS)
    n = max_w + 10
    closes = np.exp(r * np.arange(n))
    ohlcv = np.column_stack([closes, closes, closes, closes, np.full(n, 100.0)])
    feats = compute_features(ohlcv)

    for window, name in zip(DATA.MULTISCALE_RETURN_WINDOWS, DATA.FEATURE_NAMES[5:], strict=True):
        idx = DATA.FEATURE_NAMES.index(name)
        # First fully-covered row: j = window - 1 (sums close_lr[0:window]).
        assert np.isclose(feats[window - 1, idx], window * r, atol=1e-8)
        # Last incomplete-window row: j = window - 2.
        assert feats[window - 2, idx] == DATA.VOL_CHANGE_DEGENERATE_FILL
        assert feats[0, idx] == DATA.VOL_CHANGE_DEGENERATE_FILL


def test_multiscale_return_is_forward_only() -> None:
    # Mutating a FUTURE ohlcv row must not change an earlier feature row's multi-scale
    # (or any) columns -- no leakage.
    rng = np.random.default_rng(1)
    n = 300
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
