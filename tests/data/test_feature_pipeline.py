"""Exit criteria for the feature pipeline (feature-pipeline.md "Inputs")."""
from __future__ import annotations

import numpy as np

from constants import DATA
from src.data.feature_pipeline import compute_features


def _ohlcv(rows: list[list[float]]) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


def test_feature_count_matches_constant() -> None:
    assert len(DATA.FEATURE_NAMES) == DATA.NUM_INPUT_FEATURES


def test_output_shape_is_n_minus_one_by_five() -> None:
    ohlcv = _ohlcv([[10, 11, 9, 10, 100]] * 6)
    feats = compute_features(ohlcv)
    assert feats.shape == (5, DATA.NUM_INPUT_FEATURES)


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
    # volume_t = 0 -> log1p(-1) = -inf ; volume_{t-1} = 0 -> +inf : both must be handled
    vol_idx = DATA.FEATURE_NAMES.index("vol_change")
    ohlcv = _ohlcv([[10, 11, 9, 10, 100], [10, 11, 9, 10, 0], [10, 11, 9, 10, 50]])
    feats = compute_features(ohlcv)
    assert np.isfinite(feats[:, vol_idx]).all()


def test_no_nan_or_inf_in_output() -> None:
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 1, 50))
    ohlcv = np.column_stack(
        [closes, closes + 1, closes - 1, closes, rng.uniform(1, 100, 50)]
    ).astype(np.float64)
    feats = compute_features(ohlcv)
    assert np.isfinite(feats).all()
