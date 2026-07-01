"""Exit criterion for scripts/train_predictor.py::prepare() real-data path —
DECISIONS.md 'historical_start': pre-2018-01-01 candles must never enter a fold
(committed before the fix, per the test-first discipline)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import scripts.train_predictor as train_predictor
from constants import DATA


def test_prepare_real_data_drops_pre_historical_start_candles(monkeypatch) -> None:
    cutoff = np.datetime64(DATA.HISTORICAL_START, "m")
    n_pre = 5_000  # candles before HISTORICAL_START, must be dropped
    n_post = (
        DATA.LOCKED_TEST_CANDLES
        + DATA.WALK_FORWARD_TRAIN
        + DATA.WALK_FORWARD_VAL
        + DATA.WALK_FORWARD_TEST
        + 1_000  # margin so at least one fold exists after the locked-test carve-out
    )
    n_total = n_pre + n_post
    start = cutoff - np.timedelta64(n_pre, "m")
    timestamps = start + np.arange(n_total).astype("timedelta64[m]")
    _, ohlcv = train_predictor.synthetic_candles(n_total, seed=DATA.NUM_INPUT_FEATURES)

    monkeypatch.setattr(train_predictor, "load_real_candles", lambda path: (timestamps, ohlcv))
    monkeypatch.setattr(Path, "exists", lambda self: True)

    features, feature_ts, fold = train_predictor.prepare(
        synthetic=False, lookback=DATA.LOOKBACK, fold_index=0
    )

    # Exactly n_post rows survive: compute_features drops row 0 (always pre-cutoff here),
    # and the historical-start filter drops every remaining pre-cutoff row.
    assert feature_ts.shape[0] == n_post
    assert features.shape[0] == n_post
    assert feature_ts.min() == cutoff

    # No fold boundary can reach back into dropped, pre-cutoff territory.
    assert feature_ts[fold.train_start] >= cutoff
    assert feature_ts[fold.test_end - 1] >= cutoff


def test_prepare_search_slice_uses_only_pre_historical_start_candles(monkeypatch) -> None:
    cutoff = np.datetime64(DATA.HISTORICAL_START, "m")
    span = DATA.SEARCH_SLICE_TRAIN + DATA.SEARCH_SLICE_VAL
    n_pre = span + 5_000  # headroom before cutoff
    n_post = 2_000  # candles after cutoff, must never appear in the search slice
    n_total = n_pre + n_post
    start = cutoff - np.timedelta64(n_pre, "m")
    timestamps = start + np.arange(n_total).astype("timedelta64[m]")
    _, ohlcv = train_predictor.synthetic_candles(n_total, seed=DATA.NUM_INPUT_FEATURES)

    monkeypatch.setattr(train_predictor, "load_real_candles", lambda path: (timestamps, ohlcv))
    monkeypatch.setattr(Path, "exists", lambda self: True)

    features, feature_ts, fold = train_predictor.prepare(
        synthetic=False, lookback=DATA.LOOKBACK, fold_index=0, search_slice=True
    )

    assert feature_ts.shape[0] == span
    assert features.shape[0] == span
    assert bool(np.all(feature_ts < cutoff))
    assert fold.train_end - fold.train_start == DATA.SEARCH_SLICE_TRAIN
    assert fold.val_end - fold.val_start == DATA.SEARCH_SLICE_VAL
    assert fold.test_start == fold.test_end == fold.val_end
