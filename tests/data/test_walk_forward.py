"""Exit criteria for the walk-forward splitter + locked-test carve-out (splits-validation.md)."""
from __future__ import annotations

import numpy as np
import pytest

from constants import DATA
from src.data.walk_forward import (
    carve_locked_test,
    carve_search_slice,
    filter_by_historical_start,
    make_folds,
)


def test_carve_removes_locked_test_block() -> None:
    n_total = 1_000_000
    assert carve_locked_test(n_total) == n_total - DATA.LOCKED_TEST_CANDLES


def test_carve_raises_when_too_few_candles() -> None:
    with pytest.raises(ValueError):
        carve_locked_test(DATA.LOCKED_TEST_CANDLES)  # nothing left to train on


def test_fold_block_sizes_and_contiguity() -> None:
    f = make_folds(500_000)[0]
    assert f.train_end - f.train_start == DATA.WALK_FORWARD_TRAIN
    assert f.val_end - f.val_start == DATA.WALK_FORWARD_VAL
    assert f.test_end - f.test_start == DATA.WALK_FORWARD_TEST
    assert f.train_end == f.val_start and f.val_end == f.test_start


def test_stride_equals_validation_block() -> None:
    folds = make_folds(500_000)
    assert folds[1].train_start - folds[0].train_start == DATA.WALK_FORWARD_STRIDE
    assert DATA.WALK_FORWARD_STRIDE == DATA.WALK_FORWARD_VAL  # the non-overlap invariant


def test_validation_slices_non_overlapping_and_contiguous() -> None:
    folds = make_folds(500_000)
    for a, b in zip(folds, folds[1:], strict=False):
        assert a.val_end == b.val_start


def test_no_fold_extends_past_usable_region() -> None:
    n_usable = 400_000
    assert all(f.test_end <= n_usable for f in make_folds(n_usable))


def test_minimum_one_fold_starts_at_zero() -> None:
    n = DATA.WALK_FORWARD_TRAIN + DATA.WALK_FORWARD_VAL + DATA.WALK_FORWARD_TEST
    folds = make_folds(n)
    assert len(folds) == 1 and folds[0].train_start == 0


def test_too_few_candles_yields_no_folds() -> None:
    assert make_folds(DATA.WALK_FORWARD_TRAIN) == []


def test_filter_by_historical_start_drops_earlier_candles() -> None:
    ts = np.array(
        ["2016-01-01T00:00", "2017-12-31T23:59", "2018-01-01T00:00", "2018-06-01T00:00"],
        dtype="datetime64[m]",
    )
    features = np.arange(4 * DATA.NUM_INPUT_FEATURES, dtype=np.float64).reshape(
        4, DATA.NUM_INPUT_FEATURES
    )
    filtered_ts, filtered_features = filter_by_historical_start(ts, features)
    assert filtered_ts.shape[0] == 2
    assert filtered_features.shape[0] == 2
    assert filtered_ts.min() == np.datetime64(DATA.HISTORICAL_START, "m")
    np.testing.assert_array_equal(filtered_features, features[2:])


def test_filter_by_historical_start_keeps_all_when_already_post_cutoff() -> None:
    ts = np.array(["2019-01-01T00:00", "2019-01-01T00:01"], dtype="datetime64[m]")
    features = np.zeros((2, DATA.NUM_INPUT_FEATURES), dtype=np.float64)
    filtered_ts, filtered_features = filter_by_historical_start(ts, features)
    assert filtered_ts.shape[0] == 2
    assert filtered_features.shape[0] == 2


def _pre_cutoff_series(n: int) -> tuple[np.ndarray, np.ndarray]:
    """n contiguous 1-minute candles ending exactly 1 minute before HISTORICAL_START."""
    cutoff = np.datetime64(DATA.HISTORICAL_START, "m")
    ts = cutoff - np.arange(n, 0, -1).astype("timedelta64[m]")
    features = np.arange(n * DATA.NUM_INPUT_FEATURES, dtype=np.float64).reshape(
        n, DATA.NUM_INPUT_FEATURES
    )
    return ts, features


def test_carve_search_slice_sizes_match_constants() -> None:
    span = DATA.SEARCH_SLICE_TRAIN + DATA.SEARCH_SLICE_VAL
    ts, features = _pre_cutoff_series(span + 10_000)  # extra headroom before cutoff
    _, _, fold = carve_search_slice(ts, features)
    assert fold.train_end - fold.train_start == DATA.SEARCH_SLICE_TRAIN
    assert fold.val_end - fold.val_start == DATA.SEARCH_SLICE_VAL
    assert fold.train_end == fold.val_start
    assert fold.test_start == fold.test_end == fold.val_end  # no test split for search


def test_carve_search_slice_uses_most_recent_pre_cutoff_candles() -> None:
    span = DATA.SEARCH_SLICE_TRAIN + DATA.SEARCH_SLICE_VAL
    extra = 10_000
    ts, features = _pre_cutoff_series(span + extra)
    slice_ts, slice_features, _ = carve_search_slice(ts, features)
    assert slice_ts.shape[0] == span
    np.testing.assert_array_equal(slice_ts, ts[-span:])
    np.testing.assert_array_equal(slice_features, features[-span:])


def test_carve_search_slice_never_includes_post_cutoff_candles() -> None:
    span = DATA.SEARCH_SLICE_TRAIN + DATA.SEARCH_SLICE_VAL
    cutoff = np.datetime64(DATA.HISTORICAL_START, "m")
    pre_ts, pre_features = _pre_cutoff_series(span + 5_000)
    post_ts = cutoff + np.arange(1_000).astype("timedelta64[m]")
    post_features = np.zeros((1_000, DATA.NUM_INPUT_FEATURES), dtype=np.float64)
    ts = np.concatenate([pre_ts, post_ts])
    features = np.concatenate([pre_features, post_features])
    slice_ts, _, _ = carve_search_slice(ts, features)
    assert bool(np.all(slice_ts < cutoff))


def test_carve_search_slice_raises_when_insufficient_pre_cutoff_candles() -> None:
    span = DATA.SEARCH_SLICE_TRAIN + DATA.SEARCH_SLICE_VAL
    ts, features = _pre_cutoff_series(span - 1)
    with pytest.raises(ValueError):
        carve_search_slice(ts, features)
