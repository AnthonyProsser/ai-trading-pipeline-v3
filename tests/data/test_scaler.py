"""Exit criteria for the per-fold MinMaxScaler (feature-pipeline.md "Scaler contract").

The fit-window is the *whole fold* [fold_start, fold_end]: stats are computed on the
train slice only, but transform must accept the same fold's val + test slices. A
timestamp outside the fold window raises, which catches next-fold leakage by construction.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.scaler import PerFoldMinMaxScaler

MIN = np.timedelta64(1, "m")
T0 = np.datetime64("2018-01-01T00:00", "m")


def _ts(offset: int, n: int) -> np.ndarray:
    return T0 + offset * MIN + MIN * np.arange(n)


def test_fit_transform_train_maps_to_unit_range() -> None:
    x_train = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
    sc = PerFoldMinMaxScaler(fold_start=T0, fold_end=T0 + 100 * MIN).fit(x_train)
    out = sc.transform(x_train, _ts(0, 3))
    assert np.allclose(out.min(axis=0), 0.0)
    assert np.allclose(out.max(axis=0), 1.0)


def test_transform_uses_train_stats_not_later_data() -> None:
    x_train = np.array([[0.0], [1.0], [2.0]])
    sc = PerFoldMinMaxScaler(fold_start=T0, fold_end=T0 + 100 * MIN).fit(x_train)
    # val value 4.0 is outside train range [0,2]: (4-0)/(2-0) = 2.0 proves train stats are used
    out = sc.transform(np.array([[4.0]]), _ts(10, 1))
    assert np.isclose(out[0, 0], 2.0)


def test_fit_records_train_min_max_only() -> None:
    x_train = np.array([[0.0], [2.0]])
    sc = PerFoldMinMaxScaler(fold_start=T0, fold_end=T0 + 100 * MIN).fit(x_train)
    assert np.allclose(sc.data_min_, 0.0)
    assert np.allclose(sc.data_max_, 2.0)


def test_transform_after_fold_end_raises() -> None:
    sc = PerFoldMinMaxScaler(fold_start=T0, fold_end=T0 + 5 * MIN).fit(np.array([[0.0], [1.0]]))
    with pytest.raises(ValueError):
        sc.transform(np.array([[0.5]]), _ts(99, 1))


def test_transform_before_fold_start_raises() -> None:
    sc = PerFoldMinMaxScaler(fold_start=T0, fold_end=T0 + 5 * MIN).fit(np.array([[0.0], [1.0]]))
    with pytest.raises(ValueError):
        sc.transform(np.array([[0.5]]), np.array([T0 - MIN], dtype="datetime64[m]"))


def test_transform_inference_skips_window_check_and_matches_transform() -> None:
    sc = PerFoldMinMaxScaler(fold_start=T0, fold_end=T0 + 5 * MIN).fit(np.array([[0.0], [2.0]]))
    x = np.array([[1.0], [4.0]])
    in_window = sc.transform(x, _ts(0, 2))
    inference = sc.transform_inference(x)
    assert np.allclose(in_window, inference)
    # No timestamps, no fold-window assertion: an out-of-fold window scales with train stats.
    out = sc.transform_inference(np.array([[10.0]]))
    assert np.isclose(out[0, 0], 5.0)  # (10 - 0) / (2 - 0)


def test_fit_transform_matches_fit_then_transform() -> None:
    x = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
    ts = _ts(0, 3)
    a = PerFoldMinMaxScaler(fold_start=T0, fold_end=T0 + 100 * MIN).fit_transform(x, ts)
    sc = PerFoldMinMaxScaler(fold_start=T0, fold_end=T0 + 100 * MIN).fit(x)
    assert np.allclose(a, sc.transform(x, ts))


def test_constant_column_does_not_divide_by_zero() -> None:
    x_train = np.array([[5.0], [5.0], [5.0]])
    sc = PerFoldMinMaxScaler(fold_start=T0, fold_end=T0 + 10 * MIN).fit(x_train)
    out = sc.transform(x_train, _ts(0, 3))
    assert np.isfinite(out).all()
