"""Walk-forward splitter + locked-test carve-out (splits-validation.md).

Stride = validation block size, so validation slices are non-overlapping (no Bonferroni
inflation in the permutation test). The terminal locked test block is reserved first and
never enters any fold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from constants import DATA


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


def filter_by_historical_start(
    timestamps: npt.NDArray[np.datetime64], features: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.datetime64], npt.NDArray[np.float64]]:
    """Drop candles before DATA.HISTORICAL_START (DECISIONS.md 'historical_start':
    pre-2017 microstructure rejected). Any caller building folds from timestamped
    features should filter through this before carve_locked_test/make_folds, so the
    exclusion is enforced regardless of caller.
    """
    cutoff = np.datetime64(DATA.HISTORICAL_START, "m")
    mask = timestamps >= cutoff
    return timestamps[mask], features[mask]


def carve_search_slice(
    timestamps: npt.NDArray[np.datetime64], features: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.datetime64], npt.NDArray[np.float64], Fold]:
    """Carve the search-loop dev slice: the most recent SEARCH_SLICE_TRAIN +
    SEARCH_SLICE_VAL candles strictly before HISTORICAL_START (DECISIONS.md
    'search_dev_slice'). This range is excluded from every real walk-forward fold and
    the locked test set, so it is safe for an unattended search loop to iterate on.
    No test split — the search loop judges candidates on val_total only.
    """
    cutoff = np.datetime64(DATA.HISTORICAL_START, "m")
    mask = timestamps < cutoff
    pre_ts = timestamps[mask]
    pre_features = features[mask]
    span = DATA.SEARCH_SLICE_TRAIN + DATA.SEARCH_SLICE_VAL
    if pre_ts.shape[0] < span:
        raise ValueError(
            f"only {pre_ts.shape[0]} pre-{DATA.HISTORICAL_START} candles available, "
            f"need {span} for the search dev-slice"
        )
    slice_ts = pre_ts[-span:]
    slice_features = pre_features[-span:]
    train_end = DATA.SEARCH_SLICE_TRAIN
    fold = Fold(0, 0, train_end, train_end, span, span, span)
    return slice_ts, slice_features, fold


def carve_locked_test(n_total: int) -> int:
    """Return the usable candle count after reserving the terminal locked test block."""
    n_usable = n_total - DATA.LOCKED_TEST_CANDLES
    if n_usable <= 0:
        raise ValueError(
            f"n_total={n_total} leaves no usable candles after the "
            f"{DATA.LOCKED_TEST_CANDLES}-candle locked test set"
        )
    return n_usable


def make_folds(n_usable: int) -> list[Fold]:
    """Walk-forward folds over [0, n_usable): train/val/test of fixed size, advancing by stride."""
    train, val, test = DATA.WALK_FORWARD_TRAIN, DATA.WALK_FORWARD_VAL, DATA.WALK_FORWARD_TEST
    span = train + val + test
    folds: list[Fold] = []
    start = 0
    while start + span <= n_usable:
        train_end = start + train
        val_end = train_end + val
        test_end = val_end + test
        folds.append(Fold(len(folds), start, train_end, train_end, val_end, val_end, test_end))
        start += DATA.WALK_FORWARD_STRIDE
    return folds
