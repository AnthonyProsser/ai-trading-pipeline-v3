"""Walk-forward splitter + locked-test carve-out (splits-validation.md).

Stride = validation block size, so validation slices are non-overlapping (no Bonferroni
inflation in the permutation test). The terminal locked test block is reserved first and
never enters any fold.
"""
from __future__ import annotations

from dataclasses import dataclass

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
