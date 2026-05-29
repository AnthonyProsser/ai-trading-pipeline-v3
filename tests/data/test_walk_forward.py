"""Exit criteria for the walk-forward splitter + locked-test carve-out (splits-validation.md)."""
from __future__ import annotations

import pytest

from constants import DATA
from src.data.walk_forward import carve_locked_test, make_folds


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
    for a, b in zip(folds, folds[1:]):
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
