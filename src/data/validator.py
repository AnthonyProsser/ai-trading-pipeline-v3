"""CandleValidator — corruption + gap rules over a 1-minute OHLCV grid.

Contract: feature-pipeline.md "Validator". Runs before feature computation and
produces a complete, sorted 1-minute grid with an ``is_interpolated`` flag per row.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from constants import DATA


class OhlcvCol(IntEnum):
    """Column order for raw OHLCV candle arrays, shared across the data pipeline."""

    OPEN = 0
    HIGH = 1
    LOW = 2
    CLOSE = 3
    VOLUME = 4


@dataclass(frozen=True)
class ValidationReport:
    n_input: int
    n_corrupt_dropped: int
    n_duplicates_dropped: int
    n_interpolated: int
    long_gap_hours: tuple[float, ...]

    @property
    def requires_acknowledgement(self) -> bool:
        """A gap longer than the interpolation threshold needs explicit user sign-off."""
        return len(self.long_gap_hours) > 0


@dataclass(frozen=True)
class ValidatedCandles:
    timestamps: npt.NDArray[np.datetime64]
    ohlcv: npt.NDArray[np.float64]
    is_interpolated: npt.NDArray[np.bool_]
    report: ValidationReport


def _corrupt_mask(ohlcv: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    op, hi, lo = ohlcv[:, OhlcvCol.OPEN], ohlcv[:, OhlcvCol.HIGH], ohlcv[:, OhlcvCol.LOW]
    cl, vol = ohlcv[:, OhlcvCol.CLOSE], ohlcv[:, OhlcvCol.VOLUME]
    bad: npt.NDArray[np.bool_] = (
        (hi < np.maximum(op, cl)) | (lo > np.minimum(op, cl)) | (vol < 0.0) | (cl <= 0.0)
    )
    return bad


def _long_gap_hours(timestamps: npt.NDArray[np.datetime64]) -> tuple[float, ...]:
    if timestamps.shape[0] < 2:
        return ()
    hours = np.diff(timestamps) / np.timedelta64(1, "h")
    return tuple(float(h) for h in hours[hours > DATA.GAP_INTERPOLATE_MAX_HOURS])


def validate_candles(
    timestamps: npt.NDArray[np.datetime64], ohlcv: npt.NDArray[np.float64]
) -> ValidatedCandles:
    minute = np.timedelta64(1, "m")
    ts = timestamps.astype("datetime64[m]")
    ohlcv = np.asarray(ohlcv, dtype=np.float64)
    if ts.shape[0] != ohlcv.shape[0]:
        raise ValueError(f"timestamps and ohlcv must have the same length: {ts.shape[0]} != {ohlcv.shape[0]}")
    if ohlcv.ndim != 2 or ohlcv.shape[1] != len(OhlcvCol):
        raise ValueError(f"ohlcv must have shape (N, {len(OhlcvCol)}), got {ohlcv.shape}")
    n_input = int(ts.shape[0])

    # Sort by time (stable, so a duplicated minute keeps input order before we drop it).
    order = np.argsort(ts, kind="stable")
    ts, ohlcv = ts[order], ohlcv[order]

    # Drop duplicate timestamps, keeping the last occurrence.
    keep = np.ones(ts.shape[0], dtype=bool)
    keep[:-1] = ts[1:] != ts[:-1]
    n_duplicates = int((~keep).sum())
    ts, ohlcv = ts[keep], ohlcv[keep]

    # Drop corrupt candles (they become gaps to be interpolated below).
    corrupt = _corrupt_mask(ohlcv)
    n_corrupt = int(corrupt.sum())
    ts, ohlcv = ts[~corrupt], ohlcv[~corrupt]

    if ts.shape[0] == 0:
        report = ValidationReport(n_input, n_corrupt, n_duplicates, 0, ())
        return ValidatedCandles(
            ts, np.empty((0, len(OhlcvCol)), dtype=np.float64), np.empty(0, dtype=bool), report
        )

    long_gaps = _long_gap_hours(ts)

    # Reindex onto the full 1-minute grid [first_valid, last_valid].
    grid = np.arange(ts[0], ts[-1] + minute, minute, dtype="datetime64[m]")
    pos = np.searchsorted(grid, ts)
    present = np.zeros(grid.shape[0], dtype=bool)
    present[pos] = True
    filled = np.empty((grid.shape[0], len(OhlcvCol)), dtype=np.float64)
    filled[pos] = ohlcv
    is_interp = ~present

    # Forward-fill gaps as flat candles at the prior close, volume 0.
    last_present = np.where(present, np.arange(grid.shape[0]), -1)
    np.maximum.accumulate(last_present, out=last_present)
    carried_close = filled[last_present, OhlcvCol.CLOSE]
    for col in (OhlcvCol.OPEN, OhlcvCol.HIGH, OhlcvCol.LOW, OhlcvCol.CLOSE):
        filled[is_interp, col] = carried_close[is_interp]
    filled[is_interp, OhlcvCol.VOLUME] = 0.0

    report = ValidationReport(n_input, n_corrupt, n_duplicates, int(is_interp.sum()), long_gaps)
    return ValidatedCandles(grid, filled, is_interp, report)
