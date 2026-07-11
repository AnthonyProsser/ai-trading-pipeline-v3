"""1-minute -> 1-hour OHLCV aggregation (idea-07-daily-hourly experiment).

Groups rows by UTC hour (timestamp floored to the hour) and aggregates each hourly
bar from only its own minutes (forward-only, no leakage across hour boundaries).
Column order matches src.data.validator.OhlcvCol.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.data.validator import OhlcvCol

_MINUTES_PER_HOUR = 60


def resample_to_hourly(
    timestamps: npt.NDArray[np.datetime64], ohlcv: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.datetime64], npt.NDArray[np.float64]]:
    """(N,) minute timestamps + (N, 5) OHLCV -> (M,) hour-start timestamps + (M, 5) OHLCV.

    Assumes `timestamps` is sorted ascending, so every hour's minutes are contiguous.
    Per hour: open = first row's open, high = max(high), low = min(low), close = last
    row's close, volume = sum(volume). The trailing hour is dropped if it has fewer
    than 60 minutes (an incomplete hour at the end of the series) -- every other hour
    is kept as-is even if a mid-series gap left it with fewer than 60 rows.
    """
    if timestamps.shape[0] != ohlcv.shape[0]:
        raise ValueError(
            f"timestamps and ohlcv must have the same length: "
            f"{timestamps.shape[0]} != {ohlcv.shape[0]}"
        )
    n = timestamps.shape[0]
    empty_ts: npt.NDArray[np.datetime64] = np.empty(0, dtype="datetime64[h]")
    empty_ohlcv: npt.NDArray[np.float64] = np.empty((0, len(OhlcvCol)), dtype=np.float64)
    if n == 0:
        return empty_ts, empty_ohlcv

    hour = timestamps.astype("datetime64[h]")
    is_group_start = np.empty(n, dtype=bool)
    is_group_start[0] = True
    is_group_start[1:] = hour[1:] != hour[:-1]
    group_starts = np.flatnonzero(is_group_start)
    group_ends = np.append(group_starts[1:], n)  # exclusive end per group
    group_sizes = group_ends - group_starts

    if group_sizes[-1] < _MINUTES_PER_HOUR:
        group_starts = group_starts[:-1]
        group_ends = group_ends[:-1]

    if group_starts.shape[0] == 0:
        return empty_ts, empty_ohlcv

    cutoff = int(group_ends[-1])  # end of the last kept group == start of any dropped tail
    open_col = ohlcv[:cutoff, OhlcvCol.OPEN]
    high_col = ohlcv[:cutoff, OhlcvCol.HIGH]
    low_col = ohlcv[:cutoff, OhlcvCol.LOW]
    close_col = ohlcv[:cutoff, OhlcvCol.CLOSE]
    volume_col = ohlcv[:cutoff, OhlcvCol.VOLUME]

    hourly_open = open_col[group_starts]
    hourly_close = close_col[group_ends - 1]
    hourly_high = np.maximum.reduceat(high_col, group_starts)
    hourly_low = np.minimum.reduceat(low_col, group_starts)
    hourly_volume = np.add.reduceat(volume_col, group_starts)

    hourly_ts = hour[group_starts]
    hourly_ohlcv = np.column_stack(
        [hourly_open, hourly_high, hourly_low, hourly_close, hourly_volume]
    ).astype(np.float64)
    return hourly_ts, hourly_ohlcv
