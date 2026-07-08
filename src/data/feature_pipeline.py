"""Forward-only log-return features + cyclical clock features per candle
(feature-pipeline.md "Inputs"; idea-01-timefeatures for the clock features)."""
from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from constants import DATA
from src.data.validator import OhlcvCol


def compute_features(
    ohlcv: npt.NDArray[np.float64], timestamps: npt.NDArray[np.datetime64]
) -> npt.NDArray[np.float64]:
    """(N, 5) OHLCV + (N,) timestamps -> (N-1, 9) features. Row t uses close_{t-1}
    (forward-only) for the 5 OHLCV columns and timestamps[t] for the 4 clock columns.

    Feature order/names are the single source of truth in ``DATA.FEATURE_NAMES``:
    the 5 OHLCV log-return/vol-change columns, then 4 cyclical clock columns
    (tod_sin, tod_cos, dow_sin, dow_cos) encoding *where in the clock/week* the
    window sits -- inputs only, the model still predicts just the 5 OHLCV dims.
    """
    prev_close = ohlcv[:-1, OhlcvCol.CLOSE]
    prev_volume = ohlcv[:-1, OhlcvCol.VOLUME]
    cur = ohlcv[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        open_lr = np.log(cur[:, OhlcvCol.OPEN] / prev_close)
        high_lr = np.log(cur[:, OhlcvCol.HIGH] / prev_close)
        low_lr = np.log(cur[:, OhlcvCol.LOW] / prev_close)
        close_lr = np.log(cur[:, OhlcvCol.CLOSE] / prev_close)
        vol_change = np.log1p(cur[:, OhlcvCol.VOLUME] / prev_volume - 1.0)
    # Merged Phase 0 volume edge-case decisions (DECISIONS.md):
    #   volume_{t-1} == 0  -> "no information" (VOL_CHANGE_DEGENERATE_FILL)
    #   volume_t == 0 / underflow -> clip log to a finite floor (VOL_LOGRET_FLOOR), never -inf
    vol_change[prev_volume == 0.0] = DATA.VOL_CHANGE_DEGENERATE_FILL
    vol_change = np.maximum(vol_change, DATA.VOL_LOGRET_FLOOR)

    cur_ts = timestamps[1:]
    day_start = cur_ts.astype("datetime64[D]")
    tod_frac = (cur_ts - day_start) / np.timedelta64(1, "D")  # fraction of day, [0, 1)
    tod_sin = np.sin(math.tau * tod_frac)
    tod_cos = np.cos(math.tau * tod_frac)
    # Days since the 1970-01-01 epoch (a Thursday); fixed Monday=0 phase (+3 shifts
    # Thursday=3 to Monday=0). The phase choice is arbitrary but must be fixed --
    # only the period matters for seasonality.
    days = day_start.astype(np.int64)
    dow = (days + 3) % DATA.DAYS_PER_WEEK
    dow_sin = np.sin(math.tau * dow / DATA.DAYS_PER_WEEK)
    dow_cos = np.cos(math.tau * dow / DATA.DAYS_PER_WEEK)

    feats = np.column_stack(
        [open_lr, high_lr, low_lr, close_lr, vol_change, tod_sin, tod_cos, dow_sin, dow_cos]
    ).astype(np.float64)
    assert feats.shape[1] == DATA.NUM_INPUT_FEATURES
    return feats
