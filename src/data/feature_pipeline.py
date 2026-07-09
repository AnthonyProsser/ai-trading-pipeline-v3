"""Seven forward-only log-return features per candle (feature-pipeline.md "Inputs"):
5 OHLCV + 2 swing-high/low distance features (idea-05-swinglevels)."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from constants import DATA
from src.data.validator import OhlcvCol


def _rolling_swing_high_low(
    high: npt.NDArray[np.float64], low: npt.NDArray[np.float64], window: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Trailing rolling max(high)/min(low) over the last ``window`` bars, including the
    current bar (forward-only: row i only reads high/low indices <= i).

    The first ``window - 1`` rows use a partial window (all bars available so far),
    equivalent to pandas ``rolling(window, min_periods=1)`` -- never NaN.
    """
    n = high.shape[0]
    rolling_high = np.empty(n, dtype=np.float64)
    rolling_low = np.empty(n, dtype=np.float64)
    prefix = min(window - 1, n)
    rolling_high[:prefix] = np.maximum.accumulate(high[:prefix])
    rolling_low[:prefix] = np.minimum.accumulate(low[:prefix])
    if n >= window:
        windows_high = np.lib.stride_tricks.sliding_window_view(high, window)
        windows_low = np.lib.stride_tricks.sliding_window_view(low, window)
        rolling_high[window - 1 :] = windows_high.max(axis=1)
        rolling_low[window - 1 :] = windows_low.min(axis=1)
    return rolling_high, rolling_low


def compute_features(ohlcv: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """(N, 5) OHLCV -> (N-1, 7) features. Row t uses close_{t-1} (forward-only).

    Feature order/names are the single source of truth in ``DATA.FEATURE_NAMES``: the
    5 OHLCV log-return/volume-change columns, followed by 2 swing-high/low distance
    columns (signed normalized distance from close to the rolling
    ``DATA.SWING_WINDOW``-bar high/low).
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

    rolling_high, rolling_low = _rolling_swing_high_low(
        ohlcv[:, OhlcvCol.HIGH], ohlcv[:, OhlcvCol.LOW], DATA.SWING_WINDOW
    )
    cur_close = cur[:, OhlcvCol.CLOSE]
    dist_swing_high = (cur_close - rolling_high[1:]) / cur_close
    dist_swing_low = (cur_close - rolling_low[1:]) / cur_close

    feats = np.column_stack(
        [open_lr, high_lr, low_lr, close_lr, vol_change, dist_swing_high, dist_swing_low]
    ).astype(np.float64)
    assert feats.shape[1] == DATA.NUM_INPUT_FEATURES
    return feats
