"""Five forward-only log-return features per candle (feature-pipeline.md "Inputs")."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from constants import DATA
from src.data.validator import OhlcvCol


def compute_features(ohlcv: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """(N, 5) OHLCV -> (N-1, 5) features. Row t uses close_{t-1} (forward-only).

    Feature order/names are the single source of truth in ``DATA.FEATURE_NAMES``.
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
    feats = np.column_stack([open_lr, high_lr, low_lr, close_lr, vol_change]).astype(np.float64)
    assert feats.shape[1] == DATA.NUM_INPUT_FEATURES
    return feats
