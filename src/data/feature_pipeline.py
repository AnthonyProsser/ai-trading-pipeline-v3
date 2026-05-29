"""Five forward-only log-return features per candle (feature-pipeline.md "Inputs")."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from constants import DATA
from src.data.validator import OhlcvCol

FEATURE_NAMES: tuple[str, ...] = (
    "open_logret",
    "high_logret",
    "low_logret",
    "close_logret",
    "vol_change",
)


def compute_features(ohlcv: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """(N, 5) OHLCV -> (N-1, 5) features. Row t uses close_{t-1} (forward-only)."""
    prev_close = ohlcv[:-1, OhlcvCol.CLOSE]
    prev_volume = ohlcv[:-1, OhlcvCol.VOLUME]
    cur = ohlcv[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        feats = np.column_stack(
            [
                np.log(cur[:, OhlcvCol.OPEN] / prev_close),
                np.log(cur[:, OhlcvCol.HIGH] / prev_close),
                np.log(cur[:, OhlcvCol.LOW] / prev_close),
                np.log(cur[:, OhlcvCol.CLOSE] / prev_close),
                np.log1p(cur[:, OhlcvCol.VOLUME] / prev_volume - 1.0),
            ]
        ).astype(np.float64)
    # Degenerate volume ratios (zero current/prior volume) yield +/-inf or nan -> neutral 0.
    feats[~np.isfinite(feats)] = 0.0
    assert feats.shape[1] == DATA.NUM_INPUT_FEATURES
    return feats
