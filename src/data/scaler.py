"""Per-fold MinMax scaler with a strict fold-window assertion.

Contract: feature-pipeline.md "Scaler contract". Stats are fit on a fold's TRAIN
slice only; the same scaler transforms that fold's val + test slices. The allowed
window is the whole fold [fold_start, fold_end]; a transform() whose timestamps fall
outside it raises, which catches next-fold leakage by construction.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


class PerFoldMinMaxScaler:
    def __init__(self, *, fold_start: np.datetime64, fold_end: np.datetime64) -> None:
        if fold_end < fold_start:
            raise ValueError("fold_end precedes fold_start")
        self.fold_start = fold_start
        self.fold_end = fold_end
        self.data_min_: npt.NDArray[np.float64] | None = None
        self.data_max_: npt.NDArray[np.float64] | None = None

    def fit(self, x_train: npt.NDArray[np.float64]) -> PerFoldMinMaxScaler:
        x = np.asarray(x_train, dtype=np.float64)
        self.data_min_ = x.min(axis=0)
        self.data_max_ = x.max(axis=0)
        return self

    def _scale(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        assert self.data_min_ is not None and self.data_max_ is not None
        span = self.data_max_ - self.data_min_
        span[span == 0.0] = 1.0  # constant column -> map to 0 instead of dividing by zero
        scaled: npt.NDArray[np.float64] = (np.asarray(x, dtype=np.float64) - self.data_min_) / span
        return scaled

    def transform(
        self, x: npt.NDArray[np.float64], timestamps: npt.NDArray[np.datetime64]
    ) -> npt.NDArray[np.float64]:
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("transform() called before fit()")
        ts = timestamps.astype("datetime64[m]")
        if ts.size and (ts.min() < self.fold_start or ts.max() > self.fold_end):
            raise ValueError(
                "transform timestamps fall outside the fold window "
                f"[{self.fold_start}, {self.fold_end}] -- possible next-fold leakage"
            )
        return self._scale(x)

    def transform_inference(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Scale with the fitted fold stats, skipping the fold-window timestamp check.

        For locked-test / live inference, where windows legitimately fall outside the
        training fold but must use the trained min/max. transform() keeps the window
        assertion (the leakage tripwire); both share the one scaling formula via _scale.
        """
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("transform_inference() called before fit()")
        return self._scale(x)

    def fit_transform(
        self, x_train: npt.NDArray[np.float64], timestamps: npt.NDArray[np.datetime64]
    ) -> npt.NDArray[np.float64]:
        return self.fit(x_train).transform(x_train, timestamps)
