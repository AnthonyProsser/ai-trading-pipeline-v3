"""Early-stopping helper for the predictor training loop (predictor-training.md
regression #3: the v2 premature-early-stopping bug).

Patience is sourced from `constants.PredictorConfig.EARLY_STOPPING_PATIENCE`; this
module holds no magic numbers. It lives under `src/` (not the train script) so the
stopper is unit-testable and importable, per the `src/`-never-imports-`scripts/` rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class EarlyStopper:
    """Stop after `patience` consecutive non-improving validation observations.

    An observation "improves" when ``val_loss < best_so_far - min_delta``. `step`
    returns ``True`` once `patience` non-improving observations have accrued since
    the last improvement.
    """

    patience: int
    min_delta: float = 0.0
    _best: float = field(default=math.inf, init=False)
    _num_bad: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")

    def step(self, val_loss: float) -> bool:
        """Record a validation loss; return whether training should stop."""
        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._num_bad = 0
        else:
            self._num_bad += 1
        return self._num_bad >= self.patience
