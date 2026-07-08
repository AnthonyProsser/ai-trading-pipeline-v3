"""PatchTST predictor — encoder-only Transformer with channel-mixing patch embedding,
RevIN-style instance normalization, and a monotone (non-crossing) quantile head.

A single forward pass maps the lookback window to the full multi-step quantile
forecast (autoregression is banned):

    x : (batch, lookback, NUM_INPUT_FEATURES)
        -> y : (batch, HORIZON, NUM_OUTPUT_DIMS, NUM_QUANTILES)

Output semantics (``PredictorConfig.TARGET_SEMANTICS``): quantiles of the CUMULATIVE
log-return path from the forecast origin — ``y[:, h]`` forecasts the total (h+1)-step
move, the quantity the trader actually trades. Quantiles are not additive, so per-step
quantiles could never provide a calibrated interval for the horizon move; predicting
the cumulative path models it directly.

Three structural choices beyond the plain PatchTST encoder:

1. RevIN input normalization: each window is standardised per feature over the
   lookback dim with a learnable affine. This is invariant to the per-fold MinMax
   scaler's fixed affine (so the data pipeline is untouched) and adapts the encoder's
   input distribution to the current volatility regime instead of to fold-global
   min/max outliers.
2. Volatility-conditioned output scale: predicted quantiles are multiplied by the
   window's own per-feature sigma, so interval width tracks current volatility by
   construction — the head only has to learn regime-relative shape.
3. Monotone quantile head: the median is predicted directly and the outer quantiles
   are offset from it by softplus increments, so q10 <= q50 <= q90 holds for every
   output coordinate of any input. Crossing quantiles are impossible, not merely
   discouraged by the pinball loss.

`lookback` must be divisible by ``PATCH_SIZE``; it is split into
``lookback / PATCH_SIZE`` non-overlapping patches. In ``channel_mixing`` mode each
patch flattens its ``PATCH_SIZE`` timesteps across all input features into one token,
so intra-candle cross-feature (OHLCV) interaction is modelled. patch_size=16 drops
attention from O(lookback^2) to O(num_tokens^2) -- the choice that makes 8GB VRAM
feasible. All hyperparameters come from constants.py; this module hardcodes none.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from constants import DATA, PREDICTOR


class PatchTST(nn.Module):
    """Channel-mixing PatchTST encoder producing a (HORIZON, NUM_OUTPUT_DIMS,
    NUM_QUANTILES) monotone quantile tensor per sample in one forward pass."""

    def __init__(self, lookback: int = DATA.LOOKBACK) -> None:
        super().__init__()
        if lookback % PREDICTOR.PATCH_SIZE != 0:
            raise ValueError(
                f"lookback={lookback} must be divisible by PATCH_SIZE="
                f"{PREDICTOR.PATCH_SIZE} to keep the token count integer"
            )
        if PREDICTOR.PATCH_EMBED_MODE != "channel_mixing":
            raise NotImplementedError(
                f"only channel_mixing patch embedding is implemented, not "
                f"{PREDICTOR.PATCH_EMBED_MODE!r}"
            )
        if PREDICTOR.NUM_OUTPUT_DIMS > DATA.NUM_INPUT_FEATURES:
            raise ValueError(
                "output dims must be a prefix of the input features (volatility-"
                f"conditioned output scaling maps each output dim 1:1 to its input "
                f"feature); NUM_OUTPUT_DIMS={PREDICTOR.NUM_OUTPUT_DIMS} > "
                f"NUM_INPUT_FEATURES={DATA.NUM_INPUT_FEATURES}"
            )
        if tuple(sorted(PREDICTOR.QUANTILES)) != PREDICTOR.QUANTILES:
            raise ValueError(
                f"QUANTILES must be strictly ascending for the monotone head, got "
                f"{PREDICTOR.QUANTILES}"
            )

        self.lookback = lookback
        self.num_tokens = lookback // PREDICTOR.PATCH_SIZE
        self.patch_dim = PREDICTOR.PATCH_SIZE * DATA.NUM_INPUT_FEATURES
        n_quantiles = len(PREDICTOR.QUANTILES)
        self.out_per_step = PREDICTOR.NUM_OUTPUT_DIMS * n_quantiles
        # Median anchor for the monotone head; quantiles above/below it are built from
        # cumulative softplus offsets.
        self._anchor = n_quantiles // 2

        # RevIN learnable affine (applied after per-window standardisation). Sized to
        # NUM_OUTPUT_DIMS, not NUM_INPUT_FEATURES: RevIN only normalises the OHLCV
        # prefix of the input -- clock features bypass it (see forward()).
        self.revin_weight = nn.Parameter(torch.ones(PREDICTOR.NUM_OUTPUT_DIMS))
        self.revin_bias = nn.Parameter(torch.zeros(PREDICTOR.NUM_OUTPUT_DIMS))

        self.patch_embed = nn.Linear(self.patch_dim, PREDICTOR.D_MODEL)
        # Learnable positional embedding, zero-initialised (no bare init literal).
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, PREDICTOR.D_MODEL))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=PREDICTOR.D_MODEL,
            nhead=PREDICTOR.N_HEADS,
            dim_feedforward=PREDICTOR.D_FF,
            dropout=PREDICTOR.DROPOUT,
            activation=PREDICTOR.ACTIVATION,
            batch_first=True,
            norm_first=PREDICTOR.NORM_FIRST,  # pre-LN: steadier gradients, fewer NaN blow-ups
        )
        # norm=LayerNorm gives the pre-LN stack its final normalisation, so the head sees
        # a normalised residual stream (without it the last block's output is unnormalised).
        # enable_nested_tensor=False is a correctness guard: with batch_first and no mask it
        # avoids a version-dependent NestedTensor return that would break the later reshape.
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=PREDICTOR.N_LAYERS,
            norm=nn.LayerNorm(PREDICTOR.D_MODEL),
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(
            self.num_tokens * PREDICTOR.D_MODEL, PREDICTOR.HORIZON * self.out_per_step
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        n_out = PREDICTOR.NUM_OUTPUT_DIMS
        # RevIN applies only to the OHLCV prefix. Clock features (tod/dow sin/cos) are
        # bounded and already MinMax-scaled by the fold scaler; a near-constant window
        # (e.g. day-of-week within a 24h lookback) would drive RevIN's sigma toward 0
        # and blow up (x - mean) / sigma, so they bypass RevIN and reach the patch
        # embedding as-is.
        x_ohlcv = x[..., :n_out]
        x_clock = x[..., n_out:]
        # RevIN: standardise each window per feature. Any fixed affine the fold scaler
        # applied cancels out here, so the encoder sees the same distribution whether it
        # is fed raw or MinMax-scaled features.
        mean = x_ohlcv.mean(dim=1, keepdim=True)  # (batch, 1, out_dims)
        sigma = x_ohlcv.std(dim=1, keepdim=True) + PREDICTOR.REVIN_EPS  # (batch, 1, out_dims)
        z_ohlcv = (x_ohlcv - mean) / sigma * self.revin_weight + self.revin_bias
        z = torch.cat([z_ohlcv, x_clock], dim=-1)

        # (batch, lookback, feat) -> (batch, num_tokens, patch_size*feat): non-overlapping
        # patches of PATCH_SIZE consecutive timesteps with all features mixed in.
        patches = z.reshape(batch, self.num_tokens, self.patch_dim)
        tokens = self.patch_embed(patches) + self.pos_embed
        encoded = self.encoder(tokens)  # (batch, num_tokens, d_model)
        flat = encoded.reshape(batch, self.num_tokens * PREDICTOR.D_MODEL)
        raw = self.head(flat).reshape(
            batch, PREDICTOR.HORIZON, PREDICTOR.NUM_OUTPUT_DIMS, len(PREDICTOR.QUANTILES)
        )

        # Monotone quantiles in normalised space: the anchor channel is the median,
        # channels above/below add/subtract cumulative softplus offsets.
        levels: list[torch.Tensor | None] = [None] * len(PREDICTOR.QUANTILES)
        levels[self._anchor] = raw[..., self._anchor]
        for j in range(self._anchor + 1, len(PREDICTOR.QUANTILES)):
            prev_up = levels[j - 1]
            assert prev_up is not None
            levels[j] = prev_up + F.softplus(raw[..., j])
        for j in range(self._anchor - 1, -1, -1):
            prev_down = levels[j + 1]
            assert prev_down is not None
            levels[j] = prev_down - F.softplus(raw[..., j])
        y_norm = torch.stack([level for level in levels if level is not None], dim=-1)

        # Volatility-conditioned output scale: each output dim is rescaled by its input
        # feature's window sigma, so quantile spread tracks the current regime. The fold
        # scaler's fixed per-feature factor folds into the learned head weights.
        scale = sigma.squeeze(1)[:, None, :, None]  # (batch, 1, dims, 1)
        return y_norm * scale
