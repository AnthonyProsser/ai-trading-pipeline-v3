"""PatchTST predictor — encoder-only Transformer with channel-mixing patch embedding
(predictor-training.md §"Architecture", predictor-contract.md §Input/Output).

A single forward pass maps the lookback window to the full multi-step quantile
forecast (autoregression is banned):

    x : (batch, lookback, NUM_INPUT_FEATURES)
        -> y : (batch, HORIZON, NUM_OUTPUT_DIMS, NUM_QUANTILES)

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

from constants import DATA, PREDICTOR


class PatchTST(nn.Module):
    """Channel-mixing PatchTST encoder producing a (HORIZON, NUM_OUTPUT_DIMS,
    NUM_QUANTILES) quantile tensor per sample in one forward pass."""

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

        self.lookback = lookback
        self.num_tokens = lookback // PREDICTOR.PATCH_SIZE
        self.patch_dim = PREDICTOR.PATCH_SIZE * DATA.NUM_INPUT_FEATURES
        n_quantiles = len(PREDICTOR.QUANTILES)
        self.out_per_step = PREDICTOR.NUM_OUTPUT_DIMS * n_quantiles

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
        # (batch, lookback, feat) -> (batch, num_tokens, patch_size*feat): non-overlapping
        # patches of PATCH_SIZE consecutive timesteps with all features mixed in.
        patches = x.reshape(batch, self.num_tokens, self.patch_dim)
        tokens = self.patch_embed(patches) + self.pos_embed
        encoded = self.encoder(tokens)  # (batch, num_tokens, d_model)
        flat = encoded.reshape(batch, self.num_tokens * PREDICTOR.D_MODEL)
        out: torch.Tensor = self.head(flat)  # (batch, HORIZON * NUM_OUTPUT_DIMS * NUM_QUANTILES)
        return out.reshape(
            batch, PREDICTOR.HORIZON, PREDICTOR.NUM_OUTPUT_DIMS, len(PREDICTOR.QUANTILES)
        )
