"""
constants.py — single source of truth for every magic number in btc-bot-v3.

All values are grouped into frozen dataclasses. `frozen=True` prevents
mutation at runtime, so a stray `cfg.FEE_RATE_TAKER = 0.001` raises.

Any change to any value here must be paired with a CHANGELOG.md entry
in the same commit. Pre-merge check enforces.

Sentinel values for unresolved decisions are typed `None` so a
consumer that forgets to handle them fails loudly rather than silently
defaulting. Resolution gates are documented in DECISIONS.md and CLAUDE.md §9.
"""
from dataclasses import dataclass


# ============================================================
# Data pipeline
# ============================================================
@dataclass(frozen=True)
class DataConfig:
    HISTORICAL_START: str = "2018-01-01"

    # Bootstrap ingest
    # KRAKEN_DATA_PATH env var points to the local Google Drive sync of Kraken_OHLCVT.zip
    # (Google Drive for Desktop, G:\My Drive\Kraken_OHLCVT.zip).
    KRAKEN_LOCAL_ZIP_ENV_VAR: str = "KRAKEN_DATA_PATH"
    KRAKEN_HISTORY_GDRIVE_ID: str = "1ptNqWYidLkhb2VAKuLCxmp2OXEfGO-AP"
    KRAKEN_HISTORY_INNER_PATH: str = "master_q4/XBTUSD_1.csv"
    KRAKEN_HISTORY_CSV_NAME: str = "XBTUSD_1.csv"
    KRAKEN_HISTORY_ZIP_STEM: str = "kraken_master_q4"
    KRAKEN_HISTORY_OUT_DIR: str = "data/raw"
    KRAKEN_HISTORY_CACHE_DIR: str = "data/.cache"

    # Walk-forward splits (candles)
    WALK_FORWARD_TRAIN: int = 150_000
    WALK_FORWARD_VAL: int = 50_000
    WALK_FORWARD_TEST: int = 10_000
    WALK_FORWARD_STRIDE: int = 50_000  # = VAL block; non-overlapping validation

    # Locked test set: 84 days × 1440 = 12 × 1-week non-overlapping windows
    LOCKED_TEST_CANDLES: int = 120_960

    # Search dev-slice (DECISIONS.md 'search_dev_slice'): the most recent
    # SEARCH_SLICE_TRAIN + SEARCH_SLICE_VAL candles immediately before HISTORICAL_START.
    # Excluded from every real walk-forward fold and the locked test set, so an
    # autoresearch-style hyperparameter/architecture search loop can iterate on real
    # market structure without touching production data. No Bonferroni exposure to the
    # walk-forward gate — this data never enters it.
    SEARCH_SLICE_TRAIN: int = 20_000
    SEARCH_SLICE_VAL: int = 8_000
    # Bonferroni-style noise defense for the search loop (DECISIONS.md
    # 'search_confirm_seeds'): a first-seed improvement is provisional only. It is kept
    # if a strict majority of SEARCH_CONFIRM_SEEDS repeat-seed runs also improve on the
    # current best; otherwise reverted as noise.
    SEARCH_CONFIRM_SEEDS: int = 3

    # Feature pipeline
    NUM_INPUT_FEATURES: int = 5  # OHLC log-returns + log1p volume change
    FEATURE_NAMES: tuple[str, ...] = (
        "open_logret",
        "high_logret",
        "low_logret",
        "close_logret",
        "vol_change",
    )
    # vol_change is +/-inf when current/prior volume is 0; degenerate value filled neutral.
    VOL_CHANGE_DEGENERATE_FILL: float = 0.0
    LOOKBACK: int = 1_440  # SWEEP [240, 720, 1440] before long training run
    # Floor for log(volume_t / volume_{t-1}). When volume_t = 0 or the ratio is
    # tiny, the raw log goes to -inf; clip to this finite floor so the scaler
    # has a stable input distribution. e^-10 ≈ 4.5e-5× — well below any
    # observable Kraken BTC/USD 1-minute volume drop.
    VOL_LOGRET_FLOOR: float = -10.0

    # Validator
    GAP_INTERPOLATE_MAX_HOURS: int = 12  # gaps > this trigger is_interpolated=True


# ============================================================
# Predictor
# ============================================================
@dataclass(frozen=True)
class PredictorConfig:
    HORIZON: int = 15  # direct multi-step; autoregression banned
    PATCH_SIZE: int = 16  # PatchTST: 1440 / 16 = 90 tokens

    # PatchTST encoder architecture (predictor-training.md §"Architecture").
    # channel_mixing: each (PATCH_SIZE x NUM_INPUT_FEATURES) patch -> one token, so
    # cross-feature (OHLCV) interaction is modelled within a candle. ~3M params at the
    # values below; comfortable on an RTX 4060 (8GB) at batch 32 / lookback 1440.
    PATCH_EMBED_MODE: str = "channel_mixing"
    D_MODEL: int = 128
    N_HEADS: int = 8  # head_dim = D_MODEL // N_HEADS = 16
    N_LAYERS: int = 3  # encoder blocks
    D_FF: int = 256  # feed-forward dim (2 x D_MODEL)
    DROPOUT: float = 0.1
    # Behaviour-defining encoder choices live here (not in model.py) so the SHA256
    # manifest, which hashes constants.py, binds them: a post-training change would
    # otherwise silently alter the architecture without invalidating the manifest.
    ACTIVATION: str = "gelu"
    NORM_FIRST: bool = True  # pre-LN; the encoder also applies a final LayerNorm
    # RevIN-style per-window instance normalization (model-internal): each lookback
    # window is standardised per feature (learnable affine), and the predicted quantiles
    # are rescaled by the window's own volatility, so interval width tracks the current
    # regime by construction. Epsilon keeps a degenerate (flat) window's sigma finite.
    REVIN_EPS: float = 1e-5

    # Output head: q10, q50, q90 per OHLCV dimension per future step. Quantiles are
    # non-crossing by construction (median anchor +/- cumulative softplus offsets).
    QUANTILES: tuple[float, float, float] = (0.10, 0.50, 0.90)
    NUM_OUTPUT_DIMS: int = 5  # OHLCV
    # Prediction/target convention (DECISIONS.md 'target'): quantiles of the CUMULATIVE
    # log-return path from the forecast origin, per horizon step. Embedded in every
    # checkpoint; deploy refuses a checkpoint trained under a different convention.
    TARGET_SEMANTICS: str = "cumulative_logret"

    # Rollout geometry enforcement (predictor-contract.md §"Geometry enforcement")
    # H >= max(O,C) and L <= min(O,C) per emitted step and quantile; a violating
    # stochastic sample is redrawn up to this many times before a deterministic clamp.
    GEOMETRY_RESAMPLE_CAP: int = 5

    # Loss: pinball on the cumulative path + direction penalty on the FINAL-horizon
    # cumulative close only (DECISIONS.md 'loss'). The fee is a per-trade round-trip
    # cost, so it is compared against the whole horizon move, never a single step.
    DIRECTION_PENALTY_LAMBDA: float = 1.75  # range [1.5, 2.0]
    # Coverage penalty (DECISIONS.md 'loss', amended 2026-07-01): squared gap between
    # smooth empirical batch coverage and the nominal tail levels (0.10/0.90), close
    # dim only, averaged over horizon steps. Directly optimizes the two calibration
    # gate metrics — the capped bake-off measured under-coverage on the cumulative
    # model (q90_coverage 0.866 vs 0.90 nominal, calibration_rate 0.747 vs 0.80) —
    # and is self-limiting: the gradient is proportional to the coverage gap, so it
    # vanishes once coverage reaches nominal instead of fighting pinball's optimum.
    COVERAGE_PENALTY_WEIGHT: float = 1.0
    # Sigmoid indicator width as a FRACTION of the batch's per-step close-target std:
    # relative (not absolute) so the indicator stays proportionally sharp as the
    # cumulative path's variance grows across horizon steps.
    COVERAGE_PENALTY_TEMPERATURE_FRAC: float = 0.02
    # Floors the indicator width so a degenerate flat batch (zero std) cannot /0.
    COVERAGE_PENALTY_STD_FLOOR: float = 1e-8

    # Training loop. AdamW; linear warmup then constant LR with plateau decay (the
    # previous cosine schedule annealed over a MAX_EPOCHS horizon that early stopping
    # never let it reach, so real runs trained at near-peak LR throughout); AMP (bf16)
    # on the 4060. MAX_EPOCHS is an upper bound — the EarlyStopper (patience below)
    # terminates earlier in practice.
    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY: float = 1e-2
    WARMUP_FRAC: float = 0.01  # linear warmup over the first 1% of planned steps
    # Multiply the LR by FACTOR after PATIENCE consecutive non-improving validation
    # epochs. PATIENCE is deliberately < EARLY_STOPPING_PATIENCE so the LR gets at
    # least two decays' worth of chances to un-stick a plateau before training stops.
    LR_PLATEAU_FACTOR: float = 0.5
    LR_PLATEAU_PATIENCE: int = 3
    GRAD_CLIP_NORM: float = 1.0
    MAX_EPOCHS: int = 100
    USE_AMP: bool = True
    SEED: int = 0
    SMOKE_BATCH_SIZE: int = 32  # smoke default; drop to fallback on OOM
    SMOKE_BATCH_SIZE_FALLBACK: int = 16  # Azure A100 decision point if this still OOMs
    # Production (real walk-forward run) training batch size — distinct from the smoke
    # value above, which existed only for the 1-epoch/1-fold pre-launch gate. The smoke
    # value (32) had leaked into the real run via the training UI, leaving the 4060 badly
    # under-fed. 256 keeps a ~3M-param PatchTST at lookback 1440 well under 8GB (measured
    # 474 MB peak on the 4060) while cutting fixed per-step overhead vs. 32.
    # PROD_BATCH_SIZE_FALLBACK (128) is the documented manual OOM step-down; it is NOT yet
    # auto-wired into an OOM-retry loop in the dashboard run (batch 256 has wide VRAM
    # headroom, so a retry path is unbuilt for now). DECISIONS.md
    # 'predictor_prod_batch_size' — user-confirmed 256 on 2026-07-01.
    PROD_BATCH_SIZE: int = 256
    PROD_BATCH_SIZE_FALLBACK: int = 128
    WANDB_PROJECT: str = "btc-bot-v3-predictor"

    # Checkpoint persistence (train -> deploy handoff). Filenames are run_tag-based
    # ({git_sha}-s{scaler8}-c{constants8}-fold{id}) so runs keep history and bind
    # weights<->scaler<->constants by hash. Weights are a wrapped dict (state_dict +
    # provenance) loadable under torch weights_only=True; the scaler is pickled.
    # deploy_predictor.py reads both back from this dir. See DECISIONS.md §Predictor.
    CHECKPOINT_DIR: str = "checkpoints"
    CHECKPOINT_WEIGHTS_SUFFIX: str = ".pt"
    CHECKPOINT_SCALER_SUFFIX: str = ".scaler.pkl"

    # Training UI fold-history export (training-dashboard.md §"Fold history export").
    # Written under CHECKPOINT_DIR; the handoff artifact for post-training analysis.
    TRAINING_METRICS_FILENAME: str = "training_metrics.json"

    # Bug regression tests (must be exposed for tests to assert against)
    EARLY_STOPPING_PATIENCE: int = 10
    VARIANCE_FLOOR_FIRST_N_STEPS: int = 100  # assert loss > 0 across these

    # Retrain triggers
    RETRAIN_NLL_TRIGGER_MULT: float = 2.0  # 7-day NLL > 2.0× baseline
    RETRAIN_CALENDAR_DAYS: int = 30
    RETRAIN_FINETUNE_WINDOW_DAYS: int = 14  # [t-21, t-7]
    RETRAIN_GATE_WINDOW_DAYS: int = 7  # [t-7, t], strictly non-overlapping

    # Deploy gates (all three required, simultaneously)
    DEPLOY_GATE_COVERAGE_TOLERANCE: float = 0.05  # ±5% from training-time
    DEPLOY_GATE_DA_THRESHOLD: float = 0.535  # > 53.5% on |q50| > FEE_THRESHOLD
    DEPLOY_GATE_CAL_LOWER: float = 0.75
    DEPLOY_GATE_CAL_UPPER: float = 0.85

    # Pre-training gate
    BASELINE_DA_GATE: float = 0.52  # signal-first sanity check

    # Holdout walk-forward gate
    HOLDOUT_WINDOW_DAYS: int = 7
    HOLDOUT_NUM_WINDOWS: int = 12  # 12 × 1-week = 84 days = 120,960 candles


# ============================================================
# Training UI (Phase 1.5 — training-dashboard.md)
# ============================================================
@dataclass(frozen=True)
class TrainingUIConfig:
    """Presentation/behavior thresholds for the training dashboard. Values marked
    "cosmetic default" are not written down in training-dashboard.md / DECISIONS.md;
    they were chosen during implementation and have no correctness/safety impact
    (display-only) — adjust freely without a DECISIONS.md amendment.
    """

    # Loss chart (training-dashboard.md §"B — Loss chart")
    LOSS_CHART_EMA_WINDOW: int = 5  # "smoothed with a 5-epoch EMA"
    LOSS_CHART_VISIBLE_EPOCHS: int = 200  # "the last 200 epochs are always visible"
    BATCH_TRACE_MAX_POINTS: int = 40  # cosmetic default: raw batch-loss trace length
    # Emit a 'batch' SSE message (and take the CUDA sync it entails) only every Nth
    # optimizer step instead of every step. A per-step sync + JSON broadcast is pure
    # overhead ~4,642×/epoch and starves the GPU; throttling to every 50th step removes
    # it from the hot path while a browser 60Hz UI still updates far faster than a human
    # reads. The 'batch' payload schema is unchanged — only its cadence. The final step
    # of every epoch is always flushed so the trace ends on the true last value.
    # DECISIONS.md 'training_ui_batch_log_interval'. Not display-correctness critical.
    BATCH_LOG_INTERVAL: int = 50

    # Alert banners (§"J — Alert banner")
    ALERT_AUTO_DISMISS_SECONDS: float = 8.0
    ALERT_MAX_STACK: int = 3
    ALERT_BEEP_FREQUENCY_HZ: int = 880  # cosmetic default: winsound.Beep pitch
    ALERT_BEEP_DURATION_MS: int = 200  # cosmetic default: winsound.Beep length

    # Loss component strip grad-norm coloring (§"E — Loss component strip").
    # Amber past PREDICTOR.GRAD_CLIP_NORM (clipping engaged); red past this multiplier
    # of it (possible instability).
    GRAD_NORM_INSTABILITY_MULTIPLIER: float = 10.0

    # Fold history table coloring (§"G — Fold history table")
    FOLD_TABLE_DA_AMBER_FLOOR: float = 0.50  # amber 50-53.5%; green above
    # PREDICTOR.DEPLOY_GATE_DA_THRESHOLD (red below amber floor)
    FOLD_TABLE_QCOV_HEALTHY_LOWER: float = 0.85  # outside [lower, upper] -> red
    FOLD_TABLE_QCOV_HEALTHY_UPPER: float = 0.95

    # Patience bar coloring (§"H — Patience bar"). Bounds are counts against
    # PREDICTOR.EARLY_STOPPING_PATIENCE (10): 0-4 blue, 5-7 amber, 8-9 orange, 10 red.
    PATIENCE_AMBER_FLOOR: int = 5
    PATIENCE_ORANGE_FLOOR: int = 8
    # "A subtle amber glow appears behind the bar when patience >= 7"
    PATIENCE_GLOW_FLOOR: int = 7

    # ETA strip (§"F — ETA strip")
    ETA_MIN_EPOCHS_FOR_ESTIMATE: int = 2  # "—" shown below this
    ETA_FOLD_DURATION_EMA_ALPHA: float = 0.3  # cosmetic default: smoothing factor

    # SSE transport (src/training_ui/app.py). Bounds queue.Queue.get() so a disconnected
    # client's executor thread + subscriber entry are reclaimed within this many seconds
    # instead of blocking forever — not in training-dashboard.md; a transport-layer
    # correctness parameter, not display-only, but has no user-visible effect either way.
    SSE_POLL_TIMEOUT_S: float = 1.0


# ============================================================
# Trader (rules-based for v3)
# ============================================================
@dataclass(frozen=True)
class TraderConfig:
    # Position sizing
    POSITION_SIZE_BASE: float = 0.01  # 1% per trade fixed-fractional
    MAX_ALLOCATION: float = 0.04  # ±4% hard cap

    # Confidence gate
    # spread = (q90 - q10) / |q50|
    # Above this, force allocation to zero. Calibrate empirically before paper.
    CONFIDENCE_THRESHOLD: float | None = None  # TBD before paper trading

    # Exit priority stack
    SIGNAL_REVERSAL_CANDLES: int = 3  # consecutive opposite signals to exit
    TIME_BASED_EXIT_MINUTES: int | None = None  # TBD if used; lowest priority

    # Predictor staleness decay
    STALENESS_DECAY_FLOOR: float = 0.50  # multiplier at retrain_date + 30d
    STALENESS_DECAY_DAYS: int = 30  # linear from 1.0 to floor


# ============================================================
# Execution engine
# ============================================================
@dataclass(frozen=True)
class ExecutionConfig:
    # Fee model — Kraken base-tier
    FEE_RATE_TAKER: float = 0.0026  # 0.26% per side
    SLIPPAGE_FLOOR: float = 0.0005  # 0.05% on every market order
    # Round-trip drag: 2 × 0.26% + 2 × 0.05%
    FEE_THRESHOLD: float = 0.0062  # consumed by training loss + DA evaluation gate

    # Spread model: spread = SPREAD_BASE + SPREAD_ATR_SCALE × atr_ratio
    # atr_ratio = current_ATR / rolling_median_ATR (1440 candles)
    SPREAD_BASE: float = 0.0005
    SPREAD_ATR_SCALE: float = 0.0001
    ATR_ROLLING_MEDIAN_WINDOW: int = 1_440

    # Stale candle handling
    STALE_HALT_SECONDS: int = 90
    STALE_AUTO_CLOSE_SECONDS: int = 300  # 5 minutes

    # Cycle timing (60s loop)
    CYCLE_TARGET_SECONDS: int = 60
    CYCLE_WARNING_SECONDS: int = 45
    CYCLE_HARD_SECONDS: int = 55

    # Kill switch
    KILL_FLAG_PATH: str = "KILL_SWITCH.flag"
    KILL_FLAG_TMP_PATH: str = "KILL_SWITCH.flag.tmp"  # atomic write
    KILL_POLL_SECONDS: int = 2

    # Stop-loss confirmation timeout
    STOP_LOSS_CONFIRMATION_TIMEOUT_SECONDS: int = 5

    # Network
    DASHBOARD_BIND_HOST: str = "127.0.0.1"
    DASHBOARD_BIND_PORT: int = 8000
    # Training UI runs as a separate FastAPI process on its own port, same bind host.
    TRAINING_UI_BIND_PORT: int = 8001

    # agent_config.json schema version. deploy_predictor refuses to overwrite a config
    # whose schema_version differs (surfaces silent schema drift). See agent-config.md.
    AGENT_CONFIG_SCHEMA_VERSION: str = "1.0"

    # Kill criteria — auto-shutdown (no operator override path)
    K1_SESSION_DRAWDOWN: float = 0.03  # 3% in 24h rolling
    K2_TOTAL_DRAWDOWN: float = 0.10  # 10% total
    K4_NLL_BASELINE_MULT: float = 2.0  # 7-day NLL > 2.0× baseline
    K8_STALE_MINUTES: int = 5

    # Kill criteria — alert + manual review
    K3_PNL_ANOMALY_DAYS: int = 7
    K5_CALIBRATION_FLOOR: float = 0.50  # < 50% × 3 days
    K5_CALIBRATION_BREACH_DAYS: int = 3
    K6_ZERO_TRADE_HOURS: int = 4
    K7_WINRATE_FLOOR: float = 0.40  # < 40% × 3 days
    K7_WINRATE_BREACH_DAYS: int = 3
    K9_LATENCY_BREACH_CYCLES: int = 3


# ============================================================
# RL trader (deferred; v3 ships rules-based)
# ============================================================
@dataclass(frozen=True)
class RLConfig:
    """Reserved for the RL trader iteration after rules-based demonstrates
    stable signal on validated quantile output (≥3 months paper trading)."""
    pass


# ============================================================
# Module-level singletons — import these, do not instantiate ad hoc.
# ============================================================
DATA = DataConfig()
PREDICTOR = PredictorConfig()
TRAINING_UI = TrainingUIConfig()
TRADER = TraderConfig()
EXECUTION = ExecutionConfig()
RL = RLConfig()
