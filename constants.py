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
    KRAKEN_HISTORY_GDRIVE_ID: str = "1ptNqWYidLkhb2VAKuLCxmp2OXEfGO-AP"
    KRAKEN_HISTORY_INNER_PATH: str = "master_q4/BTCUSD_1.csv"
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

    # Output head: q10, q50, q90 per OHLCV dimension per future step
    QUANTILES: tuple[float, float, float] = (0.10, 0.50, 0.90)
    NUM_OUTPUT_DIMS: int = 5  # OHLCV

    # Rollout geometry enforcement (predictor-contract.md §"Geometry enforcement")
    # H >= max(O,C) and L <= min(O,C) per emitted step and quantile; a violating
    # stochastic sample is redrawn up to this many times before a deterministic clamp.
    GEOMETRY_RESAMPLE_CAP: int = 5

    # Loss
    DIRECTION_PENALTY_LAMBDA: float = 1.75  # range [1.5, 2.0]

    # Training loop (predictor-training.md §"Smoke run"). AdamW + cosine schedule with
    # linear warmup; AMP (bf16) on the 4060. MAX_EPOCHS is an upper bound — the
    # EarlyStopper (patience below) terminates earlier in practice.
    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY: float = 1e-2
    WARMUP_FRAC: float = 0.05  # linear warmup over the first 5% of total steps
    GRAD_CLIP_NORM: float = 1.0
    MAX_EPOCHS: int = 100
    USE_AMP: bool = True
    SEED: int = 0
    SMOKE_BATCH_SIZE: int = 32  # smoke default; drop to fallback on OOM
    SMOKE_BATCH_SIZE_FALLBACK: int = 16  # Azure A100 decision point if this still OOMs
    WANDB_PROJECT: str = "btc-bot-v3-predictor"

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
TRADER = TraderConfig()
EXECUTION = ExecutionConfig()
RL = RLConfig()
