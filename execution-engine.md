# Execution engine

The 60s loop, asyncio orchestration, fee model, kill switch, exchange-native stops. The only place `asyncio` lives — predictor and trader stay synchronous.

## 60s loop

Per cycle (target 60s):

1. Pull current candle (Kraken WebSocket v2 OHLC channel)
2. Validate + append to history; mark `is_interpolated` if forward-filled
3. Run predictor forward pass on the lookback window
4. Run trader rules → desired position
5. Reconcile desired vs. actual position; emit orders
6. Update SQLite + dashboard WebSocket payload

Timing budget: `CYCLE_WARNING_SECONDS = 45` (Telegram alert), `CYCLE_HARD_SECONDS = 55` (escalate; K9 fires after 3 consecutive breaches). 55 is below 60 because alert delivery itself takes some time.

## Kill switch — file-flag + watchdog

The kill switch is a file (`KILL_SWITCH.flag`) at the repo root. Two independent processes poll it every 2 seconds:

- The 60s loop (skips trade emission, closes positions on first observation)
- A dedicated OS-level watchdog process (force-kills the loop if it doesn't observe the flag within N polls)

**Atomic write.** The file is written via `KILL_SWITCH.flag.tmp` + rename, never directly truncated. A partial write must never leave a half-flag visible.

**Capital safety never depends on dashboard availability.** The dashboard kill button is allowed only as a thin shim that writes the file-flag — never as a direct API call into the loop.

**4-case test plan** (must all pass before paper trading):

1. Loop detects flag and closes positions within 2 polls
2. Watchdog detects flag if loop is hung
3. **Write flag from command line while dashboard is offline** — capital still protected
4. Atomic write under contention — no torn read

## Fee model — Kraken base-tier

```
fee_per_side  = 0.0026  (0.26%)
slippage      = 0.0005  (0.05% floor on every market order)
fee_threshold = 0.0062  (round-trip drag)
```

The slippage floor applies to **every** market order, not only conditionally — the v2 condition (`atr_normalized > 1.5 AND atr_ratio < 2.0`) excluded the worst events.

`FEE_THRESHOLD` is consumed by:

- The training loss (subtracted from predicted per-step PnL)
- The DA evaluation gate (filter: only count predictions where `|q50| > FEE_THRESHOLD`)

## Spread model

```
spread = SPREAD_BASE + SPREAD_ATR_SCALE × atr_ratio
       = 0.0005 + 0.0001 × atr_ratio
```

Where `atr_ratio = current_ATR / rolling_median_ATR` over the last 1440 candles. Captures low-liquidity hours without needing live bid/ask (Kraken OHLCV doesn't provide them).

## Stale candle handling

- `>90s` since last candle close: halt new trades. Banner on dashboard. Telegram alert.
- `≥5min` since last candle close: auto-close all open positions. K8 fires (auto-shutdown, no override).

Unmanaged open position during outage was the largest unspecified failure mode in v2.

## Exchange-native stop-loss — mandatory

Every entry order places an attached close order at Kraken. The execution engine **refuses the order entirely if the stop-loss cannot be placed or is invalid**. There are no naked positions on the book under any code path.

A position is not marked open in local state until Kraken confirms the stop-loss order. If confirmation does not arrive within `STOP_LOSS_CONFIRMATION_TIMEOUT_SECONDS` (5s), any partial fill is auto-closed and Telegram alerts.

## API ingest

- **Real-time:** Kraken WebSocket v2 OHLC channel.
- **Gap backfill:** REST `GetOHLCData` for gaps ≤12h after WebSocket reconnect (WebSocket does not backfill on reconnect — explicit detection required).
- **Gaps >12h:** forward-fill with `is_interpolated=True` (same rule as historical CandleValidator).

## Position reconciliation on startup

Query `GetOpenOrders` and `GetOpenPositions`. Refuse to start if local SQLite mismatch is unexplained. Kraken is treated as authoritative.

## Persistence

- SQLite for paper. Schema uses binary BLOBs (`history_blob`, `futures_blob`, `context_blob`), not `state_json TEXT`.
- PostgreSQL+WAL before live capital. Migration script written in week 4 of training; run at paper→live transition.
- The replay scrubber reads from an in-memory rolling deque cached for last N minutes; SQLite only for older history. Live writes otherwise contend with scrubber reads.

## Secrets

- Kraken API key: Windows Credential Manager via `keyring`. Encrypted at rest.
- API key scope verified programmatically at startup: trade-only, no withdraw, no deposit.
- `.env` reserved for non-secret config.

## Network exposure

FastAPI bound to `127.0.0.1` only. Remote access via SSH tunnel. Threat model: another device on the LAN can otherwise trigger live trades or kill switch.

## ExecutionBackend contract

`ExecutionBackend` is an abstract class. `PaperBackend` and `LiveBackend` are sibling implementations (no shared conditional path). A parity contract test must run identical scenarios through both before any live capital. Kills the conditional-branch divergence failure mode by construction.

## SHA256 manifest verification

Verified at startup AND on every weight reload. Manifest covers weights + scaler PKL + `constants.py`. A one-line constants change otherwise silently changes reward/risk shape post-training.

## Backups

Checkpoints, scaler, `agent_config.json` synced to OneDrive/Google Drive after every write (rclone or filewatcher). Single-machine deployment otherwise = single point of failure for months of training.

## Alert thresholds (Telegram)

| Trigger | Severity |
|---|---|
| Cycle latency > 45s | Warning |
| Cycle latency > 55s × 3 (K9) | Alert |
| Stale candle > 90s | Warning |
| Stale candle ≥ 5min (K8) | Auto-shutdown alert |
| Stop-loss confirmation timeout | Alert + auto-close partial fill |
| Position reconciliation mismatch on startup | Refuse start, alert |
| Kill switch fired | Confirmation alert |
