# Predictor training

## Architecture: PatchTST encoder

Encoder-only Transformer with patch embedding. Inputs the 5-feature × `lookback` tensor (see `predictor-contract.md`); produces the (15, 5, 3) quantile tensor in a single forward pass.

- `patch_size = 16`. At `lookback=1440`: 1440/16 = 90 tokens. Drops attention from O(1440²) ≈ 2M to O(90²) ≈ 8K — ~256× reduction. This is the binding choice that makes 8GB VRAM feasible.
- Encoder-only, no causal mask required because the head emits horizon directly rather than autoregressively.
- Output head: linear projection from encoder representation to `(horizon × num_dims × num_quantiles)`, reshaped.

## Loss

```
L = pinball(q10, q50, q90 vs. log_return_target) + λ × direction_penalty
```

- **Pinball loss** evaluated independently per (step, dim, quantile). Quantile τ ∈ {0.10, 0.50, 0.90}.
- **Direction penalty:** sign disagreement between predicted q50 and realized log-return, weighted by magnitude. Preserves v2's slope-composite intent without coupling to NLL.
- **λ = 1.75** within the documented [1.5, 2.0] range. Tunable via `PredictorConfig.DIRECTION_PENALTY_LAMBDA`.
- **FEE_THRESHOLD subtracted from predicted per-step PnL inside the loss.** This is the loss-side fee accounting: the gradient pushes toward predictions that survive round-trip fees rather than toward the literal next return. The Trader has no hard `|q50| > FEE_THRESHOLD` entry threshold — that filter exists only on the deploy DA gate.

## Three regression tests (must pass before training loop ships)

These exist because all three v2 predictor-training bugs were in this exact loop. Re-encountering them mid-training costs days; preventing them costs hours.

1. **Variance-floor.** `assert loss > 0` for the first 100 steps of every training run. Failure → output collapsed to a constant; v2's negative-NLL bug.
2. **Trend-loss synthetic input.** Feed constant candles. Direction penalty must emit a known non-zero output (it should detect the lack of trend and produce a calibrated baseline). v2's zero-trend-loss bug.
3. **Patience exposed.** `EARLY_STOPPING_PATIENCE` is a `constants.py` value, asserted to be > 1, and consumed by the training loop's stopper. v2's premature early-stopping bug. Test reads the constant and verifies the trainer respects it.

Failing tests live at `tests/predictor/test_training_bugs.py`. They are committed before the training loop, per the test-first discipline.

## Smoke run (1 epoch, 1 fold) — pre-launch gate

Catches OOM, NaN losses, dataloader issues before committing the 4-week run.

- Batch size = 32 by default. If OOM → drop to 16. If still OOM at 16 → this is the Azure A100 decision point. **Do NOT start the 4-week run on a borderline-OOM config.**
- Logs to W&B from step 1. Run tag = git SHA + scaler hash + `constants.py` hash + fold ID.
- Loss components logged separately (pinball, direction, total). Required for post-mortem if the long run fails halfway.

## Lookback sweep

Before the long training run, sweep `lookback ∈ [240, 720, 1440]` with early-stopping. Pick by validation MAE + direction accuracy on the corresponding walk-forward folds. <1 week of GPU; chasing it after a 4-week training run costs 4 weeks.

## Training data

- Walk-forward splitter (see `splits-validation.md`). 150k/50k/10k with stride 50k.
- The locked test set (120,960 candles) is **never** referenced from `src/`. Enforced by `grep -r 'test_locked' src/` in CI plus a `sys.settrace` runtime guard during training.
- Per-fold MinMaxScaler. The training loop receives a `(scaler, train_loader, val_loader)` tuple per fold; it never instantiates a scaler.

## Retraining

- **Conditional trigger:** 7-day NLL or quantile-coverage > 2.0× baseline → manual approval to retrain.
- **Calendar trigger:** 30-day max gap regardless of drift.
- **Window:** fine-tune on `[t-21, t-7]`, gate on `[t-7, t]`. **Strictly non-overlapping** — the v2 carry-forward language is self-deception.
- **Warm-start** from prior checkpoint.
- **Three deploy gates** (all required, simultaneously): coverage on locked test set within ±5%, DA > 53.5% on `|q50| > FEE_THRESHOLD`, calibration 75–85%.
- A single failing gate blocks deployment. No tweak-and-rerun (Bonferroni defense).

## Hyperparameter tuning during walk-forward — forbidden

If a fold gate fails, EITHER training continues unchanged OR the model is rejected. Tweaking a hyperparameter and resuming a fold turns walk-forward validation into in-sample fitting. Encoded as a CLAUDE.md non-negotiable rule.
