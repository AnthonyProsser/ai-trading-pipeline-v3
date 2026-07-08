# Questions for Anthony (accumulated while you were away)

Answer whenever you're back at the computer; none block the loop.

1. **Screening protocol going forward.** Full 78-fold runs are ~6h each, so ~3-4 ideas/day max. OK, or do you want a middle tier (e.g. 10-fold spread subset, full epochs) as the screen — ~1h/idea, fair across regimes — with full 78 folds reserved for winners?
2. **Old capped rejections (ideas 1 clock, 3 lookback720, 4 aux-head).** Fold-77 capped screening was pessimistic. Re-run any of these as full runs? My default queue: idea-1 full run (done or in progress by the time you read this), skip 3 and 4 (4 broke calibration — real defect, not fold luck).
3. **Benchmark app branch constraint.** A run's checkpoints can only be benchmarked with its matching branch checked out (9-feature checkpoints won't load under 5-feature code). Acceptable workflow (checkout branch → run benchmark → switch back), or do you want the app/registry made architecture-aware later (bigger change, needs the auditor trio)?
4. **`checkpoints/finished/` mixing runs.** All full runs promote there; the app groups by (git_sha, constants_sha8) so rows stay separate, but the dir grows ~78 files per run. Prune policy: keep all, or keep only runs you've reviewed?
5. **Fee sensitivity.** Mean DA 0.52 is real but likely sub-fee at 0.62% round trip. Worth a one-off analysis of what fee level (maker tier? longer horizon) the current edge WOULD clear, to know how far away we are? (Analysis only — the fee model itself stays locked.)
6. **Krafer video Tier-1 features.** Priority order when I get to them? My default: (2) swing high/low distance first (simplest), then (1) volume-at-price, (3) absorption, (4) rejection wicks. Or grouped as one idea to save GPU-days, at the cost of attribution?
7. **Merging idea-2 (multi-scale).** It beat baseline (0.5201 vs 0.5155, 69/78 vs 63/78 folds >0.5). Merging to `main` requires the suspended auditor trio re-run per CLAUDE.md. Want that started, or hold until more ideas are compared?
8. **Auto-benchmark after each run?** I can run benchmark scoring headlessly after each training completes (adds GPU time between trainings). Confirm you want it always-on so every run row is scored when you look.
