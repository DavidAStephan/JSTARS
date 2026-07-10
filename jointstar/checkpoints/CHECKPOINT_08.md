# Checkpoint 8 — Full validation report (DELIVERED)

**Deliverable:** `docs/03_validation_vs_baseline.mlx` (also exported to
`docs/03_validation_vs_baseline.pdf`; the plain-text source is the
matching `.m`). It reads everything from `results/cp7b/` — the final
specification incorporating all owner rulings — and covers the brief's
required content:

1. **Timing.** End-to-end estimation (2,000 particles, all 400+
   parameters including the covariance layer): ~48 min on a 6-core
   laptop; diagonal-covariance variant: 5.9 min; likelihood evaluation
   9.5 ms/core (target 50–100 ms). The baseline takes hours with a
   single poorly-converging MH chain. A 32-worker machine meets the
   30-minute production budget with room to spare.
2. **Parameter comparison vs the baseline.** 17/23 quantities have overlapping
   90% intervals. Systematic disagreements are discussed in prose: the
   gap-AR hump (same total persistence, different shape — a likelihood
   ridge where priors decide), the ρ_U/Okun-loading split (trade-off pair),
   and the Phillips slope (−0.16 vs −0.09; prior-driven on a weakly
   identified parameter).
3. **Trend comparison.** NAIRU profile matches the baseline chart; end-of-
   sample r* = 0.97% [−1.9, 3.6] vs baseline ~1%; output gap +0.4% at end,
   COVID trough consistent with the baseline's −7.5%.
4. **Horseshoe diagnostic.** ~17 of 106 off-diagonals identified;
   measurement and cross blocks mostly empty; correlations concentrated
   among trend and drift shocks. Heatmaps in `results/cp7b/figures/`.
5. **Failure modes.** The AR-shape and ρ_U disagreements (with proposed
   prior fixes if baseline comparability matters); r* still the least-
   identified object absent a neutral-rate proxy; the resolved d_t/κ
   interplay documented in CHECKPOINT_07.

**Project status: all eight checkpoints complete.** One estimate() call
reproduces everything: `jointstar.estimate('data.csv', 'NParticles',
2000, 'Seed', 42, 'Horseshoe', true, 'HierKappa', true)`. Rerunning on
updated data requires no code changes; re-rulings (exclusions, sign
restrictions, calibrated constants) are single labelled lines in
`horseshoePriors.m` / `defaultPriors.m`.

---

## Addendum — Gelman-style multi-seed convergence check (2026-07-11)

Three fully independent full-spec runs (seeds 42/7/101; `results/cp7b`,
`results/rhat_seed7`, `results/rhat_seed101`) were compared with an
R̂-style statistic. **Single runs are NOT seed-stable on the structural
parameters** (max R̂ ≈ 5.3; e.g. ρ_hpp 0.16/0.89/0.58, κ_y2020
9.1/3.8/2.2, m84_z 1.2/2.5/0.7): the posterior lives on long ridges and
one 2000-particle cloud with 2 MH sweeps/stage occupies too narrow a
footprint. The latent states are far more robust (r* end 0.97/0.63/0.83,
each well inside the others' bands). Consequences adopted:

* quotable parameter table = `results/pooled_posterior.csv` (equal-weight
  pool of the three runs, a valid stratified estimator; intervals wider
  and more credible vs the baseline, e.g. κ_y2020 [1.8, 10.0]);
* production recipe: ≥3 seeds pooled (embarrassingly parallel; still
  ~30 min total on 32 workers), optionally MSteps 4–6 for tighter
  single-run footprints;
* the validation Live Script (section 2) now states this prominently.

This mirrors — and finally explains — the baseline MCMC's convergence
struggles: the geometry is the model's, not the sampler's. SMC surfaces
it with a computable diagnostic instead of a silently stuck chain.
