# Checkpoint 11 — Drop the horseshoe; diagonal covariance is production (2026-07-14)

Follows the methodology review (CHECKPOINT_10). Owner decided to **drop
the horseshoe covariance layer entirely** and use a **diagonal
innovation covariance**, matching the original in-house model.

## Why (owner decision, evidence-based)

- The horseshoe-on-innovation-covariance was the **one component with
  no published precedent in any setting** (CHECKPOINT_10) — the piece
  the owner was wary of being "first" on.
- It carried a real kernel-invariance bug (draw-then-clamp of the
  truncated hyper-scales), CHECKPOINT_10 audit item 2.
- The original RBA model uses a diagonal covariance (owner-confirmed;
  the brief had left this as an open question that defaulted to
  diagonal at CP4).
- Empirically it identified only ~17/106 off-diagonals and — once the
  gap-shock correlations were excluded by ruling to stop them
  distorting the cycle — did **not** materially change the headline
  latent states or policy parameters vs the diagonal model.

Diagonal is fully precedented (every UC model; the Chan UC papers;
Zaman 2022 at the Cleveland Fed), removes the novelty and the bug, and
is ~4× faster.

## Verification (before deleting anything)

1. **Diagonal reproduces the headline.** 3-seed diagonal final spec
   (Horseshoe=false, HierKappa=true, PieObs=true) gives r* end
   0.73 [−3.1, 4.4] (cp7b horseshoe 0.97 [−1.9, 3.6]), NAIRU 5.46
   (5.83), gap 0.62 (0.40), Phillips slope −0.157 (−0.163). Economics
   preserved; bands somewhat wider; ν now spans zero (−0.027
   [−0.09, 0.03] vs cp7b −0.075 excl 0 — less identified without the
   covariance channel).
2. **GATE: bitwise-identical.** After removing the horseshoe code, a
   diagonal seed-42 run reproduced the pre-removal diagonal run's
   smc_log.csv **bitwise** on all non-timing columns (29 stages, 85
   columns, zero differences) — proving the deletion did not perturb
   the diagonal path or its RNG consumption.
3. **Suite green** 20/20 (was 26; the 6 horseshoe test methods removed).

## IMPORTANT finding — diagonal does NOT fix the seed-instability

Predicted that the ~78-dim diagonal model would mix far better than the
~410-dim horseshoe one. **It does not:** max cross-seed R̂ = 5.82,
essentially unchanged from cp7b's ~5.3. The seed-instability is a
property of the **likelihood-ridge geometry** (the gap-AR persistence
split φ1/φ2, the ρ_U/Okun trade-off, the r* band), not the horseshoe's
dimensionality. Consequences: the ≥3-seed pooling discipline remains
necessary; and **the ridge geometry is now THE open problem** — the
natural next target if convergence is to be improved (reparameterization
of the ridges, not more particles).

## What was removed / kept

- **Deleted (code):** `horseshoeSample.m`, `horseshoeMutate.m`,
  `horseshoePriors.m`, `hsUnpack.m`, `horseshoeDiag.m`; the `Horseshoe`
  option and its prior-selection/mutation wiring in `estimate.m`,
  `runSMC.m`, `mhMutate.m`, `priorSample.m`, `priorLogPdf.m`; the
  horseshoe tests (`testHorseshoeSMC`, `testHorseshoeShrinkage`).
- **Deleted (results/docs):** `results/cp7b`, `results/rhat_seed7`,
  `results/rhat_seed101`, the old root-level `pooled_posterior.csv` /
  `convergence_rhat.csv`, the horseshoe heatmap figures, and
  `docs/03_validation_vs_baseline.{m,mlx,pdf}` (the CP8 report was
  cp7b/horseshoe-specific; the Table-3 validation had already been
  removed from the production pipeline).
- **Kept as dead-but-harmless (entangled with the diagonal path):** the
  full correlated-R (`Rfull`) branch in `computeLogLik.m` and the
  LDL′ scaffolding in `ModelSpec.m` (the diagonal model is the LDL′
  path with identity factors), plus `tests/testFullRVsKalman.m` which
  covers that branch. Removing them was judged to risk the diagonal
  path for no functional gain; they can be excised later if desired.

## Diagonal production (the new headline)

`jointstar.production('data.csv')` → `results/production/` (3 seeds
pooled): 79-parameter coefficient table, cross-seed R̂ table, pooled
smoothed states. Regenerated latent-states comparison figure at
`results/figures/latent_states_comparison.png`. The model is now a
clean assembly of standard, precedented components with no novel
statistical methodology (see METHODOLOGY_NOTE.md).

## Still open (separate items)

- **The kappa 2023-boundary question** — RESOLVED 2026-07-15: owner
  confirmed the current cutoff (κ=1 from 2023Q4; 2023Q3 last elevated,
  matching the brief). No code change. Now an owner ruling in CLAUDE.md.
- **The ridge geometry / seed-instability** — the real remaining
  convergence problem, unaffected by the covariance choice.
