# Checkpoint 12 — Seed-instability diagnosed: connected ridge, NOT multimodality (2026-07-15)

Investigates CHECKPOINT_11's open problem ("is the cross-seed
disagreement genuine multimodality?") from the saved production
particle clouds (`results/production/seed{42,7,101}/particles_stage_29.mat`)
— no re-estimation. Evidence artifacts: `results/multimodality_diag/`.

## Verdict: one connected likelihood ridge; genuine multimodality REFUTED

The decisive test is a **bridging log-posterior profile**: evaluate
ll+lp (exact production evaluator — sanity-gated bitwise against the
stored per-particle ll/lp) along straight lines connecting the seed-42
and seed-101 posterior regions (the two most-separated clouds; seed101
alone carries the worst gwbar/sig_xi split). Four paths × 40 points:
MAP-to-MAP and mean-to-mean in full 79-D, plus the restricted
[gwbar, gzbar, sig_xi] block in both directions.

**Result: zero dip below the lower endpoint on every path** (separated
modes would require a valley >10 log points below both endpoints). The
full-79-D paths even show an interior *hump* — up to 17.9 log points
ABOVE the better endpoint at t≈0.38 — i.e. the straight line passes
through higher-density territory than either seed's own best particle:
positive evidence of connectivity AND of per-seed under-exploration of
the ridge crest.

## Supporting evidence (3-agent particle-cloud analysis + adversarial verification)

- **Sampler health ruled out as confounder**: ESS 60–85%, 97–99% unique
  particles (no ancestral collapse), φ=1 all seeds, mapping of P columns
  to `posterior_summary.csv` order verified to ~4e-15.
- **Within-seed elongation aligns with the between-seed axis** (gap-AR
  pair: |cos(within-PC1, between-axis)| = 0.99–1.00, anisotropy to
  7.4×) — three samples strung along one ridge, not three basins.
- **Pooled marginals are connected** for gzbar and phi2 (zero empty
  bins between clouds) despite R̂ 2.6/2.0.
- **Seed101 sits 19–28 log points lower** in posterior height (outside
  the ~10-point LML noise floor) — an under-explored lower ridge
  segment, now confirmed monotonically reachable from seed42's region.
- Caveat on earlier 1-D reads: pooled GMM-BIC "k=3" was largely
  re-detecting the 3-seed design (+ resampling-duplicate artifacts);
  marginal seed-disjointness does NOT discriminate modes from a ridge.

## Reconciliation with the 2026-07-14 "shelf" finding (the question this answers)

The 07-14 transformed-kernel session (commit 4ff28d2: `MutationTransform`
+ `StructuredBlocks`, arm-B 3-seed A/B) left open a "genuine multimodal
core" hypothesis: arm-B seed101 found a posterior **shelf** at
log-target ≈ −300 (wmean; max −276) while all other runs sat at
−340..−365. Verified this session from the rescued arm-B clouds
(`results/armB_transformed_kernel/`, recovered from the old session
scratchpad before it evaporated): arm-B wmean lpost by seed =
−365.05 / −347.72 / −300.34 (42/7/101).

**The shelf is the crest of the same connected ridge, not a separate
mode**: today's bridge path between the two *production* clouds passes
through lpost ≈ −287 — shelf-height territory — while rising and
falling monotonically, no barrier. What looked like a mode only one
lucky seed could find is a higher crest region that the raw-scale
kernel fails to climb reliably.

## Consequences

- No mode-jumping/parallel-tempering machinery is needed. The problem
  is **hill-climbing/mixing along one connected ridge** — sampler-side
  fixes are the right family, and pooled 3-seed results remain a valid
  (conservative) uncertainty envelope, not a mixture of distinct
  economic regimes.
- **The 07-14 arm-B evidence now has a clean interpretation**: the
  transformed kernel (max R̂ 5.82 → 2.84; pooled γ2 −0.157 → −0.268)
  is not "finding a different answer," it is climbing higher up the
  same ridge — the production table's γ2 is partially a sticky-kernel
  artifact, as the 07-14 session suspected. Note arm-B's own seeds
  still span −365..−300: better, not converged; even arm-B doesn't get
  every seed to the crest.
- **Recommendation to owner (sign-off required — changes the quotable
  table)**: enable `'MutationTransform', true, 'StructuredBlocks', true`
  in `production.m` and regenerate `results/production/`. Optionally
  combine with a late-stage MSteps ladder to push all seeds up the
  crest. Further ranked options (gzbar/gwbar sum/split rotation
  mirroring phisum/phi2 — the model identifies only the sum via
  `ck = 0.025·(gzbar+gwbar)/(1−α)`; non-centered sig_xi; cross-seed
  exchange as last resort) recorded in the session log — none reverses
  an owner ruling.

## Also this session

- **κ 2023 boundary RESOLVED** (owner ruling, 2026-07-15): keep the
  current cutoff — 2023Q3 last elevated, κ=1 from 2023Q4. No code
  change. Recorded in CLAUDE.md owner rulings.
