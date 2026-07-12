# Checkpoint 5 — Horseshoe on innovation-covariance off-diagonals (PASSED, big findings)

**Run:** grouped horseshoe (measurement / trend / drift / cross-block, own
τ_g each) on all 91 transition + 21 measurement off-diagonals of the two
LDL′ factors; Makalic–Schmidt Gibbs-within-SMC on the scales; block MH with
per-particle proposal scaling for the L's. N=2000, 410-column particle
space, 28 stages, **51.9 min** on 6 cores, acceptance ~0.20–0.37 throughout.
Outputs: `results/cp5/` incl. `hs_shrinkage.csv` and heatmaps.

**Shrinkage verdict: 12 of 112 off-diagonals identified** (90% band excludes
zero); the other 100 shrink to ≈0. The measurement and cross blocks are
essentially empty (τ_g at the truncation floor); the action is in the
trend-shock block: gap↔NAIRU (−0.39), gap↔π^e (+0.24), gap↔k* (+0.18),
gap↔ξ (+0.12), z*↔U* (−0.23), z*↔π^e (+0.27), k*-trend↔k-drift (−0.52),
pr*↔z* (−0.25), hpp*↔wapop* (−0.29), plus two small measurement pairs
(U↔y, lpr↔U).

**Two absorption findings (flagged, not re-parameterised):**
1. **The IS coefficient ν collapsed to ≈0** (CP4: −0.07 excluding zero) once
   gap↔ξ shock correlation was freed — ν and corr(η^c, η^ξ) compete for the
   same r*/gap comovement. Options: (a) exclude the gap↔ξ pair from the
   horseshoe (theory says the IS channel is the mechanism); (b) tighten ν's
   prior; (c) let the surveys (CP7) discipline ξ first, then revisit.
2. **Okun loading ξ1 flipped sign** (−0.19 → +0.12) with the −0.39 gap↔NAIRU
   shock correlation absorbing the negative comovement.

Also: σ_c halves (0.54→0.32) and σ_gz falls (0.053→0.034) as correlated
shocks soak variance. γ2 (−0.33) and α (0.38) are stable.

**Sampler notes for the record:** naive SMC froze twice — fixes were block
MH + adaptive scale, per-particle (τλ)-scaled L proposals, λ/τ ~ C+
truncated to [0.05, 10], and scale-consistent initial L draws (exact
densities throughout; init underdispersion washed out by resample-move).

**Next:** Checkpoint 6 (hierarchical κ shrinkage) — but the ν finding may
warrant a ruling first.
