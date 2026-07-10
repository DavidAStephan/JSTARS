# Checkpoint 4 — SMC on full JointSTAR, no horseshoe (PASSED; rerun with Rees 2019 priors)

**Definitive run** (`results/cp4_rees/`): Rees (2019) priors as specified —
Beta on the gap-AR(1) coefficients, γ2 = −|γ2| with
|γ2| ~ Beta(.2,.1), gap AR(2) in (sum, second-lag) parameterisation, the
"(t)+(t−1)" Normals placed on loading sums. N=2000, 66 parameters, seed 42:
**converged in 20 stages, 6.9 minutes**, acceptance ≈ 0.48, final ESS ≈ 2000.
(An earlier run with provisional priors is kept in `results/cp4/` for the
prior-sensitivity comparison.)

**Headline posteriors:** γ1 = 0.59, γ2 = −0.33; Okun ξ1 = −0.19; IS ν = −0.07
(90% band excludes 0); α = 0.38; gap AR(2): φ1+φ2 = 0.82, φ2 = 0.05.
Smoothed r*: ~3.9% (2004) declining to **~2.0% now** [−1.3, 5.5]; NAIRU
9.0 → 5.5pp; trend inflation ≈ 2.3%. Factor-of-2 baseline check still
pending the baseline model's published estimates.

**Findings:**
1. **Prior sensitivity is material.** Under unrestricted priors the Phillips
   slope was −0.08 with a band spanning zero; the paper's sign-restricted
   Beta gives −0.33. Gap dynamics also moved (hump-shaped (1.21, −0.43) →
   flatter (0.77, 0.05)) and end-of-sample r* fell from ~3.0 to ~2.0. The
   likelihood is ridged along these directions; the priors are doing real
   work. Worth a paragraph in the validation report.
2. **σ_gz and σ_ξ remain prior-dominated** (pile-up); r* bands ±3-4pp.
   The survey series (Checkpoint 7) are the intended fix.
3. **d_t/κ competition:** φ_y ≈ 0 while κ_y,2020 is large — the variance
   channel absorbs GDP's COVID level shift. Watch at Checkpoint 7.

**Next:** Checkpoint 5 (horseshoe LDL′ on innovation covariance).
