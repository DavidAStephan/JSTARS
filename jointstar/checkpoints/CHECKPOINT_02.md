# Checkpoint 2 — Precision likelihood on the full JointSTAR (PASSED, with one caveat)

**Validated.** The full model specification is implemented as a 14-state
system (`ModelSpec.jointstar`): gap AR(2) with the IS term, five trend/drift
pairs (exact 2×2 SPD innovation blocks from the contemporaneous-drift timing),
NAIRU, ξ, and the regime-switching π^e; measurement equations 28–34 with the
COVID level shifters, the error-correction terms implemented as lagged-state
loadings (`Zlag`, validated to 1e-8 against a companion-form Kalman filter),
the 1984Q1/1993Q1 volatility breaks, and the 12 κ windows (hard-capped at
2023Q3 per the brief). Data enters via `loadData` using the
agreed transforms (wapop = lf/(pr/100), 400·Δlog CPI index, r = cash − π^e),
sample trimmed to 1974Q3–2025Q4 (T=208).

At the prior-init θ the log-likelihood is finite and plausible (per-observation
log-density in a sane range), smoothed potential output tracks observed GDP,
and ≥26/30 prior draws give finite likelihoods (the rest are chol-guard
rejections, as designed). **Timing: 9.5 ms per evaluation** (system build +
likelihood, one core) vs the 50–100 ms target.

**Caveat:** the brief's specific test — likelihood at the baseline model's
*posterior mode* — is still impossible because the baseline outputs have not
been supplied. The IS term is dropped pre-1993Q3 (no real-rate data); flagged.

**Next:** Checkpoint 4 SMC run (in progress).
