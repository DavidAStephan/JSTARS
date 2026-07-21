# Checkpoint 16 — The "more draws" ladder: what buys mixing in SMC (2026-07-21)

Owner question: in MCMC, poor mixing → run 1M+ draws (8h). What is the
SMC equivalent? Four levers tried IN ORDER, each a paired 10-seed sweep
(seeds 42/7/101/1-6/8) of the production config (diagonal + trio kernel
+ WasteFree + PoolAcuteOnly, N=2000, MSteps=2) with ONE lever changed,
compared against the same-seed baseline in
`results/experiments/PoolAcuteOnly/`. Stats: `benchmarks/variantStats.m`
(extracted verbatim from the Opus-verified comparator). Runs in
`results/experiments/mixing_ladder/{msteps4,ess07,n4000}/`.

## Results (paired, same 10 seeds)

| lever | cost/seed | avg stages | max rank R̂ | max classic R̂ | median N_eff |
|---|---|---|---|---|---|
| baseline (production) | ~6.0 min | 29 | 1.208 | 1.257 | 7.7 |
| G=20 (step 1, CP15 — done) | — | — | 1.188 (G=20) | — | 8.2 |
| **MSteps 2→4** (step 2) | 12.0 min | 29 | 1.111 | 1.097 | **60.3** |
| **ESSTargetFrac 0.5→0.7** (step 3) | 8.9 min | 40.5 | **1.095** | **1.076** | 18.0 |
| NParticles 2000→4000 (step 4, control) | 13.2 min | 29 | 1.168 | 1.168 | 11.7 |

## Findings

1. **The MCMC-to-SMC lever mapping is confirmed empirically.** "More
   draws" maps to mutation effort (MSteps) and tempering resolution
   (ESSTargetFrac), NOT to more particles: N=4000 was the most expensive
   lever (2.2×) and bought the least R̂ (1.208→1.168) — particles behave
   like more chains, not longer chains.
2. **ESSTargetFrac 0.7 is the best R̂-per-minute** (1.208→1.095 at 1.5×
   cost): smaller φ increments mean the cloud never bridges a large
   target change, so seeds agree. Modest N_eff gain (→18).
3. **MSteps=4 is the N_eff monster** (7.7→60, ~8×, at 2.0× cost): under
   waste-free parity, doubling MSteps doubles chain length AND the
   retained cloud (~25k particles at φ=1). R̂ 1.208→1.111.
4. **All three levers shift the gap-AR block the same direction**
   (phi2 −0.42→−0.47..−0.53, phisum →0.96, rhoU →0.61-0.64,
   gamma2 stable ≈ −0.22): the production table still sits slightly
   short of the ridge crest. Any adoption therefore re-touches the
   quotable table and needs the usual G=20 regeneration.
5. Residual worst-5 under either good lever = the COVID-κ scale factors
   (~1.03–1.11) — the structurally weak-ID block (CP15), which no
   sampler budget converges.

## Cost framing vs the old MCMC habit

Old: 1M-draw MH chain ≈ 8h. Here: production (3 seeds pooled) is ~20 min;
the strongest single lever (ESS 0.7) makes it ~30 min; the belt-and-braces
publication run (MSteps=4 + ESS 0.7, G=20 seeds) is ~5h of unattended,
resumable, embarrassingly-parallel compute — and the honest convergence
report (N_eff / rank R̂) comes with it.

## Recommendation (PENDING OWNER)

- Production default: current config is already publication-grade under
  the field-native standard (G=20 rank R̂ 1.19). No change required.
- If adopting one lever: **ESSTargetFrac 0.7** (+50% runtime, R̂ ≈ 1.09,
  minimal code surface — the option is now plumbed through estimate.m).
- For the final paper table: one G=20 run at **MSteps=4 + ESS 0.7**
  (untested in combination — run 3 seeds first) for headline N_eff,
  regenerating pooled_posterior.csv and re-running the G=20 N_eff report.
- 10-seed deltas carry noise; per the convergence discipline, any adopted
  change gets a G=20 confirmation before the table is re-quoted.

## Code added this session (uncommitted)

- `+jointstar/estimate.m`: 'ESSTargetFrac' pass-through (default [] =
  unchanged; byte-identity verified).
- `benchmarks/runMixingLadder.m` (driver), `benchmarks/variantStats.m`
  (per-run-name convergence stats, extracted from compareCovidFlags).
