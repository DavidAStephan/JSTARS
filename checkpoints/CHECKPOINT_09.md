# Checkpoint 9 — Post-completion audit-and-improve pass (2026-07-11)

Orchestrated session (Fable + Haiku/Sonnet sub-agents) auditing the
completed toolbox and testing one improvement candidate end-to-end.
Everything below was produced under the convergence discipline: no
single-seed number is treated as evidence.

## 1. Code audit (mechanical sweep, all of +jointstar/ + benchmarks/)

**Two findings, both minor:**

1. `estimate.m` wrote `smc_log.csv` into `OutDir` with no guard against
   an existing file — concurrent runs sharing an `OutDir` (easy via the
   `'results'` default) would silently interleave/overwrite. **Fixed
   this session**: `warning('jointstar:existingLog', ...)` at run start.
2. `docs/03_validation_vs_baseline.m` leading-digit filename can't be
   `run` directly — already documented in CHECKPOINT_08; cosmetic, left
   as-is.

**Verified still fixed / clean** (the full list matters because these
were real bugs once): `'fixed'` priors excluded from `mutateIdx` on both
the plain and horseshoe paths; φ_y/φ_u sign restrictions enforced at all
three evaluation points (priorSample/priorLogPdf/mhMutate);
EXCLUDE/EXCLUDE_GAP pairs structurally unroutable into mutable L columns
(pre-filtered indices in hsUnpack); NaN-initialised state accumulators;
explicit upstream data masking, no silent truncation; no lag/lead
misalignment in the IS term; all hardcoded constants documented and
intentional. No new bug class found.

## 2. Refreshed baseline (what any change must beat)

From `results/cp7b` (single seed) and `results/pooled_posterior.csv`
(3-seed pool, the quotable table): 17/23 Table-3 90%-CI overlaps; the 6
misses are all on the known ridges — ξ1+ξ2, ρ_U, ρ_hpp, φ1, φ2, κ_u2020.
cp7b SMC health confirmed by direct log read: 30 stages, final ESS
1597/2000, MH acceptance ≈ 0.22, ~48 min. End-of-sample r* 0.97%
[−1.9, 3.6]; NAIRU 5.83 [4.83, 6.88]; gap +0.40 [−0.64, 1.65]; mean r*
90%-band width 4.34 pp.

## 3. Improvement candidates considered

Ranked list came from a read-only proposal pass grounded in the
checkpoints and `results/cp7b/hs_shrinkage.csv`:

- **A1 ridge-atom MH blocking — implemented and tested this session
  (§4–6).**
- **A2 MSteps 4→6** — already endorsed by the CHECKPOINT_08 addendum;
  not re-tested (would prove what's expected at ~50% wall-clock cost);
  remains the recommended knob when tighter single-run footprints are
  worth ~70–75 min/seed.
- **A3 split the horseshoe cross block by π^e involvement** — DROPPED on
  evidence, not deferred: the new τ_g diagnostic (§4) shows the cross
  block's global scale (median 0.109, 5% 0.069) is not pinned at the
  0.05 truncation floor, so the "π^e pairs starved by a shared jammed
  scale" premise fails. (The measurement group's 5% quantile does sit at
  the floor — but that binds in the more-shrinkage direction, harmless.)
- **A4 surface τ_g by group** — implemented (§4).
- **B tier (prior recentering toward the baseline: ρ_U, φ2, γ2)** —
  NOT run; requires owner sign-off (§8). These work by overriding flat
  likelihood directions with the baseline's answer — answer-matching,
  not identification — and re-litigate the Rees-priors adoption.
- **C tier (r* identification, surveys/pre-1993 cash rate)** —
  data-blocked, no action possible.

## 4. Changes shipped (all code, no model/prior/spec change; 24/24 tests)

- `runSMC.m` + new `blockPartition.m`: `'RidgeAtoms'` option
  (**default off** — default path verified byte-identical, exact
  `randperm`/`linspace` reproduction). Atoms glue documented ridge sets
  into shared MH blocks: {φsum,φ2}, {ρ_U,ξ1,ξ2}, {ρ_pr,θ1,θ2},
  {ρ_hpp,λ1,λ2}, {ρ_k,χ1,χ2}, {m84_z,σ_z}, {m84_c,σ_c}, plus one atom
  per COVID-κ window (κ + its lm/la hypers), resolved by name from the
  prior spec.
- **LML now recorded**: per-stage `lml_inc` column in `smc_log.csv`,
  `out.lml` on the result struct (the tempering-identity total already
  existed internally as `logZ`; it was never surfaced — so no
  pre-existing run's LML is recoverable, all future runs carry it).
- `horseshoeDiag.m`: group-level τ_g median/5/95 table, written to
  `tau_group.csv`. cp7b values: meas 0.061 [0.050, 0.096]; trend 0.158
  [0.096, 0.272]; drift 0.575 [0.239, 1.263]; cross 0.109 [0.069,
  0.183].
- `estimate.m`: existing-log warning (§1), `'RidgeAtoms'` plumbing.
- New tests: `testRidgeAtoms.m`, `testLmlRecording.m`. Suite: 24 green.
- New benchmark: `benchmarks/checkAtomsConvergence.m` (R̂ + pooling over
  the atoms runs; reusable for future variants).

## 5. RidgeAtoms evaluation — gate then full check

**Single-seed gate** (paired N=1000/MSteps=2/seed 42, atoms off vs on,
`results/gate_atoms_*`): health identical (27 vs 29 stages, ~21 vs ~23
min, acc 0.29 vs 0.28). The atoms leg landed closer to trusted values on
φ2/κ_y2020/ν-sign, but a single pair is not evidence — gate passed on
health only.

**Full 3-seed check** (seeds 42/7/101, N=2000, MSteps=2, matched to the
original R̂ benchmark; `results/atoms_seed*`,
`results/convergence_rhat_atoms.csv`, `results/pooled_posterior_atoms.csv`):

| | R̂>1.5 | R̂>2 | max R̂ |
|---|---|---|---|
| no atoms (CP8 addendum) | 101 (25%) | 54 (13%) | 5.31 (ρ_hpp) |
| atoms | 104 (26%) | 54 (13%) | 5.03 (g_w-bar) |

**No population-level improvement — the instability relocated.**
Per-atom: {φsum,φ2}, {ρ_hpp,λ1,λ2}, {m84_c,σ_c} and the 2020-κ group
genuinely improved; {ρ_U,ξ1,ξ2} and {ρ_pr,θ1,θ2} all-members worsened
(θ2 → new top-3 offender *despite being inside an atom*); collateral
destabilisation of previously-stable non-atom parameters (σ_gpr
1.24→4.92, g_w-bar 2.52→5.03, α, φ_hpp) — mechanistically expected,
since forcing atom contiguity perturbs everyone else's block
assignment. Pooled-vs-pooled: consistent within noise except **ρ_U
(0.86 → 0.74, ≈1.9 old-pool SD, per-seed means scattered 0.56/0.86/0.81
vs the old tight 0.90/0.81/0.88)** — flagged for the owner in §8.
Latent states robust throughout (r* end 0.46/0.83/0.85 vs old
0.97/0.63/0.83, all inside each other's bands).

**Decision (spec-comparator concurring): RidgeAtoms ships OFF by
default.** It is a legitimate zero-cost targeted tool (e.g. a focused
look at ρ_hpp or the 2020-κ block) but not a defensible default: it
doesn't reduce the population-level instability, worsens a known-issue
ridge, and flipping the default would desynchronise the production
recipe from the existing quotable baseline. Production recipe is
unchanged: **≥3 seeds pooled, optionally MSteps 4–6.**

## 6. LML noise floor (new, quantitatively useful)

Same-spec LML estimates across seeds: −637.9 / −635.1 / −645.7 at
N=2000 (spread ≈ 10.6); the N=1000 gate pair differed by ≈ 24.5.
**Consequence: single-run LML differences of order 10–25 log points are
sampler noise for this model.** Any future Bayes-factor-style spec
comparison needs multi-seed LML (means across ≥3 seeds), exactly like
the parameters.

## 7. Artifacts

New/changed code: `+jointstar/{runSMC,estimate,horseshoeDiag}.m`,
`+jointstar/blockPartition.m` (new), `tests/{testRidgeAtoms,
testLmlRecording}.m` (new), `benchmarks/checkAtomsConvergence.m` (new).
Results: `results/gate_atoms_{off,on}`, `results/atoms_seed{42,7,101}`,
`results/convergence_rhat_atoms.csv`, `results/pooled_posterior_atoms.csv`,
`results/audit_tmp` (τ_g scratch, deletable). Quotable table remains
`results/pooled_posterior.csv` (unchanged).

## 8. Items requiring owner sign-off

1. **B-tier prior recentering** (not run): tighten/recenter ρ_U prior
   toward the baseline's ~0.20 split; recenter φ2 toward the baseline
   hump (−0.65, sd 0.35); recenter |γ2| toward 0.10. Each would likely
   move its non-overlapping Table-3 row into overlap — by asserting the
   baseline's answer on a flat likelihood direction. Recommended order
   if approved: ρ_U, then γ2, then φ2; test individually, ≥3 seeds
   each (~2.5 h per variant).
2. **The ρ_U pooled shift under atoms** (0.86 → 0.74, ≈1.9 SD): most
   likely atoms worsened that ridge's mixing, but the alternative —
   that the old pool under-explored the ridge and the wider new pool is
   the more honest stratified estimate — can't be excluded from two
   3-run pools. If ρ_U matters for the write-up, a 6-run pool (both
   triples combined, valid since both target the same posterior) is the
   cheap next step; say the word.
3. **Atom-subset retest** (optional follow-up): keep only the atoms
   that empirically helped ({ρ_hpp,λ1,λ2}, {m84_c,σ_c}, {φsum,φ2},
   COVID-κ groups), drop the two that hurt. Concrete and testable
   (~2.5 h for a 3-seed check) — but only worth it if a tighter
   single-run footprint has operational value beyond what pooling
   already provides.
4. Whether to commit this session's code changes to git (nothing
   committed yet).
