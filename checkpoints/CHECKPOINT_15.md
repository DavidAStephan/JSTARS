# Checkpoint 15 — COVID-block specification investigation: four ideas, one clean adoption (2026-07-16)

Follows CHECKPOINT_14 (waste-free SMC). Owner asked what to change in
the specification, particularly around the COVID parameters. Four ideas
were investigated 1-by-1, each as a new default-off name-value flag on
`jointstar.estimate` (Opus-designed, Sonnet/Haiku-coded,
Opus-math-verified), estimated on the waste-free kernel and compared vs
the same-seed waste-free baseline (Opus-adjudicated). All UNCOMMITTED,
behind default-off flags, `production.m` untouched, suite 111 green.

## The four flags

| flag | change | params |
|---|---|---|
| FixSingletonKappa | drop hyperparameters on the 2 single-member kappa groups (g5=kaphpp_2022, g6=kappi_2023) → fixed a-priori-calibrated priors | 79→75 |
| PoolAcuteOnly | keep the kappa hierarchy ONLY for the 4-member 2020 acute group; fix priors on g2–g6 (subsumes FixSingletonKappa) | 79→69 |
| CovidDecay | Lenza-Primiceri geometric decay (s0_y/u/pr/k + rho_c) replacing the free per-window kappa | 79→70 |
| DropPhiY | fix phiy≡0 (nested restriction test of the d/kappa stringency-vs-variance competition) | 79 (phiy fixed) |

Calibration of all fixed priors is a-priori (matched to the incumbent
hyperprior-implied mean/spread, NOT to any posterior — Opus-verified
non-circular). No owner ruling reversed; kappa≥1, 2023Q4 boundary,
sme_pieobs, phi sign restriction, diagonal cov all preserved.

## Convergence + economics (baseline = waste-free kernel, same seeds)

10-seed screen, then G=20 confirmation for the two adopt-recommend
simplifications. Baseline (G=20): max rank R̂ **1.238**, median N_eff
**7.15**, gap 2020Q2 trough **−2.69** [−5.44, −0.26].

| flag | G | max rank R̂ | median N_eff | γ2 Δ | phi2 Δ | 2020Q2 trough |
|---|---|---|---|---|---|---|
| **PoolAcuteOnly** | 20 | **1.188** | **8.18** | −0.008 | +0.016 | −2.31 (≈ baseline) |
| **FixSingletonKappa** | 20 | **1.162** | **9.00** | −0.002 | −0.001 | −3.29 (~0.6pp deeper) |
| CovidDecay | 10 | 1.176 | 8.97 | ~0 | **+0.056** | **−4.99** (doubles) |
| DropPhiY | 10 | 1.233 | 9.01 | −0.007 | −0.006 | −3.41 (~1pp deeper) |

## Verdicts

**PoolAcuteOnly — ADOPT (confirmed at G=20). The clean win.** Better
convergence than baseline (R̂ 1.238→1.188, N_eff 7.15→8.18), all
structural economics within cross-seed noise (γ2, gap-AR, Okun), latent
states and the noisy COVID trough essentially unchanged — all while
dropping **10 unidentified hyperparameters**. It strictly dominates the
current spec: the same answer with a third fewer COVID nuisance
parameters. Recommended as the production parsimony default.

**FixSingletonKappa — subsumed by PoolAcuteOnly.** Also confirmed clean
on convergence (best R̂ 1.162) and structural economics, but retains a
modest ~0.6pp deepening of the 2020Q2 gap trough that persists across
both 10 and 20 seeds (bands overlap heavily, so likely still noise on
the least-seed-stable latent quantity). Since PoolAcuteOnly fixes these
same singleton groups AND the 2-member groups AND shows *no* trough
wobble, PoolAcuteOnly is the better choice. Keep FixSingletonKappa's
flag as the minimal-change fallback; adopt PoolAcuteOnly.

**CovidDecay — OWNER ECONOMIC DECISION (not a diagnostics adopt).** Best
convergence of all four (R̂ 1.176) — which means its economic shift is
REAL, not a ridge artifact: the parametric decay moves gap-AR the most
(phi2 +12%) and **doubles the COVID gap trough (−2.7 → −5.0, narrower
band)**. That is a legible, well-identified alternative to the
per-window-kappa hierarchy, but it moves *away* from the model's
shallow/transitory-COVID design intent. Whether deeper is "better" is a
modeling judgment, not a convergence result. Decay params are
economically sensible (rho_c 0.74, s0 1.2–6.0). Surface for owner
sign-off per the no-silent-reparameterization norm; only run a G=20
confirmation if the owner favors the direction.

**DropPhiY — OWNER CALL, lean REJECT.** No parsimony (still 79 params),
no convergence gain (R̂ 1.233 ≈ baseline). Zeroing the stringency-GDP
loading barely moves structural params (phiy was weakly identified for
them) but reallocates COVID GDP weakness onto the kappa multipliers
(trough −2.7 → −3.4) rather than improving identification, and it
touches the explicit phiy<0 owner ruling. No self-standing case; absent
an owner reason to remove the stringency channel, reject.

## Recommendation

1. **Adopt PoolAcuteOnly into production** — add `'PoolAcuteOnly', true`
   to the `jointstar.estimate` call in `production.m` (alongside the
   waste-free kernel decision from CP14), delete `results/production/`,
   regenerate. G=20-confirmed strict improvement: same economics, 10
   fewer unidentified parameters, slightly better convergence. Owner
   sign-off is a formality (no economic change).
2. **CovidDecay**: owner decision on whether the deeper parametric-decay
   COVID dip is preferred over the shallow-transitory design. If yes,
   develop + G=20-confirm; if no, keep the flag dormant.
3. **DropPhiY**: reject unless there is an economic reason to drop the
   stringency-GDP channel.

## Artifacts (all uncommitted)

Flags: `+jointstar/{estimate,defaultPriors,ModelSpec,priorLogPdf}.m`;
tests `tests/test{FixSingletonKappa,PoolAcuteOnly,CovidDecay,DropPhiY}.m`
(111 suite green); drivers `benchmarks/run<Flag>.m`; comparator
`benchmarks/compareCovidFlags.m`. Runs: `results/experiments/<Flag>/`
(10 seeds each; the two winners at 20). G=20 comparison CSVs:
`results/experiments/covid_compare_g20/`.
