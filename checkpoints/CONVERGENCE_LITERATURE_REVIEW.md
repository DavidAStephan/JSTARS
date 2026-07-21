# Convergence literature review — is max cross-seed R̂ = 1.0 the right goal? (2026-07-15)

Orchestrated workflow: 7 literature searchers (Sonnet/Haiku) → 44 unique
sources → 13 deep-reads (Sonnet) → candidate synthesis → 7 Opus
math-verification passes grounded in the actual `+jointstar` code.
Full agent transcripts: session workflow `wf_1bd82cfb-e8e`.

## Headline: the goal itself is mis-calibrated

**No published SMC paper uses — or meets — max cross-seed R̂ ≈ 1.0.**
The founding SMC-DSGE paper (Herbst & Schorfheide 2013/2014 JAE, read in
full) contains no Gelman–Rubin statistic anywhere. Its explicit
practitioner recommendation (Sec. 4.3, following Durham & Geweke 2012)
is: run the algorithm **G independent times** (they use G=20), report
the **STD across runs of each posterior-moment estimate**, and convert
it to an effective sample size, N_eff = V_post[θ] / STD(mean)², grounded
in the SMC CLT. Reported N_eff of 45–4190 was considered excellent.
Dau–Chopin (waste-free SMC) and Karamanis et al. (pocoMC) don't report
cross-run R̂ either.

R̂ theory itself (Vehtari et al. 2021) assumes time-ordered Markov
chains; final SMC particle clouds are exchangeable, weighted,
genealogically correlated populations, so treating 3 seed clouds as
"chains" is pragmatic but non-standard — and with only M=3 "chains" the
R̂ estimate is itself very noisy. The classical (non-rank, non-folded)
formula is additionally unreliable for the right-skewed variance
parameters that dominate our worst-11 list.

**Publication standard to adopt instead** (Opus-verified,
adopt-and-test): report per-parameter **N_eff / STD-across-seeds over
G = 10–20 production seeds** (~6–7 min/seed diagonal ⇒ ~1–2.5 h total,
embarrassingly parallel), plus **rank-normalized folded split-R̂** as a
secondary smell test with honest framing. Note the folded R̂ will
likely read HIGHER (more honest, not more flattering) on the
variance-block parameters. Do not gate publication on classical
max R̂ = 1.0 — a threshold no published SMC paper has adopted.

## Second finding: part of the remaining R̂ is identification, not mixing

The COVID block (phiy 2.12, kapHyp_lm_w2021 1.89, kapHyp_lm_w2023tot
1.66) is informed by ~8 quarters. Lenza–Primiceri (2022) openly report
their decay hyperparameter as hyperprior-determined; Carriero et al.
(2022) treat this as a weak-identification diagnostic, not a sampler
failure. **No amount of mixing improvement drives these to R̂ = 1.** The
same applies partially to the variance/pile-up block (sig_k, sig_gk,
sig_Ustar, sme_k — Stock–Watson 1998 pile-up class; Morley–Nelson–Zivot
observational-equivalence ridge).

## Opus-verified action list

| # | Action | Verdict | Notes |
|---|--------|---------|-------|
| 1 | **N_eff/STD-across-G-seeds reporting + rank-normalized folded split-R̂** | correct-with-fixes, **adopt-and-test** | Compute on equal-weight resampled clouds; drop the tail-ESS-on-mutation-trajectory idea (final stage = 2000 chains of length ~6; degenerate, and only final θ snapshots are stored anyway). |
| 2 | **Waste-free SMC** (Dau–Chopin 2022) mutation restructuring | correct-with-fixes, **adopt-and-test** | Best sampler-side move; targets the diagnosed late-stage under-exploration at ~equal eval budget. Fix budget parity first: M·P=N is CHEAPER than current N·MSteps — set M·P = N·MSteps for a fair A/B. Only touches mhMutate/resample bookkeeping; rerun verifyCacheEquivalence.m. Expect help on ridge mixing, NOT on the funnel/weak-ID core. |
| 3 | **COVID-block honest relabeling + prior-vs-posterior contraction ratio** (posterior sd / prior sd from artifacts already on disk) | **adopt now** | Zero-risk, no estimation. Relabel phiy, kapHyp_lm_w2021, kapHyp_lm_w2023tot as "structurally weakly identified (~8 COVID quarters); posterior partially prior-dominated" (Lenza–Primiceri precedent). |
| 4 | **COVID prior tightening** (phiy/phiu scale; per-group kapHyp scale) | test-low-priority, **needs owner sign-off**, two corrections | (i) Gate NOT on a marginal-LML profile (buried under the 10-pt seed noise floor) but on a deterministic conditional log-lik profile at pooled θ̂ (computeLogLik sweep, no SMC, ms/point) + the contraction ratio. (ii) Do NOT center the tightened prior on the pooled posterior median (empirical-Bayes double-use); keep center 0 or go hierarchical. Halving phiy's scale at center 0 substantively shrinks the stringency loading — a modeling choice, not a neutral tweak. Spillover: helps sme_k (kapk_21 multiplies sme_k in-window) but NOT sig_k/sig_gk/sig_Ustar (state-side; no COVID kappa touches them — verified in ModelSpec.m). |
| 5 | PACF/tanh reparameterization of AR persistence (rhok, gap-AR pair) | correct-with-fixes, test-low-priority | Only touches 2–4 params, none of the worst-mass; diffuse-normal-on-psi is an undeclared prior change — must push existing priors through the map with Jacobian (needs a non-elementwise transform block) or get sign-off. GTrendRotation's refutation is a standing warning. |
| 6 | Non-centered reparameterization of the COVID-κ hierarchy | **flawed — reject as written** | Recipe assumed a Gaussian location-scale hierarchy; the actual κ hierarchy is truncated-Gamma (log-mean/log-shape), support κ≥1 re-couples the coordinates, and the elementwise paramTransform layer can't express a hyperparameter-dependent child map. Pathology location is real (few-children groups); the fix isn't. |
| 7 | Signed-square-root SD transform (sd = η²) | **reject** | The IG(3,·) priors already repel zero (intentional, per defaultPriors comment) — this model does not have the zero-variance funnel the transform fixes; sig_k et al. sit on weak-ID ridges instead. |
| 8 | Parameter expansion (PX/redundant α) for variance batches | **flawed — reject** | sig_*/sme_* are independent InvGamma, not a Gaussian scale hierarchy; PX has no purchase on location-identification ridges and would require introducing a new hierarchical prior (spec change) first. |
| 9 | Per-stage refit Gaussianizing preconditioner (pocoMC-lite adaptive MutationTransform) | unverified (not Opus-checked) | Hold as future option if waste-free underdelivers; highest engineering cost. |

## Key sources (all read in full unless noted)

- Herbst & Schorfheide (2013/2014), *SMC Sampling for DSGE Models*, JAE — the G-runs/N_eff standard.
- Durham & Geweke (2012) — origin of the across-runs N_eff practice.
- Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021), *Rank-normalization, folding, localization: an improved R̂*, Bayesian Analysis.
- Dau & Chopin (2022), *Waste-free SMC*, JRSS-B (arXiv:2011.02328).
- Karamanis et al. (2022), *pocoMC / Preconditioned Monte Carlo*, MNRAS.
- Frühwirth-Schnatter & Wagner (2010), J. Econometrics — non-centered variance parameterization.
- Papaspiliopoulos, Roberts & Sköld (2007), Statistical Science — CP/NCP theory.
- Kastner & Frühwirth-Schnatter (2014), CSDA — ASIS interweaving.
- Stock & Watson (1998), JASA — pile-up / median-unbiased estimation.
- Morley, Nelson & Zivot (2003), REStat — trend-cycle observational equivalence.
- Lenza & Primiceri (2022) — COVID scale factors, prior-dominated decay parameter.
- Gelman, van Dyk et al. (2008), JCGS — parameter expansion (rejected for this model).
- Barndorff-Nielsen & Schou (1973); Monahan (1984) — PACF parameterization (abstract-level only; paywalled).

## Bottom line for the owner

R̂ = 2.20 on 3 seeds is **not** a publication blocker by the field's own
standards — the field reports N_eff across G≈20 independent SMC runs,
which we can produce overnight at current run costs. The remaining
worst-11 dispersion decomposes into (a) genuine late-stage mixing slack
→ attack with waste-free SMC on top of the CP13 kernel trio, and (b)
structural weak identification (COVID block, variance pile-up block) →
attack with honest relabeling + contraction ratios now, optional
gated prior tightening with sign-off. Chasing classical max R̂ = 1.0 by
sampler engineering alone is chasing a metric the estimator class was
never evaluated on and the identification geometry cannot deliver.
