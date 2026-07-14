# Checkpoint 10 — Methodology review: pedigree, precedents, validity audit (2026-07-14)

Orchestrated review (Fable coordinating; Sonnet literature + reference-
code agents with web access; Haiku code-mapping; Opus adversarial
mathematics audit) of the owner's concern: the estimation approach
looked like a novel combination (Chan-Jeliazkov precision likelihood
inside Herbst-Schorfheide tempering SMC), and the owner did not want to
be "first" — preferring (a) motivation from existing work or (b)
alignment with someone else's documented code.

## Headline verdict

**The method is not the novel synthesis the old METHODOLOGY_NOTE
claimed.** For a linear Gaussian model the precision likelihood and the
Kalman likelihood are the same mathematical object (two exact
factorizations of one Gaussian integral; equality machine-verified to
1e-8 in our tests). Herbst-Schorfheide (2014)'s own first empirical
illustration applies their tempering SMC to a plain linear Gaussian
state-space model with the states-marginalized likelihood. Therefore
the sampler is the published HS algorithm on its original model class;
the likelihood engine is an implementation choice with no statistical
content. Every sampler design choice (adaptive ESS tempering,
systematic resampling at 0.5N, random blocked mutation with cloud
covariance, 0.25-region acceptance targeting, the LML identity) has a
documented twin in HS 2014/2015 and/or the NY Fed's production SMC.jl
(Cai et al. 2021, BSD-3). Model-class precedent for the precision
likelihood at a central bank: Zaman (2022, Cleveland Fed), a large
multivariate stars UC model estimated with Chan-Jeliazkov banded
routines. Full citation map + cross-check table: METHODOLOGY_NOTE.md.

**What is genuinely ours** (and must be owned as a contribution, not
hidden): (1) grouped-horseshoe priors on innovation-covariance Cholesky
off-diagonals estimated within SMC (no horseshoe-via-SMC precedent
found anywhere), with the hyper-scale Gibbs hook + tempered-MH split
and its φ-invariance argument; (2) per-particle hierarchically-scaled
proposals for those columns; (3) the hierarchical truncated-Gamma
COVID-κ block. All engineering/modeling novelties, no new statistical
theory claimed.

**Recommendation adopted**: option (a) — motivate from HS14 + Cai et
al./SMC.jl + Chan/Zaman — with SMC.jl cited as the documented
reference implementation for the sampler shell (option (b) flavor).
METHODOLOGY_NOTE.md rewritten accordingly.

## Mathematics audit findings (Opus, adversarial)

1. **Equivalence reframing — VALID algebra, corollary trimmed.** The
   precision-vs-Kalman substitution carries no statistical content,
   but it does NOT transfer HS validity to the whole sampler: validity
   questions live in the mutation/initialization layer. Coverage note:
   the independent Kalman oracle test is first-order only; the AR(2)
   gap block is validated via the companion-embedding test, not a
   second-order oracle.
2. **Horseshoe Gibbs hook — INVALID as implemented (MUST-FIX, open).**
   `horseshoeSample.m` (and `priorSample.m` at init) draws the λ²/τ²
   conditionals UNtruncated and then clamps to bounds ([0.05,10] on
   λ,τ), while `priorLogPdf.m` evaluates the UNtruncated density with
   no indicator/normalizer. The MH layer and the Gibbs layer therefore
   target different distributions; no single target is invariant. The
   clamp is not a far-tail technicality: the measurement group's τ
   posterior 5% quantile sits AT the 0.05 floor, so the atom binds and
   biases measurement-block correlations toward zero. Fix options:
   (i) truncated-IG conditional draws via gammaincinv + make
   priorLogPdf consistent (indicator + (ν,ξ)-dependent truncation
   normalizer) — keeps the truncation rationale; or (ii) remove the
   clamp, keep only massless overflow guards — but the floor currently
   binds, so (ii) changes results more and the PD-guard motivation
   must be re-checked. EITHER fix changes sampled posteriors →
   requires a production re-run + revalidation. AWAITING OWNER
   DECISION.
3. **Initialization underdispersion — VALID-WITH-CAVEAT.** The φ=0
   horseshoe-L cloud is deliberately tighter than the prior. Posterior
   imprint decays with mutation effort (not exact validity — and the
   ridge R̂≈5 suggests mutation is not fully washing it out); the LML
   is first-order biased by the init gap (plus omitted normalizer
   constants). Adopted: logZ documented as an internal diagnostic, not
   a calibrated marginal likelihood. Optional exact-in-N repair
   (importance-correcting the initial weights by the closed-form
   Gaussian ratio) noted but NOT adopted — it would reintroduce the
   stage-0 weight degeneracy the tightening exists to avoid.
4. **Multi-seed pooling — relabelled.** R̂≈5 ⇒ between/within ≈24: the
   seeds explore different ridge sub-regions. The pool is an
   inter-seed-uncertainty envelope over under-converged runs —
   preferable to any single seed, but not a converged posterior, and
   pooling reduces seed noise, not finite-N bias; pooled intervals are
   not guaranteed conservative on dimensions all seeds jointly miss.
   Comments in poolRuns.m/production.m updated to say this.
5. **Novelty claim — corrected.** Old note's "not combined this way in
   any published paper" retired; replaced with the narrowed statement
   (standard sampler + standard evaluator; novelty confined to the
   shrinkage-prior modeling layer).

Verifier's note (Fable): Opus's draft mislabelled Dynare's `hssmc` as
"horseshoe-in-SMC" — `hssmc` is Herbst-Schorfheide SMC (no horseshoe).
Corrected here; the conclusion it touched (no horseshoe-via-SMC
precedent) is unaffected and confirmed by the literature sweep.

## Changes applied in this checkpoint

- METHODOLOGY_NOTE.md rewritten (pedigree, precedent map, cross-check
  table, honest novelty statement, validity caveats).
- Comment-level relabels (no behavior change): poolRuns.m and
  production.m no longer call the pool a "valid stratified estimator";
  priorSample.m init comment softened per audit item 3(a).
- This checkpoint records the audit.

## Open items awaiting owner sign-off

1. **The horseshoe truncation fix (audit item 2)** — recommended
   option (i) (truncated draws + consistent density). Costs: small code
   change + test updates + a fresh `jointstar.production` run (~80
   min) + comparison of the new pooled table (expect measurement-block
   correlations to move away from zero somewhat; other blocks little
   changed).
2. Whether to also add a second-order Kalman oracle test (cosmetic
   coverage; a few hours of work, no production impact).
