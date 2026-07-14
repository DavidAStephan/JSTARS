# Methodology note: pedigree, precedents, and what is actually ours

*Rewritten 2026-07-14 following an orchestrated methodology review
(literature sweep, reference-implementation comparison, and an
adversarial mathematics audit; see `checkpoints/CHECKPOINT_10.md` for
the full findings). This supersedes the earlier version of this note,
which described the estimation approach as a novel synthesis "not
combined this way in any published paper" — a claim the review found
to be an overstatement.*

## The method in one paragraph

The estimator is a standard adaptive likelihood-tempering SMC sampler
of the Herbst–Schorfheide (2014) type — adaptive ESS-targeted
tempering, systematic resampling, cloud-covariance blocked Metropolis
mutation, and the usual tempering marginal-likelihood identity, all
with direct precedent in the NY Fed's production SMC.jl code and its
companion paper (Cai et al. 2021) — whose inner likelihood is
evaluated by the Chan–Jeliazkov (2009) sparse-precision method. For a
linear Gaussian state-space model the precision evaluation returns the
numerically identical marginal likelihood a Kalman filter would
(verified in our test suite to 1e-8 against a textbook filter,
including missing data and a matched proper initial prior), and
therefore contributes efficiency but no statistical novelty. The
project's actual contribution is a modeling package layered on this
standard sampler: grouped-horseshoe shrinkage priors on the
innovation-covariance Cholesky factors with a Makalic–Schmidt
hyper-scale Gibbs step embedded in the SMC mutation, per-particle
hierarchically-scaled proposals for those factors, and a hierarchical
truncated-Gamma treatment of the COVID scaling parameters. We claim no
new statistical theory.

## Why the likelihood engine adds no statistical novelty

For a linear Gaussian state-space model the integrated likelihood
p(y|θ) is a single Gaussian integral over the latent state path. The
Chan–Jeliazkov precision route and the Kalman filter are two exact
factorizations of that same integral; the choice between them alters
floating-point conditioning and speed, but no statistical property of
any sampler built on top. Substituting one for the other inside
tempering SMC is an implementation choice of the same kind as the NY
Fed's use of optimized Kalman routines (StateSpaceRoutines.jl) or
Herbst's (2015) Chandrasekhar recursions — a literature that exists
precisely because swapping fast, exact likelihood evaluators in and
out of samplers is normal practice.

Verification in this repo: `tests/testPrecisionVsKalman.m` (AR(1)+noise
over a parameter grid, nonzero initial mean, scattered and contiguous
missing data; loglik equal to 1e-8; 10,000 state draws match the
smoother's moments) and `tests/testFullRVsKalman.m` (bivariate
local-level with full correlated measurement covariance and partially
missing observation vectors). Coverage caveat, per the math audit: the
independent Kalman oracle is first-order only; the production model's
AR(2) gap block is validated indirectly via the companion-form
embedding test (`testZlagVsCompanion`), not against a second-order
oracle.

## Precedent map

**The sampler applied to exactly our model class.** Herbst &
Schorfheide (2014)'s *first empirical illustration* (their Section 5.1)
is a plain linear Gaussian state-space model — not a DSGE — run
through the identical correction/selection/mutation tempering
algorithm with the states-marginalized likelihood. "Tempering SMC
wrapped around a linear-Gaussian p(y|θ)" is page one of the source
paper, not our invention.

**The sampler in central-bank production code.**
- NY Fed `SMC.jl`/`DSGE.jl` (BSD-3, github.com/FRBNY-DSGE): adaptive
  ESS-targeted tempering (choose φ_n by root-finding so ESS falls to a
  target fraction — Cai, Del Negro, Herbst, Matlin, Sarfati,
  Schorfheide 2021, *Econometrics Journal*), systematic resampling at
  threshold 0.5·N, blocked random-partition RW-MH mutation with
  cloud-covariance proposals, 0.25 acceptance targeting, and the same
  log-marginal-likelihood accumulation identity we implement.
- Dynare ≥ 6.0 ships the Herbst–Schorfheide sampler as
  `posterior_sampling_method='hssmc'` (fixed λ-schedule variant).
- Random blocked mutation partitions are core Herbst–Schorfheide
  machinery ("we use random blocking of parameters … during the MH
  mutation step"; N_blocks = 3–6 in their Smets–Wouters application),
  with Chib & Ramamurthy (2010) as the antecedent.

**The likelihood evaluator in exactly our model class.**
- Chan & Jeliazkov (2009): the precision method itself; Chan's MATLAB
  code is publicly posted (citation-ware) for UC-class models.
- Chan, Koop & Potter (2016, JAE): bivariate UC model of trend
  inflation and NAIRU estimated by banded-precision MCMC — an explicit
  statement that precision/banded methods are the standard non-Kalman
  route for this model class.
- Grant & Chan (2017, JEDC): UC/HP output-gap model, precision sampler.
- **Zaman (2022, Cleveland Fed WP 21-23R)**: a large multivariate UC
  "stars" model (r*, u*, g*, π*) jointly estimated at a Federal
  Reserve bank using exactly the Chan–Jeliazkov band/sparse routines —
  the closest production-scale precedent for our model class + our
  likelihood technique (outer sampler: Gibbs).
- Mertens (2023, JEDC / Bundesbank): precision-based sampling extended
  to singular state-space models, trend-cycle applications, posted code.

**Modern samplers on UC/stars models.**
- Jahan-Parvar, Knipp & Szerszeń (2024, FEDS 2024-100): multivariate
  UC model estimated by SMC² — see the distinction below.
- Lombardi & Sgherri (2007, ECB WP 794): natural-rate tracking via
  sequential (particle-filter) SMC.
- Bognanni & Herbst (2018, JAE): SMC replacing Gibbs for Markov-
  switching VARs — "SMC instead of a model-specific Gibbs sampler for
  a reduced-form macro model" as an established move.
- Brault (2024, Bank of Canada SWP 24-13): parallel tempering for DSGE
  estimation — tempering-family samplers under active central-bank use.

**The gap that remains, stated honestly.** No published paper we could
find runs batch likelihood-tempering SMC with a precision-evaluated
(rather than Kalman-evaluated) linear-Gaussian likelihood — the
pairing itself appears unpublished, though by the equivalence argument
it is an implementation footnote rather than a methodological novelty.
And no precedent exists anywhere, in any sampler family, for horseshoe
shrinkage priors estimated via SMC: that piece is genuinely ours (see
below).

## Reference-implementation cross-check

| Design choice | Ours (`runSMC.m`) | Herbst–Schorfheide 2014/15 | NY Fed SMC.jl (Cai et al. 2021) | Dynare `hssmc` |
|---|---|---|---|---|
| Tempering schedule | adaptive: bisection so post-reweight ESS = 0.5·N | fixed φ_n = (n/N_φ)^λ | both; adaptive targets ESS ratio α·ESS_{n−1} | fixed (n/N_φ)^λ, 25 stages |
| Resampling | systematic, trigger ESS ≤ 0.5·N | multinomial baseline (alternatives discussed) | systematic default, threshold 0.5 | not documented |
| Mutation | blocked RW-MH, random partitions ~40 cols, cloud covariance, 2.38²/d scaling | blocked RW-MH, random partitions, cloud covariance; N_blocks 3–6 | blocked RW-MH, `n_blocks`, cloud covariance | RW-MH, scale c |
| Acceptance targeting | step rule, no-change band 0.20–0.35 | smooth logistic, target 0.25 | target 0.25 | target 0.25 |
| Likelihood | Chan–Jeliazkov precision (≡ Kalman, verified 1e-8) | Kalman filter | Kalman filter (StateSpaceRoutines.jl) | Kalman filter |
| LML | tempering identity, log-sum-exp | same | same | same (presumed, HS-implementing) |

Deviations from the references are variants, not departures: our
absolute-ESS bisection vs. SMC.jl's relative-ESS rule; our step-rule
acceptance adaptation vs. HS's logistic (same target region).

## What is genuinely ours

1. **Grouped-horseshoe shrinkage priors on the innovation-covariance
   Cholesky off-diagonals (Lq/Lr), estimated within tempering SMC**,
   with the Makalic–Schmidt hyper-scale updates riding in a per-stage
   Gibbs hook while the L coefficients themselves move by tempered MH.
   No published precedent found for horseshoe-via-SMC in any setting.
   The validity argument: the likelihood does not depend on the
   hyper-scales, so their full conditional under any tempered target
   equals the prior conditional — the Gibbs step is φ-invariant. (See
   the caveat below on the truncation implementation.)
2. **Per-particle hierarchically-scaled RW proposals** for the L
   columns (each particle's own conditional-prior sd as the proposal
   metric, instead of the cross-particle cloud covariance) — a
   sensible adaptation for hierarchically-scaled coordinates with no
   direct reference twin.
3. **Hierarchical truncated-Gamma COVID-κ block** with window-shared
   hyperparameters.

## Known validity caveats (from the 2026-07-14 mathematics audit)

1. **Horseshoe truncation implementation (must-fix, open):** the code
   draws the hyper-scale conditionals untruncated and then clamps λ, τ
   to [0.05, 10], while `priorLogPdf` evaluates the untruncated
   density. The clamp puts point mass at the bounds, and the floor
   demonstrably binds for the measurement group — so the composite
   kernel is not exactly invariant for any single target, and
   measurement-block correlations are biased toward zero. Fix: draw
   truncated-IG conditionals by CDF inversion and make `priorLogPdf`
   consistent (indicator + truncation normalizer), or drop the clamp.
2. **Initialization approximation:** the φ=0 horseshoe-L draws are
   deliberately tighter than the prior (numerical-PD guard). The
   posterior imprint decays with mutation effort but is not exactly
   zero, and the reported logZ is a biased estimate of the marginal
   likelihood (first-order in the init gap, plus omitted truncation/
   stationarity normalizer constants). **Read `out.lml`/logZ as an
   internal diagnostic, not a calibrated marginal likelihood.**
3. **Multi-seed pooling:** the 3-seed pooled table is an inter-seed
   uncertainty envelope over runs that individually do not converge on
   ridge parameters (cross-seed R̂ up to ~5 ⇒ between/within variance
   ratio ~24) — honest and preferable to any single seed, but it is
   not a converged posterior, and pooling reduces seed noise, not
   finite-N bias. The durable fix is better ridge mixing
   (reparameterization, more mutation), not more pooling.

## Why this isn't SMC² (unchanged from the original note)

SMC² (Chopin, Jacob & Papaspiliopoulos 2013) targets online/sequential
estimation where states cannot be integrated out analytically: each
parameter particle carries a nested particle filter over states. Our
model is linear Gaussian, states are integrated out exactly at every
evaluation, and the sampler is a batch (offline) tempering run. The
Jahan-Parvar et al. (2024) UC application uses SMC², not batch
tempering — related, but a different algorithm solving a harder
problem than ours requires.

## Core citations

- Chan, J.C.C., Jeliazkov, I. (2009). Efficient simulation and
  integrated likelihood estimation in state space models. *IJMMNO* 1,
  101–120.
- Chan, J.C.C., Koop, G., Potter, S. (2016). A bounded model of time
  variation in trend inflation, NAIRU and the Phillips curve. *JAE*
  31, 551–565.
- Grant, A.L., Chan, J.C.C. (2017). Reconciling output gaps. *JEDC*
  75, 114–121.
- Herbst, E. (2015). Using the "Chandrasekhar recursions" for
  likelihood evaluation of DSGE models. *Computational Economics* 45,
  693–705.
- Herbst, E., Schorfheide, F. (2014). Sequential Monte Carlo sampling
  for DSGE models. *JAE* 29, 1073–1098.
- Herbst, E., Schorfheide, F. (2015). *Bayesian Estimation of DSGE
  Models*. Princeton University Press.
- Cai, M., Del Negro, M., Herbst, E., Matlin, E., Sarfati, R.,
  Schorfheide, F. (2021). Online estimation of DSGE models.
  *Econometrics Journal* 24, C33–C68. (SMC.jl companion paper.)
- Chib, S., Ramamurthy, S. (2010). Tailored randomized block MCMC
  methods with application to DSGE models. *Journal of Econometrics*
  155, 19–38.
- Zaman, S. (2022). A unified framework to estimate macroeconomic
  stars. FRB Cleveland WP 21-23R.
- Mertens, E. (2023). Precision-based sampling for state space models
  that have no measurement error. *JEDC*, 104720.
- Bognanni, M., Herbst, E. (2018). A sequential Monte Carlo approach
  to inference in multiple-equation Markov-switching models. *JAE*.
- Makalic, E., Schmidt, D.F. (2016). A simple sampler for the
  horseshoe estimator. *IEEE Signal Processing Letters* 23, 179–182.
- Carvalho, C.M., Polson, N.G., Scott, J.G. (2010). The horseshoe
  estimator for sparse signals. *Biometrika* 97, 465–480.
- Chopin, N., Jacob, P.E., Papaspiliopoulos, O. (2013). SMC²: an
  efficient algorithm for sequential analysis of state-space models.
  *JRSS-B* 75, 397–426.
- Jahan-Parvar, M.R., Knipp, C., Szerszeń, P.J. (2024). Trend-cycle
  decomposition and forecasting using Bayesian multivariate unobserved
  components. FEDS 2024-100.
- Software: FRBNY `DSGE.jl`/`SMC.jl` (BSD-3), Dynare ≥6.0 (`hssmc`,
  GPL), J. Chan's MATLAB precision-sampler code (joshuachan.org,
  citation-ware), E. Herbst's `fortress`.
