# Methodology note: pedigree and precedents

*Rewritten 2026-07-14 after the methodology review (CHECKPOINT_10) and
the subsequent decision to drop the horseshoe covariance layer
(CHECKPOINT_11). The production model now uses a **diagonal** innovation
covariance — matching the original in-house model — so it is assembled
entirely from standard, published, centrally-used components with **no
novel statistical methodology**. An earlier version of this note
described the estimator as a "novel synthesis"; that framing was found
to be an overstatement and has been removed.*

## The method in one paragraph

The estimator is a standard adaptive likelihood-tempering SMC sampler
of the Herbst–Schorfheide (2014) type — adaptive ESS-targeted
tempering, systematic resampling, cloud-covariance blocked Metropolis
mutation, and the usual tempering marginal-likelihood identity, all
with direct precedent in the NY Fed's production SMC.jl code and its
companion paper (Cai et al. 2021) — whose inner likelihood is the
Chan–Jeliazkov (2009) sparse-precision evaluation of a linear Gaussian
unobserved-components model with a **diagonal innovation covariance**.
For a linear Gaussian model the precision evaluation returns the
numerically identical marginal likelihood a Kalman filter would
(verified in the test suite to 1e-8), so it contributes efficiency, not
statistical content. Every ingredient — the sampler, the likelihood
evaluator, and the diagonal covariance — is standard and precedented.
The only model-specific modelling choices are a hierarchical
truncated-Gamma treatment of the COVID variance-scaling parameters and
the use of a constructed inflation-expectations series as a direct
measurement of the trend-inflation state; neither is a new statistical
method.

## Why the likelihood engine adds no statistical novelty

For a linear Gaussian state-space model the integrated likelihood
p(y|θ) is a single Gaussian integral over the latent state path. The
Chan–Jeliazkov precision route and the Kalman filter are two exact
factorizations of that same integral; the choice between them alters
floating-point conditioning and speed, but no statistical property of
any sampler built on top. Substituting one for the other inside
tempering SMC is an implementation choice of the same kind as the NY
Fed's use of optimized Kalman routines or Herbst's (2015) Chandrasekhar
recursions.

Verification: `tests/testPrecisionVsKalman.m` (AR(1)+noise over a
parameter grid, nonzero initial mean, scattered and contiguous missing
data; loglik equal to 1e-8; 10,000 state draws match the smoother's
moments). Coverage caveat: the independent Kalman oracle is first-order
only; the production model's AR(2) gap block is validated indirectly
via the companion-form embedding test (`testZlagVsCompanion`).

## Precedent map

**The sampler applied to exactly our model class.** Herbst &
Schorfheide (2014)'s *first empirical illustration* (their Section 5.1)
is a plain linear Gaussian state-space model — not a DSGE — run
through the identical tempering algorithm with the states-marginalized
likelihood. "Tempering SMC wrapped around a linear-Gaussian p(y|θ)" is
page one of the source paper.

**The sampler in central-bank production code.**
- NY Fed `SMC.jl`/`DSGE.jl` (BSD-3, github.com/FRBNY-DSGE): adaptive
  ESS-targeted tempering (Cai, Del Negro, Herbst, Matlin, Sarfati,
  Schorfheide 2021, *Econometrics Journal*), systematic resampling at
  threshold 0.5·N, blocked random-partition RW-MH mutation with
  cloud-covariance proposals, 0.25 acceptance targeting, and the same
  log-marginal-likelihood accumulation identity we implement.
- Dynare ≥ 6.0 ships the Herbst–Schorfheide sampler as
  `posterior_sampling_method='hssmc'`.
- Random blocked mutation partitions are core Herbst–Schorfheide
  machinery (Nblocks = 3–6 in their Smets–Wouters application), with
  Chib & Ramamurthy (2010) as the antecedent.

**The likelihood evaluator in exactly our model class.**
- Chan & Jeliazkov (2009): the precision method itself; Chan's MATLAB
  code is publicly posted for UC-class models.
- Chan, Koop & Potter (2016, JAE) and Grant & Chan (2017, JEDC): UC /
  trend-cycle / NAIRU models estimated by banded-precision methods.
- **Zaman (2022, Cleveland Fed WP 21-23R)**: a large multivariate UC
  "stars" model (r*, u*, g*, π*) estimated at a Federal Reserve bank
  using exactly the Chan–Jeliazkov band/sparse routines — the closest
  production-scale precedent for our model class and likelihood
  technique.
- Mertens (2023, JEDC / Bundesbank): precision-based sampling for
  singular state-space / trend-cycle models.

**The diagonal innovation covariance** is the default for essentially
every UC/trend-cycle model in the literature (HLW, the Chan UC papers
above, the original in-house JointSTAR) — the most standard choice
possible, and the one this project now uses.

## Reference-implementation cross-check

| Design choice | Ours (`runSMC.m`) | Herbst–Schorfheide 2014/15 | NY Fed SMC.jl (Cai et al. 2021) | Dynare `hssmc` |
|---|---|---|---|---|
| Tempering schedule | adaptive: bisection so post-reweight ESS = 0.5·N | fixed φ_n = (n/N_φ)^λ | both; adaptive targets ESS ratio | fixed (n/N_φ)^λ |
| Resampling | systematic, trigger ESS ≤ 0.5·N | multinomial baseline | systematic default, threshold 0.5 | — |
| Mutation | blocked RW-MH, random partitions, cloud covariance, 2.38²/d scaling | blocked RW-MH, random partitions, cloud covariance | blocked RW-MH, cloud covariance | RW-MH |
| Acceptance targeting | step rule, band 0.20–0.35 | smooth logistic, target 0.25 | target 0.25 | target 0.25 |
| Likelihood | Chan–Jeliazkov precision (≡ Kalman, verified 1e-8) | Kalman filter | Kalman filter | Kalman filter |
| Innovation covariance | diagonal | (DSGE structural) | (DSGE structural) | (DSGE structural) |

Deviations from the references are variants, not departures: our
absolute-ESS bisection vs. SMC.jl's relative-ESS rule; our step-rule
acceptance adaptation vs. HS's logistic (same 0.25 target region).

## The horseshoe exploration (done, and dropped)

Between Checkpoints 5–8 this project explored a grouped-horseshoe
shrinkage prior on the innovation-covariance Cholesky off-diagonals —
a discovery tool for "which cross-shock correlations does the data
identify." It was the one component with **no published precedent in
any setting** (horseshoe-via-SMC), and the methodology review
(CHECKPOINT_10) additionally found it carried a kernel-invariance bug.
Empirically it identified only ~17 of 106 off-diagonals, concentrated
in the trend/drift blocks, and — once the gap-shock correlations were
excluded by owner ruling to stop them distorting the cycle dynamics —
it did **not** materially change the headline latent states or the
policy-relevant parameters relative to the diagonal model. It was
therefore dropped (CHECKPOINT_11) in favour of the diagonal covariance
that matches the original model and is fully precedented. The
scientific content of that exploration — "the data identify few
cross-shock correlations and none that move the headline" — is a
positive result that *justifies* the diagonal choice with evidence
rather than assumption.

## Known validity caveat

**Seed-stability.** The model's structural parameters are not
seed-stable in a single SMC run (cross-seed R̂ up to ~5.8), because the
posterior lives on long likelihood ridges (the gap-AR persistence
split, the ρ_U/Okun trade-off, the r* band). This is a property of the
model's geometry, present in the diagonal model too (dropping the
horseshoe did not change it), and is the natural next target for
improvement (reparameterization of the ridges). The reported posterior
is a ≥3-seed equal-weight pool — an inter-seed uncertainty envelope
that widens intervals where seeds disagree, preferable to any single
seed, but not a converged posterior; pooling averages seed noise, it
does not remove finite-N bias. The tempering marginal-likelihood
`out.lml`/logZ is an internal diagnostic, not a calibrated marginal
likelihood (it omits prior truncation/stationarity normalizer
constants).

## Why this isn't SMC²

SMC² (Chopin, Jacob & Papaspiliopoulos 2013) targets online/sequential
estimation where states cannot be integrated out and each parameter
particle carries a nested particle filter. Our model is linear
Gaussian, states are integrated out exactly at every evaluation, and
the sampler is a batch (offline) tempering run. The Jahan-Parvar et al.
(2024) UC application uses SMC², not batch tempering.

## Core citations

- Chan, J.C.C., Jeliazkov, I. (2009). Efficient simulation and
  integrated likelihood estimation in state space models. *IJMMNO* 1,
  101–120.
- Chan, J.C.C., Koop, G., Potter, S. (2016). A bounded model of time
  variation in trend inflation, NAIRU and the Phillips curve. *JAE* 31,
  551–565.
- Grant, A.L., Chan, J.C.C. (2017). Reconciling output gaps. *JEDC* 75,
  114–121.
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
- Chopin, N., Jacob, P.E., Papaspiliopoulos, O. (2013). SMC²: an
  efficient algorithm for sequential analysis of state-space models.
  *JRSS-B* 75, 397–426.
- Jahan-Parvar, M.R., Knipp, C., Szerszeń, P.J. (2024). Trend-cycle
  decomposition and forecasting using Bayesian multivariate unobserved
  components. FEDS 2024-100.
- Software: FRBNY `DSGE.jl`/`SMC.jl` (BSD-3), Dynare ≥6.0 (`hssmc`,
  GPL), J. Chan's MATLAB precision-sampler code (joshuachan.org).
