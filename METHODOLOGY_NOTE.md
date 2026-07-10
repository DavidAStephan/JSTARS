# Methodology note: estimation approach and related literature

## What we're actually combining

The estimation approach used in `jointstar2` (and inherited by `jointstar_lm`) combines two
methodological ingredients that come from **separate literatures and are not, as far as we can
tell, combined this way in any published paper**. Worth being explicit about this — both for
accurate citation in any write-up, and so nobody assumes we're just implementing an existing
named method.

1. **Precision-based state-space computation** — Chan and Jeliazkov (2009), building on Chib and
   Jeliazkov (2006). For a linear Gaussian state-space model, the entire latent state path can be
   integrated out analytically via a sparse precision matrix and a banded/sparse Cholesky
   factorisation, giving both the marginal likelihood `p(y|θ)` and state draws without running a
   Kalman filter or a particle filter. This is a pure computational-efficiency result: it doesn't
   change what's being estimated, only how fast each likelihood evaluation is.

2. **Adaptive-tempering SMC over the parameter vector** — Herbst and Schorfheide (2014, and their
   2015 book *Bayesian Estimation of DSGE Models*). A population of particles is moved through a
   sequence of tempered posteriors, bridging from the prior to the full posterior, with resampling
   and MCMC-mutation steps at each stage. Designed as a more robust alternative to single-chain
   Metropolis-Hastings for posteriors with ridges, multimodality, or other awkward geometry.

**The combination — using Chan's fast precision-based likelihood evaluation as the inner
likelihood call inside a Herbst-Schorfheide-style batch tempering SMC — is our synthesis, not a
lift from a specific paper.** It's a natural fit (SMC needs many cheap likelihood evaluations;
Chan's method supplies exactly that for linear Gaussian state-space models), but if writing this
up formally, cite both source papers separately rather than implying a single unified method.

## Why this isn't the same as SMC²

Worth flagging because it's an easy conflation. There's an established SMC variant used in
several recent macro UC papers — **SMC²** (Chopin, Jacob, and Papaspiliopoulos, 2013) — which is
*not* the same algorithm as what's described above, despite both being "SMC applied to a UC
model."

- **What we use (batch tempering SMC + Chan precision likelihood)**: fixed full sample, particles
  over θ only, states integrated out analytically at each likelihood evaluation via sparse
  precision matrices. Runs once per estimation, offline.
- **SMC²**: designed for *online/sequential* estimation as new data arrives. Maintains `M`
  parameter particles, and **each parameter particle carries its own nested cloud of `N` state
  particles** — a full particle filter inside every parameter particle. Weights update as each new
  observation arrives; periodic PMCMC "rejuvenation" moves refresh the parameter particle
  population when weight degeneracy gets too severe.

SMC² exists because it's built to handle **nonlinear or non-Gaussian** state evolution, where the
states can't be integrated out in closed form and a particle filter is the only option. For a
linear Gaussian model — which JointSTAR and the labour-slack model both are — SMC² is solving a
harder problem than we have. Chan's precision approach is the more efficient tool for our case
specifically because it avoids needing a particle filter over states at all.

## Related literature: a direct empirical precedent

**Jahan-Parvar, Knipp, and Szerszeń (2024), "Trend-Cycle Decomposition and Forecasting Using
Bayesian Multivariate Unobserved Components,"** Federal Reserve FEDS working paper 2024-100.
([PDF](https://www.federalreserve.gov/econres/feds/files/2024100pap.pdf))

This is the closest published precedent to what we're doing, worth reading before any RDP-style
write-up. Key points:

- A multivariate UC model of US output, unemployment, and PCE inflation, sharing a common
  stochastic cycle with stochastic volatility, estimated online via **SMC²** (not the batch
  tempering + precision-sampler combination we use, per the distinction above).
- **Directly relevant finding**: fully Bayesian estimation that properly accounts for parameter
  uncertainty produces materially lower out-of-sample RMSFE than MLE / point-estimate approaches,
  especially at longer forecast horizons and for multivariate specifications. Their MLE-based
  RMSFEs are described as "generally much larger" than the fully-Bayesian SMC-based ones (their
  Table 2, Panel A vs Panel B).
- They also find univariate models "severely underestimate peaks and troughs" of the gap measures
  compared to their multivariate model, and that simpler/alternative models "showed much smaller
  unemployment gaps" during COVID, "implying inability to capture the abrupt turns" in
  macroeconomic variables' trends — explicitly noting that "if employed as policy tools, the
  failures of these alternative models could have serious implications."

That last point is a US-unemployment-gap instance of essentially the same failure mode that
motivated the JointSTAR modernisation in the first place (NAIRU/gap estimates lagging reality
during fast-moving episodes), via a related but different mechanism (a shared stochastic cycle
with time-varying volatility, rather than our multi-indicator factor-loading structure). Useful
as an independent, quantified citation for why properly-converged Bayesian treatment of these
models is not a cosmetic improvement — it changes forecast accuracy in ways that matter if the
model is used as a policy input.

## Core citations

- Chib, S., Jeliazkov, I., 2006. Accept–reject Metropolis–Hastings sampling and marginal
  likelihood estimation. *Statistica Neerlandica* 55, 12–26.
- Chan, J.C.C., Jeliazkov, I., 2009. Efficient simulation and integrated likelihood estimation in
  state space models. *International Journal of Mathematical Modelling and Numerical
  Optimisation* 1, 101–120.
- Herbst, E., Schorfheide, F., 2014. Sequential Monte Carlo sampling for DSGE models. *Journal of
  Applied Econometrics* 29, 1073–1098.
- Herbst, E., Schorfheide, F., 2015. *Bayesian Estimation of DSGE Models*. Princeton University
  Press.
- Chopin, N., Jacob, P.E., Papaspiliopoulos, O., 2013. SMC²: an efficient algorithm for sequential
  analysis of state-space models. *Journal of the Royal Statistical Society: Series B* 75,
  397–426.
- Jahan-Parvar, M.R., Knipp, C., Szerszeń, P.J., 2024. Trend-cycle decomposition and forecasting
  using Bayesian multivariate unobserved components. FEDS Working Paper 2024-100, Federal Reserve
  Board.
