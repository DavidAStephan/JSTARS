# jointstar — fast Bayesian estimation of the JointSTAR unobserved-components model

MATLAB toolbox replacing the Metropolis–Hastings JointSTAR estimator with a
precision-based state-space likelihood (Chan–Jeliazkov 2009) inside Sequential
Monte Carlo with adaptive tempering (Herbst–Schorfheide 2014), diagonal
innovation covariance (matching the original RBA model), and external survey
series as direct trend measurements.

## Status: all eight checkpoints complete

| Piece | Status |
|---|---|
| 1. Precision-based likelihood + state sampler | **Done** (CP1–2: matches Kalman to 1e-8 incl. `Zlag` lagged loadings and full-R; 9.5 ms/eval vs 50–100 ms target) |
| 2. SMC with adaptive tempering | **Done** (CP3 bimodal toy; block MH + adaptive scale + per-particle scaled proposals for hierarchical coords) |
| 3. Innovation covariance | Diagonal Q/R (matching the original RBA model). A grouped-horseshoe extension on the off-diagonals was explored and dropped — see `CHECKPOINT_11`. |
| 3b. Hierarchical COVID κ | **Done** (CP6: Gamma(a_g,b_g)·1[κ≥1], window-shared hypers; extremes shrink toward window means) |
| 4. Trend measurement row | **Done** (CP7: constructed `pi_e` anchors trend inflation — Consensus unavailable; error calibrated 30bp; φ_y, φ_u sign-restricted per ruling) |
| Validation | Retired along with the horseshoe model (was CP8-specific, `docs/03_validation_vs_baseline.mlx`); current production diagnostics are the pooled posterior and cross-seed R̂ in `results/production/`. |

**Production estimation (one command, no options):**
```matlab
out = jointstar.production('data.csv');   % run from the repo root
```
Runs the full diagonal-covariance spec from 3 seeds (42/7/101), pools them
into the quotable posterior, and writes the **coefficient table**
(`out.coefficients` / `results/production/pooled_posterior.csv`), R̂
convergence, and smoothed states to `results/production/`. ~80 min on 6
cores; idempotent (skips seeds already computed). The lower-level
`jointstar.estimate('data.csv', 'NParticles', 2000, 'Seed', 42, 'HierKappa',
true)` runs a single seed for A/B and diagnostics. 79 parameters; cross-seed
R̂ is elevated on ridge parameters (max ~5.8) — a mixing property of the
posterior, not something the diagonal-vs-horseshoe choice fixes or causes;
r* ends the sample at 0.73%.

Checkpoint history + owner rulings: `checkpoints/`. Spec-decision switches
live in `defaultPriors.m` (sign restrictions, calibrated 30bp).

## Layout

```
+jointstar/          the package: ModelSpec, computeLogLik, drawStates,
                      runSMC, mhMutate, diagnostics
tests/                matlab.unittest suite; run with runtests('tests')
benchmarks/           benchJointstarSize.m — timing at full model size
checkpoints/          CHECKPOINT_XX.md summaries + QUESTIONS.md
results/              production posteriors, R̂ convergence, smoothed states
startup.m             paths, RNG, optional parpool
```

## Quick start

```matlab
startup                          % run from the repo root; startup(8) opens a parpool
runtests('tests')                 % full validation suite (~5 s)
benchJointstarSize                % timing at JointSTAR size
```

The estimation machinery is generic: `computeLogLik`/`drawStates` consume the
system struct from `ModelSpec.system()`, and `runSMC` consumes function
handles `logPrior`/`logLik`/`samplePrior`. The JointSTAR-specific work is a
single static constructor `ModelSpec.jointstar(theta, data, ...)` plus
`defaultPriors` — no changes to the samplers.
