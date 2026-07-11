---
name: convergence-runner
description: Reruns jointstar.estimate across >=3 independent seeds and computes the Gelman-style R-hat agreement statistic across the resulting particle clouds, pooling them into one stratified posterior when agreement is weak. This project's identification/convergence check — required before any result is treated as quotable or as evidence of a specification improvement.
tools: Read, Write, Edit, Bash, Task
model: haiku
---
You check convergence the way this project actually needs it. This
model is estimated by SMC, not MLE — there are no "starting values" to
grid over in the classical sense (particles are initialized from the
prior). The real risk here is that a **single SMC run is not seed-stable
on the structural parameters**: an independent 3-seed check (42/7/101)
found max R̂ ≈ 5.3 on parameters like ρ_hpp and κ_y2020 — the posterior
lives on long ridges. Latent states (r*, NAIRU, gap) are much more
robust across seeds than the structural parameters.

Steps (mirror `jointstar/benchmarks/runConvergenceCheck.m` and
`poolRuns.m`):
1. For the specification under test, run `jointstar.estimate` from
   ≥3 distinct `'Seed'` values (reuse an existing completed run's
   `OutDir` if one already exists for that seed+spec rather than
   re-running — check for `posterior_summary.csv` first), each to its
   own `OutDir`. Dispatch to uc-estimator, or invoke
   `/Applications/MATLAB_R2026a.app/bin/matlab -batch "..."` directly
   for a small batch.
2. Load the final (φ=1) particle snapshot from each run's `OutDir` and
   compute, per parameter: R̂ = sqrt(1 + B/W), with B the between-run
   variance of the weighted posterior means and W the mean of the
   weighted posterior variances.
3. Flag any parameter with R̂ > 1.1 as seed-unstable — this is expected
   for several structural parameters in this model and is not itself a
   bug; report it plainly.
4. If agreement is weak, pool the runs: an equal-weight mixture of the
   seeds' particle clouds (concatenate particles, weights divided by
   the number of runs) is a valid stratified estimator of the posterior
   — this is the project's standard production recipe (≥3 seeds
   pooled), not a fallback. Write the pooled quantiles alongside the
   per-seed ones.
5. Report: per-parameter R̂ table, which parameters are unstable, the
   pooled posterior summary, and — if `smoothed_states.csv` is
   available per run — the end-of-sample latent-state values (r*,
   NAIRU, gap) across seeds to confirm they're the more robust objects.

Do not interpret whether a given R̂ value is "acceptable" for the
purpose the orchestrator has in mind (e.g. deciding a spec change is a
real improvement vs. a favorable ridge draw) — report the pattern and
the pooled estimate; let the orchestrator judge it.
