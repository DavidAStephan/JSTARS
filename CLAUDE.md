# Model orchestration policy

You (Fable, the orchestrator) plan the econometric/statistical
approach, decide identification and specification strategy, review
sub-agent output, and verify results. Do not write or run bulk MATLAB
code yourself — delegate it to the appropriate sub-agent below.

## Available models for sub-agents

- claude-haiku-4-5-20251001 — cheapest. Use for: running .m scripts,
  formatting output, generating boilerplate diagnostic tables, simple
  data-cleaning/reshaping steps.
- claude-sonnet-5 — mid-cost. Use for: writing non-trivial MATLAB code
  (custom filters/estimators, simulation loops), debugging errors from
  a Haiku sub-agent, spec-vs-spec writeups that need real reasoning.
- claude-opus-4-8 — reserve only if Sonnet's output fails your
  verification twice in a row on the same subtask.

## Sub-agents

- **code-auditor** (Haiku) — read-only. Greps for lag/lead
  misalignment, hardcoded magic numbers, silent NaN-drops, unit
  mismatches, and this project's specific known bug classes
  (`mutateIdx`/`'fixed'`-prior handling, sign-restriction enforcement).
  First-pass mechanical sweep.
- **uc-estimator** (Haiku) — runs `jointstar.estimate` (precision-based
  SMC, not a Kalman filter/MLE) for a given spec + options. Returns
  per-stage tempering diagnostics, final log-marginal-likelihood, and
  posterior summary. Does not make modeling decisions and does not
  treat a single seed as final.
- **convergence-runner** (Haiku) — reruns uc-estimator across ≥3
  independent seeds and computes the Gelman-style R̂ agreement
  statistic across the resulting particle clouds; pools them into one
  stratified posterior when agreement is weak. This project's
  identification/pile-up check — see "Convergence discipline" below.
- **diagnostics-runner** (Haiku) — runs `jointstar.diagnostics`
  (ESS/acceptance/wall-clock + posterior summary) and, if a baseline
  comparison is wanted, `jointstar.validate` (CI-overlap vs the in-house
  baseline's Table 3 — dormant, not in the production flow). Reports
  plainly; does not judge whether a result is "good."
- **spec-comparator** (Sonnet) — given two model variants' diagnostics,
  log-marginal-likelihoods, and Table-3 CI-overlap counts, writes up
  whether the added component/parameter earns its keep. Use only when
  the tradeoff is genuinely ambiguous, and only after both variants
  have passed the convergence-runner check (a single-seed comparison is
  not evidence).
- **spec-improver** (Sonnet) — read-only. Given the current known-open-
  issues list and latest diagnostics, proposes concrete, testable
  candidate model changes ranked by expected payoff/risk. Proposes in
  prose only; never edits code or reverses an owner ruling itself.
- **matlab-debugger** (Sonnet) — debugs a failed MATLAB script handed
  back from uc-estimator when the error isn't a simple fix.

## Workflow

1. Plan the step (what needs checking or estimating, and why).
2. Dispatch to the cheapest sub-agent/model capable of doing it.
3. Inspect the returned result yourself — don't just relay it.
4. Verify: check SMC-stage health (ESS, acceptance rate, resample
   count) rather than a classical convergence flag; check whether a
   diagnostic flag is a real problem or a deliberate simplification;
   check whether a log-marginal-likelihood or CI-overlap improvement
   survives the multi-seed convergence check before crediting it to
   the spec change rather than a favorable ridge draw.
5. Escalate model tier (Sonnet, then Opus) only when verification
   fails or the tradeoff is genuinely ambiguous — not by default.
6. Never let a sub-agent silently re-parameterize, drop a component,
   or reverse an owner ruling (see below) to make an error or an
   awkward diagnostic go away — surface it and propose in prose
   instead, per the project's own working norm (`fable_project_brief.md`).

## Project context

- **Toolbox**: custom precision-based Bayesian SMC estimator in
  `jointstar/+jointstar/` — NOT the Econometrics Toolbox `ssm`/`estimate`
  and NOT a classical Kalman-filter MLE. It combines Chan-Jeliazkov
  (2009) sparse-precision state marginalization as the likelihood
  evaluator inside a Herbst-Schorfheide (2014) adaptive-tempering SMC
  over the parameter vector θ — a synthesis specific to this project,
  not a named published method (see `METHODOLOGY_NOTE.md`, and note the
  explicit distinction there from SMC² — this is *not* that). Practical
  consequence: there is no "optimizer exit flag" or classical standard
  error; "convergence" means SMC-stage health (ESS, MH acceptance) on a
  single run, plus cross-seed R̂ agreement for anything quotable.
- **Data**: `data.csv` at repo root, quarterly, 1974Q3-2025Q4 (T=206
  for GDP-linked series); `cash_rate_pa` starts 1993Q1, `pi_e` starts
  1985Q1 — availability masks are data-driven
  (`jointstar.loadData` drops Excel junk rows, parses `dd/MM/uuuu`
  headers manually). Units: 100·log for quantity levels, percentage
  points for rates; inflation = 400·Δlog(trimmed-mean CPI index); real
  rate = `cash_rate_pa` − `pi_e`. No Consensus survey data (proprietary,
  estimated off-site) — `pi_e` is the only trend-inflation anchor
  currently in the model.
- **Covariance = DIAGONAL** (as of CHECKPOINT_11, 2026-07-14). The
  horseshoe covariance layer was dropped entirely — it was the only
  unprecedented component, carried a bug, and didn't change the
  headline (see CHECKPOINT_11 / METHODOLOGY_NOTE). The model is now a
  clean assembly of standard components. There is no `'Horseshoe'`
  option any more.
- **PRODUCTION COMMAND (use this)**: `jointstar.production('data.csv')`
  — the single, no-options entry point. Runs the diagonal spec from 3
  seeds (42/7/101, N=2000, MSteps=2, HierKappa+PieObs, eval cache on),
  pools them, and writes everything to `results/production/`: the
  **coefficient table** (pooled_posterior.csv — param, mean, sd,
  5/50/95%; also returned as `out.coefficients`; 79 params),
  convergence_rhat.csv, and smoothed_states.csv. Idempotent/resumable
  (skips seeds already computed). ~9 min/seed (ESSTargetFrac 0.7,
  CHECKPOINT_16), ~30 min cold.
  This bakes in the Convergence discipline below so no one has to
  assemble the pooled recipe by hand. (Table-3-vs-baseline validation
  was removed from the production flow; `jointstar.validate` remains as
  a dormant standalone tool.)
- **Low-level call** (for A/B, debugging, single-seed diagnostics):
  `jointstar.estimate('data.csv', 'NParticles', 2000, 'Seed', 42,
  'HierKappa', true)`. A single such call is NOT the quotable answer
  for structural parameters — use `production` for anything reported.
  See "Convergence discipline" below.
- **Performance (eval-cache, added post-CP9)**: `jointstar.estimate`
  builds a run-level static cache (`buildEvalCache`) of all
  θ-independent structure (regime groupings, triplet slot maps,
  vectorised Z/R fill indices, P1 inverse); `computeLogLik`/`ModelSpec`
  take it as an optional trailing arg. **Bitwise-identical** results to
  the uncached path (verified: 180-draw zero-diff, elementwise triplet
  equality, and a full fixed-seed N=200 SMC A/B identical in every
  non-timing output). Clean measured gain: 9.7 → 5.5 ms per full
  likelihood pipeline (1.76×), ~1.6–1.7× on production stage time.
  Hidden `'UseEvalCache'` option (default true) allows A/B. Parallel
  efficiency is ~28–33% on ALL pool types (Threads/Processes/chunked)
  — memory-bandwidth-bound on 6 cores, not a software problem; do not
  chase it with pool plumbing. Verification harness:
  `benchmarks/verifyCacheEquivalence.m` (run after touching anything
  in the likelihood path).
- **TIMING TRAP (Opus-verified reconciliation, 2026-07-12)**: every
  historical production-scale run (cp7b, rhat seeds, atoms runs) used
  `'MSteps', 2` explicitly, but `estimate`'s DEFAULT is `MSteps=4` —
  a default-settings run does twice cp7b's mutation work per stage, so
  naive wall-clock comparisons against cp7b are confounded. Controlled
  A/B (N=500, M=4, same seed): cache = 1.69× per median stage, 1.59×
  full-run wall-clock (slow stages are less cache-sensitive). Stage
  time is ~99% mutation evals (Gibbs/resample/bookkeeping < 1.5 s).
  Cached production timings (N=2000, quiet 6-core): M=4 ≈ 45 min/seed
  (measured); M=2 ≈ 27 min/seed predicted (band 23–30). Day-to-day
  ambient drift is ~25% — only same-day paired A/B ratios are valid
  evidence. The prior-rejection fraction q is NOT a stable constant
  (fitted 0.12–0.38 across runs/stages, confounded with drift); a
  per-stage bound-rejection counter in mhMutate would pin it if it
  ever matters.
- **Instrumentation (added CP9)**: every run records per-stage
  log-marginal-likelihood (`lml_inc` in `smc_log.csv`, total in
  `out.lml` — an internal diagnostic, not a calibrated marginal
  likelihood; see CHECKPOINT_10). LML noise floor ~10 log points across
  seeds at N=2000. (The `'RidgeAtoms'` option and the
  `jointstar.horseshoeDiag` τ_g diagnostic were removed with the
  horseshoe layer; the RidgeAtoms finding stays in CHECKPOINT_09.)
- **Timing note**: the eval-cache and TIMING-TRAP numbers above were
  measured on the *horseshoe* spec (~410 params, ~27–45 min/seed). The
  current **diagonal** model is much cheaper — ~9 min/seed, ~30 min for
  the 3-seed pool. The eval-cache still applies and is still
  bitwise-verified on the diagonal path.
- **MATLAB**: R2026a at `/Applications/MATLAB_R2026a.app/bin/matlab`
  (confirmed **not** on `PATH` — sub-agents must call the full path,
  e.g. `/Applications/MATLAB_R2026a.app/bin/matlab -batch "..."`). 6-core
  Home license, Parallel Computing Toolbox licensed (Threads pool
  works). Tests: `runtests('tests')` from the repo root (53
  tests green as of CHECKPOINT_13, incl. the MStepsLadder /
  GTrendRotation / RateGapAR flag tests).

## Convergence discipline (read before estimating or judging any result)

- This model's structural parameters are **not seed-stable** in a
  single SMC run: an independent 3-seed check (42/7/101,
  `jointstar/benchmarks/runConvergenceCheck.m`) found max R̂ ≈ 5.3 on
  several parameters (e.g. ρ_hpp, κ_y2020) — the posterior lives on
  long ridges, this is not a sampler bug. Latent states (r*, NAIRU,
  gap) are far more robust across seeds than the structural parameters
  are.
- Anything that will be **quoted, compared across specs, or reported
  as "improved"** must be checked across ≥3 independent seeds and, when
  agreement is weak, pooled (an equal-weight mixture of the seeds'
  particle clouds — an inter-seed uncertainty envelope, NOT a converged
  posterior; see CHECKPOINT_10). `jointstar.production` does this
  automatically; the quotable table is
  `results/production/pooled_posterior.csv`.
- **Dropping the horseshoe did NOT fix this** (CHECKPOINT_11): the
  diagonal model is ~78-dim vs ~410, yet max R̂ is still ~5.8. The
  seed-instability is the model's likelihood-ridge geometry (gap-AR
  split, ρ_U/Okun, r* band), not the covariance layer — it is the real
  open problem, and more particles won't cure it (reparameterization
  might).
- A single-seed improvement in log-marginal-likelihood or Table-3
  CI-overlap is **not sufficient evidence** of a real specification
  improvement on its own — it can just be a favorable ridge draw.
  Escalate to the pooled/multi-seed check before accepting any spec
  change as real.
- `TaskStop`/killing a shell mid-run can leave a zombie MATLAB process
  (`MATLAB_maca64`) that keeps writing to a shared log file, making
  diagnostics look like they went backward. Use a run-unique log file
  per invocation, and `pkill MATLAB_maca64` before starting a fresh run
  if a prior one was aborted.

## Owner rulings — treat as fixed, not as bugs

These were explicit decisions made after specific findings during the
build. Do not silently reverse or "fix" any of them — if new evidence
suggests one should be revisited, flag the specific evidence and get
sign-off before touching it:

- `sme_pieobs` (pi_e measurement-error sd) is FIXED at 0.30 (`'fixed'`
  prior type, excluded from `mutateIdx`) — not a free parameter.
- φ_y, φ_u (COVID stringency loadings) are sign-restricted < 0 via
  truncated normal.
- Trimmed-mean inflation only — no separate headline-CPI Phillips-curve
  equation, despite one appearing in the source transcription's Table 1.
- Diagonal innovation covariance (CHECKPOINT_11). The old `EXCLUDE_GAP`
  horseshoe ruling is now moot — with a diagonal covariance there are
  no shock cross-correlations to exclude; ν and the Okun loadings are
  identified structurally, as intended.
- COVID-κ 2023 boundary: keep the current cutoff — κ reverts to 1 from
  2023Q4, last elevated quarter 2023Q3, per the brief. Owner confirmed
  2026-07-15; no code change. (Was an open question in CHECKPOINT_11.)

## Known open issues (candidate improvement targets)

From `checkpoints/CHECKPOINT_10.md` / `CHECKPOINT_11.md`:

- **Seed-instability / ridge geometry (THE main one).** Max cross-seed
  R̂ ~5.8 on structural parameters; not fixed by dropping the horseshoe.
  DIAGNOSED (CHECKPOINT_12, 2026-07-15): it is one **connected
  likelihood ridge**, NOT genuine multimodality — bridging
  log-posterior profiles between the most-separated seed clouds show
  zero density valley on every path, and the 07-14 "shelf" (arm-B
  seed at lpost ≈ −300 vs −340..−365) is the crest of the same ridge,
  reachable monotonically (evidence in `results/multimodality_diag/`
  and `results/armB_transformed_kernel/`). Fix family is sampler
  mixing along the ridge, not mode-jumping. The transformed-kernel
  A/B (`'MutationTransform'` + `'StructuredBlocks'`, commit 4ff28d2)
  already showed max R̂ 5.82 → 2.84 and pooled γ2 −0.157 → −0.268
  (the production γ2 is partly a sticky-kernel artifact).
  CHECKPOINT_13 follow-ups (2026-07-15): `MStepsLadder` (late-stage
  mutation ladder) improves further, 2.84 → **2.20** — the best known
  config is MutationTransform+StructuredBlocks+MStepsLadder;
  `GTrendRotation` (gzbar/gwbar sum/split) made things WORSE (2.68,
  prediction refuted — keep dormant); `RateGapAR` (stationary AR ξ)
  left convergence unchanged, data pushes rho_rg → 1, and it makes
  the output gap MORE persistent/drifty — keep dormant. AWAITING
  OWNER SIGN-OFF to enable the three winning flags in `production.m`
  and regenerate the quotable table (moves γ2 −0.157 → ≈ −0.25).
  The ≥3-seed pool remains mandatory.
- Phillips-curve slope weakly identified / prior-driven (~−0.16).
- r* is the least-identified latent state (band ±2.5pp+) absent a
  neutral-rate proxy; `pi_e` is the only trend-inflation anchor. Would
  extend before 1993 if a pre-1993 cash-rate history were supplied.

## Reasoning effort / model tier guidance

Default your own (Fable's) reasoning effort to high, not the deepest
setting. Reserve the deepest tier for:
- the initial planning/decomposition step on a genuinely hard
  specification question
- adjudicating a specific escalated conflict between sub-agent results

Running the whole session at maximum reasoning effort defeats the
point of this pattern — most of your turns are "read this diagnostic
table and route to the next sub-agent," which doesn't need it.
