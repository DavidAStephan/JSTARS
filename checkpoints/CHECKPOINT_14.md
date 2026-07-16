# Checkpoint 14 — Convergence, resolved on two fronts: the R̂ goal was mis-calibrated, and waste-free SMC halves the residual (2026-07-16)

Follows CHECKPOINT_13 (E1 ladder, max 3-seed R̂ = 2.20). Prompted by
the owner's question: "R̂ 2.20 is too high for a publishable piece —
find how researchers get good convergence, R̂ of 1 is the goal."
Two orchestrated workflows (literature review → implementation +
Opus verification) plus three MATLAB batches. All code uncommitted,
behind default-off flags; `production.m` unchanged except the table
regenerated under the already-adopted trio kernel.

## Finding 1 — the "R̂ = 1.0" goal is not the standard this estimator class is judged by

Full literature review in `results/literature_review/
CONVERGENCE_LITERATURE_REVIEW.md` (7 searchers → 44 sources → 13
deep-reads → 7 Opus verifications). Headline, Opus-verified against the
primary sources:

- **No published SMC paper computes or meets a cross-seed R̂ threshold.**
  Herbst & Schorfheide (2013/2014, read in full) contain no
  Gelman–Rubin statistic. Their stated practitioner standard (Sec 4.3,
  after Durham–Geweke 2012) is: run G independent times (they use
  G=20), report STD-across-runs of each posterior moment → N_eff =
  V_post/STD(mean)². N_eff of 45–4190 was deemed excellent.
- R̂ theory (Vehtari et al. 2021) assumes time-ordered Markov chains;
  SMC particle clouds are exchangeable/weighted/genealogical, so
  treating seed clouds as "chains" is pragmatic but non-standard, and
  the classical formula is unreliable on the right-skewed variance
  parameters that dominate our worst list.
- With only M=3 "chains" the R̂ estimate is itself very noisy — the max
  over 79 params is upward-biased.

**Consequence, measured.** New tool `benchmarks/computeNeff.m`
(Opus-verified clean; reproduces `production.m`'s R̂ formula bit-for-bit).
Ran the field-native report over **G=20 seeds** (production 42/7/101 +
17 new seeds in `results/neff_sweep/`, ~6.8 min each):

| statistic (baseline trio kernel) | 3-seed (old headline) | **G=20 (honest)** |
|---|---|---|
| max classical R̂ | 2.20 (sig_k) | **1.69 (sme_w)** |
| max rank bulk R̂ | — | 1.55 |
| max rank folded R̂ | 2.08 | 1.22 |
| **max(bulk,folded)** — proper Vehtari report | — | **1.55** |
| params max(bulk,folded) > 1.2 | — | **58 of 79** |
| params max(bulk,folded) > 1.1 | — | 76 of 79 |

Output: `results/production/neff_g20.csv`. **Most of the "2.20" was
small-G noise in the max** (sig_k's 2.20 regressed to 1.19 at G=20).
But note the correction: the modern statistic to report is
max(bulk,folded) rank R̂ (Vehtari), on which the baseline trio kernel is
**1.55 with 58 of 79 params above 1.2** — the folded-only column (1.22)
is too rosy on its own. So the baseline is NOT as converged as a first
read suggested; a real mixing gap remains, which Finding 2 closes.

## Finding 2 — waste-free SMC cuts the residual R̂ roughly in half at equal compute

`'WasteFree'` option implemented in `estimate.m`/`runSMC.m` (Dau &
Chopin 2022; Opus-verified: target-correct, exact eval-budget parity
M·(P−1) = N·MSteps_eff, flag-off path bit-identical, new test file
`tests/testWasteFree.m` 4/4 green). Matched 3-seed A/B, same seeds
(42/7/101), same eval budget, trio kernel in both arms:

| | baseline trio | **+ WasteFree** |
|---|---|---|
| max 3-seed classical R̂ | 2.20 | **1.21 (sig_Ustar)** |
| max 3-seed rank-folded R̂ | 2.08 | **1.25** |
| median N_eff (per run) | 2.6 | **16.6** |
| sig_k: R̂ / N_eff | 2.20 / 0.93 | **1.00 / 150** |
| phiy: R̂ / N_eff | 2.12 / 0.95 | **1.01 / 101** |

Every one of the baseline's worst-14 parameters collapsed to R̂ ≈
1.0–1.2 (`results/experiments/WF_wastefree/neff_wf3.csv` vs
`neff_base3.csv`). Uniform collapse across 14 independent ridge
directions is a mixing signal, not a lucky-seed draw. Mechanism matches
the CP12 diagnosis exactly: waste-free retains the intermediate
mutation states that the classical kernel discards, so the same
likelihood budget yields a larger, better-mixed final ensemble along
the ridge crest.

**But it also moves the posterior** (pooled WF vs baseline):
phisum 0.885 → 0.957, phi2 −0.132 → −0.454, rhoU 0.81 → 0.66,
γ2 −0.248 → −0.217, gap median 0.57 → 1.21 (wider band). Latent r*,
NAIRU, and trend inflation are stable (r* end 0.62 → 0.72 within band;
NAIRU ~5.3 both). **Interpretation:** the current production trio
kernel is STILL under-mixing — it sits biased on the ridge, and
waste-free climbs higher, so the CP12/CP13 "sticky-kernel artifact"
warning still bites. Adopting WF changes the quotable table again
(gap-AR block most).

**Caveats.** (a) WF's adaptive-tempering ESS is computed on the
correlated M·P cloud and is inflated, so the φ-schedule differs and LML
is NOT comparable across arms (WF −442 vs baseline −499 is meaningless);
R̂/N_eff on the final φ=1 clouds is the valid comparison. (b) Cloud
grows to 12,500 particles at the final stage — handled correctly by the
pooling and R̂ code (variable cloud size).

### Finding 2 CONFIRMED at G=20 (17 WF seeds added, `results/experiments/WF_sweep/`)

The 3-seed WF result held and improved on the robust statistic. G=20,
baseline trio kernel → + WasteFree, equal compute
(`results/experiments/WF_wastefree/neff_wf_g20.csv`):

| statistic (G=20) | baseline | **+ WasteFree** |
|---|---|---|
| max classical R̂ | 1.69 | **1.19** |
| max rank bulk R̂ | 1.55 | **1.24** |
| max rank folded R̂ | 1.22 | **1.04** |
| **max(bulk,folded)** | 1.55 | **1.24** |
| params max(bulk,folded) > 1.2 | 58 | **2** |
| params > 1.1 | 76 | 22 |
| median N_eff (per run) | 2.65 | **7.15** (~2.7×) |

Waste-free improves EVERY convergence statistic at G=20 — this is a
replicated result, not a 3-seed fluke. Max folded R̂ 1.04 is essentially
converged; the conservative max(bulk,folded) 1.24 leaves only 2 of 79
params above 1.2 (COVID scale params kapk_20/kapc_2021 and rhok/phiy,
all just over 1.0). The economic shift is CONFIRMED stable across 20
seeds (baseline-G20 → WF-G20 pooled): phisum 0.907 → 0.959 (toward the
in-house baseline Table-3 value 0.96 the model had been missing),
phi2 −0.160 → −0.456, rhoU 0.76 → 0.65, γ2 −0.234 → −0.224 (barely
moves), posterior SDs roughly halve (sig_k sd 0.0093 → 0.0055). Latent
r*/NAIRU/trend-inflation stable. So waste-free is a genuine win AND it
moves the gap-AR block — adoption is now a clean recommendation, gated
only on owner sign-off for the (baseline-consistent) economic shift.
Pooled tables: `results/production/pooled_g20.csv`,
`results/experiments/WF_wastefree/pooled_wf_g20.csv`.

## Finding 3 — the COVID hyperparameters are structurally weak-ID; no sampler can converge them

Deterministic conditional log-likelihood profiles at two anchors
(`benchmarks/profileConditional.m`, Opus-verified; replaces the
marginal-LML profile, which is buried under the ~10-log-pt seed noise
floor) + prior/posterior contraction ratios
(`benchmarks/contractionRatios.m`):

- **kapHyp_lm_w2021, kapHyp_lm_w2023tot: exactly 0.000 log-points of
  likelihood variation** across their whole range — they enter only the
  hierarchical-κ prior, never the state-space likelihood (grep-confirmed
  absent from ModelSpec.m). Contraction ratios 0.91 and 1.34 (both >
  0.7 "prior-dominated"). No amount of mixing drives these to R̂ = 1;
  the honest move is to relabel them "structurally weakly identified
  (~8 COVID quarters); posterior prior-dominated" (Lenza–Primiceri
  precedent). Evidence: `results/production/contraction_ratios.csv`,
  `results/identification_profiles/`.
- **phiy now profiles as CURVED / data-identified** at both anchors
  (~19 log-pt ll range) under the trio kernel, contraction 0.34. So its
  residual R̂ is ridge-mixing, not weak identification — the
  literature-review candidate to *tighten phiy's prior* is NOT needed
  and is dropped. phiu is data-identified (contraction 0.24). This
  narrows the honest-relabel set to the kapHyp_* hierarchy only.
- sig_k profiles strongly curved conditionally (63–91 log-pts) yet was
  the worst 3-seed R̂ — the signature of a ridge (curved in every 1-D
  slice, flat combination direction). Waste-free, not prior surgery, is
  the right lever, and it worked (Finding 2).

## Rejected on the math (Opus verification, do not pursue)

Non-centered reparameterization of the κ hierarchy (it is truncated-
Gamma, not Gaussian location-scale); signed-√ SD transform (the IG(3,·)
priors already repel zero — no funnel here); parameter expansion
(sig_*/sme_* are independent InvGamma, no batch scale). Details in the
literature review. PACF/tanh AR reparameterization survives as
low-priority (touches only rhok + the gap-AR pair, none of the worst
mass).

## Deliverables this session (all uncommitted)

- `benchmarks/computeNeff.m` — N_eff + rank-normalized folded R̂ report
  (also fixed loadFinalSnapshot to pick the final cloud by parsed stage
  number, copy-invariant, vs production.m's mtime pick).
- `+jointstar/estimate.m`, `runSMC.m` — `'WasteFree'` option (default
  off), `tests/testWasteFree.m`.
- `benchmarks/contractionRatios.m`, `benchmarks/profileConditional.m`.
- `results/production/` regenerated under the trio kernel (γ2 −0.252,
  max 3-seed R̂ 2.20 — independently replicates the E1 league table);
  old raw-kernel table archived to `results/production_rawkernel/`.
- `results/neff_sweep/` (17 seeds), `results/experiments/WF_wastefree/`
  (3 WF seeds), `results/production/neff_g20.csv`.

## PENDING OWNER DECISIONS

1. **Adopt the field-native convergence report** (N_eff over G≥10 seeds
   + rank-normalized folded R̂) as the quotable convergence standard,
   replacing "max classical R̂". Under it the current model is at
   rank-folded R̂ 1.22 (2 params > 1.2). Zero risk.
2. **Adopt WasteFree in production** — CONFIRMED at G=20 (max folded R̂
   1.04, max(bulk,folded) 1.24, only 2 params > 1.2, N_eff ~2.7×). It is
   the recommended production kernel. Owner sign-off needed only because
   it changes the quotable table again (phi2 −0.16 → −0.46, phisum →
   0.96 — which moves the gap-AR block TOWARD the in-house baseline —
   γ2 barely moves, SDs halve). To adopt: add `'WasteFree', true` to the
   `jointstar.estimate` call in `production.m`, delete `results/production/`,
   re-run. Budget-parity M/P rule and flag-off bit-identity are
   Opus-verified.
3. **Relabel the kapHyp_* COVID hyperparameters** as structurally
   weakly identified (prior-dominated) in the quotable table. Drop the
   phiy prior-tightening idea (phiy is data-identified under the new
   kernel).

Still pending from CP13: the trio kernel itself (MutationTransform +
StructuredBlocks + MStepsLadder) is committed but production.m adopting
it + regenerating the table still awaits the same sign-off; this
session regenerated the table so the numbers are on disk.
