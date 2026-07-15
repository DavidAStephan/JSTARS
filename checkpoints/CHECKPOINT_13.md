# Checkpoint 13 — Convergence follow-up experiments: ladder wins, rotation fails informatively, rate-gap-AR answers the economics question (2026-07-15)

Follows CHECKPOINT_12 (connected ridge confirmed). Three gated,
individually-flagged experiments, each on 3 seeds (42/7/101, N=2000,
MSteps=2, HierKappa+PieObs) stacked on the transformed kernel
(`MutationTransform`+`StructuredBlocks`). All code is behind
default-off options with the default path proven untouched (test
suite grew 37 → 53, all green; flag-off bit-identity tests included).

## League table (max cross-seed R̂; worst-5 in each CSV)

| variant | kernel/spec | max R̂ | R̂>1.5 | pooled γ2 |
|---|---|---|---|---|
| production | raw kernel, baseline spec | 5.82 | — | −0.157 |
| arm B (CP12) | transformed kernel | 2.84 | — | −0.268 |
| **E1** | + `MStepsLadder` | **2.20** | 11/79 | −0.252 |
| E2 | E1 + `GTrendRotation` | 2.68 | 17/79 | −0.223 |
| E4 | E1 + `RateGapAR` (spec change) | 2.20 | 11/80 | −0.214 |

Evidence: `results/experiments/{E1_ladder,E2_rotation,E4_rgapar}/`
(convergence_rhat.csv, pooled_posterior.csv per variant; E4 adds
economics_comparison.csv).

## E1 — MSteps ladder (ADOPT-candidate)

`MStepsLadder`: M×1 below φ=0.7, ×2 to 0.95, ×3 above. Max R̂
2.84 → 2.20, R̂>1.5 down to 11/79, ~+40% mutation cost (~7 min/seed
diagonal). Economics stable. The best sampler-side configuration
found: **MutationTransform + StructuredBlocks + MStepsLadder**.

## E2 — gzbar/gwbar (sum,split) rotation (REJECT, keep dormant)

Opus-verified exact prior rotation, but the prediction FAILED: the
disagreement did not concentrate in the split (R̂ 1.25) — it stayed
in `gtrend_sum` (R̂ 2.08), and the worst-offender mass moved into the
wage block (sig_w 2.68, sme_w 2.58). Max R̂ worse than E1. Conclusion:
the binding ridge direction is the *level* of total trend growth vs
other blocks, not the gz/gw split. Genuine negative result; flag stays
in the code, default off, do not enable.

## E4 — stationary AR real-rate gap (owner-instructed spec probe)

`RateGapAR`: ξ (non-growth r* component) AR(1) `rho_rg` instead of RW
(Opus-approved design; new param, 80 total).

- **Convergence**: identical to E1 (max R̂ 2.20, 11 > 1.5). `rho_rg`
  itself is seed-unstable (R̂ 1.68) and piles toward 1: pooled mean
  0.878, q95 **0.989** — the data does not want ξ stationary; it
  pulls back toward the unit root.
- **Economics (the owner's concern, ANSWERED — direction inverted)**:
  r* band collapses (end-2025 5–95 width 2.9pp vs 5.6 E1 / 7.5
  production) — mechanically, from imposing a finite unconditional
  variance. But the low-frequency variation does not disappear — it
  **migrates into the output gap**, which becomes MORE persistent
  (lag-1 autocorr 0.945 vs 0.892 E1) and MORE drifty in the quiet
  decade (mean |gap| 2010–19: 0.336 vs 0.252 E1), ending 2025 at
  +0.91. I.e. the RW-ξ was not making the gap strange; forcing ξ
  stationary makes it stranger.
- **Fit**: mean LML +17.6 vs E1 — above the ~10-pt noise floor but
  inside the ~46–59-pt cross-seed spreads, sign flips by seed:
  inconclusive, no degradation.
- Verdict: does not help convergence, hurts gap properties, data
  rejects the stationarity restriction (rho_rg → 1). Keep the flag
  dormant unless a neutral-rate proxy arrives to discipline ξ some
  other way.

## Caveats

- Each comparison is one 3-seed triple per variant; R̂ itself is
  noisy. The E1-vs-armB improvement (2.84→2.20) is directionally
  consistent with the mechanism but has not been replicated on a
  second seed triple.
- LML cross-seed ranges remain 46–59 points in every variant —
  consistent with the connected-ridge geometry; single-seed LML
  comparisons remain meaningless here.

## PENDING OWNER DECISION

Production candidate = `MutationTransform` + `StructuredBlocks` +
`MStepsLadder` in `production.m`, then regenerate
`results/production/`. This moves the quotable table (γ2 from −0.157
to ≈ −0.25, several σ/κ params shift; the old values are partly
sticky-kernel artifacts per CP12). Nothing committed to git; E1/E2/E4
code sits in the working tree behind default-off flags.
