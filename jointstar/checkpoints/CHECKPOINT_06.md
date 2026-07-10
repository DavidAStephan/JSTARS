# Checkpoint 6 — Hierarchical COVID-κ shrinkage (PASSED on its own terms; one big finding)

**Run:** grouped horseshoe (gap↔ξ excluded per owner ruling) + hierarchical
κ priors — κ_v ~ Gamma(a_g, b_g)·1[κ≥1] with (a_g, b_g) shared within each
COVID time window and estimated in θ (log-mean/log-shape, Normal priors;
truncation normaliser included). N=2000, 419 parameters, 27 stages,
**55.9 min**. Outputs: `results/cp6/`, including
`validation_vs_baseline.csv` (jointstar.validate vs the baseline model's
published estimates).

**κ deliverable (vs baseline modes, per the baseline's own advice):** 10 of 12
κ's have overlapping 90% intervals. The hierarchy does what it was built
for — the implausible extremes shrink toward window means: κ^y_2020
10.7 → 5.1, κ^u_2020 16 → 3.2, κ^pr_2020 10 → 5.6, while well-identified
κ's barely move (κ^π 2.3 → 2.5, κ^c 1.7 → 1.8). Flag: κ^u_2020's intervals
are disjoint ([12,29] vs [2.0,4.6]) — the 2020 group (y, u, pr, k) may be
over-pooled; note also the baseline's own κ posteriors carry documented
convergence problems, so some gap is expected.

**The big finding — the covariance structure competes with the gap
dynamics.** With 19 of 111 off-diagonals identified, the posterior migrates
to a low-persistence gap: φ1+φ2 = 0.43 [0.34, 0.53] vs baseline 0.96
[0.94, 0.98], with the Okun loading sum doubling (−1.22 vs −0.52) and
ρ_U tripling (0.63 vs 0.20). Stage-by-stage checks show this is not
late-stage mode collapse — the hump-shaped-dynamics region loses mass from
φ ≈ 0.04 onward. ν also stays ≈ 0 despite the gap↔ξ exclusion: with a
weakly persistent gap there is little left for the r-gap channel to
explain. In short: correlated-shocks-with-weak-dynamics and
uncorrelated-shocks-with-strong-dynamics are competing representations,
and the horseshoe posterior prefers the former; the work model lives at
the latter. Overall baseline score: 14/23 CI overlaps, 13/23 within a
factor of 2 (κ's mostly pass; dynamics fail).

**Options for the owner (no change made):**
(a) exclude ALL gap-shock (c) rows from the horseshoe so the covariance
    modernisation cannot rewrite the cycle — preserves work-model dynamics,
    keeps correlations among trends/drifts/measurement errors;
(b) tighten the trend-block global scale (τ ~ C+(0, 0.1));
(c) accept the correlated-shock representation as a candidate re-spec.

**Next:** Checkpoint 7 with `pi_e` as a direct π^e measurement — decision
on (a)/(b)/(c) affects what CP7 inherits.
