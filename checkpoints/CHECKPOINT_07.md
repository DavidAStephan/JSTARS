# Checkpoint 7 — pi_e as a trend-inflation measurement (RUN COMPLETE; decision needed)

**Design.** No Consensus data is available off-site, so the survey block
reduces to one anchor: pi_e_t = π^e_t + ε (error-sd prior centred 30bp),
observed from 1985Q1. Two runs isolate its effect, both under the (a)
ruling (no gap-shock correlations in the horseshoe): `results/cp6a/`
without the anchor, `results/cp7/` with it. 37.6 + ~42 min wall-clock.

**Ruling (a) worked.** With the gap correlations excluded, the IS channel
is back: ν = −0.10 [−0.17, −0.02] (cp6a) — negative, excluding zero — and
gap persistence recovered (φ1+φ2 = 0.75 → 0.90 with the anchor, vs
baseline 0.96 and the 0.43 of Checkpoint 6). Baseline score improves to
13/23 CI overlaps; the remaining misses are concentrated in the gap-AR
shape (φ1 = 0.93 vs 1.65 — the data under this spec prefers persistence
without the hump) and ρ_U (0.83 vs 0.20).

**What the anchor buys (the payoff question):**
* σ_gz 0.052 → **0.039** and σ_ξ 0.158 → **0.135**, both bands tighter —
  the pile-up variances are finally data-disciplined, not prior-dominated.
* trend inflation glued to pi_e (measurement error posterior ~15bp);
  σ_πe falls 0.29 → 0.23; κ^π falls 3.3 → 2.2.
* ν attenuates to −0.06 [−0.11, −0.01], still excluding zero.

**What it does NOT buy: r\*.** End-of-sample r* median falls 0.74% →
−0.5% and the 90% band *widens* (6.4 → 9.8pp at end). With no
neutral-rate proxy, anchoring π^e re-attributes the Phillips/NAIRU block
but leaves ξ undisciplined — as warned when the Consensus data fell
through. The baseline model's published r* is ≈1%; cp6a is closer than cp7.

**SEVERE d_t/κ interplay — the brief's "watch for this" case fired.**
Adding the anchor flipped COVID level-shifter signs while κ's moved too:
φ_u −0.027 → **+0.019** (both bands exclude 0), φ_y ≈ 0 → **+0.065**
(excludes 0, wrong sign vs the model specification's φ_y < 0), while
κ^y_2020 rose 2.9 → 5.2. Per the brief, nothing was re-parameterised;
options for the owner:
  (i)  sign-restrict φ_y, φ_u < 0 (truncated priors), per the
       model specification's stated signs — cheapest, kills the flips;
  (ii) tighten κ priors in quarters where d_t is large (brief's own
       suggestion), so the level channel keeps the mean shift;
  (iii) drop the shifters that do no work anywhere (φ_pr, φ_k, φ_hpp
       hover near zero in all runs);
  (iv) fix the pi_e measurement error at 30bp instead of estimating it —
       the 15bp posterior may be over-trusting a constructed series, and
       the Phillips re-attribution is what destabilises the φ's.

**Recommendation:** (i) + (iv) together; then Checkpoint 8 (validation
report). Awaiting the owner's call.

---

## Addendum — final CP7 run after rulings (1)+(4) (`results/cp7b/`)

Sign restrictions (φ_y, φ_u < 0) plus the calibrated 30bp anchor error
resolved the interplay and improved everything at once (47.6 min, 28 stages):

* **Baseline agreement: 17/23 CI overlaps** (from 13/23).
* **r\* ends at 0.97% [−1.9, 3.6]** — on the baseline's ~1% — and the mean
  band width narrows to 4.3pp (vs 5.4 without the anchor): with the φ
  channel sign-identified, the anchor now disciplines rather than distorts.
* σ_ξ = 0.093 (from 0.158 pre-anchor), ν = −0.075 [−0.106, −0.040],
  γ2 = −0.16 (moving toward the baseline's −0.09), φ_y = −0.037,
  φ_u = −0.010, κ_y2020 = 9.1 (baseline mode 10.7).

This configuration is the Checkpoint 8 input.
