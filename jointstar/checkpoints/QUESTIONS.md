# Open questions

Resolved so far: the full model specification (incl. Rees 2019 priors,
adopted 2026-07-10, and the κ count corrected to 12 — matches
implementation); data transforms (wapop = lf/(part_rate/100),
400·Δlog CPI index, r = cash − pi_e).

## Resolved 2026-07-10 (afternoon)

1. **Baseline model outputs** — supplied (POST-COVID published estimates
   are the target; κ's compared on modes per the baseline's own notes) and
   the baseline's smoothed-state charts (output gap, NAIRU, real neutral
   rate). Wired into `jointstar.validate`.

2. **Survey columns** — NOT available (Consensus is proprietary; estimation
   is off-site). Checkpoint 7 will instead use the constructed `pi_e`
   column as a direct noisy measurement of the trend-inflation state:
   pi_e_t = π^e_t + ε, half-normal prior on its error sd (~30bp), start
   1985Q1. No trend-growth or neutral-rate proxy exists, so the g^z/ξ
   ridge keeps only the inflation-expectations anchor — expectations for
   how much the r* band can tighten should be set accordingly.

3. **ν ruling** — keep the IS channel; the gap↔ξ off-diagonal is excluded
   from the horseshoe (`horseshoePriors`, EXCLUDE list) so the structural
   coefficient is not absorbed by a reduced-form shock correlation.

## Data facts you may want to change

3. **IS term starts 1993Q3 because `cash_rate_pa` starts 1993Q1** in
   data.csv (132 obs; `pi_e` starts 1985Q1, but the real rate needs both).
   If you supply a cash-rate history back to the mid-1980s the IS term
   extends automatically — the availability mask is data-driven, no code
   change. Earlier than pi_e's 1985 start would need a decision on how to
   proxy expectations.

## New question from the priors

4. **Headline inflation equation.** [Resolved: trimmed-mean inflation only,
   per owner ruling — no headline CPI observable added.] The source priors
   originally listed a "Headline inflation" Phillips curve, but the model's
   signal equations and the data have only trimmed-mean inflation.

## Standing items

5. **Covariance structure** of the current model's innovations — still
   unknown; Checkpoint 4 baseline uses diagonal Q, horseshoe LDL′ arrives
   in Checkpoint 5 regardless.
6. **Priors not in the source table**: shock sds, break multipliers,
   π̄*/α_π, COVID φ's and κ's keep my documented defaults in
   `defaultPriors.m` — review welcome.
7. **Sample start 1974Q3** (first GDP obs) and drift persistence fixed at
   0.975/0.95 as specified.
