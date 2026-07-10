# Checkpoint 3 — SMC recovers a known bimodal posterior (PASSED)

**Validated.** `jointstar.runSMC` (adaptive tempering via ESS-targeted
bisection, systematic resampling, adaptive random-walk MH mutation with
(2.38²/d)·Σ̂ proposals) was run on a 2-D two-Gaussian-mixture target with
asymmetric weights (0.7/0.3) and well-separated modes, under a wide Gaussian
prior, so the exact posterior mass split (0.776) is available in closed form.
With N=2000 particles and 5 MH steps the sampler populated both modes
(1373/627 particles) and recovered the split as **0.786 vs 0.776 true** —
inside the 0.05 Monte Carlo tolerance — with within-mode means matching the
analytic product-of-Gaussians means to <0.15. Runs are bit-reproducible given
a seed (proposal noise is pre-generated, so parfor scheduling cannot change
results). A full pipeline test (SMC over the precision likelihood on a
local-level model) recovers known variance parameters within posterior
uncertainty.

**Surprise / fix worth knowing:** with adaptive tempering the post-reweight
ESS lands *exactly on* the target, so a strict `ESS < N/2` resampling rule
never fires and the schedule stalls — the threshold must be inclusive. Also,
tempering finished in only 3 stages on this toy; expect 15–30 on the real
model.

**Next:** Checkpoint 4 (SMC × Piece 1 on full JointSTAR) — blocked on the
full model specification; see `QUESTIONS.md`.
