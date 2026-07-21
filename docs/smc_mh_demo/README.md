# MH vs SMC — a runnable demo

A small, self-contained MATLAB demo that shows *how* Metropolis–Hastings (MH)
and tempered Sequential Monte Carlo (SMC) explore a posterior differently — the
companion example to the JointSTAR sampler.

Run it:

```matlab
>> smc_vs_mh_demo
```

Base MATLAB only, no toolboxes. Tested on R2026a. Takes ~15 seconds.

## The model (tiny, but with interesting geometry)

We observe `n = 25` noisy measurements of the **product** of two parameters:

```
y_i = theta1 * theta2 + noise ,   noise ~ N(0, 1.2^2)
theta1, theta2 ~ N(0, 3^2)         (independent priors)
```

Only the *product* is well identified, so the posterior sits on a curved
**ridge** (`theta1 * theta2 ≈ 3`). And because `(theta1, theta2)` and
`(-theta1, -theta2)` give the same product, there are **two** symmetric ridges
(modes) separated by a low-density barrier near the axes.

This is a deliberately compact stand-in for the weak-identification / ridge
geometry that shows up in real macro "stars" models — the exact thing that makes
single-chain MCMC mix slowly.

## What the demo shows

**MH** sends **one walker** across the posterior. Started in one arm, it accepts
~37% of its steps and crawls *along* the curved ridge — but it essentially never
crosses the barrier to the other arm. After 20,000 iterations it reports only
**one** of the two equally-valid answers (100% of its samples in a single mode).

**SMC** starts a **cloud of 1,200 particles** at the broad, symmetric prior and
turns the data's influence up gradually (`phi: 0 → 1`, here in 13 adaptive
stages). At high temperature the barrier is invisible, so the cloud spans both
arms; as it cools it settles onto **both** ridges at once — the honest,
symmetric posterior (a clean ~50/50 split).

Both recover the product `theta1*theta2 ≈ 3.1` (true value 3). Same model, same
data, same target posterior — the difference is entirely in *how* they explore.

## Outputs (written next to the script)

- **`smc_vs_mh_animation.gif`** — side-by-side animation: the MH chain filling one
  arm while the SMC cloud anneals from the prior onto both. Drop it into slides.
- **`smc_vs_mh_summary.png`** — the four-panel still: the posterior, MH's one arm,
  SMC's two arms, and the giveaway `theta1` marginal (MH one-sided, SMC symmetric).

## The SMC recipe inside `run_smc`

It is the JointSTAR sampler in miniature, so the script doubles as readable
pseudocode for the real thing:

1. draw particles from the prior (exact — no burn-in),
2. choose the next temperature `phi` by bisection so the reweighted effective
   sample size stays at half the particles,
3. systematic-resample (clone the fit, drop the unfit),
4. refresh each particle with a few cloud-scaled random-walk MH moves at the
   current `phi`,
5. repeat until `phi = 1`.

Note that MH lives *inside* SMC as step 4 — demoted from the whole algorithm to a
short local refresh, because tempering + resampling now do the global work.

## Things to try

- Start the MH chain in the other arm (`th0 = [-2.5; -2.5]`) — it gets stuck
  there instead. Neither start is "wrong"; that's the point.
- Shrink the MH step (`propSdMH = 0.08`) — higher acceptance, but even slower
  crawl along the ridge.
- Widen the noise (`sigma = 2.5`) — the two arms merge; both methods agree again.
- Drop the SMC ESS target (`essFrac = 0.5`) — fewer, coarser tempering stages;
  watch whether the cloud still finds both arms.
