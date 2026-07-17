# Methodology note: model, estimator, convergence, and precedents

*Current as of CHECKPOINT_15 (2026-07-16). The production estimator is a
standard adaptive likelihood-tempering SMC sampler with a states-
marginalized (Chan–Jeliazkov) linear-Gaussian likelihood, a diagonal
innovation covariance, an unconstrained-coordinate block Metropolis
mutation, a **waste-free** mutation scheme (Dau–Chopin 2022), and a
parsimonious hierarchical treatment of the COVID variance-scaling
parameters. Every ingredient is standard and published; there is **no
novel statistical methodology**. This note gives the description, the
math, and the references for each component, and states the convergence
standard the project reports against.*

The single production entry point is `jointstar.production('data.csv')`,
which runs the configuration below from three seeds and pools them.

---

## 1. The method in one paragraph

Write the model as a linear Gaussian unobserved-components state-space
system for Australian quarterly macro data (r\*, NAIRU, potential
output, trend growth, output gap, trend inflation). For a fixed
parameter vector θ the latent state path is integrated out **exactly**
in one sparse-precision Gaussian computation (Chan–Jeliazkov 2009),
returning `p(y|θ)` — numerically identical to a Kalman filter (verified
to 1e-8). Posterior inference on θ uses adaptive likelihood-tempering
Sequential Monte Carlo (Herbst–Schorfheide 2014): a cloud of θ-particles
is carried through a sequence of bridging densities
`π_n(θ) ∝ p(θ) p(y|θ)^{φ_n}` from the prior (φ=0) to the posterior
(φ=1), with adaptive ESS-targeted tempering, systematic resampling, and
block random-walk Metropolis mutation. Three tuning refinements improve
mixing on this model's ridge-shaped posterior — mutation in
**unconstrained coordinates** (with the exact Jacobian), **structured
parameter blocks**, and a late-stage **mutation-step ladder** — together
with **waste-free** mutation (Dau–Chopin 2022), which retains every
intermediate Metropolis state as a particle instead of discarding it, at
equal likelihood-evaluation budget. The only model-specific modelling
choices are a **parsimonious hierarchical truncated-Gamma** treatment of
the COVID variance-scaling parameters and the use of a constructed
inflation-expectations series as a direct measurement of the
trend-inflation state; neither is a new statistical method. Convergence
is reported the way the SMC literature reports it — effective sample
size across independent runs and a rank-normalized R̂ — not against an
MCMC-style "R̂ = 1" target that no published SMC estimator uses.

---

## 2. The estimated model — exact specification

*This section transcribes the model exactly as built in
`+jointstar/ModelSpec.m` (`jointstar` static constructor) and
`+jointstar/loadData.m`, so the functional form can be checked against
the source model it replicates. The generic container is a linear
Gaussian state-space system*

```
α_t = c_t + A1_t α_{t−1} + A2_t α_{t−2} + η_t,   η_t ~ N(0, Q_t),   α_1 ~ N(a1, P1)
y_t = d_t + Z_t α_t + Z^L_t α_{t−1} + ε_t,        ε_t ~ N(0, R_t)
```

*with missing observations masked (never imputed). Below, `α` and `α_t`
components are named; `a = alpha` is the capital share; a "drift" is the
growth rate of a trend.*

### 2.1 State vector (m = 14)

`α = ( c, U*, pie, z*, g^z, k*, g^k, w*, g^w, hpp*, g^{hpp}, pr*, g^{pr}, ξ )`

- `c` — output gap (cyclical component)
- `U*` — NAIRU (trend unemployment)
- `pie` — trend inflation (the code's state name; also measured directly
  by the π^e series). Distinct from the *observed* trimmed-mean inflation
  `π` (measurement row 2).
- `z*, k*, w*, hpp*, pr*` — trend levels of (labour-augmenting)
  productivity, capital, working-age population, hours per person,
  participation; each with its drift `g^z, g^k, g^w, g^{hpp}, g^{pr}`
- `ξ` — the non-growth component of the neutral real rate

Two derived combinations are used but are **not** separate states:
potential output `τ*_t = z*_t + a·k*_t + (1−a)(w*_t + pr*_t + hpp*_t − U*_t)`
and the neutral real rate `r*_t = [4/(1−a)]·g^z_t + ξ_t`.

Constants (from θ): `c_z = 0.025·ḡ_z`, `c_w = 0.025·ḡ_w`,
`c_k = 0.025·(ḡ_z+ḡ_w)/(1−a)`; gap AR(2) is parameterised as
`(φ_sum, φ_2)` with `φ_1 = φ_sum − φ_2` (Rees 2019).

### 2.2 State (transition) equations — exactly as coded

**Output gap** — AR(2) plus an IS (real-rate-gap) term:
```
c_t = φ_1 c_{t−1} + φ_2 c_{t−2} + (ν/2) Σ_{j=1,2} ( r_{t−j} − r*_{t−j} ) + η^c_t
```
with `r*_{t−j} = [4/(1−a)]·g^z_{t−j} + ξ_{t−j}` and `r_{t−j}` the observed
real cash rate. The IS term is **active only in quarters where both
`r_{t−1}` and `r_{t−2}` are observed** (the real cash rate starts 1993Q1);
where they are not, it is dropped, not imputed.

**NAIRU** — random walk:  `U*_t = U*_{t−1} + η^{U*}_t`

**Trend inflation** (`pie` state) — regime-switching at 1993Q1:
```
pie_t = pie_{t−1} + η^{pie}_t                                 (before 1993Q1, RW)
pie_t = α_π · π̄ + (1 − α_π) pie_{t−1} + η^{pie}_t             (from 1993Q1, AR(1) to π̄ = pistar)
```

**Productivity / capital / population trends** — unit-root trend with a
persistent, contemporaneous drift (persistence `ρ_d = 0.975`):
```
g^z_t = c_z + 0.975 g^z_{t−1} + η^{g^z}_t ;   z*_t = z*_{t−1} + g^z_t + η^{z}_t
g^k_t = c_k + 0.975 g^k_{t−1} + η^{g^k}_t ;   k*_t = k*_{t−1} + g^k_t + η^{k}_t
g^w_t = c_w + 0.975 g^w_{t−1} + η^{g^w}_t ;   w*_t = w*_{t−1} + g^w_t + η^{w}_t
```
**Hours and participation trends** — same form with `ρ_d = 0.95` and
**zero drift intercept**:
```
g^{hpp}_t = 0.95 g^{hpp}_{t−1} + η^{g^{hpp}}_t ;   hpp*_t = hpp*_{t−1} + g^{hpp}_t + η^{hpp}_t
g^{pr}_t  = 0.95 g^{pr}_{t−1}  + η^{g^{pr}}_t  ;   pr*_t  = pr*_{t−1}  + g^{pr}_t  + η^{pr}_t
```
"Contemporaneous drift" (`trend_t = trend_{t−1} + g_t + η`) is implemented
exactly in the matrices by the lagged coefficient `A1(trend, g) = ρ_d`
together with the innovation map `M` (§2.4) injecting the drift shock
`η^{g}` into the trend row — the two combine to `trend_t = trend_{t−1} +
g_t + η^{trend}`.

**Neutral-rate residual** — random walk:  `ξ_t = ξ_{t−1} + η^{ξ}_t`

### 2.3 Signal (measurement) equations — exactly as coded

Observables and units (`+jointstar/loadData.m`): `y` = 100·log real
non-farm GDP; `π` = trimmed-mean inflation (400·Δlog index, quarterly
annualised); `wapop` = 100·log working-age population; `U` = unemployment
rate (pp); `lpr` = 100·log(participation/100); `hpp` = 100·log average
hours; `k` = 100·log real capital; `pie_obs` = π^e (optional 8th row).
`D` is the COVID government-stringency index (0 outside the pandemic),
`D^L` its lag; lagged **observed** series (`π_{t−1}`, `U_{t−1}`, `w_{t−1}`,
…) enter the intercepts as exogenous data.

**(1) GDP** — potential + gap + COVID shifter:
```
y_t = τ*_t + c_t + φ_y D_t + ε^y_t ,   τ*_t = z*_t + a k*_t + (1−a)(w*_t + pr*_t + hpp*_t − U*_t)
```

**(2) Inflation** — expectations-augmented Phillips curve (observed
inflation `π`; trend-inflation state `pie`):
```
π_t = γ_1 pie_t + (1 − γ_1) π_{t−1} + γ_2 (U_t − U*_t) + γ_2 φ_u D_t + ε^π_t
```
(`γ_1` = weight on the trend-inflation state vs. lagged observed
inflation; `γ_2 ≤ 0` the Phillips slope on the unemployment gap; the last
term a COVID stringency shifter.)

**(3) Working-age population** — AR(1) in its own gap, no cycle loading:
```
wapop_t = w*_t + ρ_w ( wapop_{t−1} − w*_{t−1} ) + ε^w_t
```

**(4)–(7) Unemployment, participation, hours, capital** — each its trend
+ an AR(1) gap (quasi-differenced via the observed lag) + a loading on
the common output gap `c` (Okun-type) + a COVID shifter:
```
(4) U_t   = U*_t   + ρ_U  (U_{t−1}   − U*_{t−1})   + ξ_1 c_t + ξ_2 c_{t−1} − φ_u D_t   + ρ_U  φ_u  D_{t−1}  + ε^U_t
(5) lpr_t = pr*_t  + ρ_pr (lpr_{t−1} − pr*_{t−1})  + θ_1 c_t + θ_2 c_{t−1} + φ_pr D_t  − ρ_pr φ_pr D_{t−1}  + ε^{lpr}_t
(6) hpp_t = hpp*_t + ρ_hpp(hpp_{t−1} − hpp*_{t−1}) + λ_1 c_t + λ_2 c_{t−1} + φ_hpp D_t − ρ_hpp φ_hpp D_{t−1} + ε^{hpp}_t
(7) k_t   = k*_t   + ρ_k  (k_{t−1}   − k*_{t−1})   + χ_1 c_t + χ_2 c_{t−1} + φ_k D_t   − ρ_k  φ_k  D_{t−1}  + ε^k_t
```

**(8) Inflation expectations** (optional, from 1985Q1) — direct noisy
measurement of the trend-inflation state:
```
pie_obs_t = pie_t + ε^{pie}_t ,   sd(ε^{pie}) = sme_pieobs = 0.30 (fixed)
```

The stringency loadings `φ_y, φ_u < 0` are sign-restricted (owner ruling).
The `d_t = 0` outside 2020–23 collapses the COVID-block rows to their
pre-pandemic form (except the AR(1) intercepts, which always carry the
observed lag).

### 2.4 Error terms and covariance — the careful bit

**State innovations `η_t ~ N(0, Q_t)`.** The underlying structural shocks
are **mutually independent** with standard deviations
`sig_c, sig_Ustar, sig_pie, sig_z, sig_gz, sig_k, sig_gk, sig_w, sig_gw,
sig_hpp, sig_ghpp, sig_pr, sig_gpr, sig_xi` (14, one per state). The
innovation covariance is assembled as

```
Q_t = M · ( Σ_0 ⊙ (k_t k_t') ) · M' ,   Σ_0 = diag(sig²)
```

- **`Σ_0` is diagonal** — the innovation covariance carries **no free
  cross-shock correlations** (the grouped-horseshoe off-diagonal layer
  explored in CP5–8 was dropped, §6). ν and the Okun/loadings are
  identified structurally, not through a shock covariance.
- **`M` (contemporaneous drift) is the only source of off-diagonal `Q`.**
  It adds each drift shock into its trend row, so every trend/drift pair
  has an exact **2×2 block**
  ```
  Var(η^{trend}) = sig_trend² + sig_drift²,  Cov(η^{trend}, η^{drift}) = sig_drift²,  Var(η^{drift}) = sig_drift².
  ```
  These within-pair covariances are a **structural consequence of the
  timing** (drift enters the trend contemporaneously), not estimated
  parameters. `Q_t` is therefore block-diagonal: scalar variances for
  `c, U*, π, ξ`, and five 2×2 trend/drift blocks. Everything else is zero.
- **`k_t` (variance scaling)** is 1 everywhere except: pre-1984 gap
  (`×m84_c`) and `z*` (`×m84_z`); pre-1993 NAIRU (`×m93_U`); and the
  2020–21 gap innovation (`×kapc_2021`, COVID). Scaling is multiplicative
  on the shock sd (`Σ_0 ⊙ k k'`), preserving the block structure.

**Measurement errors `ε_t ~ N(0, R_t)`.** In production `R_t` is
**strictly diagonal** — **no cross-equation measurement-error
correlation**:

```
R_t = diag( sme² ⊙ (k^R_t)² )
```

with per-equation sds `sme_y, sme_pi, sme_w, sme_U, sme_pr, sme_hpp,
sme_k` (and `sme_pieobs = 0.30` fixed for the 8th row). The multipliers
`k^R_t = 1` except for the volatility breaks (pre-1984 GDP `×m84_y`,
pre-1993 inflation `×m93_pi`) and the COVID `κ ≥ 1` windows scaling the
relevant measurement variances: GDP `×kapy_20 / kapy_21`, U `×kapu_20 /
kapu_2122`, participation `×kappr_20 / kappr_2122`, capital `×kapk_20 /
kapk_21`, inflation `×kappi_2023`, wapop `×kappop_2021`, hpp
`×kaphpp_2022`. The COVID κ's are given the parsimonious hierarchical
prior of §5; the π^e row takes no COVID or break multiplier.

**A full non-diagonal measurement covariance `R_full`** is supported by
the code (the CP5–8 horseshoe path supplied a Cholesky factor `L_r`) but
is **not** used in production — production runs the fast diagonal path.

**Initial state.** `a1` anchors the trend levels to first-observation
data (`z*` backed out from the GDP identity) with small drift means;
`P1 = diag([25,25,25,100,0.25,100,0.25,100,0.25,100,0.25,100,0.25,4])`
— wide on levels, tight on drifts (a near-diffuse initialisation).

### 2.5 Correspondence to the source model, and documented deviations

The constructor maps to the source transcription's numbered equations
(GDP/Phillips + the labour-market/capital block are eqs 1–7 / 28–34, the
IS term is eq 8, wapop's AR(1) is eq 34, the U/lpr/hpp/k cyclical
measurement are eqs 30–33). Where the source is ambiguous, the following
choices were made (all recorded in the `ModelSpec` docstring and
`checkpoints/QUESTIONS.md`) — this is the checklist for a functional-form
comparison against the original:

- a **single** IS coefficient ν (eq 8 as written);
- drift persistence fixed at **0.975** (z, k, w) / **0.95** (hpp, pr);
- `lpr*` ≡ the participation trend `pr*` throughout;
- the COVID-block measurement equations are applied over the whole
  sample (with `d_t = 0` outside 2020–23 they reduce to eqs 1–7, except
  wapop's AR(1));
- the IS term is **dropped, not imputed**, when `r_{t−1}` or `r_{t−2}`
  is unavailable;
- trend/drift pairs use **contemporaneous drift**, giving the exact 2×2
  SPD blocks in `Q` (§2.4);
- **trimmed-mean inflation only** — no separate headline-CPI Phillips
  equation, although one appears in the source Table 1 (owner ruling);
- the gap AR(2) is reparameterised as `(φ_sum, φ_2)` with
  `φ_1 = φ_sum − φ_2` (Rees 2019 prior parameterisation).

Everything else is a textbook multivariate UC model in the
Holston–Laubach–Williams / Chan–Koop–Potter tradition.

---

## 3. The likelihood engine — Chan–Jeliazkov precision sampling

For a linear Gaussian SSM the integrated likelihood

```
p(y | θ) = ∫ p(y | α, θ) p(α | θ) dα
```

is a single Gaussian integral over the stacked state path
`α = (α_1', …, α_T')'`. The transition equation implies a Gaussian prior
`α ~ N(a, K⁻¹)` whose precision `K` is **block-banded sparse** (only
adjacent-in-time blocks are nonzero). Adding the measurement information
gives the conditional precision

```
P = K + Z' H⁻¹ Z          (also sparse, banded)
```

and the log integrated likelihood follows from a sparse Cholesky
`P = L L'`:

```
log p(y|θ) = −½ [ c + log|H| + log|K⁻¹ … | − (terms) + e' Ω e ]
```

evaluated in `O(T)` via banded factorization rather than the dense
`O(T³)` cost. For a linear Gaussian model this returns the numerically
**identical** value a Kalman filter would; the precision route is an
efficiency choice, not a statistical one.

- **Verification:** `tests/testPrecisionVsKalman.m` (AR(1)+noise over a
  parameter grid, nonzero initial mean, scattered and contiguous
  missing data; log-likelihood equal to 1e-8; 10⁴ state draws match the
  smoother moments). The AR(2) gap block is validated indirectly via the
  companion-form embedding test `testZlagVsCompanion` (the independent
  Kalman oracle is first-order only).
- **Efficiency layer:** a run-level static cache (`buildEvalCache`) of
  the θ-independent structure; **bitwise-identical** to the uncached
  path (`benchmarks/verifyCacheEquivalence.m`), ~1.7× faster per
  likelihood.

**References:** Chan & Jeliazkov (2009); Chan, Koop & Potter (2016);
Grant & Chan (2017); McCausland, Miller & Pelletier (2011); Mertens
(2023).

---

## 4. The sampler — adaptive tempering SMC

### 4.1 Tempering path and adaptive schedule

The sampler targets a sequence of bridging densities

```
π_n(θ) ∝ p(θ) · p(y|θ)^{φ_n},     0 = φ_0 < φ_1 < … < φ_{N_φ} = 1.
```

A cloud of `N = 2000` particles `{θ^(i)}` is carried from the prior
(φ=0) to the posterior (φ=1). At each stage `n`:

1. **Reweight** by the incremental likelihood power
   `w̃_n^(i) ∝ p(y|θ^(i))^{φ_n − φ_{n−1}}`.
2. **Choose φ_n adaptively** by 1-D bisection so the post-reweighting
   effective sample size hits a target fraction of `N`:
   ```
   ESS(φ_n) = (Σ_i w_n^(i))² / Σ_i (w_n^(i))²  =  γ · N,   γ = 0.5.
   ```
3. **Resample** (systematic) whenever `ESS ≤ γN`.
4. **Mutate**: `M_stage` sweeps of block random-walk Metropolis with
   invariant density `π_n` (§4.2–4.4).

### 4.2 Mutation in unconstrained coordinates (`MutationTransform`)

Rather than propose on the constrained scale, each mutated coordinate is
mapped to an unconstrained `η = T(θ)` by an elementwise, bounds-driven
bijection (`+jointstar/paramTransform.m`):

```
both bounds finite (lo,hi)   →  η = logit((θ−lo)/(hi−lo))
lo finite, hi = +∞           →  η = log(θ − lo)
hi finite, lo = −∞           →  η = log(hi − θ)
```

so σ's map by `log`, the COVID κ's (support `[1,∞)`) by `log(θ−1)`,
`beta`/`negbeta`/bounded-`tnorm` parameters by an affine logit, and the
sign-restricted stringency loadings (`φ_y, φ_u < 0`) by `log(−θ)`. The
Metropolis proposal is a Gaussian random walk in η, and the acceptance
ratio carries the exact Jacobian of the inverse map, so the target on
the η-scale is `π_n(θ(η))·|dθ/dη|` and

```
a = min{ 1,  [π_n(θ') |J(η')|] / [π_n(θ) |J(η)|] },   |J(η)| = |dθ/dη|.
```

This removes the boundary "stickiness" of raw-scale random walk against
hard constraints (σ→0, κ→1, ρ→1, φ→0). It is a coordinate change that
leaves the posterior invariant — the standard reparameterization
device of Papaspiliopoulos–Roberts–Sköld (2007). The transform is
verified prior-invariant per-type and jointly
(`benchmarks/verifyTransformInvariance*.m`).

### 4.3 Structured blocks and the mutation-step ladder

- **`StructuredBlocks`**: instead of purely random partitions, known
  ridge-coupled parameters (the gap-AR pair, the ρ_U/Okun set) are
  co-blocked so a single Metropolis move can travel *along* a ridge.
  Tailored/structured blocking is standard SMC/MCMC practice
  (Chib–Ramamurthy 2010; Herbst–Schorfheide use `N_blocks = 3–6`).
- **`MStepsLadder`**: the number of mutation sweeps rises where mixing
  is hardest, using the tempering exponent already reached:
  ```
  M_stage = MSteps           for φ < 0.70
          = 2·MSteps         for 0.70 ≤ φ < 0.95
          = 3·MSteps         for φ ≥ 0.95.
  ```
  (Production base `MSteps = 2`.) Spending more mutation effort at high
  φ, where the target is sharpest, is a pure tuning choice.

### 4.4 Waste-free mutation (Dau–Chopin 2022)

Classical SMC resamples `N` particles, runs each through `M_stage`
Metropolis steps, and keeps only the **endpoint** — discarding the
intermediate states (and the likelihood evaluations that produced them).
Waste-free SMC instead resamples a smaller number of **ancestor chains**
`M = ⌊WF_MFRAC · N⌋` (`WF_MFRAC = 0.25`, so `M = 500`), runs each for
`P−1` steps, and **retains all `M·P` visited states** as the next
particle system (equal weights). The chain length is set for
**exact likelihood-evaluation-budget parity** with the classical scheme:

```
M · (P − 1)  =  N · M_stage      ⟹      P = N·M_stage / M + 1.
```

Example (`N=2000, M_stage=2`): `M=500`, `P=9`, so `500·8 = 4000 = 2000·2`
mutation sweeps, producing a `4500`-particle cloud. The next tempering
increment importance-reweights all `M·P` particles, and the tempering
marginal-likelihood estimator (§4.5) is unchanged and remains consistent
under this weighting (Dau–Chopin, Prop. 1–2). Retaining the otherwise-
discarded states lowers estimator variance at no extra likelihood cost,
which is what closes most of the residual cross-seed disagreement (§7).
The flag-off path is bitwise-identical to classical mutation
(`tests/testWasteFree.m`). Downstream logic handles the variable cloud
size (`|cloud| ≠ N`).

### 4.5 Marginal-likelihood identity

The tempering run yields the standard normalizing-constant estimator

```
log p(y)  =  Σ_n  log ( (1/N) Σ_i  w̃_n^(i) ),
```

recorded per stage as `lml_inc` and cumulatively as `out.lml`. **This is
an internal diagnostic, not a calibrated marginal likelihood** — it
omits prior truncation/stationarity normalizer constants and carries a
finite-N tempering bias; single-run values are not comparable across
specifications (cross-seed spread ~46–59 log points on this model).

**References:** Herbst & Schorfheide (2014, 2015); Cai et al. (2021, the
NY Fed `SMC.jl` companion paper); Dau & Chopin (2022);
Papaspiliopoulos, Roberts & Sköld (2007); Chib & Ramamurthy (2010).

---

## 5. COVID variance model — hierarchical truncated-Gamma κ

The pandemic quarters are handled, as is standard, by inflating the
relevant innovation/measurement variances by regime scale factors
`κ ≥ 1` over defined windows (the Lenza–Primiceri 2022 "scale up the
disturbance variances" device). Within a time-window group `g` the
factors are drawn from a shared truncated Gamma:

```
κ_v  ~  Gamma(a_g, b_g) · 1[κ_v ≥ 1],
```

with `(a_g, b_g)` shared across the variables in group `g` and carried
in θ through Normal priors on their log-mean and log-shape (the
truncation normalizer, which depends on the hyperparameters via the
incomplete Gamma function, is included in the density). This shrinks the
within-window κ's toward a common value.

**Parsimony refinement (`PoolAcuteOnly`, production default, CP15).**
The pooling only earns its keep where several κ's genuinely share a
scale. Diagnostics (conditional log-likelihood profiles and
prior→posterior contraction ratios, `benchmarks/{profileConditional,
contractionRatios}.m`) showed the hyperparameters of the small/single-
member groups have **zero likelihood curvature** — they enter only the
prior — and so are structurally unidentified. The production model
therefore keeps the hierarchy **only for the 4-member 2020 acute group**
`{y, u, pr, k}` and replaces every sparser group's hyperparameters with
a fixed, a-priori-calibrated `Gamma(2.0, 1.25)·1[κ≥1]` prior on the κ
directly (calibrated to the incumbent hyperprior-implied mean/spread,
**not** to any posterior — no empirical-Bayes circularity). This drops
10 unidentified hyperparameters (79→**69** parameters) with the
structural economics, latent states, and COVID gap trough unchanged
within cross-seed noise, and slightly better convergence (§7). Fixing an
unidentified hyperparameter to a calibrated value is the same trade
already made for the π^e measurement-error sd (fixed at 0.30).

Alternative COVID parameterizations are implemented behind default-off
flags for future evaluation but are **not** in the production model: a
Lenza–Primiceri-style geometric-decay scale (`CovidDecay`) and a
stringency-loading restriction test (`DropPhiY`); see CHECKPOINT_15.

**References:** Lenza & Primiceri (2022); Carriero, Clark, Marcellino &
Mertens (2022).

---

## 6. Diagonal innovation covariance (and the horseshoe, dropped)

`Q_t` is **diagonal** — the default for essentially every UC/trend-cycle
model (HLW, the Chan UC papers, the original in-house JointSTAR) and the
most standard choice possible.

Between Checkpoints 5–8 the project explored a grouped-**horseshoe**
shrinkage prior on the innovation-covariance Cholesky off-diagonals as a
discovery tool for "which cross-shock correlations does the data
identify." This was the one component with **no published precedent**
(horseshoe-via-SMC), and the methodology review (CP10) found it also
carried a kernel-invariance bug. Empirically it identified only ~17 of
106 off-diagonals, concentrated in the trend/drift blocks, and — with the
gap-shock correlations excluded by owner ruling — did **not** move the
headline latent states or policy-relevant parameters versus the diagonal
model. It was dropped (CP11). The scientific content of that exploration
— "the data identify few cross-shock correlations and none that move the
headline" — is a positive result that *justifies* the diagonal choice
with evidence rather than assumption.

---

## 7. Convergence — reported the SMC way, not the MCMC way

**The standard.** No published SMC estimator reports, or meets, an
MCMC-style cross-run `R̂ ≈ 1` target; the founding SMC-for-macro
reference (Herbst–Schorfheide) contains no Gelman–Rubin statistic. The
field standard (Herbst–Schorfheide 2014 §4.3, after Durham–Geweke) is to
run the sampler `G` times independently and report, per posterior
functional `h`, an **effective sample size**

```
N_eff(h)  =  Var_π[h] / Var_{across G runs}[ ĥ_g ],
```

estimated per parameter as `N_eff = V_post / std_g(mean_g)²`, with
`V_post` from the pooled cloud and `std_g` the between-run standard
deviation of the per-run posterior means. This project reports `N_eff`
over `G = 20` independent seeds (`benchmarks/computeNeff.m`).

**Secondary smell-test.** As a robust cross-seed agreement statistic we
also compute the **rank-normalized, folded** split-less `R̂` of Vehtari
et al. (2021): equal-weight-resample each seed's final cloud; take
fractional ranks `r` of the pooled draws and normalize
`z = Φ⁻¹((r − 3/8)/(S + 1/4))`; compute

```
R̂ = sqrt( [ (n−1)/n · W  +  B/n ] / W )
```

on `z` (**bulk**) and on the folded values `|θ − median|` (**tail**),
and report `max(bulk, folded)`. The usual split-half step is deliberately
**omitted**: an SMC final cloud is exchangeable, not time-ordered, so
splitting it in half detects nothing — an intentional, documented
deviation (`computeNeff.m` header). Rank-normalization is what makes the
statistic trustworthy on the right-skewed variance parameters that
dominate this model's disagreement.

**What the numbers are.** On the raw kernel this model's structural
parameters were badly seed-unstable (`max R̂ ≈ 5.8`) because the
posterior lives on long, connected likelihood ridges (the gap-AR
persistence split, the ρ_U/Okun trade-off, the r\* band) — diagnosed
(CP12) as **one connected ridge, not multimodality** (bridging
log-posterior profiles show no density valley between seed clouds). The
tuning refinements close most of it: the transformed kernel + structured
blocks + ladder took `max(bulk,folded) R̂` from ~5.8 to ~1.55 at G=20,
and **waste-free mutation** took it to **≈1.19–1.24 at G=20** (median
`N_eff ≈ 8`) — the residual disagreement is now at the level any
published SMC estimator carries. Much of the historically-quoted
"`R̂ = 2.2`" was small-sample noise in a 3-seed maximum; at G=20 with
rank-normalization the honest figure is far lower.

**Reporting rule.** Anything quoted or compared across specifications is
taken from the ≥3-seed **pool** — an equal-weight mixture of the seeds'
final clouds, an inter-seed uncertainty envelope that widens intervals
where seeds disagree. It is preferable to any single seed but is **not** a
converged posterior: pooling averages seed noise, it does not remove
finite-N bias. Latent states (r\*, NAIRU, gap, trend inflation) are far
more seed-stable than the structural parameters.

**References:** Herbst & Schorfheide (2014, §4.3); Durham & Geweke
(2014); Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021);
Papaspiliopoulos, Roberts & Sköld (2007).

---

## 8. Precedent map

**The sampler applied to exactly this model class.** Herbst &
Schorfheide (2014) §5.1 — their *first* empirical illustration — is a
plain linear Gaussian state-space model (not a DSGE) run through the
identical tempering algorithm with a states-marginalized likelihood.
"Tempering SMC wrapped around a linear-Gaussian `p(y|θ)`" is page one of
the source paper.

**The sampler in central-bank production code.** NY Fed `SMC.jl`/
`DSGE.jl` (BSD-3): adaptive ESS-targeted tempering (Cai et al. 2021),
systematic resampling at `0.5·N`, blocked random-partition RW-MH with
cloud-covariance proposals, 0.25 acceptance targeting, and the same
log-marginal-likelihood identity. Dynare ≥ 6.0 ships the HS sampler as
`hssmc`.

**Waste-free mutation** is Dau & Chopin (2022, *JRSS-B*), a published,
general-purpose SMC improvement — a drop-in change to the mutation/
resample loop, not a bespoke device.

**The likelihood evaluator in exactly this model class.** Chan &
Jeliazkov (2009) (the method); Chan, Koop & Potter (2016) and Grant &
Chan (2017) (UC/NAIRU/trend-cycle by banded precision); **Zaman (2022,
Cleveland Fed)** — a large multivariate UC "stars" model (r\*, u\*, g\*,
π\*) estimated at a Fed bank using exactly these Chan–Jeliazkov routines,
the closest production-scale precedent; Mertens (2023).

**The COVID scale factors** follow Lenza–Primiceri (2022); the
convergence reporting follows Herbst–Schorfheide/Durham–Geweke and
Vehtari et al. (2021). Every ingredient is standard and published.

---

## 9. Reference-implementation cross-check

| Design choice | Ours (`runSMC.m`) | Herbst–Schorfheide 2014/15 | NY Fed SMC.jl (Cai et al. 2021) | Dynare `hssmc` |
|---|---|---|---|---|
| Tempering schedule | adaptive: bisection so post-reweight ESS = 0.5·N | fixed φ_n=(n/N_φ)^λ | both; adaptive targets ESS ratio | fixed (n/N_φ)^λ |
| Resampling | systematic, trigger ESS ≤ 0.5·N | multinomial baseline | systematic default, threshold 0.5 | — |
| Mutation | block RW-MH, structured+random partitions, **unconstrained coords + Jacobian**, cloud covariance | blocked RW-MH, random partitions, cloud covariance | blocked RW-MH, cloud covariance | RW-MH |
| Waste-free | **yes (Dau–Chopin 2022), budget-parity** | no | no | no |
| Late-stage effort | MSteps ladder (×1/×2/×3 by φ) | fixed M | fixed M | fixed |
| Acceptance targeting | step rule, target ~0.25 (scale halved below 0.10, grown above 0.35) | logistic, target 0.25 | target 0.25 | target 0.25 |
| Likelihood | Chan–Jeliazkov precision (≡ Kalman, 1e-8) | Kalman | Kalman | Kalman |
| Innovation covariance | diagonal | (DSGE structural) | (DSGE structural) | (DSGE structural) |
| Convergence report | N_eff over G=20 + rank-folded R̂ | N_eff over G runs | N_eff / RNE | — |

Deviations from the references are variants, not departures: absolute-ESS
bisection vs. SMC.jl's relative-ESS rule; step-rule vs. logistic
acceptance adaptation (same 0.25 target region); the unconstrained-
coordinate mutation and waste-free scheme are published add-ons layered
on the same algorithm.

---

## 10. Why this isn't SMC²

SMC² (Chopin, Jacob & Papaspiliopoulos 2013) targets online/sequential
estimation where states **cannot** be integrated out and each parameter
particle carries a nested particle filter. Our model is linear Gaussian,
the states are integrated out **exactly** at every evaluation, and the
sampler is a batch (offline) tempering run. The Jahan-Parvar et al.
(2024) UC application uses SMC², not batch tempering.

---

## 11. Known validity caveats

- **Seed-stability.** Largely resolved by the tuning refinements (§7):
  `max(bulk,folded) R̂ ≈ 1.2` at G=20, versus ~5.8 on the raw kernel. The
  residual is at the level any SMC estimator carries; the reported table
  is a ≥3-seed pool (inter-seed envelope, not a converged posterior).
- **Weakly identified directions.** The Phillips slope (γ2 ≈ −0.24) is
  prior-influenced; r\* is the least-identified latent state (band
  ±2.5pp) absent a neutral-rate proxy — `pi_e` is the only trend-
  inflation anchor, and r\* is data-identified only from 1993 (cash-rate
  coverage).
- **COVID hyperparameters** informed by ~8 quarters are prior-dominated
  by construction (Lenza–Primiceri make the same acknowledgement); this
  is a correct weak-identification diagnostic, addressed by the
  `PoolAcuteOnly` parsimony (§5), not a sampler failure.
- **`out.lml`** is an internal diagnostic, not a calibrated marginal
  likelihood (§4.5).

---

## 12. Core citations

- Chan, J.C.C., Jeliazkov, I. (2009). Efficient simulation and integrated
  likelihood estimation in state space models. *IJMMNO* 1, 101–120.
- Chan, J.C.C., Koop, G., Potter, S. (2016). A bounded model of time
  variation in trend inflation, NAIRU and the Phillips curve. *JAE* 31,
  551–565.
- Grant, A.L., Chan, J.C.C. (2017). Reconciling output gaps: seemingly
  unrelated trend-cycle decomposition. *JEDC* 75, 114–121.
- McCausland, W.J., Miller, S., Pelletier, D. (2011). Simulation
  smoothing for state-space models: a computational efficiency analysis.
  *Comput. Stat. & Data Anal.* 55, 199–212.
- Mertens, E. (2023). Precision-based sampling for state space models
  that have no measurement error. *JEDC* 154, 104720.
- Herbst, E., Schorfheide, F. (2014). Sequential Monte Carlo sampling for
  DSGE models. *JAE* 29, 1073–1098.
- Herbst, E., Schorfheide, F. (2015). *Bayesian Estimation of DSGE
  Models*. Princeton University Press.
- Cai, M., Del Negro, M., Herbst, E., Matlin, E., Sarfati, R.,
  Schorfheide, F. (2021). Online estimation of DSGE models. *Econometrics
  Journal* 24, C33–C68. (SMC.jl companion.)
- Chib, S., Ramamurthy, S. (2010). Tailored randomized block MCMC methods
  with application to DSGE models. *J. Econometrics* 155, 19–38.
- Dau, H.-D., Chopin, N. (2022). Waste-free sequential Monte Carlo.
  *JRSS-B* 84, 114–148. (arXiv:2011.02328)
- Papaspiliopoulos, O., Roberts, G.O., Sköld, M. (2007). A general
  framework for the parametrization of hierarchical models. *Statistical
  Science* 22, 59–73.
- Durham, G., Geweke, J. (2014). Adaptive sequential posterior simulators
  for massively parallel computing environments. *Advances in
  Econometrics* 34, 1–44.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., Bürkner, P.-C.
  (2021). Rank-normalization, folding, and localization: an improved R̂
  for assessing convergence of MCMC. *Bayesian Analysis* 16, 667–718.
- Lenza, M., Primiceri, G.E. (2022). How to estimate a VAR after March
  2020. *JAE* 37, 688–699.
- Carriero, A., Clark, T.E., Marcellino, M., Mertens, E. (2022).
  Addressing COVID-19 outliers in BVARs with stochastic volatility.
  *Review of Economics and Statistics*, 1–38.
- Zaman, S. (2022). A unified framework to estimate macroeconomic stars.
  FRB Cleveland WP 21-23R.
- Chopin, N., Jacob, P.E., Papaspiliopoulos, O. (2013). SMC²: an efficient
  algorithm for sequential analysis of state-space models. *JRSS-B* 75,
  397–426.
- Jahan-Parvar, M.R., Knipp, C., Szerszeń, P.J. (2024). Trend-cycle
  decomposition and forecasting using Bayesian multivariate unobserved
  components. FEDS 2024-100.
- Software: FRBNY `DSGE.jl`/`SMC.jl` (BSD-3), Dynare ≥6.0 (`hssmc`, GPL),
  J. Chan's MATLAB precision-sampler code (joshuachan.org).
