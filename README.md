# Bayesian Joint State-Space Model (R + CmdStanR)

This repository provides a **joint Bayesian state-space model** in Stan (for CmdStanR) to estimate:

- Potential output
- Output gap
- NAIRU
- r-star (neutral real rate, `r*`)
- Trend participation rate
- Trend average hours worked

The model is written to estimate all latent states jointly from observed macro data.

## Model overview

Let the observed series be:

- `y_obs_t`: observed (log) output
- `u_obs_t`: observed unemployment rate
- `r_obs_t`: observed real policy/short rate
- `pr_obs_t`: observed participation rate
- `h_obs_t`: observed average hours worked

Latent states include:

- `y_pot_t`: potential output level
- `g_pot_t`: potential output trend growth
- `y_gap_t`: output gap
- `nairu_t`: non-accelerating inflation rate of unemployment
- `rstar_t`: neutral real rate
- `pr_tr_t`: trend participation
- `h_tr_t`: trend average hours

Core equations:

1. **Measurement equations**
   - `y_obs_t = y_pot_t + y_gap_t + e_y_t`
   - `u_obs_t = nairu_t - beta_u_gap * y_gap_t + e_u_t`
   - `r_obs_t = rstar_t + beta_r_gap * y_gap_t + e_r_t`
   - `pr_obs_t = pr_tr_t + beta_pr_gap * y_gap_t + e_pr_t`
   - `h_obs_t = h_tr_t + beta_h_gap * y_gap_t + e_h_t`

2. **State equations**
   - Potential output local-linear trend:
     - `y_pot_t = y_pot_{t-1} + g_pot_{t-1} + eta_y_pot_t`
     - `g_pot_t = g_pot_{t-1} + eta_g_pot_t`
   - Output gap AR(2):
     - `y_gap_t = phi1 * y_gap_{t-1} + phi2 * y_gap_{t-2} + eta_gap_t`
   - Slow-moving trends:
     - `nairu_t = nairu_{t-1} + eta_nairu_t`
     - `rstar_t = rstar_{t-1} + eta_rstar_t`
     - `pr_tr_t = pr_tr_{t-1} + eta_pr_t`
     - `h_tr_t = h_tr_{t-1} + eta_h_t`

## Files

- `stan/joint_state_space.stan`: Stan model.
- `R/fit_joint_state_space.R`: CmdStanR helpers to build data list, compile, and fit.

## Quick start

```r
install.packages(c("cmdstanr", "posterior"))
cmdstanr::install_cmdstan()  # once

source("R/fit_joint_state_space.R")

# Replace with your real data vectors (same length T)
T <- 120
set.seed(123)
example_df <- data.frame(
  y_obs  = cumsum(rnorm(T, 0.005, 0.01)),
  u_obs  = 5 + rnorm(T, 0, 0.2),
  r_obs  = 1 + rnorm(T, 0, 0.2),
  pr_obs = 63 + rnorm(T, 0, 0.15),
  h_obs  = 34 + rnorm(T, 0, 0.1)
)

stan_data <- make_state_space_data(example_df)
fit <- fit_joint_state_space(stan_data)

# Extract posterior summaries
fit$summary(variables = c("phi1", "phi2", "beta_u_gap", "beta_r_gap"))
```

## Practical notes

- Standardize or mean-center input series before fitting for better geometry.
- Use informative priors if identification is weak in your application.
- If you include inflation and wage equations, this can further anchor NAIRU and output gap.
