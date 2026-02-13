# Bayesian joint state-space estimation using CmdStanR
# Requires: cmdstanr

make_state_space_data <- function(df) {
  required_cols <- c("y_obs", "u_obs", "r_obs", "pr_obs", "h_obs")
  missing_cols <- setdiff(required_cols, names(df))
  if (length(missing_cols) > 0) {
    stop(sprintf("Missing required columns: %s", paste(missing_cols, collapse = ", ")))
  }

  if (any(!stats::complete.cases(df[, required_cols]))) {
    stop("Input data contains NA values in required columns.")
  }

  T <- nrow(df)
  if (T < 3) {
    stop("Need at least 3 time periods (T >= 3).")
  }

  list(
    T = T,
    y_obs = as.numeric(df$y_obs),
    u_obs = as.numeric(df$u_obs),
    r_obs = as.numeric(df$r_obs),
    pr_obs = as.numeric(df$pr_obs),
    h_obs = as.numeric(df$h_obs)
  )
}

fit_joint_state_space <- function(
    data_list,
    stan_file = "stan/joint_state_space.stan",
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1000,
    adapt_delta = 0.95,
    max_treedepth = 12,
    seed = 1234) {

  if (!requireNamespace("cmdstanr", quietly = TRUE)) {
    stop("Package 'cmdstanr' is required. Install via install.packages('cmdstanr').")
  }

  mod <- cmdstanr::cmdstan_model(stan_file)

  fit <- mod$sample(
    data = data_list,
    chains = chains,
    parallel_chains = parallel_chains,
    iter_warmup = iter_warmup,
    iter_sampling = iter_sampling,
    adapt_delta = adapt_delta,
    max_treedepth = max_treedepth,
    seed = seed
  )

  fit
}
