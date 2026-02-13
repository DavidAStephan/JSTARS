data {
  int<lower=3> T;
  vector[T] y_obs;
  vector[T] u_obs;
  vector[T] r_obs;
  vector[T] pr_obs;
  vector[T] h_obs;
}

parameters {
  // Initial states
  real y_pot_1;
  real g_pot_1;
  real y_gap_1;
  real y_gap_2;
  real nairu_1;
  real rstar_1;
  real pr_tr_1;
  real h_tr_1;

  // Dynamic coefficients
  real<lower=-1.5, upper=1.5> phi1;
  real<lower=-1.0, upper=1.0> phi2;

  real beta_u_gap;
  real beta_r_gap;
  real beta_pr_gap;
  real beta_h_gap;

  // State shock scales
  real<lower=0> sigma_y_pot;
  real<lower=0> sigma_g_pot;
  real<lower=0> sigma_gap;
  real<lower=0> sigma_nairu;
  real<lower=0> sigma_rstar;
  real<lower=0> sigma_pr_tr;
  real<lower=0> sigma_h_tr;

  // Measurement shock scales
  real<lower=0> sigma_y;
  real<lower=0> sigma_u;
  real<lower=0> sigma_r;
  real<lower=0> sigma_pr;
  real<lower=0> sigma_h;

  // Non-centered shocks
  vector[T - 1] z_y_pot;
  vector[T - 1] z_g_pot;
  vector[T - 2] z_gap;
  vector[T - 1] z_nairu;
  vector[T - 1] z_rstar;
  vector[T - 1] z_pr_tr;
  vector[T - 1] z_h_tr;
}

transformed parameters {
  vector[T] y_pot;
  vector[T] g_pot;
  vector[T] y_gap;
  vector[T] nairu;
  vector[T] rstar;
  vector[T] pr_tr;
  vector[T] h_tr;

  y_pot[1] = y_pot_1;
  g_pot[1] = g_pot_1;

  y_gap[1] = y_gap_1;
  y_gap[2] = y_gap_2;

  nairu[1] = nairu_1;
  rstar[1] = rstar_1;
  pr_tr[1] = pr_tr_1;
  h_tr[1] = h_tr_1;

  for (t in 2:T) {
    g_pot[t] = g_pot[t - 1] + sigma_g_pot * z_g_pot[t - 1];
    y_pot[t] = y_pot[t - 1] + g_pot[t - 1] + sigma_y_pot * z_y_pot[t - 1];

    nairu[t] = nairu[t - 1] + sigma_nairu * z_nairu[t - 1];
    rstar[t] = rstar[t - 1] + sigma_rstar * z_rstar[t - 1];
    pr_tr[t] = pr_tr[t - 1] + sigma_pr_tr * z_pr_tr[t - 1];
    h_tr[t] = h_tr[t - 1] + sigma_h_tr * z_h_tr[t - 1];
  }

  for (t in 3:T) {
    y_gap[t] = phi1 * y_gap[t - 1] + phi2 * y_gap[t - 2] + sigma_gap * z_gap[t - 2];
  }
}

model {
  // Priors for initials
  y_pot_1 ~ normal(y_obs[1], 1.0);
  g_pot_1 ~ normal(0, 0.1);
  y_gap_1 ~ normal(0, 0.5);
  y_gap_2 ~ normal(0, 0.5);

  nairu_1 ~ normal(u_obs[1], 0.5);
  rstar_1 ~ normal(r_obs[1], 0.5);
  pr_tr_1 ~ normal(pr_obs[1], 0.5);
  h_tr_1 ~ normal(h_obs[1], 0.5);

  // Dynamic parameters
  phi1 ~ normal(0.6, 0.3);
  phi2 ~ normal(0.0, 0.2);

  beta_u_gap ~ normal(0.5, 0.5);
  beta_r_gap ~ normal(0.3, 0.5);
  beta_pr_gap ~ normal(0.1, 0.5);
  beta_h_gap ~ normal(0.1, 0.5);

  // Scale priors
  sigma_y_pot ~ normal(0, 0.1);
  sigma_g_pot ~ normal(0, 0.05);
  sigma_gap ~ normal(0, 0.2);
  sigma_nairu ~ normal(0, 0.05);
  sigma_rstar ~ normal(0, 0.05);
  sigma_pr_tr ~ normal(0, 0.05);
  sigma_h_tr ~ normal(0, 0.05);

  sigma_y ~ normal(0, 0.1);
  sigma_u ~ normal(0, 0.2);
  sigma_r ~ normal(0, 0.2);
  sigma_pr ~ normal(0, 0.2);
  sigma_h ~ normal(0, 0.2);

  // Standard normal innovations
  z_y_pot ~ std_normal();
  z_g_pot ~ std_normal();
  z_gap ~ std_normal();
  z_nairu ~ std_normal();
  z_rstar ~ std_normal();
  z_pr_tr ~ std_normal();
  z_h_tr ~ std_normal();

  // Measurement equations
  y_obs ~ normal(y_pot + y_gap, sigma_y);
  u_obs ~ normal(nairu - beta_u_gap * y_gap, sigma_u);
  r_obs ~ normal(rstar + beta_r_gap * y_gap, sigma_r);
  pr_obs ~ normal(pr_tr + beta_pr_gap * y_gap, sigma_pr);
  h_obs ~ normal(h_tr + beta_h_gap * y_gap, sigma_h);
}

generated quantities {
  vector[T] unemployment_gap;
  vector[T] real_rate_gap;

  for (t in 1:T) {
    unemployment_gap[t] = u_obs[t] - nairu[t];
    real_rate_gap[t] = r_obs[t] - rstar[t];
  }
}
