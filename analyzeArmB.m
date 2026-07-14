%% Convergence A/B analysis: arm B (new kernel) vs arm A (baseline production)
% Compute Gelman R̂, rank-normalized R̂, pooled posterior, and health metrics

cd('/Users/davidstephan/Documents/JSTARS');
outDir = '/private/tmp/claude-501/-Users-davidstephan-Documents-JSTARS/1883750c-1387-4d11-8beb-57d3da17699c/scratchpad/armB';

% Load priors
P = jointstar.defaultPriors('HierKappa', true);
d = P.d;

fprintf('\n=== ARM A/B CONVERGENCE ANALYSIS ===\n');
fprintf('priors: %d parameters, HierKappa=true\n', d);

% -------- ARM B --------
fprintf('\nARM B (MutationTransform=true, StructuredBlocks=true):\n');
armB_dirs = {
    fullfile(outDir, 'seed42')
    fullfile(outDir, 'seed7')
    fullfile(outDir, 'seed101')
};
seeds = [42, 7, 101];

% Extract arm B health metrics and load particles
armB_particles = cell(1, 3);
armB_logw = cell(1, 3);
armB_lml = zeros(1, 3);
armB_stages = zeros(1, 3);
armB_acc_ranges = cell(1, 3);
armB_ess = zeros(1, 3);
armB_unique_rows = zeros(1, 3);

for k = 1:3
    fprintf('  seed %d: ', seeds(k));
    [armB_particles{k}, armB_logw{k}, armB_lml(k), armB_stages(k), ...
        armB_acc_ranges{k}, armB_ess(k), armB_unique_rows(k)] ...
        = extractMetrics(armB_dirs{k}, P);
    fprintf('stages=%d, unique=%d, ESS=%.0f/2000, LML=%.2f\n', ...
        armB_stages(k), armB_unique_rows(k), armB_ess(k), armB_lml(k));
end

% -------- ARM A (production baseline) --------
fprintf('\nARM A (baseline production, diagonal spec):\n');
armA_dirs = {
    'results/production/seed42'
    'results/production/seed7'
    'results/production/seed101'
};

% Extract arm A health metrics and load particles
armA_particles = cell(1, 3);
armA_logw = cell(1, 3);
armA_lml = zeros(1, 3);
armA_stages = zeros(1, 3);
armA_acc_ranges = cell(1, 3);
armA_ess = zeros(1, 3);
armA_unique_rows = zeros(1, 3);

for k = 1:3
    fprintf('  seed %d: ', seeds(k));
    [armA_particles{k}, armA_logw{k}, armA_lml(k), armA_stages(k), ...
        armA_acc_ranges{k}, armA_ess(k), armA_unique_rows(k)] ...
        = extractMetrics(armA_dirs{k}, P);
    fprintf('stages=%d, unique=%d, ESS=%.0f/2000, LML=%.2f\n', ...
        armA_stages(k), armA_unique_rows(k), armA_ess(k), armA_lml(k));
end

% -------- Gelman R̂ (standard) --------
fprintf('\n--- Standard R̂ (Gelman formula) ---\n');
[armB_rhat, armB_means] = computeRhat(armB_particles, armB_logw);
[armA_rhat, armA_means] = computeRhat(armA_particles, armA_logw);

fprintf('ARM B: max R̂ = %.3f, count R̂>1.2 = %d, count R̂>1.5 = %d\n', ...
    max(armB_rhat), nnz(armB_rhat > 1.2), nnz(armB_rhat > 1.5));
fprintf('ARM A: max R̂ = %.3f, count R̂>1.2 = %d, count R̂>1.5 = %d\n', ...
    max(armA_rhat), nnz(armA_rhat > 1.2), nnz(armA_rhat > 1.5));

% -------- Rank-normalized R̂ --------
fprintf('\n--- Rank-normalized R̂ (Vehtari et al. 2021) ---\n');
[armB_rnorm_rhat] = computeRankNormRhat(armB_particles, armB_logw);
[armA_rnorm_rhat] = computeRankNormRhat(armA_particles, armA_logw);

fprintf('ARM B: max rank-R̂ = %.3f, count >1.2 = %d, count >1.5 = %d\n', ...
    max(armB_rnorm_rhat), nnz(armB_rnorm_rhat > 1.2), nnz(armB_rnorm_rhat > 1.5));
fprintf('ARM A: max rank-R̂ = %.3f, count >1.2 = %d, count >1.5 = %d\n', ...
    max(armA_rnorm_rhat), nnz(armA_rnorm_rhat > 1.2), nnz(armA_rnorm_rhat > 1.5));

% -------- ARM B Pooled Posterior --------
fprintf('\n--- ARM B Pooled Posterior ---\n');
armB_pooled = poolPosterior(armB_particles, armB_logw, P);
armB_pooled_file = fullfile(outDir, 'armB_pooled_posterior.csv');
writetable(armB_pooled, armB_pooled_file);
fprintf('Wrote: %s\n', armB_pooled_file);

% -------- Write R̂ tables --------
fprintf('\n--- Writing convergence metrics ---\n');

% Standard R̂ for arm B
armB_rhat_file = fullfile(outDir, 'armB_convergence_rhat.csv');
tbl = table(string(P.names(:)), armB_rhat(:), armB_means', ...
    'VariableNames', {'param', 'rhat', 'means_by_seed'});
writetable(splitvars(tbl), armB_rhat_file);
fprintf('Wrote: %s\n', armB_rhat_file);

% Rank-normalized R̂ for both arms
rnorm_tbl = table(string(P.names(:)), armA_rnorm_rhat(:), armB_rnorm_rhat(:), ...
    'VariableNames', {'param', 'armA_rnorm_rhat', 'armB_rnorm_rhat'});
rnorm_file = fullfile(outDir, 'armB_rhat_ranknorm.csv');
writetable(rnorm_tbl, rnorm_file);
fprintf('Wrote: %s\n', rnorm_file);

% -------- Comparison table: worst 15 R̂ parameters --------
fprintf('\n--- Comparison table (worst 15 arm-A R̂) ---\n');
[~, idx_worst] = sort(armA_rhat, 'descend');
idx_worst = idx_worst(1:min(15, d));

% Construct table with correct indexing
params_worst = P.names(idx_worst);
comp_rows = {};
for i = 1:numel(idx_worst)
    idx = idx_worst(i);
    comp_rows{i} = {params_worst{i}, armA_rhat(idx), armB_rhat(idx), ...
        armA_rnorm_rhat(idx), armB_rnorm_rhat(idx), ...
        sprintf('[%.3f %.3f %.3f]', armA_means(:, idx)), ...
        sprintf('[%.3f %.3f %.3f]', armB_means(:, idx))};
end
comp_tbl = cell2table(vertcat(comp_rows{:}), ...
    'VariableNames', {'param', 'armA_rhat', 'armB_rhat', 'armA_rnorm_rhat', 'armB_rnorm_rhat', ...
    'armA_means_by_seed', 'armB_means_by_seed'});
fprintf(repmat('=', 1, 140)); fprintf('\n');
disp(comp_tbl);
fprintf(repmat('=', 1, 140)); fprintf('\n');

% -------- Seed-mean dispersion: select 5 parameters --------
fprintf('\n--- Seed-mean dispersion (in within-seed-sd units) ---\n');
target_params = {'gwbar', 'kappr_2122', 'kapHyp_lm_w2023tot', 'sig_xi', 'gzbar'};

for kp = 1:numel(target_params)
    pname = target_params{kp};
    idx = find(strcmp(P.names, pname));
    if isempty(idx)
        fprintf('%s: NOT FOUND\n', pname);
        continue
    end
    % Seed means from analysis above (armA_means is 3 x 79)
    armA_seed_dispersion = std(armA_means(:, idx));
    armB_seed_dispersion = std(armB_means(:, idx));

    % Within-seed sd is approximately the mean of seed sds
    % Compute for each seed and average
    armA_within_sds = zeros(1, 3);
    armB_within_sds = zeros(1, 3);
    for r = 1:3
        w_a = exp(armA_logw{r} - max(armA_logw{r}));
        w_a = w_a / sum(w_a);
        armA_within_sds(r) = sqrt(w_a' * (armA_particles{r}(:, idx) - armA_means(r, idx)).^2);

        w_b = exp(armB_logw{r} - max(armB_logw{r}));
        w_b = w_b / sum(w_b);
        armB_within_sds(r) = sqrt(w_b' * (armB_particles{r}(:, idx) - armB_means(r, idx)).^2);
    end
    armA_mean_within_sd = mean(armA_within_sds);
    armB_mean_within_sd = mean(armB_within_sds);

    armA_ratio = armA_seed_dispersion / armA_mean_within_sd;
    armB_ratio = armB_seed_dispersion / armB_mean_within_sd;

    fprintf('%s: armA %.2f, armB %.2f (within-seed sd: armA=%.4f, armB=%.4f)\n', ...
        pname, armA_ratio, armB_ratio, armA_mean_within_sd, armB_mean_within_sd);
end

% -------- LML summary --------
fprintf('\n--- Log-Marginal-Likelihood (LML) ---\n');
fprintf('ARM B:\n');
for k = 1:3
    fprintf('  seed %d: LML = %.2f\n', seeds(k), armB_lml(k));
end
fprintf('  cross-seed range: %.2f (max-min)\n', max(armB_lml) - min(armB_lml));

fprintf('ARM A:\n');
for k = 1:3
    fprintf('  seed %d: LML = %.2f\n', seeds(k), armA_lml(k));
end
fprintf('  cross-seed range: %.2f (max-min)\n', max(armA_lml) - min(armA_lml));

fprintf('\n=== ANALYSIS COMPLETE ===\n');

% ========================================================================
% Helper functions
% ========================================================================

function [P, logw, lml, stages, acc_range, ess, n_unique] = extractMetrics(dirpath, priors)
% Load final particle snapshot and extract metrics
snaps = dir(fullfile(dirpath, 'particles_stage_*.mat'));
assert(~isempty(snaps), 'No particle snapshots in %s', dirpath);
datenums = [snaps.datenum];
[~, iL] = max(datenums);
S = load(fullfile(dirpath, snaps(iL).name));
assert(abs(S.phi - 1) < 1e-9, 'Last snapshot is not phi=1');

P = S.P;
logw = S.logw;

% Load LML from smc_log.csv
log_tbl = readtable(fullfile(dirpath, 'smc_log.csv'));
lml = sum(log_tbl.lml_inc);
stages = height(log_tbl);

% Acceptance rates
acc_range = [min(log_tbl.acc_rate), max(log_tbl.acc_rate)];

% ESS: compute from weights
w = exp(logw - max(logw));
w = w / sum(w);
ess = 1 / sum(w.^2);

% Unique rows
n_unique = size(unique(P, 'rows'), 1);
end

function [rhat, seed_means] = computeRhat(particles, logws)
% Gelman R̂ = sqrt(1 + B/W)
d = size(particles{1}, 2);
R = numel(particles);

mu = zeros(R, d);
v = zeros(R, d);
for r = 1:R
    w = exp(logws{r} - max(logws{r}));
    w = w / sum(w);
    mu(r, :) = w' * particles{r};
    v(r, :) = w' * (particles{r} - mu(r, :)).^2;
end

B = var(mu, 0, 1);
W = mean(v, 1);
ok = W > 1e-12;
rhat = ones(1, d);
rhat(ok) = sqrt(1 + B(ok) ./ W(ok));

seed_means = mu;
end

function rnorm_rhat = computeRankNormRhat(particles, logws)
% Rank-normalized R̂: transform to normal scores, then compute standard R̂
d = size(particles{1}, 2);
R = numel(particles);

rnorm_rhat = zeros(1, d);

for j = 1:d
    % Rank-transform each parameter
    all_vals = [];
    for r = 1:R
        all_vals = [all_vals; particles{r}(:, j)];
    end

    [~, ranks] = sort(all_vals);
    N = numel(all_vals);
    z = norminv(((1:N)' - 0.5) / N);
    z_vals = nan(N, 1);
    z_vals(ranks) = z;

    % Split back into R chains and compute R̂
    mu_r = zeros(1, R);
    v_r = zeros(1, R);
    idx = 1;
    for r = 1:R
        n_r = size(particles{r}, 1);
        w = exp(logws{r} - max(logws{r}));
        w = w / sum(w);
        z_chain = z_vals(idx:idx+n_r-1);
        mu_r(r) = w' * z_chain;
        v_r(r) = w' * (z_chain - mu_r(r)).^2;
        idx = idx + n_r;
    end

    B_r = var(mu_r, 0, 1);
    W_r = mean(v_r, 1);
    if W_r > 1e-12
        rnorm_rhat(j) = sqrt(1 + B_r / W_r);
    else
        rnorm_rhat(j) = 1;
    end
end
end

function pooled_tbl = poolPosterior(particles, logws, P)
% Equal-weight pooled posterior (1/R each seed)
d = P.d;
R = numel(particles);

all_P = [];
all_w = [];
for r = 1:R
    w = exp(logws{r} - max(logws{r}));
    w = w / sum(w);
    all_P = [all_P; particles{r}];
    all_w = [all_w; w / R];
end

mu = (all_w' * all_P)';
sd = sqrt(all_w' * (all_P - mu').^2)';
q = zeros(d, 3);
pset = [0.05, 0.5, 0.95];
for j = 1:d
    [xs, i] = sort(all_P(:, j));
    cw = cumsum(all_w(i)); cw = cw / cw(end);
    for kq = 1:3
        q(j, kq) = xs(find(cw >= pset(kq), 1, 'first'));
    end
end

pooled_tbl = table(string(P.names(:)), mu, sd, q(:, 1), q(:, 2), q(:, 3), ...
    'VariableNames', {'param', 'mean', 'sd', 'q05', 'q50', 'q95'});
end
