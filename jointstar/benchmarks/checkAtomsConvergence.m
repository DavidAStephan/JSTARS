% Convergence check for RidgeAtoms specification: compute R-hat across
% three independent seeds and pool them into a stratified posterior.
%
% Mirrors runConvergenceCheck.m and poolRuns.m but:
% - Points to three atoms_seed* directories (existing runs, full spec)
% - Asserts posterior_summary.csv exists (no re-estimation)
% - Reads smc_log.csv and sums lml_inc per run
% - Writes pooled posterior to results/pooled_posterior_atoms.csv
% - Prints top 15 parameters by R-hat with per-seed means
% - Prints per-run LML totals and end-of-sample latent states

% Use absolute paths to ensure consistent directory resolution
basedir = '/Users/davidstephan/Documents/JSTARS/jointstar';
addpath(basedir, fullfile(basedir, '..'));

dirs = {
    fullfile(basedir, 'results/atoms_seed42')
    fullfile(basedir, 'results/atoms_seed7')
    fullfile(basedir, 'results/atoms_seed101')
};

% Verify posterior_summary.csv files exist
for r = 1:numel(dirs)
    assert(isfile(fullfile(dirs{r}, 'posterior_summary.csv')), ...
        sprintf('Missing posterior_summary.csv in %s', dirs{r}));
end
fprintf('=== all runs verified (no re-estimation) ===\n');

P = jointstar.horseshoePriors('HierKappa', true);
R = numel(dirs);
mu = zeros(R, P.d);
v = zeros(R, P.d);
uniqFrac = zeros(R, 1);
rstarEnd = zeros(R, 3);
nairuEnd = zeros(R, 3);
gapEnd = zeros(R, 3);
lmlTotal = zeros(R, 1);

% Load particle snapshots, compute per-run statistics
for r = 1:R
    % Load final particle snapshot (phi = 1)
    snaps = dir(fullfile(dirs{r}, 'particles_stage_*.mat'));
    [~, iL] = max([snaps.datenum]);
    S = load(fullfile(dirs{r}, snaps(iL).name));
    assert(abs(S.phi - 1) < 1e-9, ...
        sprintf('Final stage phi != 1 in run %d', r));

    % Normalize weights
    w = exp(S.logw - max(S.logw));
    w = w / sum(w);

    % Weighted posterior mean and variance
    mu(r, :) = w' * S.P;
    v(r, :) = w' * (S.P - mu(r, :)).^2;

    % Fraction of unique particles at phi=1
    uniqFrac(r) = size(unique(S.P, 'rows'), 1) / size(S.P, 1);

    % Sum log-marginal-likelihood increments
    log_tbl = readtable(fullfile(dirs{r}, 'smc_log.csv'));
    lmlTotal(r) = sum(log_tbl.lml_inc);

    % End-of-sample latent states
    st = readtable(fullfile(dirs{r}, 'smoothed_states.csv'));
    rstarEnd(r, :) = [st.rstar_q05(end), st.rstar_med(end), st.rstar_q95(end)];
    nairuEnd(r, :) = [st.nairu_q05(end), st.nairu_med(end), st.nairu_q95(end)];
    gapEnd(r, :) = [st.gap_q05(end), st.gap_med(end), st.gap_q95(end)];
end

% Compute R-hat = sqrt(1 + B/W)
B = var(mu, 0, 1);                % between-run variance of means
W = mean(v, 1);                   % mean within-run variance
ok = W > 1e-12;                   % skip calibrated constant
rhat = ones(1, P.d);
rhat(ok) = sqrt(1 + B(ok) ./ W(ok));

% Write convergence table to CSV
isBase = 1:P.d <= P.baseD;
tbl = table(string(P.names(:)), rhat(:), mu', ...
    'VariableNames', {'param', 'rhat', 'means_by_seed'});
writetable(splitvars(tbl), fullfile(basedir, 'results/convergence_rhat_atoms.csv'));

fprintf('\nunique-particle fraction at phi=1 per run: ');
fprintf('%.2f ', uniqFrac);
fprintf('\n');

fprintf('\nlog-marginal-likelihood (total per run):\n');
for r = 1:R
    fprintf('  %-22s %10.2f\n', dirs{r}, lmlTotal(r));
end

fprintf('\nr* end of sample by seed:\n');
for r = 1:R
    fprintf('  %-22s %6.3f [%6.3f, %6.3f]\n', dirs{r}, ...
        rstarEnd(r, 2), rstarEnd(r, 1), rstarEnd(r, 3));
end

fprintf('\nNAIRU end of sample by seed:\n');
for r = 1:R
    fprintf('  %-22s %6.3f [%6.3f, %6.3f]\n', dirs{r}, ...
        nairuEnd(r, 2), nairuEnd(r, 1), nairuEnd(r, 3));
end

fprintf('\noutput gap end of sample by seed:\n');
for r = 1:R
    fprintf('  %-22s %6.3f [%6.3f, %6.3f]\n', dirs{r}, ...
        gapEnd(r, 2), gapEnd(r, 1), gapEnd(r, 3));
end

fprintf('\nRhat summary: base params (n=%d): max %.3f, #>1.05: %d, #>1.10: %d\n', ...
    nnz(isBase & ok), max(rhat(isBase & ok)), ...
    nnz(rhat > 1.05 & isBase & ok), nnz(rhat > 1.10 & isBase & ok));
fprintf('Rhat summary: ALL params (n=%d):  max %.3f, #>1.05: %d, #>1.10: %d\n', ...
    nnz(ok), max(rhat(ok)), nnz(rhat > 1.05 & ok), nnz(rhat > 1.10 & ok));

% Top 15 parameters by R-hat
[~, order] = sort(rhat, 'descend');
fprintf('\ntop 15 parameters by R-hat:\n');
for j = order(1:15)
    fprintf('  %-18s rhat=%.3f  per-seed means: %8.3f %8.3f %8.3f\n', ...
        P.names{j}, rhat(j), mu(1, j), mu(2, j), mu(3, j));
end

fprintf('\noverall max R-hat: %.3f\n', max(rhat(ok)));

% =========================================================================
% Pool the three independent-seed runs into one stratified posterior
% =========================================================================

allP = [];
allW = [];
for r = 1:R
    snaps = dir(fullfile(dirs{r}, 'particles_stage_*.mat'));
    [~, iL] = max([snaps.datenum]);
    S = load(fullfile(dirs{r}, snaps(iL).name));
    assert(abs(S.phi - 1) < 1e-9);
    w = exp(S.logw - max(S.logw));
    w = w / sum(w);
    allP = [allP; S.P]; %#ok<AGROW>
    allW = [allW; w / R]; %#ok<AGROW>
end

% Pooled posterior: mean, sd, quantiles
d = P.d;
mu_pool = (allW' * allP)';
sd_pool = sqrt(allW' * (allP - mu_pool').^2)';
q_pool = zeros(d, 3);
for j = 1:d
    [xs, i] = sort(allP(:, j));
    cw = cumsum(allW(i));
    cw = cw / cw(end);
    pset = [0.05, 0.5, 0.95];
    for kq = 1:3
        q_pool(j, kq) = xs(find(cw >= pset(kq), 1, 'first'));
    end
end

tbl_pool = table(string(P.names(:)), mu_pool, sd_pool, ...
    q_pool(:, 1), q_pool(:, 2), q_pool(:, 3), ...
    'VariableNames', {'param', 'mean', 'sd', 'q05', 'q50', 'q95'});
writetable(tbl_pool, fullfile(basedir, 'results/pooled_posterior_atoms.csv'));

fprintf('\n=== pooled posterior written to %s ===\n', ...
    fullfile(basedir, 'results/pooled_posterior_atoms.csv'));
