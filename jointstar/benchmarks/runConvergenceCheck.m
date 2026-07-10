% Gelman-style convergence check for the SMC estimator: run the FULL
% final specification from independent seeds and compare posteriors.
% Between-run agreement is the SMC analogue of R-hat -- the runs share no
% randomness at all (independent prior draws, tempering schedules,
% resampling and mutation streams), so agreement cannot come from
% anything but the target posterior.
%
% Statistic per parameter: Rhat = sqrt(1 + B/W), with B the variance of
% the (weighted) posterior means across runs and W the mean of the
% (weighted) posterior variances.  Rhat ~ 1 for agreement; > 1.1 flags a
% parameter whose posterior location moves materially across seeds.

seeds = [7, 101];                       % seed 42 already exists (cp7b)
dirs = {'results/cp7b'};
for s = seeds
    od = sprintf('results/rhat_seed%d', s);
    if ~isfile(fullfile(od, 'posterior_summary.csv'))
        jointstar.estimate('../data.csv', 'NParticles', 2000, ...
            'MSteps', 2, 'Seed', s, 'OutDir', od, ...
            'Horseshoe', true, 'HierKappa', true, 'PieObs', true);
    end
    dirs{end + 1} = od; %#ok<AGROW>
end
fprintf('=== all runs done ===\n');

P = jointstar.horseshoePriors('HierKappa', true);
R = numel(dirs);
mu = zeros(R, P.d); v = zeros(R, P.d);
uniqFrac = zeros(R, 1); rstarEnd = zeros(R, 3);
for r = 1:R
    snaps = dir(fullfile(dirs{r}, 'particles_stage_*.mat'));
    [~, iL] = max([snaps.datenum]);
    S = load(fullfile(dirs{r}, snaps(iL).name));
    assert(abs(S.phi - 1) < 1e-9);
    w = exp(S.logw - max(S.logw)); w = w / sum(w);
    mu(r, :) = w' * S.P;
    v(r, :) = w' * (S.P - mu(r, :)).^2;
    uniqFrac(r) = size(unique(S.P, 'rows'), 1) / size(S.P, 1);
    st = readtable(fullfile(dirs{r}, 'smoothed_states.csv'));
    rstarEnd(r, :) = [st.rstar_q05(end), st.rstar_med(end), st.rstar_q95(end)];
end

B = var(mu, 0, 1);                       % between-run variance of means
W = mean(v, 1);                          % mean within-run variance
ok = W > 1e-12;                          % skip the calibrated constant
rhat = ones(1, P.d);
rhat(ok) = sqrt(1 + B(ok) ./ W(ok));

isBase = 1:P.d <= P.baseD;
tbl = table(string(P.names(:)), rhat(:), mu', 'VariableNames', ...
    {'param', 'rhat', 'means_by_seed'});
writetable(splitvars(tbl), 'results/convergence_rhat.csv');

fprintf('\nunique-particle fraction at phi=1 per run: ');
fprintf('%.2f ', uniqFrac); fprintf('\n');
fprintf('r* end of sample by seed:\n');
for r = 1:R
    fprintf('  %-22s %.2f [%.2f, %.2f]\n', dirs{r}, ...
        rstarEnd(r, 2), rstarEnd(r, 1), rstarEnd(r, 3));
end
fprintf('\nRhat summary: base params (n=%d): max %.3f, #>1.05: %d, #>1.10: %d\n', ...
    nnz(isBase & ok), max(rhat(isBase & ok)), ...
    nnz(rhat > 1.05 & isBase & ok), nnz(rhat > 1.10 & isBase & ok));
fprintf('Rhat summary: ALL params (n=%d):  max %.3f, #>1.05: %d, #>1.10: %d\n', ...
    nnz(ok), max(rhat(ok)), nnz(rhat > 1.05 & ok), nnz(rhat > 1.10 & ok));
[~, order] = sort(rhat, 'descend');
fprintf('\nworst 12 parameters:\n');
for j = order(1:12)
    fprintf('  %-18s rhat=%.3f  means:', P.names{j}, rhat(j));
    fprintf(' %8.3f', mu(:, j)); fprintf('\n');
end
