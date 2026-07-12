% Pool the independent-seed SMC runs into one stratified posterior.
% Each run is a valid (equally-weighted) importance approximation of the
% same posterior, so their equal-weight mixture is too -- and the
% between-run disagreement that Rhat flags becomes honest within-pool
% spread instead of being hidden inside any single run's too-tight cloud.

dirs = {'results/cp7b', 'results/rhat_seed7', 'results/rhat_seed101'};
P = jointstar.horseshoePriors('HierKappa', true);
allP = []; allW = [];
for r = 1:numel(dirs)
    snaps = dir(fullfile(dirs{r}, 'particles_stage_*.mat'));
    [~, iL] = max([snaps.datenum]);
    S = load(fullfile(dirs{r}, snaps(iL).name));
    assert(abs(S.phi - 1) < 1e-9);
    w = exp(S.logw - max(S.logw)); w = w / sum(w);
    allP = [allP; S.P]; %#ok<AGROW>
    allW = [allW; w / numel(dirs)]; %#ok<AGROW>
end

d = P.d;
mu = (allW' * allP)';
sd = sqrt(allW' * (allP - mu').^2)';
q = zeros(d, 3);
for j = 1:d
    [xs, i] = sort(allP(:, j));
    cw = cumsum(allW(i)); cw = cw / cw(end);
    for kq = 1:3
        pset = [0.05, 0.5, 0.95];
        q(j, kq) = xs(find(cw >= pset(kq), 1, 'first'));
    end
end
tbl = table(string(P.names(:)), mu, sd, q(:, 1), q(:, 2), q(:, 3), ...
    'VariableNames', {'param', 'mean', 'sd', 'q05', 'q50', 'q95'});
writetable(tbl, 'results/pooled_posterior.csv');

sel = {'gamma1', 'gamma2', 'nu', 'phisum', 'phi2', 'xi1', 'rhoU', ...
    'rhohpp', 'sig_gz', 'sig_xi', 'sig_c', 'sme_y', 'kapy_20', 'kapu_20'};
fprintf('pooled posterior (3 seeds x 2000 particles):\n');
for i = 1:numel(sel)
    r = tbl(strcmp(tbl.param, sel{i}), :);
    fprintf('  %-8s %8.3f  [%8.3f, %8.3f]\n', sel{i}, r.mean, r.q05, r.q95);
end
