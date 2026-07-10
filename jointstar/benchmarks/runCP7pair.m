% Checkpoint 7 pair: (1) cp6a = post-ruling reference without the pi_e
% measurement row; (2) cp7 = identical spec plus the row.  Differences
% between the two isolate what the trend-inflation anchor does to r*,
% the pile-up variances, and the d_t/kappa interplay.

r6a = jointstar.estimate('../data.csv', 'NParticles', 2000, 'MSteps', 2, ...
    'Seed', 42, 'OutDir', 'results/cp6a', 'Horseshoe', true, ...
    'HierKappa', true, 'PieObs', false);
jointstar.horseshoeDiag(r6a.smc, r6a.priors, 'results/cp6a');
s = r6a.summary; st = r6a.states; save('results/cp6a/results.mat', 's', 'st', '-v7.3');
fprintf('=== cp6a done ===\n');

r7 = jointstar.estimate('../data.csv', 'NParticles', 2000, 'MSteps', 2, ...
    'Seed', 42, 'OutDir', 'results/cp7', 'Horseshoe', true, ...
    'HierKappa', true, 'PieObs', true);
jointstar.horseshoeDiag(r7.smc, r7.priors, 'results/cp7');
s = r7.summary; st = r7.states; save('results/cp7/results.mat', 's', 'st', '-v7.3');
t = jointstar.validate('results/cp7');
writetable(t, 'results/cp7/validation_vs_baseline.csv');
fprintf('=== cp7 done ===\n');

% ---- CP7 deliverable: before/after comparison --------------------------
sel = {'nu', 'gamma2', 'sig_gz', 'sig_xi', 'sig_pie', 'phisum', 'phi2', ...
    'phiy', 'phiu', 'phipr', 'phihpp', 'phik', 'kapy_20', 'kapu_20', ...
    'kappi_2023', 'sme_pieobs'};
fprintf('\n%-12s %28s %28s\n', 'param', 'cp6a (no anchor)', 'cp7 (pi_e anchor)');
for i = 1:numel(sel)
    a = r6a.summary(strcmp(r6a.summary.param, sel{i}), :);
    b = r7.summary(strcmp(r7.summary.param, sel{i}), :);
    if isempty(a)
        fprintf('%-12s %28s %10.3f [%6.3f,%6.3f]\n', sel{i}, '--', ...
            b.mean, b.q05, b.q95);
    else
        fprintf('%-12s %10.3f [%6.3f,%6.3f] %10.3f [%6.3f,%6.3f]\n', sel{i}, ...
            a.mean, a.q05, a.q95, b.mean, b.q05, b.q95);
    end
end

% r* level and band width, end of sample and 2010s average
for R = {{'cp6a', r6a.states}, {'cp7', r7.states}}
    nm = R{1}{1}; st = R{1}{2};
    bw = st.rstar_q95 - st.rstar_q05;
    fprintf('%s: r* end = %.2f [%.2f, %.2f] (band %.2f); mean band width %.2f\n', ...
        nm, st.rstar_med(end), st.rstar_q05(end), st.rstar_q95(end), ...
        bw(end), mean(bw, 'omitnan'));
end
