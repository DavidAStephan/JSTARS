%% JointSTAR re-estimation: validation against the current in-house model
% *jointstar toolbox -- Checkpoint 8 validation report.*
%
% This report compares the new estimator with the current in-house
% JointSTAR implementation ("the baseline"). The model is unchanged in
% structure -- the same signal and transition equations, COVID adjustments
% and variance breaks -- but the estimation machinery is new: the states
% are integrated out with a sparse precision-matrix likelihood
% (Chan-Jeliazkov), and the parameters are sampled by sequential Monte
% Carlo (SMC) with adaptive tempering rather than a single
% Metropolis-Hastings chain. Two additions were mandated by the project
% brief: horseshoe priors that let the data decide which innovation
% cross-correlations exist, and (in place of the unavailable Consensus
% surveys) the constructed inflation-expectations series as a direct
% measurement of trend inflation.
%
% Everything below loads from |results/cp7b| -- the final specification,
% reflecting the owner rulings logged in |checkpoints/|: no correlations
% with the output-gap shock, COVID level-shifters sign-restricted
% negative for GDP and unemployment, and the expectations anchor's
% measurement error calibrated at 30 basis points.

% locate the toolbox root from the package (works wherever this script
% is executed from, incl. Live Editor temp copies)
pkg = which('jointstar.estimate');
assert(~isempty(pkg), 'add the jointstar toolbox root to the path first');
root = fileparts(fileparts(pkg));
run = fullfile(root, 'results', 'cp7b');
summary = readtable(fullfile(run, 'posterior_summary.csv'));
states = readtable(fullfile(run, 'smoothed_states.csv'));
valid = readtable(fullfile(run, 'validation_vs_baseline.csv'));
shrink = readtable(fullfile(run, 'hs_shrinkage.csv'));
smclog = readtable(fullfile(run, 'smc_log.csv'));

%% 1. Timing: minutes, not hours
% The baseline Metropolis-Hastings estimation takes hours per run and
% converges poorly (its own results table warns that several posterior
% means are unreliable). The new estimator runs end-to-end -- 2,000
% particles, every parameter including the covariance layer -- in under an
% hour on a 6-core laptop, and the diagonal-covariance variant of
% Checkpoint 4 took under six minutes. A single likelihood evaluation
% costs ~10 ms on one core (target was 50-100 ms), so a 32-worker machine
% comfortably meets the 30-minute production budget.
fprintf('SMC stages: %d;  total mutation wall-clock: %.1f min\n', ...
    height(smclog), sum(smclog.wallclock_s) / 60);
fprintf('Final tempering exponent phi = %.4f (1 = full posterior)\n', ...
    smclog.phi(end));

%% 2. Convergence diagnostics the baseline never had
% Adaptive tempering chooses each step so the effective sample size stays
% at half the cloud; Metropolis acceptance within each stage is held near
% 25% by block proposals with adaptive scaling. A stalling schedule or
% collapsing acceptance rate -- the failure modes of the old chain -- are
% immediately visible here, and did not occur in the final runs.
%
% *Gelman-style multi-seed check -- read this before quoting the theta
% table.* Three fully independent estimations (seeds 42, 7, 101) were
% compared; an R-hat-style statistic (between-run variance of posterior
% means against within-run variance) flags most structural parameters
% (max ~5.3): the posterior sits on long ridges -- rho_U vs the Okun
% loadings, kappas vs error variances within COVID windows, the pre-1984
% break vs sigma_z -- and a single cloud occupies too narrow a footprint
% on them. This is the same pathology that broke the baseline's MCMC;
% SMC exposes it rather than hiding it. Two consequences: (i) the
% quotable parameter table is |results/pooled_posterior.csv| -- the
% equal-weight pool of the three runs, a valid stratified estimator with
% honestly wider intervals; (ii) the LATENT STATES are far more robust
% than theta (end-of-sample r* across seeds: 0.97, 0.63, 0.83 -- well
% inside one another's bands), because points along a ridge imply nearly
% the same trends. Production recipe: run >= 3 seeds and pool (they
% parallelise perfectly); more MH sweeps per stage (MSteps 4-6) tightens
% single-run footprints at proportional cost.
figure('Color', 'w');
subplot(1, 2, 1); plot(smclog.stage, smclog.phi, '-o'); grid on
xlabel('SMC stage'); ylabel('\phi'); title('Tempering schedule')
subplot(1, 2, 2); plot(smclog.stage, smclog.acc_rate, '-o'); grid on
xlabel('SMC stage'); ylabel('MH acceptance'); title('Mutation acceptance')

%% 3. Parameters: where we agree with the baseline, and where we do not
% The table compares the posterior against the baseline's published
% POST-COVID estimates (the COVID variance factors are compared on modes,
% as the baseline's own notes advise). Two flags per row: whether the 90%
% intervals overlap, and the brief's "within a factor of two" test.
disp(valid)
fprintf('CI overlap: %d/%d;  within factor of 2: %d/%d\n', ...
    nnz(valid.ci_overlap), height(valid), ...
    nnz(valid.within_factor2), height(valid));
% Agreements: the Phillips-curve expectations weight, the participation
% and hours blocks, and most COVID variance factors -- including the ones
% the baseline itself estimated poorly, where the hierarchical prior
% pulls implausible extremes (e.g. kappa_u 2020 of 16) toward their
% window means.
%
% Systematic disagreements, with our reading:
%
% * *Gap dynamics.* The baseline finds a hump-shaped AR(2)
%   (1.65, -0.69); we find similar total persistence (sum ~ 0.9 vs 0.96)
%   but without the hump. The likelihood appears ridged along this
%   direction -- under the paper's own prior parameterisation (Beta on
%   the sum of lags), the data pulls toward first-order dynamics. Worth
%   checking whether the baseline's hump owes to its prior or its chain.
% * *Unemployment-gap persistence.* rho_U near 0.8 here vs 0.20 in the
%   baseline, with the Okun loading sum correspondingly smaller. These
%   two parameters trade off (both produce serially correlated
%   unemployment gaps); the joint fit is similar. A tighter prior on
%   rho_U would recover the baseline split if comparability matters.
% * *Phillips slope.* gamma_2 near -0.16 vs -0.09 (intervals close but
%   not overlapping). Under an unrestricted prior our estimate matched
%   the baseline (-0.08); the paper's sign-restricted Beta prior moves
%   mass toward steeper slopes. This is prior sensitivity on a weakly
%   identified parameter, documented in CHECKPOINT_04/05.

%% 4. Latent states: r*, NAIRU, output gap
% Baseline references (from the published trend charts): the output gap
% troughs near -7.5% in 2020; the NAIRU declines to roughly 4.5-5% by
% 2023; the real neutral rate ends around 1% with wide model-range
% uncertainty.
last = height(states);
fprintf('End of sample (%s):\n', string(states.date(last)));
fprintf('  r*    %.2f [%.2f, %.2f]\n', states.rstar_med(last), ...
    states.rstar_q05(last), states.rstar_q95(last));
fprintf('  NAIRU %.2f [%.2f, %.2f]\n', states.nairu_med(last), ...
    states.nairu_q05(last), states.nairu_q95(last));
fprintf('  gap   %.2f [%.2f, %.2f]\n', states.gap_med(last), ...
    states.gap_q05(last), states.gap_q95(last));
figure('Color', 'w');
tl = tiledlayout(3, 1);
panels = {'rstar', 'Real neutral rate r* (%)'; ...
    'nairu', 'NAIRU (pp)'; 'gap', 'Output gap (%)'};
for i = 1:3
    nexttile;
    lo = states.([panels{i, 1} '_q05']); hi = states.([panels{i, 1} '_q95']);
    md = states.([panels{i, 1} '_med']);
    fill([states.date; flipud(states.date)], [lo; flipud(hi)], ...
        [0.78 0.86 1], 'EdgeColor', 'none'); hold on
    plot(states.date, md, 'b', 'LineWidth', 1.2); grid on
    title(panels{i, 2});
end
title(tl, 'Smoothed latent states, median and 90% band');
% The NAIRU path matches the baseline shape (high-7s in the early 1990s
% declining through the sample), ending modestly above the baseline's.
% r* ends at ~1% -- on top of the baseline's current estimate -- after
% the Checkpoint 7 rulings (sign-restricted COVID shifters, calibrated
% 30bp anchor error); the 90% band, while narrower than any earlier run,
% remains wide (+/- ~2.5pp): without a neutral-rate proxy series the r*
% level stays the least-identified object in this system
% (CHECKPOINT_07).

%% 5. Which shock correlations does the data actually want?
% A horseshoe prior shrinks each off-diagonal of the innovation-
% covariance Cholesky factors aggressively toward zero unless the data
% insists otherwise -- so surviving entries are evidence, not decoration.
% Off-diagonals whose 90% band excludes zero:
disp(shrink(logical(shrink.band_excludes_zero), :))
% The measurement-error and trend-drift cross blocks are essentially
% empty; the identified correlations sit among trend shocks (e.g.
% productivity with NAIRU and trend inflation, capital trend with its
% drift). Heatmaps: results/cp7b/figures/hs_heatmap_Lq.png / _Lr.png.
% Correlations involving the output-gap shock are excluded by owner
% ruling: freeing them re-explains the cycle as correlated noise and
% collapses the IS channel (CHECKPOINT_05/06) -- a genuine, documented
% identification trade-off rather than a numerical artefact.

%% 6. Failure modes and what we would look at next
% * The gap-AR hump and the rho_U/Okun split are the two places the new
%   posterior systematically disagrees with the baseline. Both look like
%   likelihood ridges where priors decide; neither materially changes
%   the latent trends. If baseline comparability is required, tighter
%   priors on rho_U and the AR(2) shape would close the gap.
% * r* remains wide. The single expectations anchor disciplines the
%   drift-variance pile-up (sigma_g and sigma_xi tightened materially)
%   but not the r* level. A market-implied neutral-rate proxy -- even a
%   rough one -- would do more for r* than any further sampler work.
% * The COVID level-vs-variance channels (phi coefficients vs kappas)
%   absorb each other at the margin; the sign restrictions adopted at
%   Checkpoint 7 resolve this pragmatically, matching the model
%   specification.
