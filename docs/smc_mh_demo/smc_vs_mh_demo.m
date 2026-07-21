function smc_vs_mh_demo()
%SMC_VS_MH_DEMO  Side-by-side demo: random-walk Metropolis-Hastings (MH)
%   versus tempered Sequential Monte Carlo (SMC) on a small Bayesian model
%   whose posterior has an awkward CURVED, BIMODAL RIDGE -- the kind of
%   geometry that makes a single MCMC chain mix slowly, and that tempered
%   SMC is built to handle. This is a teaching companion to the JointSTAR
%   estimator (which lives on a long connected ridge of its own).
%
%   THE MODEL (deliberately tiny, fully Bayesian):
%     We observe n noisy measurements of the PRODUCT of two parameters,
%         y_i = theta1 * theta2 + noise,     noise ~ N(0, sigma^2),
%     with independent priors  theta1, theta2 ~ N(0, s0^2).
%   Only the product theta1*theta2 is well identified, so the posterior
%   concentrates on a curved RIDGE (a hyperbola theta1*theta2 = const).
%   And because (theta1,theta2) and (-theta1,-theta2) give the SAME
%   product, there are TWO symmetric ridges (modes) separated by a
%   low-density barrier near the axes. A compact stand-in for the weak
%   identification / ridge geometry of real macro "stars" models.
%
%   WHAT YOU SEE:
%     * MH sends ONE walker across the posterior. Started in one arm, it
%       crawls slowly ALONG the curved ridge and essentially never crosses
%       the barrier to the other arm -- it reports only ONE of the two
%       equally-valid answers.
%     * SMC starts a whole CLOUD of particles at the (broad, symmetric)
%       prior and gradually turns up the data's influence (phi: 0 -> 1).
%       At high temperature the barrier is invisible, so the cloud spans
%       both arms; as it cools it settles onto BOTH ridges at once -- the
%       honest, symmetric posterior.
%
%   OUTPUTS (written next to this file):
%     * a live side-by-side animation while it runs,
%     * smc_vs_mh_animation.gif   (drop into slides),
%     * smc_vs_mh_summary.png     (final samples + theta1 marginals).
%
%   Base MATLAB only -- no toolboxes. Tested on R2026a.
%
%   Run:  >> smc_vs_mh_demo

rng(7);                                   % reproducible
outdir = fileparts(mfilename('fullpath'));

% ---------------- 1. Model and synthetic data --------------------------
truth = [2.0; 1.5];                       % true (theta1, theta2); product 3
n     = 25;                               % number of observations
sigma = 1.2;                              % measurement-noise sd
s0    = 3.0;                              % prior sd on each parameter
y     = truth(1) * truth(2) + sigma * randn(n, 1);
Sy    = sum(y);  Syy = sum(y .^ 2);       % sufficient stats for a fast likelihood

% Unnormalised log-posterior pieces. Each takes a 2 x M batch of parameter
% columns and returns a 1 x M row -- fully vectorised over particles.
predSSE  = @(TH) Syy - 2 * (TH(1,:) .* TH(2,:)) * Sy + n * (TH(1,:) .* TH(2,:)) .^ 2;
loglik   = @(TH) -0.5 * predSSE(TH) / sigma^2;
logprior = @(TH) -0.5 * (TH(1,:) .^ 2 + TH(2,:) .^ 2) / s0^2;
logpost  = @(TH) loglik(TH) + logprior(TH);

% ---------------- 2. Run the two samplers ------------------------------
% MH: one long single chain, started deliberately in the (+,+) arm.
nIterMH = 20000;
propSdMH = 0.22;                          % isotropic random-walk step
th0 = [2.5; 2.5];
[chainMH, accMH] = run_mh(logpost, th0, nIterMH, propSdMH);
evalsMH = nIterMH;                        % one likelihood eval per iteration

% SMC: a cloud annealed from the prior to the posterior.
Nsmc = 1200; essFrac = 0.90; nMut = 6;
[snapsSMC, phiSMC, evalsSMC] = run_smc(loglik, logprior, Nsmc, s0, essFrac, nMut);
partsSMC = snapsSMC{end};

fprintf('\n--- results -------------------------------------------------\n');
fprintf('MH : %d iterations, accept rate %.2f, %d likelihood evals\n', ...
    nIterMH, accMH, evalsMH);
fprintf('SMC: %d particles, %d tempering stages, %d likelihood evals\n', ...
    Nsmc, numel(phiSMC) - 1, evalsSMC);
burn = round(nIterMH / 5);
mhKeep  = chainMH(:, burn+1:end);
fracPosMH  = mean(mhKeep(1,:)  > 0);
fracPosSMC = mean(partsSMC(1,:) > 0);
fprintf('Share of samples in the (+,+) arm:   MH %.0f%%   SMC %.0f%%\n', ...
    100*fracPosMH, 100*fracPosSMC);
fprintf('  (truth has a twin at %.1f,%.1f; the honest posterior is ~50/50)\n', ...
    -truth(1), -truth(2));
fprintf('Estimated product theta1*theta2:  MH %.2f   SMC %.2f   (true 3.00)\n', ...
    mean(mhKeep(1,:).*mhKeep(2,:)), mean(partsSMC(1,:).*partsSMC(2,:)));

% ---------------- 3. Contour background --------------------------------
gv = linspace(-6, 6, 240);
[GX, GY] = meshgrid(gv, gv);
GZ = reshape(logpost([GX(:)'; GY(:)']), size(GX));
GP = exp(GZ - max(GZ(:)));                % normalised posterior height
levels = (0.03:0.12:1) * max(GP(:));
bgCmap = makeRamp([0.99 0.99 0.97], [0.62 0.74 0.90]);   % light -> soft blue
mhCol  = [0.92 0.41 0.20];               % orange
smcCol = [0.16 0.47 0.84];               % blue

% ---------------- 4. Live animation + GIF ------------------------------
gifFile = fullfile(outdir, 'smc_vs_mh_animation.gif');
F = 64;
mhIdx    = round(linspace(2, nIterMH, F));
smcStage = round(linspace(1, numel(snapsSMC), F));

fig = figure('Color', 'w', 'Position', [80 80 1080 560]);
tmp = tempname; mkdir(tmp); frames = strings(1, F);
for f = 1:F
    clf(fig);
    tl = tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    title(tl, 'Sampling a curved, bimodal posterior:  one walker vs. a cooling crowd', ...
        'FontWeight', 'bold', 'FontSize', 13);

    % --- left: MH ---
    ax1 = nexttile(tl); drawBg(ax1, gv, GP, levels, bgCmap); hold(ax1, 'on');
    k = mhIdx(f);
    plot(ax1, chainMH(1, 1:k), chainMH(2, 1:k), '-', 'Color', [mhCol 0.5], 'LineWidth', 0.7);
    plot(ax1, chainMH(1, k), chainMH(2, k), 'o', 'MarkerFaceColor', mhCol, ...
        'MarkerEdgeColor', 'w', 'MarkerSize', 9, 'LineWidth', 1);
    finishAx(ax1, sprintf('MCMC / Metropolis-Hastings    \\rm one chain  \\bullet  iteration %d', k), mhCol, truth);

    % --- right: SMC ---
    ax2 = nexttile(tl); drawBg(ax2, gv, GP, levels, bgCmap); hold(ax2, 'on');
    s = smcStage(f); P = snapsSMC{s};
    scatter(ax2, P(1,:), P(2,:), 12, smcCol, 'filled', 'MarkerFaceAlpha', 0.45);
    finishAx(ax2, sprintf('Tempered SMC    \\rm %d particles  \\bullet  \\phi = %.2f', size(P,2), phiSMC(s)), smcCol, truth);

    drawnow;
    frames(f) = fullfile(tmp, sprintf('f%03d.png', f));
    exportgraphics(fig, frames(f), 'Resolution', 108);
end

% assemble GIF from the rendered frames (robust with or without a display)
for f = 1:F
    [A, map] = rgb2ind(imread(frames(f)), 256);
    if f == 1
        imwrite(A, map, gifFile, 'gif', 'LoopCount', Inf, 'DelayTime', 0.09);
    else
        dt = 0.09; if f == F, dt = 1.4; end     % hold the final frame
        imwrite(A, map, gifFile, 'gif', 'WriteMode', 'append', 'DelayTime', dt);
    end
end
fprintf('\nWrote %s\n', gifFile);

% ---------------- 5. Static summary figure -----------------------------
sfig = figure('Color', 'w', 'Position', [80 80 1080 760]);
tl = tiledlayout(sfig, 2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

axA = nexttile(tl); drawBg(axA, gv, GP, levels, bgCmap); hold(axA, 'on');
plot(axA, truth(1), truth(2), 'p', 'MarkerFaceColor', [0.1 0.1 0.1], 'MarkerEdgeColor','w','MarkerSize',13);
plot(axA, -truth(1), -truth(2), 'p', 'MarkerFaceColor', [0.5 0.5 0.5], 'MarkerEdgeColor','w','MarkerSize',13);
finishAx(axA, 'The posterior: two curved ridges    \rm (black star = truth, grey = its twin)', [0.2 0.2 0.2], []);

axB = nexttile(tl); drawBg(axB, gv, GP, levels, bgCmap); hold(axB, 'on');
thin = mhKeep(:, 1:8:end);
scatter(axB, thin(1,:), thin(2,:), 10, mhCol, 'filled', 'MarkerFaceAlpha', 0.35);
finishAx(axB, sprintf('MH samples: ONE arm found\\rm%.0f%% of mass in a single mode', 100*fracPosMH), mhCol, truth);

axC = nexttile(tl); drawBg(axC, gv, GP, levels, bgCmap); hold(axC, 'on');
scatter(axC, partsSMC(1,:), partsSMC(2,:), 10, smcCol, 'filled', 'MarkerFaceAlpha', 0.35);
finishAx(axC, sprintf('SMC particles: BOTH arms\\rm%.0f%% / %.0f%% across the two modes', 100*fracPosSMC, 100*(1-fracPosSMC)), smcCol, truth);

axD = nexttile(tl); hold(axD, 'on');
edges = -6:0.35:6;
histogram(axD, mhKeep(1,:),  edges, 'Normalization','pdf', 'FaceColor', mhCol,  'EdgeColor','none', 'FaceAlpha', 0.55);
histogram(axD, partsSMC(1,:), edges, 'Normalization','pdf', 'FaceColor', smcCol, 'EdgeColor','none', 'FaceAlpha', 0.55);
xline(axD, 0, ':', 'Color', [0.5 0.5 0.5]);
title(axD, 'Marginal posterior of \theta_1', 'FontWeight','bold');
xlabel(axD, '\theta_1'); ylabel(axD, 'density');
legend(axD, {'MH (one-sided)', 'SMC (symmetric)'}, 'Box','off', 'Location','north');
set(axD, 'FontSize', 11); box(axD, 'on'); axD.XLim = [-6 6];

title(tl, 'Same model, same data, same target posterior -- different exploration', ...
    'FontWeight','bold', 'FontSize', 13);
pngFile = fullfile(outdir, 'smc_vs_mh_summary.png');
exportgraphics(sfig, pngFile, 'Resolution', 150);
fprintf('Wrote %s\n\n', pngFile);
end

% ======================================================================
% Samplers
% ======================================================================
function [chain, accRate] = run_mh(logpost, th0, nIter, propSd)
% Random-walk Metropolis-Hastings: one chain, isotropic Gaussian proposal.
d = numel(th0); chain = zeros(d, nIter);
th = th0(:); lp = logpost(th); nAcc = 0;
for t = 1:nIter
    prop = th + propSd * randn(d, 1);
    lpp  = logpost(prop);
    if log(rand) < lpp - lp
        th = prop; lp = lpp; nAcc = nAcc + 1;
    end
    chain(:, t) = th;
end
accRate = nAcc / nIter;
end

function [snaps, phis, nEval] = run_smc(loglik, logprior, N, s0, essFrac, nMut)
% Adaptive-tempering SMC (the JointSTAR recipe in miniature):
%   draw from the prior; step phi 0 -> 1 by bisection so the reweighted
%   ESS stays at essFrac*N; systematic-resample; then refresh each particle
%   with a few cloud-scaled random-walk MH moves at the current phi.
d = 2;
parts = s0 * randn(d, N);            % exact draws from the prior N(0, s0^2 I)
ll = loglik(parts);
nEval = N;
phi = 0; snaps = {parts}; phis = 0;
while phi < 1
    dphi = next_dphi(ll, essFrac, N);          % adaptive temperature step
    phiNew = min(1, phi + dphi);
    logw = (phiNew - phi) * ll;                % incremental importance weights
    w = exp(logw - max(logw)); w = w / sum(w);
    idx = systematic_resample(w);              % survival of the fittest
    parts = parts(:, idx); ll = ll(idx);
    propSd = max(2.38 / sqrt(d) * std(parts, 0, 2), 1e-3);   % cloud-scaled step
    lpri = logprior(parts);
    for m = 1:nMut                             % local exploration at phiNew
        prop = parts + propSd .* randn(d, N);
        llp  = loglik(prop);  lprip = logprior(prop);
        nEval = nEval + N;
        logacc = phiNew * (llp - ll) + (lprip - lpri);
        acc = log(rand(1, N)) < logacc;
        parts(:, acc) = prop(:, acc); ll(acc) = llp(acc); lpri(acc) = lprip(acc);
    end
    phi = phiNew; snaps{end+1} = parts; phis(end+1) = phi; %#ok<AGROW>
end
end

function dphi = next_dphi(ll, essFrac, N)
% Bisection for the temperature increment that lands ESS on essFrac*N.
target = essFrac * N;
essFun = @(dp) essOf((dp) * (ll - max(ll)));
lo = 0; hi = 1;
if essFun(hi) >= target, dphi = hi; return; end   % full step keeps enough ESS
for it = 1:60
    mid = 0.5 * (lo + hi);
    if essFun(mid) < target, hi = mid; else, lo = mid; end
end
dphi = 0.5 * (lo + hi);
end

function e = essOf(logw)
w = exp(logw - max(logw));
e = sum(w)^2 / sum(w .^ 2);
end

function idx = systematic_resample(w)
% Standard systematic (low-variance) resampling.
N = numel(w); positions = ((0:N-1) + rand) / N;
edges = min(cumsum(w), 1); edges(end) = 1;
idx = zeros(1, N); i = 1; j = 1;
while i <= N
    if positions(i) < edges(j), idx(i) = j; i = i + 1; else, j = j + 1; end
end
end

% ======================================================================
% Small plotting helpers
% ======================================================================
function drawBg(ax, gv, GP, levels, cmap)
contourf(ax, gv, gv, GP, levels, 'LineStyle', 'none');
colormap(ax, cmap);
end

function finishAx(ax, ttl, col, truth)
axis(ax, [-6 6 -6 6]); axis(ax, 'square');
set(ax, 'FontSize', 11, 'Layer', 'top'); box(ax, 'on');
xlabel(ax, '\theta_1'); ylabel(ax, '\theta_2');
title(ax, ttl, 'Color', col, 'FontWeight', 'bold');
if ~isempty(truth)
    plot(ax, truth(1), truth(2), 'p', 'MarkerFaceColor', [0.12 0.12 0.12], ...
        'MarkerEdgeColor', 'w', 'MarkerSize', 11);
end
hold(ax, 'off');
end

function cmap = makeRamp(c0, c1)
t = linspace(0, 1, 256)';
cmap = (1 - t) .* c0 + t .* c1;
end
