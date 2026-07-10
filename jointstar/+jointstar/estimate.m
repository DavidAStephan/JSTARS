function results = estimate(dataFile, varargin)
%ESTIMATE One-call JointSTAR estimation: data in, posterior out.
%
%   results = jointstar.estimate('data.csv', 'NParticles', 2000, 'Seed', 42)
%
%   Pipeline: loadData -> defaultPriors -> runSMC (precision-based
%   likelihood, adaptive tempering) -> posterior summaries and smoothed
%   states written to OutDir.
%
%   Name-value options:
%     'NParticles'  (2000)   SMC particles
%     'MSteps'      (4)      MH mutation steps per stage
%     'Seed'        (42)     RNG seed (logged, reproducible)
%     'OutDir'      ('results')
%     'Priors'      (auto)   defaultPriors(), or horseshoePriors() when
%                            'Horseshoe' is true
%     'Horseshoe'   (false)  Checkpoint 5+: LDL' innovation covariances
%                            with grouped horseshoe priors on the
%                            off-diagonals, Makalic-Schmidt Gibbs on the
%                            scales between SMC stages
%     'HierKappa'   (false)  Checkpoint 6+: hierarchical COVID-kappa
%                            priors, Gamma(a_g, b_g)*1[kappa>=1] with
%                            window-shared estimated hyperparameters
%     'PieObs'      (true)   Checkpoint 7+: pi_e as a direct measurement
%                            of trend inflation (error calibrated 30bp);
%                            false reproduces the 7-observable model
%     'UseParallel' (auto)   open/use a parpool for the mutation loop
%     'NStateDraws' (500)    posterior state trajectories for the bands
%
%   Final-specification call (all owner rulings baked in):
%     jointstar.estimate('data.csv', 'NParticles', 2000, 'Seed', 42, ...
%         'Horseshoe', true, 'HierKappa', true)
%
%   Outputs written to OutDir:
%     smc_log.csv             per-stage tempering diagnostics
%     particles_stage_XX.mat  particle-cloud snapshots (-v7.3, every 5)
%     posterior_summary.csv   mean/sd/5/50/95% per parameter
%     smoothed_states.csv     median + 5%/95% bands for r*, NAIRU, potential
%                             output, output gap, trend inflation
%
%   See also jointstar.runSMC, jointstar.computeLogLik.

ip = inputParser;
ip.addParameter('NParticles', 2000);
ip.addParameter('MSteps', 4);
ip.addParameter('Seed', 42);
ip.addParameter('OutDir', 'results');
ip.addParameter('Priors', []);
ip.addParameter('Horseshoe', false);
ip.addParameter('HierKappa', false);
ip.addParameter('PieObs', true);   % pi_e as trend-inflation measurement (CP7)
ip.addParameter('UseParallel', []);
ip.addParameter('NStateDraws', 500);
ip.parse(varargin{:});
o = ip.Results;

rng(o.Seed);
if ~isfolder(o.OutDir), mkdir(o.OutDir); end

dat = jointstar.loadData(dataFile, 'PieObs', o.PieObs);
P = o.Priors;
if isempty(P)
    if o.Horseshoe
        P = jointstar.horseshoePriors('HierKappa', o.HierKappa);
    else
        P = jointstar.defaultPriors('HierKappa', o.HierKappa);
    end
end

usePar = o.UseParallel;
pool = [];
if isempty(usePar) || usePar
    pool = openPool();
    usePar = ~isempty(pool);
end

% Avoid CPU oversubscription: computeLogLik's sparse Cholesky is itself
% multithreaded, so N parfor workers x M BLAS threads compete for the
% same cores (measured ~2.4x slowdown on 6 cores; worse with more).  Cap
% to 1 BLAS thread per worker FOR THE DURATION of this run only, then
% restore the client's setting on return (or on error) -- leaving the
% whole session single-threaded would silently throttle the user's later
% work.  Skipped if the caller pre-set it (nThreadsOrig already 1).
nThreadsOrig = maxNumCompThreads;
restoreThreads = onCleanup(@() maxNumCompThreads(nThreadsOrig)); %#ok<NASGU>
if usePar
    if isa(pool, 'parallel.ThreadPool')
        maxNumCompThreads(1);          % Threads pool shares the client process
    else
        wait(parfevalOnAll(pool, @maxNumCompThreads, 0, 1));  % Processes: workers
    end
end
fprintf('jointstar.estimate: T=%d quarters, %d parameters, parallel=%d\n', ...
    dat.T, P.d, usePar);

prob = struct( ...
    'samplePrior', @(N) jointstar.priorSample(P, N), ...
    'logPrior', @(tv) jointstar.priorLogPdf(P, tv), ...
    'logLik', @(tv) logLikTheta(P, tv, dat), ...
    'paramNames', {P.names});

opts = struct('NParticles', o.NParticles, 'MSteps', o.MSteps, ...
    'Seed', o.Seed, 'LogFile', fullfile(o.OutDir, 'smc_log.csv'), ...
    'SaveDir', o.OutDir, 'UseParallel', usePar, 'Verbose', true);
% always restrict MH to the mutable columns: 'fixed' (calibrated)
% parameters and Gibbs-owned horseshoe hypers must never be proposed
if isfield(P, 'mutateIdx')
    opts.MutateIdx = P.mutateIdx;
end
if isfield(P, 'hs')
    opts.ExtraMutate = @(Pm, lp, ll, phi) ...
        jointstar.horseshoeMutate(Pm, lp, ll, phi, P);
    % the L off-diagonals need per-particle proposal scales: the
    % particle's own conditional prior sd tau_g * lambda_ij (floored so
    % shrunk coordinates still explore, capped for sanity)
    hsc = P.hs;
    opts.ScaledCols = hsc.colL;
    opts.LocalScaleFn = @(row) min(max(sqrt( ...
        row(hsc.colTau2(hsc.groups)) .* row(hsc.colLam2)), 1e-3), 0.5);
end

out = jointstar.runSMC(prob, opts);

summary = jointstar.diagnostics(out, true);
writetable(summary, fullfile(o.OutDir, 'posterior_summary.csv'));

states = smoothedStates(P, out, dat, o.NStateDraws);
writetable(states, fullfile(o.OutDir, 'smoothed_states.csv'));

results = struct('smc', out, 'summary', summary, 'states', states, ...
    'dat', dat, 'priors', P);
fprintf('done: %.1f min wall-clock, logZ = %.2f\n', out.wallclock / 60, out.logZ);
end

% ======================================================================
function ll = logLikTheta(P, tv, dat)
th = jointstar.thetaStruct(P, tv);
if isfield(P, 'hs')
    [Lq, Lr] = jointstar.hsUnpack(P, tv);
    spec = jointstar.ModelSpec.jointstar(th, dat, struct('Lq', Lq, 'Lr', Lr));
else
    spec = jointstar.ModelSpec.jointstar(th, dat);
end
ll = jointstar.computeLogLik(spec.system(), dat.y);
end

function pool = openPool()
% Open (or reuse) a parallel pool.  Defaults to a Threads pool sized to
% the machine's physical cores; degrades gracefully to serial when the
% Parallel Computing Toolbox is not licensed.  Thread-oversubscription
% capping is handled by the caller (estimate) so it can be restored.
pool = [];
if ~license('test', 'Distrib_Computing_Toolbox'), return; end
try
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool('Threads');
    end
catch err
    warning('jointstar:noPool', 'parpool unavailable (%s); running serial.', ...
        err.message);
    pool = [];
end
end

function tbl = smoothedStates(P, out, dat, nDraws)
% posterior bands for the headline latent series: resample particles by
% weight, draw one state trajectory per resampled particle
[~, ix] = jointstar.ModelSpec.jointstarStates();
N = size(out.particles, 1);
edges = cumsum(out.weights); edges(end) = 1;
u = (rand + (0:nDraws - 1)') / nDraws;
sel = discretize(u, [0; edges]);

T = dat.T;
% NaN-initialised so a skipped draw (infeasible theta, cannot happen at
% phi=1 but guarded anyway) never contaminates the quantiles with zeros
rstar = nan(nDraws, T); nairu = nan(nDraws, T); taustar = nan(nDraws, T);
gap = nan(nDraws, T); trinf = nan(nDraws, T);
for k = 1:nDraws
    tv = out.particles(sel(k), :);
    th = jointstar.thetaStruct(P, tv);
    if isfield(P, 'hs')
        [Lq, Lr] = jointstar.hsUnpack(P, tv);
        spec = jointstar.ModelSpec.jointstar(th, dat, struct('Lq', Lq, 'Lr', Lr));
    else
        spec = jointstar.ModelSpec.jointstar(th, dat);
    end
    [ll, aux] = jointstar.computeLogLik(spec.system(), dat.y);
    if ~isfinite(ll), continue; end
    a = jointstar.drawStates(aux);
    rstar(k, :) = 4 / (1 - th.alpha) * a(ix.gz, :) + a(ix.xi, :);
    nairu(k, :) = a(ix.Ustar, :);
    taustar(k, :) = a(ix.zstar, :) + th.alpha * a(ix.kstar, :) ...
        + (1 - th.alpha) * (a(ix.wstar, :) + a(ix.prstar, :) ...
        + a(ix.hppstar, :) - a(ix.Ustar, :));
    gap(k, :) = a(ix.c, :);
    trinf(k, :) = a(ix.pie, :);
end

q = @(X, p) quantile(X, p, 1)';
tbl = table(dat.dates', 'VariableNames', {'date'});
series = {'rstar', rstar; 'nairu', nairu; 'taustar', taustar; ...
    'gap', gap; 'trend_inflation', trinf};
for s = 1:size(series, 1)
    nm = series{s, 1}; X = series{s, 2};
    tbl.([nm '_q05']) = q(X, 0.05);
    tbl.([nm '_med']) = q(X, 0.50);
    tbl.([nm '_q95']) = q(X, 0.95);
end
end
