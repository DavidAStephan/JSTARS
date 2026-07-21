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
%     'Priors'      (auto)   defaultPriors('HierKappa', o.HierKappa)
%     'HierKappa'   (false)  Checkpoint 6+: hierarchical COVID-kappa
%                            priors, Gamma(a_g, b_g)*1[kappa>=1] with
%                            window-shared estimated hyperparameters
%     'PieObs'      (true)   Checkpoint 7+: pi_e as a direct measurement
%                            of trend inflation (error calibrated 30bp);
%                            false reproduces the 7-observable model
%     'UseParallel' (auto)   open/use a parpool for the mutation loop
%     'NStateDraws' (500)    posterior state trajectories for the bands
%     'UseEvalCache' (true)  build a jointstar.buildEvalCache once (after
%                            loadData/priors) and thread it through every
%                            likelihood evaluation this run makes,
%                            eliminating theta-independent recomputation
%                            in ModelSpec.jointstar/computeLogLik (see
%                            those files).  Verified bit-for-bit identical
%                            to the pre-existing (uncached) evaluator --
%                            see benchmarks/verifyCacheEquivalence.m and
%                            tests/testEvalCache.m.  Hidden/undocumented
%                            escape hatch: set false only to A/B a run
%                            against the pre-cache code path.
%     'MutationTransform' (false)  mutate in elementwise-transformed
%                            coordinates instead of raw theta (see
%                            jointstar.paramTransform / jointstar.runSMC).
%                            Posterior-invariant, flag-gated; DEFAULT
%                            FALSE reproduces the exact prior code path
%                            and RNG consumption bitwise (see
%                            DESIGN_transformed_kernel.md).
%     'StructuredBlocks' (false)  co-block known ridge/relation parameter
%                            groups instead of a per-parameter randperm
%                            partition (see jointstar.blockAtoms /
%                            jointstar.runSMC).  Posterior-invariant,
%                            flag-gated; DEFAULT FALSE reproduces the
%                            exact prior partition and RNG consumption.
%     'MStepsLadder' (false) late-stage mutation-effort ladder: make the
%                            per-stage MH step count stage-dependent,
%                            M_stage = MSteps for phi<0.7, 2*MSteps for
%                            0.7<=phi<0.95, 3*MSteps for phi>=0.95 (see
%                            jointstar.runSMC).  More mutation effort
%                            where ridge exploration is hardest (late
%                            tempering stages).  DEFAULT FALSE leaves
%                            M_stage == MSteps every stage -- identical
%                            code path and RNG consumption.
%     'GTrendRotation' (false)  reparameterise the r*-trend growth pair
%                            (gzbar, gwbar) as (gtrend_sum, gtrend_split)
%                            = (gzbar+gwbar, gzbar-gwbar), mirroring the
%                            existing (phisum, phi2) gap-AR rotation (see
%                            jointstar.defaultPriors / jointstar.ModelSpec
%                            / jointstar.blockAtoms).  ModelSpec's
%                            ck = 0.025*(gzbar+gwbar)/(1-alpha) means the
%                            model mainly identifies the SUM; this puts
%                            the RW proposal on the identified/
%                            unidentified axes directly.  DEFAULT FALSE
%                            emits the original gzbar/gwbar rows and
%                            reproduces the exact prior code path.
%     'WasteFree'   (false)  waste-free SMC mutation (Dau & Chopin 2022,
%                            arXiv:2011.02328): instead of resampling
%                            NParticles ancestors and running each through
%                            MSteps_eff sweeps keeping only the endpoint,
%                            resample a fixed M = round(0.25*NParticles)
%                            chains every stage and advance each chain
%                            P_stage-1 composite sweeps of the identical
%                            stage-invariant kernel, KEEPING every visited
%                            state; the cloud becomes M*P_stage particles
%                            with equal weight, importance-reweighted by
%                            the next stage's tempering increment. P_stage
%                            is chosen each stage for eval-budget parity
%                            with the classical path: M*(P_stage-1) ~=
%                            NParticles*MSteps_eff(phi) (see
%                            jointstar.runSMC 'WasteFree' doc for the
%                            exact rule). DEFAULT FALSE reproduces the
%                            exact prior code path and RNG consumption
%                            bitwise -- for a later A/B only, not yet
%                            used by jointstar.production.
%     'RateGapAR'   (false)  design e4d: switch the xi state (r*_t =
%                            4/(1-alpha)*gz_t + xi_t) from a driftless
%                            random walk to a stationary mean-zero AR(1),
%                            A0(xi,xi)=rho_rg (new param, Beta(9.99,1.76),
%                            mean .85 sd .10), P1(xi,xi)=
%                            sig_xi^2/(1-rho_rg^2) (see
%                            jointstar.defaultPriors / jointstar.ModelSpec
%                            / jointstar.blockAtoms, which co-blocks
%                            rho_rg with {phisum, phi2, nu} -- it enters
%                            the same output-gap IS equation).  Removes
%                            the permanent-shock component the RW
%                            contributed to the IS forcing term.  DEFAULT
%                            FALSE => isfield(th,'rho_rg') false =>
%                            A0(xi,xi)=1, P1(xi,xi)=4 exactly as before,
%                            bit-identical.
%     'FixSingletonKappa' (false)  only meaningful when HierKappa=true:
%                            the two single-member COVID-kappa window
%                            groups (kaphpp_2022, kappi_2023) have
%                            hierarchical hypers with ZERO likelihood
%                            curvature (CHECKPOINT_14 -- they enter only
%                            priorLogPdf, never ModelSpec).  This flag
%                            drops their 4 hyper dims (theta 79 -> 75
%                            under HierKappa) and gives each singleton
%                            kappa a fixed truncated-Gamma(2.0, 1.25)
%                            prior instead -- the CONDITIONAL prior at
%                            the hypers' own prior means (a-priori
%                            calibration only; see
%                            jointstar.defaultPriors).  The multi-member
%                            kappa groups are untouched.  DEFAULT FALSE
%                            (or HierKappa=false) reproduces today's
%                            add() sequence and P.kap byte-for-byte.
%     'PoolAcuteOnly' (false)  only meaningful when HierKappa=true;
%                            INDEPENDENT of FixSingletonKappa (own
%                            option, own inputParser). Keeps the kappa
%                            hierarchy ONLY for g1 = w2020 (the 4-member
%                            acute window {kapy_20, kapu_20, kappr_20,
%                            kapk_20}) and collapses ALL other groups --
%                            g2(w2021), g3(w2122), g4(w2021tot),
%                            g5(w2022tot), g6(w2023tot) -- to the same
%                            fixed truncated-Gamma(2.0, 1.25) a-priori
%                            calibration FixSingletonKappa already uses
%                            for its two singletons (see
%                            jointstar.defaultPriors). Drops 10 hyper
%                            dims (theta 79 -> 69 under HierKappa); the
%                            8 re-typed kappas remain free parameters in
%                            mutateIdx. When both FixSingletonKappa and
%                            PoolAcuteOnly are true, PoolAcuteOnly
%                            dominates (it is a strict superset).
%                            DEFAULT FALSE (or HierKappa=false)
%                            reproduces today's add() sequence and
%                            P.kap byte-for-byte.
%     'CovidDecay'  (false)  only meaningful when HierKappa=true;
%                            MUTUALLY EXCLUSIVE with PoolAcuteOnly and
%                            FixSingletonKappa (errors if combined --
%                            all three are alternative kappa
%                            reparameterizations of the same window
%                            groups). Replaces the 8 COVID kappas of the
%                            acute two-step groups g1(w2020), g2(w2021),
%                            g3(w2122) -- kapy_20/kapy_21 (row1 GDP),
%                            kapu_20/kapu_2122 (row4 U), kappr_20/
%                            kappr_2122 (row5 participation), kapk_20/
%                            kapk_21 (row7 capital) -- with a parametric
%                            geometric-decay SD multiplier
%                            (Lenza-Primiceri style): kr(row,t) = 1 +
%                            s0_row*rho_c^(t-t0), t0 = 2020Q2, over each
%                            row's original union window, kr = 1 outside
%                            (see jointstar.ModelSpec.jointstar). NEW
%                            PARAMS (5): s0_y, s0_u, s0_pr, s0_k ('logn',
%                            peak-excess SD amplitude per row) and a
%                            single shared decay rate rho_c ('beta') --
%                            see jointstar.defaultPriors for the a-priori
%                            calibration. The kept groups g4(w2021tot =
%                            {kapc_2021, kappop_2021}), g5(w2022tot =
%                            {kaphpp_2022}), g6(w2023tot = {kappi_2023})
%                            are unchanged. Parameter count 79 -> 70
%                            under HierKappa (79 - 8 kappas - 6 hypers +
%                            5 decay). DEFAULT FALSE (or HierKappa=false)
%                            reproduces today's add() sequence and
%                            P.kap byte-for-byte.
%     'DropPhiY'    (false)  a nested-restriction hypothesis test of the
%                            COVID level shifter on GDP (phiy).  When
%                            DropPhiY=false (default), phiy is sign-
%                            restricted <0 via truncated normal N(0,0.10)
%                            on (-Inf,0).  When DropPhiY=true, phiy is
%                            fixed at 0 ('fixed' prior type, excluded from
%                            mutateIdx), dropping the GDP COVID intercept
%                            term entirely from d_t.  This tests whether
%                            the stringency loading is redundant GIVEN kappa
%                            (the d_t/kappa competition on the 2020 GDP
%                            drop).  This is a NESTED RESTRICTION (H0:
%                            phiy=0 within H1: phiy<0), not a sign-flip
%                            reversal.  Composes cleanly with all other
%                            flags (orthogonal); phiu's sign restriction is
%                            unchanged.  Parameter count 79 -> 78 (free
%                            params 78 -> 77) when flag is true.  DEFAULT
%                            FALSE reproduces the prior phiy row byte-for-
%                            byte (see jointstar.defaultPriors).
%     'ESSTargetFrac' ([])   target ESS fraction for adaptive tempering;
%                            default [] = runSMC's own default 0.5; higher =
%                            smaller phi increments/more stages
%
%   Final-specification call (all owner rulings baked in; diagonal
%   innovation covariance -- see CLAUDE.md "Owner rulings"):
%     jointstar.estimate('data.csv', 'NParticles', 2000, 'Seed', 42, ...
%         'HierKappa', true)
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
ip.addParameter('HierKappa', false);
ip.addParameter('PieObs', true);   % pi_e as trend-inflation measurement (CP7)
ip.addParameter('UseParallel', []);
ip.addParameter('NStateDraws', 500);
ip.addParameter('UseEvalCache', true);
ip.addParameter('MutationTransform', false);
ip.addParameter('StructuredBlocks', false);
ip.addParameter('MStepsLadder', false);
ip.addParameter('WasteFree', false);
ip.addParameter('GTrendRotation', false);
ip.addParameter('RateGapAR', false);
ip.addParameter('FixSingletonKappa', false);
ip.addParameter('PoolAcuteOnly', false);
ip.addParameter('CovidDecay', false);
ip.addParameter('DropPhiY', false);
ip.addParameter('ESSTargetFrac', []);
ip.parse(varargin{:});
o = ip.Results;

rng(o.Seed);
if ~isfolder(o.OutDir), mkdir(o.OutDir); end

logPath = fullfile(o.OutDir, 'smc_log.csv');
if isfile(logPath)
    warning('jointstar:existingLog', ...
        ['OutDir "%s" already contains smc_log.csv; results in that ' ...
         'directory will be overwritten. Concurrent runs must use ' ...
         'distinct OutDirs.'], o.OutDir);
end

dat = jointstar.loadData(dataFile, 'PieObs', o.PieObs);
P = o.Priors;
if isempty(P)
    P = jointstar.defaultPriors('HierKappa', o.HierKappa, ...
        'GTrendRotation', o.GTrendRotation, 'RateGapAR', o.RateGapAR, ...
        'FixSingletonKappa', o.FixSingletonKappa, ...
        'PoolAcuteOnly', o.PoolAcuteOnly, ...
        'CovidDecay', o.CovidDecay, 'DropPhiY', o.DropPhiY);
end

% Run-level cache of everything theta-independent (regime groupings,
% obsMask, P1^{-1}, ...); threaded through every likelihood eval this run
% makes.  Deterministic (a fixed probe theta, no randomness) -- building
% it does not perturb the RNG stream set by rng(o.Seed) above, so a run
% with UseEvalCache true/false is byte-for-byte reproducible modulo the
% likelihood evaluator itself (see benchmarks/verifyCacheEquivalence.m).
cache = [];
if o.UseEvalCache
    cache = jointstar.buildEvalCache(dat, P);
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
    'logLik', @(tv) logLikTheta(P, tv, dat, cache), ...
    'paramNames', {P.names}, ...
    'priors', P);   % needed only by MutationTransform/StructuredBlocks

opts = struct('NParticles', o.NParticles, 'MSteps', o.MSteps, ...
    'Seed', o.Seed, 'LogFile', fullfile(o.OutDir, 'smc_log.csv'), ...
    'SaveDir', o.OutDir, 'UseParallel', usePar, 'Verbose', true, ...
    'MutationTransform', o.MutationTransform, ...
    'StructuredBlocks', o.StructuredBlocks, ...
    'MStepsLadder', o.MStepsLadder, ...
    'WasteFree', o.WasteFree);
% always restrict MH to the mutable columns: 'fixed' (calibrated)
% parameters must never be proposed
if isfield(P, 'mutateIdx')
    opts.MutateIdx = P.mutateIdx;
end
if ~isempty(o.ESSTargetFrac)
    opts.ESSTargetFrac = o.ESSTargetFrac;
end

out = jointstar.runSMC(prob, opts);

summary = jointstar.diagnostics(out, true);
writetable(summary, fullfile(o.OutDir, 'posterior_summary.csv'));

states = smoothedStates(P, out, dat, o.NStateDraws, cache);
writetable(states, fullfile(o.OutDir, 'smoothed_states.csv'));

results = struct('smc', out, 'summary', summary, 'states', states, ...
    'dat', dat, 'priors', P);
fprintf('done: %.1f min wall-clock, total LML = %.2f\n', ...
    out.wallclock / 60, out.lml);
end

function ll = logLikTheta(P, tv, dat, cache)
th = jointstar.thetaStruct(P, tv);
spec = jointstar.ModelSpec.jointstar(th, dat, [], cache);
ll = jointstar.computeLogLik(spec.system(), dat.y, cache);
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

function tbl = smoothedStates(P, out, dat, nDraws, cache)
% posterior bands for the headline latent series: resample particles by
% weight, draw one state trajectory per resampled particle.  Threading
% the run's eval cache through here too (Checkpoint: eval-cache pass)
% gives these ~nDraws (default 500) re-evaluations -- each previously a
% full from-scratch factorisation of a theta whose aux the SMC loop had
% already discarded -- the same fast path as the main mutation loop,
% rather than singling them out for a separate optimisation.
if nargin < 5, cache = []; end
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
    spec = jointstar.ModelSpec.jointstar(th, dat, [], cache);
    [ll, aux] = jointstar.computeLogLik(spec.system(), dat.y, cache);
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
