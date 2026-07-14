function out = runSMC(prob, opts)
%RUNSMC Sequential Monte Carlo with adaptive likelihood tempering.
%
%   out = jointstar.runSMC(prob, opts)
%
%   Herbst-Schorfheide (2014) style SMC over the fixed parameters theta.
%   The tempered target sequence is
%
%       pi_n(theta) ~ p(theta) * p(y|theta)^phi_n,   0 = phi_0 < ... < phi_N = 1,
%
%   with each stage doing: (1) reweight by p(y|theta)^(phi_n - phi_{n-1});
%   (2) choose phi_n adaptively by 1-D bisection so the post-reweighting
%   effective sample size hits the target; (3) systematic resampling
%   whenever ESS is at or below the target (inclusive, since adaptive
%   tempering lands exactly on it); (4) M sweeps of BLOCK random-walk
%   Metropolis-Hastings per particle -- random partition of the mutated
%   columns each stage, per-block proposal covariance
%   (c^2 * 2.38^2/d_b) * weighted cloud covariance with the scale c
%   adapted across stages toward ~25% acceptance.
%
%   prob (struct):
%     .samplePrior  @(N) -> N x d initial particles
%     .logPrior     @(theta_row) -> scalar log prior density
%     .logLik       @(theta_row) -> scalar log-likelihood (may be -Inf)
%     .paramNames   (optional) 1 x d string/cellstr for logging
%
%   opts (struct, all optional):
%     .NParticles   (default 1000)
%     .MSteps       MH mutation steps per stage (default 4)
%     .ESSTargetFrac target ESS fraction for adaptive tempering (default 0.5)
%     .Seed         RNG seed, logged (default 42)
%     .MaxStages    hard cap (default 200)
%     .LogFile      CSV path for the stage log ('' = no file)
%     .LogAppend    append instead of overwrite (default false)
%     .SaveDir      directory for particle-cloud snapshots ('' = none)
%     .SaveEvery    snapshot cadence in stages (default 5)
%     .UseParallel  parfor over particles (default: pool already open)
%     .MutateIdx    logical 1 x d mask: columns updated by the MH step
%                   (default all).  Columns outside the mask ride along
%                   unchanged through MH (e.g. 'fixed'/calibrated
%                   parameters).  The adaptive proposal covariance is
%                   built on the masked subspace.
%     .NBlocks      MH block count (default ceil(dm/40)).  Each MH step
%                   proposes block-by-block over a random partition of
%                   the mutated columns (re-drawn each stage) -- in high
%                   dimension a single joint random walk is rejected
%                   essentially always, freezing the cloud.
%     .ScaleInit    initial proposal scale multiplier c on (2.38^2/d_b)
%                   (default 1).  c adapts across stages toward ~25%
%                   acceptance (halved below 10%, grown above 35%).
%     .Verbose      print per-stage line (default true)
%     .MutationTransform  (default false) mutate in elementwise-
%                   transformed coordinates eta = T(theta) (log / logit
%                   / shifted-log per prior support; see
%                   jointstar.paramTransform) instead of raw theta.
%                   Posterior-invariant: the MH accept ratio includes the
%                   Jacobian difference (jointstar.mhMutate).  Requires
%                   prob.priors (the jointstar.defaultPriors struct).
%                   DEFAULT FALSE reproduces the exact prior code path
%                   and RNG consumption (DESIGN_transformed_kernel.md).
%     .StructuredBlocks   (default false) partition the mutated columns
%                   into random ATOMS (known ridge/relation parameter
%                   groups, jointstar.blockAtoms) instead of a per-
%                   parameter randperm, so ridge partners are never
%                   split across MH blocks.  Requires prob.priors.
%                   DEFAULT FALSE reproduces the exact prior partition
%                   (jointstar.blockPartition) and RNG consumption.
%
%   out (struct): particles (N x d), logw, weights (normalised), loglik,
%   logprior, stages (table: stage, phi, ESS pre/post, acceptance,
%   wall-clock, lml_inc, parameter means), seed, logZ and lml (identical;
%   cumulative log marginal-likelihood estimate from the tempering
%   identity -- lml is the documented name, logZ kept for compatibility).
%
%   Reproducibility: rng(opts.Seed) at entry; all proposal noise and
%   acceptance uniforms are pre-generated on the client so results are
%   identical whether or not the mutation loop runs under parfor.
%
%   See also jointstar.mhMutate, jointstar.computeLogLik.

arguments
    prob (1, 1) struct
    opts (1, 1) struct = struct()
end

o = withDefaults(opts);
rng(o.Seed);

N = o.NParticles;
P = prob.samplePrior(N);                 % N x d
d = size(P, 2);
names = paramNames(prob, d);

lp = rowApply(prob.logPrior, P);
ll = evalLikelihoods(prob.logLik, P, o.UseParallel);

logw = zeros(N, 1);
phi = 0;
stage = 0;
logZ = 0;
cScale = o.ScaleInit;
stageRows = {};
tStart = tic;

% ---- transformed-kernel / structured-blocking setup (both default off,
% in which case this whole block is inert and nothing below it runs;
% see 'MutationTransform' / 'StructuredBlocks' above) -------------------
kernelT = [];
kernelAtoms = {};
if o.MutationTransform || o.StructuredBlocks
    if ~isfield(prob, 'priors') || isempty(prob.priors)
        error('jointstar:missingPriors', ...
            ['runSMC: MutationTransform/StructuredBlocks require ' ...
             'prob.priors (the jointstar.defaultPriors struct) to ' ...
             'build the per-parameter transform/atom spec.']);
    end
    kernelP = prob.priors;
    mi0 = o.MutateIdx;
    if isempty(mi0), mi0 = true(1, d); end
    if o.MutationTransform
        kernelT = jointstar.paramTransform(kernelP, mi0);
    end
    if o.StructuredBlocks
        kernelAtoms = jointstar.blockAtoms(kernelP, find(mi0));
    end
end

if o.Verbose
    fprintf('SMC: N=%d, d=%d, M=%d MH steps, ESS target %.0f%%, seed %d\n', ...
        N, d, o.MSteps, 100 * o.ESSTargetFrac, o.Seed);
end

while phi < 1 && stage < o.MaxStages
    stage = stage + 1;
    tStage = tic;

    % ---- 1-2. adaptive tempering: find phi_new by bisection ----------
    essPre = essOf(logw);
    phiNew = nextPhi(logw, ll, phi, o.ESSTargetFrac * N);
    incr = (phiNew - phi) * ll;
    incr(isnan(incr)) = -Inf;
    % log-marginal-likelihood increment before renormalising: log of the
    % weighted mean of exp((phi_t - phi_{t-1}) * loglik) under the
    % pre-stage normalised weights, log-sum-exp stabilised
    lmlInc = logSumExpW(logw, incr);
    logZ = logZ + lmlInc;
    logw = logw + incr;
    phi = phiNew;
    essPost = essOf(logw);

    % ---- 3. systematic resampling ------------------------------------
    % NB: adaptive tempering drives ESS to exactly the target, so the
    % threshold must be inclusive or the weights never reset and the
    % next bisection stalls.
    W = normW(logw);
    if essPost <= o.ESSTargetFrac * N + 1e-9
        idx = systematicResample(W);
        P = P(idx, :); lp = lp(idx); ll = ll(idx);
        logw = zeros(N, 1);
        W = ones(N, 1) / N;
    end

    % ---- 4. block MH mutation with adaptive proposal covariance ------
    % (restricted to the MutateIdx subspace)
    mi = o.MutateIdx;
    if isempty(mi), mi = true(1, d); end
    covCols = find(mi);
    dm = numel(covCols);
    Pm = P(:, covCols);
    if o.MutationTransform
        % transform FULL-WIDTH first, then restrict: kernelT's masks
        % (cols/kind/lo/hi) index full-width theta columns, so handing
        % it the restricted N x dm matrix would silently misalign every
        % column past the first 'fixed' parameter (paramTransform now
        % also asserts on input width, making any such call loud)
        Em = kernelT.toEta(P);
        Em = Em(:, covCols);
        mu = W' * Em;
        Pc = Em - mu;
    else
        mu = W' * Pm;
        Pc = Pm - mu;
    end
    Sig = (Pc .* W)' * Pc;
    Sig = (Sig + Sig') / 2 + 1e-10 * eye(dm);

    nB = o.NBlocks;
    if isempty(nB), nB = max(1, ceil(dm / 40)); end
    if o.StructuredBlocks
        [bperm, edgesB] = jointstar.blockPartitionAtoms(kernelAtoms, nB);
    else
        [bperm, edgesB] = jointstar.blockPartition(dm, covCols, nB);
    end
    mut = struct();
    mut.M = o.MSteps;
    mut.transform = kernelT;
    mut.covBlocks = cell(1, nB); mut.covLprops = cell(1, nB);
    mut.covEpsRows = cell(1, nB);
    for b = 1:nB
        sel = bperm(edgesB(b) + 1:edgesB(b + 1));
        mut.covBlocks{b} = covCols(sel);        % absolute columns
        db = numel(sel);
        Sb = Sig(sel, sel) + 1e-12 * eye(db);
        mut.covLprops{b} = chol((cScale^2 * 2.38^2 / db) * Sb, 'lower');
        mut.covEpsRows{b} = sel;
    end
    nProp = o.MSteps * nB;

    eps3 = randn(dm, o.MSteps, N);  % pre-generated: parfor-invariant
    logu = log(rand(N, nProp));
    accN = zeros(N, 1);
    logLik = prob.logLik; logPrior = prob.logPrior;
    if o.UseParallel
        parfor i = 1:N
            [P(i, :), lp(i), ll(i), accN(i)] = jointstar.mhMutate( ...
                P(i, :), lp(i), ll(i), phi, mut, ...
                eps3(:, :, i), logu(i, :), logPrior, logLik);
        end
    else
        for i = 1:N
            [P(i, :), lp(i), ll(i), accN(i)] = jointstar.mhMutate( ...
                P(i, :), lp(i), ll(i), phi, mut, ...
                eps3(:, :, i), logu(i, :), logPrior, logLik);
        end
    end
    accRate = mean(accN) / nProp;

    % acceptance-adaptive proposal scale for the next stage
    if accRate < 0.10
        cScale = max(cScale * 0.5, 1e-3);
    elseif accRate < 0.20
        cScale = max(cScale * 0.8, 1e-3);
    elseif accRate > 0.35
        cScale = min(cScale * 1.25, 3);
    end

    % ---- diagnostics ---------------------------------------------------
    wc = toc(tStage);
    W = normW(logw);
    pm = W' * P;
    stageRows{end + 1} = [stage, phi, essPre, essPost, accRate, wc, lmlInc, pm]; %#ok<AGROW>
    if o.Verbose
        fprintf('  stage %3d  phi=%.5f  ESS %6.0f->%6.0f  acc=%.2f  %.1fs\n', ...
            stage, phi, essPre, essPost, accRate, wc);
    end
    if ~isempty(o.SaveDir) && mod(stage, o.SaveEvery) == 0
        snapshot(o.SaveDir, stage, P, logw, ll, lp, phi, o.Seed);
    end
end

if phi < 1
    error('jointstar:smcMaxStages', ...
        'SMC hit MaxStages=%d at phi=%.4f < 1.', o.MaxStages, phi);
end

stages = stageTable(stageRows, names);
if ~isempty(o.LogFile)
    writeLog(o.LogFile, stages, o.LogAppend);
end
if ~isempty(o.SaveDir)
    snapshot(o.SaveDir, stage, P, logw, ll, lp, phi, o.Seed);
end

wallclock = toc(tStart);
if o.Verbose
    fprintf('SMC done: %d stages, %.1f s wall-clock, total LML = %.4f\n', ...
        stage, wallclock, logZ);
end

out = struct('particles', P, 'logw', logw, 'weights', normW(logw), ...
    'loglik', ll, 'logprior', lp, 'stages', stages, 'seed', o.Seed, ...
    'logZ', logZ, 'lml', logZ, 'wallclock', wallclock, 'paramNames', {names});
end

% ======================================================================
function o = withDefaults(opts)
def = struct('NParticles', 1000, 'MSteps', 4, 'ESSTargetFrac', 0.5, ...
    'Seed', 42, 'MaxStages', 200, 'LogFile', '', 'LogAppend', false, ...
    'SaveDir', '', 'SaveEvery', 5, 'UseParallel', [], ...
    'MutateIdx', [], 'NBlocks', [], ...
    'ScaleInit', 1, 'Verbose', true, ...
    'MutationTransform', false, 'StructuredBlocks', false);
o = def;
fn = fieldnames(opts);
for k = 1:numel(fn), o.(fn{k}) = opts.(fn{k}); end
if isempty(o.UseParallel)
    o.UseParallel = ~isempty(gcpNoCreate());
end
end

function p = gcpNoCreate()
p = [];
try %#ok<TRYNC>  PCT may not be installed
    p = gcp('nocreate');
end
end

function names = paramNames(prob, d)
if isfield(prob, 'paramNames') && ~isempty(prob.paramNames)
    names = cellstr(prob.paramNames);
else
    names = arrayfun(@(k) sprintf('theta%d', k), 1:d, 'UniformOutput', false);
end
end

function v = rowApply(f, P)
N = size(P, 1);
v = zeros(N, 1);
for i = 1:N, v(i) = f(P(i, :)); end
end

function ll = evalLikelihoods(logLik, P, useParallel)
N = size(P, 1);
ll = zeros(N, 1);
if useParallel
    parfor i = 1:N, ll(i) = logLik(P(i, :)); end
else
    for i = 1:N, ll(i) = logLik(P(i, :)); end
end
ll(isnan(ll)) = -Inf;
end

function e = essOf(logw)
W = normW(logw);
e = 1 / sum(W.^2);
end

function W = normW(logw)
w = exp(logw - max(logw));
W = w / sum(w);
end

function phiNew = nextPhi(logw, ll, phiOld, essTarget)
% largest phi in (phiOld, 1] whose post-reweight ESS >= essTarget
f = @(ph) essOf(logw + (ph - phiOld) * llClean(ll)) - essTarget;
if f(1) >= 0
    phiNew = 1;
    return
end
lo = phiOld; hi = 1;
for k = 1:60
    mid = (lo + hi) / 2;
    if f(mid) >= 0, lo = mid; else, hi = mid; end
end
phiNew = lo;
if phiNew <= phiOld
    % likelihood dispersion so extreme that no representable increment
    % keeps ESS at target: force a minimal step and let resampling +
    % mutation absorb the hit rather than aborting the run
    phiNew = min(1, phiOld + max(1e-6 * (1 - phiOld), 10 * eps(phiOld)));
    warning('jointstar:temperingForced', ...
        'Tempering forced past phi=%.3g (ESS will undershoot target).', phiOld);
end
end

function ll = llClean(ll)
ll(~isfinite(ll)) = -1e300;   % -Inf * 0 protection inside bisection
end

function idx = systematicResample(W)
N = numel(W);
edges = cumsum(W);
edges(end) = 1;
u = (rand + (0:N - 1)') / N;
idx = discretize(u, [0; edges]);
end

function s = logSumExpW(logw, incr)
% log( sum(W .* exp(incr)) ) with W the normalised weights of logw
lw = logw - max(logw);
z = lw + incr;
mz = max(z);
s = mz + log(sum(exp(z - mz))) - log(sum(exp(lw)));
end

function tbl = stageTable(rows, names)
Mrows = vertcat(rows{:});
base = {'stage', 'phi', 'ess_pre', 'ess_post', 'acc_rate', 'wallclock_s', 'lml_inc'};
cols = [base, strcat('mean_', names)];
tbl = array2table(Mrows, 'VariableNames', cols);
end

function writeLog(fname, tbl, doAppend)
if doAppend && isfile(fname)
    writetable(tbl, fname, 'WriteMode', 'append');
else
    writetable(tbl, fname);
end
end

function snapshot(dirName, stage, P, logw, ll, lp, phi, seed) %#ok<INUSD>
if ~isfolder(dirName), mkdir(dirName); end
fname = fullfile(dirName, sprintf('particles_stage_%02d.mat', stage));
save(fname, 'P', 'logw', 'll', 'lp', 'phi', 'seed', '-v7.3');
end
