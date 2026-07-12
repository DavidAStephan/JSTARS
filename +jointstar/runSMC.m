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
%   adapted across stages toward ~25% acceptance, plus optional
%   per-particle scaled diagonal blocks for hierarchical coordinates
%   (ScaledCols/LocalScaleFn).
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
%     .ExtraMutate  @(P, lp, ll, phi) -> [P, lp, ll]; Gibbs-within-SMC
%                   hook applied after MH (e.g. horseshoe hyperparameters
%                   that enter the prior but not the likelihood)
%     .MutateIdx    logical 1 x d mask: columns updated by the MH step
%                   (default all).  Columns outside the mask ride along
%                   unchanged through MH and are the ExtraMutate hook's
%                   responsibility (Gibbs-within-SMC).  The adaptive
%                   proposal covariance is built on the masked subspace.
%     .NBlocks      MH block count (default ceil(dm/40)).  Each MH step
%                   proposes block-by-block over a random partition of
%                   the mutated columns (re-drawn each stage) -- in high
%                   dimension a single joint random walk is rejected
%                   essentially always, freezing the cloud.
%     .ScaleInit    initial proposal scale multiplier c on (2.38^2/d_b)
%                   (default 1).  c adapts across stages toward ~25%
%                   acceptance (halved below 10%, grown above 35%).
%     .ScaledCols   columns proposed with PER-PARTICLE diagonal steps
%                   instead of cloud-covariance steps: step sd =
%                   c * LocalScaleFn(particleRow).  Use for
%                   hierarchically-scaled coordinates (horseshoe L_ij),
%                   where the particle's own conditional prior sd -- not
%                   the cross-particle dispersion -- is the right metric;
%                   cloud-scale proposals on a tight-prior particle are
%                   rejected with probability ~1 and freeze the cloud.
%     .LocalScaleFn @(row) -> 1 x numel(ScaledCols) proposal sds for one
%                   particle (required with ScaledCols).
%     .Verbose      print per-stage line (default true)
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
    % (restricted to the MutateIdx subspace; other columns are the
    % ExtraMutate hook's responsibility)
    mi = o.MutateIdx;
    if isempty(mi), mi = true(1, d); end
    scCols = o.ScaledCols(:)';
    covCols = setdiff(find(mi), scCols);
    dm = numel(covCols);
    Pm = P(:, covCols);
    mu = W' * Pm;
    Pc = Pm - mu;
    Sig = (Pc .* W)' * Pc;
    Sig = (Sig + Sig') / 2 + 1e-10 * eye(dm);

    nB = o.NBlocks;
    if isempty(nB), nB = max(1, ceil(dm / 40)); end
    [bperm, edgesB] = jointstar.blockPartition(dm, covCols, nB);
    mut = struct();
    mut.M = o.MSteps;
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
    % per-particle scaled blocks (horseshoe L's etc.)
    nS = numel(scCols);
    nSB = max(1, ceil(nS / 40)) * (nS > 0);
    sperm = randperm(nS);
    edgesS = round(linspace(0, nS, nSB + 1));
    mut.scBlocks = cell(1, nSB); mut.scPos = cell(1, nSB);
    mut.scEpsRows = cell(1, nSB);
    for b = 1:nSB
        sel = sperm(edgesS(b) + 1:edgesS(b + 1));
        mut.scBlocks{b} = scCols(sel);          % absolute columns
        mut.scPos{b} = sel;                     % positions in LocalScaleFn output
        mut.scEpsRows{b} = dm + sel;
    end
    mut.localScaleFn = o.LocalScaleFn;
    mut.cScale = cScale;
    nProp = o.MSteps * (nB + nSB);

    eps3 = randn(dm + nS, o.MSteps, N);  % pre-generated: parfor-invariant
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

    % ---- optional Gibbs-within-SMC hook (horseshoe etc.) --------------
    if ~isempty(o.ExtraMutate)
        [P, lp, ll] = o.ExtraMutate(P, lp, ll, phi);
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
    'ExtraMutate', [], 'MutateIdx', [], 'NBlocks', [], ...
    'ScaleInit', 1, 'ScaledCols', [], 'LocalScaleFn', [], 'Verbose', true);
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
