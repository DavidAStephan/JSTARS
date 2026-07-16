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
%     .MStepsLadder (default false) late-stage mutation-effort ladder:
%                   make the per-stage MH sweep count M stage-dependent
%                   instead of the constant o.MSteps. Using the
%                   tempering exponent phi already reached BEFORE this
%                   stage's mutation (i.e. phi_n, just set by the
%                   adaptive-tempering step above), the effective count
%                   is M_stage = MSteps for phi < 0.7, 2*MSteps for
%                   0.7 <= phi < 0.95, 3*MSteps for phi >= 0.95 -- more
%                   mutation effort exactly where ridge exploration is
%                   hardest (late stages, phi near 1). DEFAULT FALSE
%                   leaves M_stage == o.MSteps every stage, i.e. the
%                   exact prior code path and RNG consumption
%                   (eps3/logu sizes unchanged).
%     .WasteFree    (default false) waste-free SMC mutation (Dau &
%                   Chopin 2022, "Waste-free Sequential Monte Carlo",
%                   arXiv:2011.02328). Classical mutation resamples the
%                   full N-particle cloud, runs each particle through
%                   M_stage sweeps, and keeps only the endpoint -- every
%                   intermediate MH state is discarded. Waste-free
%                   instead resamples a fixed number of ANCESTOR CHAINS
%                   wfM = round(WF_MFRAC * NParticles), WF_MFRAC = 0.25,
%                   computed once at the top of this function (constant
%                   across the whole run), and advances each chain
%                   P_stage - 1 composite sweeps of the EXACT SAME
%                   stage-invariant kernel used below (transformed-space
%                   block MH via jointstar.mhMutate with mut.M == 1 per
%                   call, i.e. one full pass over all blocks == one
%                   chain step), recording the state after every sweep.
%                   The new cloud is wfM * P_stage particles (initial
%                   ancestor state + P_stage-1 recorded sweeps per
%                   chain) with equal (zero log-) weight; the NEXT
%                   stage's tempering increment importance-reweights all
%                   of them, and the standard lml_inc estimator
%                   (logSumExpW below) remains a consistent normalising-
%                   constant estimator under this weighting scheme
%                   (Dau & Chopin, Prop. 1) -- no change needed there.
%                   Resampling to wfM ancestors happens EVERY stage
%                   (there is no ESS-gated skip: this codebase already
%                   mutates every stage regardless of essPost, so
%                   "always resample" is the direct waste-free analogue
%                   of that existing invariant).
%
%                   EVAL-BUDGET PARITY (M/P rule): P_stage is chosen each
%                   stage so total per-chain sweeps wfM*(P_stage-1)
%                   matches the classical path's per-stage mutation
%                   budget NParticles*M_stage(phi) (M_stage already
%                   includes the MStepsLadder multiplier if that flag is
%                   also on): P_stage - 1 = round(M_stage / WF_MFRAC),
%                   P_stage = that + 1. Substituting wfM =
%                   WF_MFRAC*NParticles gives wfM*(P_stage-1) ==
%                   NParticles*M_stage exactly (up to rounding).
%                   Example (NParticles=2000, MSteps=2, WF_MFRAC=0.25):
%                   M_stage=2 -> wfM=500, P_stage=9 (500*8=4000 sweeps
%                   == 2000*2, cloud 4500); with MStepsLadder engaged,
%                   M_stage=4 -> P_stage=17 (cloud 8500), M_stage=6 ->
%                   P_stage=25 (cloud 12500). The cloud therefore GROWS
%                   past NParticles -- downstream logic (ESS target,
%                   snapshot, posterior summary) is written to use the
%                   CURRENT cloud size, not a hardcoded NParticles, so
%                   this is handled generically.
%
%                   The adaptive proposal covariance driving the MH
%                   blocks is built from the FULL current (pre-ancestor-
%                   resample) weighted cloud -- the same formula as the
%                   classical path, just evaluated one step earlier in
%                   the stage (before the ancestor draw) since that
%                   cloud is the more representative covariance estimate
%                   available at this point, mirroring how the classical
%                   path always uses its full N-particle cloud.
%
%                   DEFAULT FALSE reproduces the exact prior code path
%                   and RNG consumption bitwise: the classical resample-
%                   then-mutate-keep-endpoint branch below is untouched.
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

% ---- waste-free SMC setup (default off; see 'WasteFree' doc above for
% the exact M/P eval-budget-parity rule) --------------------------------
WF_MFRAC = 0.25;   % fraction of NParticles kept as persistent chains
wfM = 0;
if o.WasteFree
    wfM = max(2, round(WF_MFRAC * N));
end

if o.Verbose
    fprintf('SMC: N=%d, d=%d, M=%d MH steps, ESS target %.0f%%, seed %d\n', ...
        N, d, o.MSteps, 100 * o.ESSTargetFrac, o.Seed);
    if o.WasteFree
        fprintf('  WasteFree: wfM=%d ancestor chains (WF_MFRAC=%.2f)\n', wfM, WF_MFRAC);
    end
end

while phi < 1 && stage < o.MaxStages
    stage = stage + 1;
    tStage = tic;

    % ---- 1-2. adaptive tempering: find phi_new by bisection ----------
    % Ncur is the CURRENT cloud size: under WasteFree it varies stage to
    % stage (wfM*P_stage from the previous stage's mutation, or N before
    % the first mutation); under the classical path the cloud is always
    % exactly N, so Ncur == N here every stage and this substitution is a
    % bit-identical no-op for the flag-off path.
    Ncur = numel(logw);
    essPre = essOf(logw);
    phiNew = nextPhi(logw, ll, phi, o.ESSTargetFrac * Ncur);
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

    W = normW(logw);
    % mi/covCols/dm are needed by both branches below (adaptive proposal
    % covariance subspace); mStage (this stage's MStepsLadder-scaled
    % sweep count) is also shared, since WasteFree's P_stage is derived
    % from it. Neither depends on WasteFree, so computing them once here
    % changes nothing about either branch's numerics.
    mi = o.MutateIdx;
    if isempty(mi), mi = true(1, d); end
    covCols = find(mi);
    dm = numel(covCols);
    if o.MStepsLadder
        if phi >= 0.95
            mStage = 3 * o.MSteps;
        elseif phi >= 0.7
            mStage = 2 * o.MSteps;
        else
            mStage = o.MSteps;
        end
    else
        mStage = o.MSteps;
    end
    Pstage = NaN;   % only meaningful/logged when o.WasteFree (see below)

    if o.WasteFree
        % ================================================================
        % ---- waste-free resample+mutate (Dau & Chopin 2022) -----------
        % See the 'WasteFree' doc above for the full derivation. Summary:
        % resample wfM ancestor chains EVERY stage (no ESS gate -- this
        % codebase already mutates every stage unconditionally, so
        % "always resample" is the direct analogue), advance each chain
        % P_stage-1 sweeps of the identical stage-invariant kernel below,
        % and keep every visited state. The adaptive proposal covariance
        % is built from the FULL current (pre-ancestor-resample) weighted
        % cloud -- same formula as the classical branch, just evaluated
        % one step earlier (before the ancestor draw), since that cloud
        % is the more representative covariance estimate available here.
        Pm = P(:, covCols);
        if o.MutationTransform
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

        mutStep = struct();
        mutStep.M = 1;   % one composite sweep per chain step (design (iii))
        mutStep.transform = kernelT;
        mutStep.covBlocks = cell(1, nB); mutStep.covLprops = cell(1, nB);
        mutStep.covEpsRows = cell(1, nB);
        for b = 1:nB
            sel = bperm(edgesB(b) + 1:edgesB(b + 1));
            mutStep.covBlocks{b} = covCols(sel);        % absolute columns
            db = numel(sel);
            Sb = Sig(sel, sel) + 1e-12 * eye(db);
            mutStep.covLprops{b} = chol((cScale^2 * 2.38^2 / db) * Sb, 'lower');
            mutStep.covEpsRows{b} = sel;
        end

        % M/P eval-budget-parity rule (see 'WasteFree' doc): P_stage - 1
        % = round(mStage/WF_MFRAC) sweeps per chain, so that
        % wfM*(P_stage-1) == NParticles*mStage (substituting wfM ==
        % WF_MFRAC*NParticles), matching the classical branch's total
        % mutation-sweep budget for this stage.
        Pstage = max(round(mStage / WF_MFRAC) + 1, 2);
        nSweeps = Pstage - 1;

        idxAnc = systematicResample(W, wfM);   % always resample to wfM
        Panc = P(idxAnc, :); lpAnc = lp(idxAnc); llAnc = ll(idxAnc);

        eps3 = randn(dm, nSweeps, wfM);  % pre-generated: parfor-invariant
        logu = log(rand(wfM, nB * nSweeps));
        accN = zeros(wfM, 1);
        logLik = prob.logLik; logPrior = prob.logPrior;

        Ptraj3 = zeros(Pstage, d, wfM);
        lpTraj3 = zeros(Pstage, wfM);
        llTraj3 = zeros(Pstage, wfM);
        if o.UseParallel
            parfor i = 1:wfM
                [Ptraj3(:, :, i), lpTraj3(:, i), llTraj3(:, i), accN(i)] = ...
                    chainTrajectory(Panc(i, :), lpAnc(i), llAnc(i), phi, ...
                    mutStep, nSweeps, nB, eps3(:, :, i), logu(i, :), ...
                    logPrior, logLik, d);
            end
        else
            for i = 1:wfM
                [Ptraj3(:, :, i), lpTraj3(:, i), llTraj3(:, i), accN(i)] = ...
                    chainTrajectory(Panc(i, :), lpAnc(i), llAnc(i), phi, ...
                    mutStep, nSweeps, nB, eps3(:, :, i), logu(i, :), ...
                    logPrior, logLik, d);
            end
        end
        accRate = sum(accN) / (wfM * nSweeps * nB);

        % flatten the (P_stage x d x wfM) chain trajectories into the new
        % (wfM*P_stage) x d cloud -- chain-major order (all P_stage
        % states of chain 1, then chain 2, ...); order is immaterial for
        % an equally-weighted exchangeable cloud
        P = zeros(wfM * Pstage, d);
        lp = zeros(wfM * Pstage, 1);
        ll = zeros(wfM * Pstage, 1);
        for i = 1:wfM
            rows = (i - 1) * Pstage + 1:i * Pstage;
            P(rows, :) = Ptraj3(:, :, i);
            lp(rows) = lpTraj3(:, i);
            ll(rows) = llTraj3(:, i);
        end
        logw = zeros(wfM * Pstage, 1);   % equal weight; next stage reweights
    else
        % ---- 3. systematic resampling ------------------------------------
        % NB: adaptive tempering drives ESS to exactly the target, so the
        % threshold must be inclusive or the weights never reset and the
        % next bisection stalls.
        if essPost <= o.ESSTargetFrac * N + 1e-9
            idx = systematicResample(W);
            P = P(idx, :); lp = lp(idx); ll = ll(idx);
            logw = zeros(N, 1);
            W = ones(N, 1) / N;
        end

        % ---- 4. block MH mutation with adaptive proposal covariance ------
        % (restricted to the MutateIdx subspace)
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
        % ---- late-stage mutation-effort ladder (default off; when off,
        % mStage == o.MSteps every stage -- identical code path/RNG
        % consumption to before this option existed; see 'MStepsLadder'
        % above). phi here is the tempering exponent just reached this
        % stage, BEFORE mutation -- exactly what the design calls for.
        % (mStage itself is computed once, shared with WasteFree, above.)

        mut = struct();
        mut.M = mStage;
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
        nProp = mStage * nB;

        eps3 = randn(dm, mStage, N);  % pre-generated: parfor-invariant
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
    end

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
    if o.WasteFree
        % extra columns only appended in the WasteFree run (schema is
        % consistent across all stages of a single run since o.WasteFree
        % cannot change mid-run); n_cloud lets a smoke test/plot see the
        % cloud-size growth directly instead of inferring it from wfM/
        % wf_p by hand.
        stageRows{end + 1} = [stage, phi, essPre, essPost, accRate, wc, ...
            lmlInc, mStage, numel(ll), wfM, Pstage, pm]; %#ok<AGROW>
    else
        stageRows{end + 1} = [stage, phi, essPre, essPost, accRate, wc, lmlInc, mStage, pm]; %#ok<AGROW>
    end
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

stages = stageTable(stageRows, names, o.WasteFree);
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
    'MutationTransform', false, 'StructuredBlocks', false, ...
    'MStepsLadder', false, 'WasteFree', false);
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

function idx = systematicResample(W, K)
% Systematic resample of W (a length-Ncur weight vector) down to K
% indices; K defaults to numel(W) (the pre-existing behaviour, used by
% the classical resampling step -- passing K explicitly is new, used
% only by WasteFree's always-resample-to-wfM-ancestors step).
if nargin < 2 || isempty(K), K = numel(W); end
edges = cumsum(W);
edges(end) = 1;
u = (rand + (0:K - 1)') / K;
idx = discretize(u, [0; edges]);
end

function [Ptraj, lpTraj, llTraj, accCount] = chainTrajectory(theta0, lp0, ll0, ...
    phi, mutStep, nSweeps, nB, epsChain, loguChain, logPrior, logLik, d)
%CHAINTRAJECTORY One waste-free ancestor chain (Dau & Chopin 2022).
%
%   Runs nSweeps composite MH sweeps of the EXACT stage-invariant kernel
%   (jointstar.mhMutate with mutStep.M == 1, i.e. one pass over all
%   blocks per call -- the same per-sweep kernel the classical branch
%   uses, called mut.M times there vs. once per call here), recording the
%   state after EVERY sweep instead of only the final one. Row 1 is the
%   pre-mutation ancestor state itself, so the trajectory has
%   nSweeps + 1 == P_stage rows total, matching the M/P rule documented
%   on the 'WasteFree' option above.
Pstage = nSweeps + 1;
Ptraj = zeros(Pstage, d);
lpTraj = zeros(Pstage, 1);
llTraj = zeros(Pstage, 1);
Ptraj(1, :) = theta0; lpTraj(1) = lp0; llTraj(1) = ll0;
theta = theta0; lp = lp0; ll = ll0;
accCount = 0;
for j = 1:nSweeps
    u0 = (j - 1) * nB;
    [theta, lp, ll, nAccJ] = jointstar.mhMutate(theta, lp, ll, phi, ...
        mutStep, epsChain(:, j), loguChain(u0 + 1:u0 + nB), logPrior, logLik);
    accCount = accCount + nAccJ;
    Ptraj(j + 1, :) = theta; lpTraj(j + 1) = lp; llTraj(j + 1) = ll;
end
end

function s = logSumExpW(logw, incr)
% log( sum(W .* exp(incr)) ) with W the normalised weights of logw
lw = logw - max(logw);
z = lw + incr;
mz = max(z);
s = mz + log(sum(exp(z - mz))) - log(sum(exp(lw)));
end

function tbl = stageTable(rows, names, wasteFree)
% wasteFree (optional, default false): appends n_cloud/wf_m/wf_p columns
% -- only ever true for a WasteFree run, whose stageRows all carry those
% extra fields consistently (o.WasteFree cannot change mid-run), so the
% schema matches the data for every row.
if nargin < 3, wasteFree = false; end
Mrows = vertcat(rows{:});
base = {'stage', 'phi', 'ess_pre', 'ess_post', 'acc_rate', 'wallclock_s', 'lml_inc', 'm_stage'};
if wasteFree
    base = [base, {'n_cloud', 'wf_m', 'wf_p'}];
end
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
