function results = verifyTransformInvarianceJoint()
%VERIFYTRANSFORMINVARIANCEJOINT Full-vector (79-parameter) joint prior
%invariance test for the blocked, transformed, cloud-covariance MH
%mutation pathway ('MutationTransform' + 'StructuredBlocks' options).
%
%   results = benchmarks.verifyTransformInvarianceJoint()
%
%   benchmarks/verifyTransformInvariance.m validates each transform kind
%   SINGLE-SITE against its prior (one varying coordinate, all others
%   fixed).  That isolates the per-coordinate Jacobian (R1-R3) but cannot
%   catch a JOINT-level bug: wrong Jacobian summation across a random
%   block partition, covariance-vs-blocking column misalignment, or a
%   clamp interaction that only shows up when every coordinate moves at
%   once under the real weighted-cloud-covariance proposal.
%
%   This benchmark instead replicates jointstar.runSMC's mutation stage
%   verbatim (see runSMC.m lines ~160-225) at phi=0 (null likelihood, so
%   the exact invariant law of the resulting Markov chain is the prior
%   itself) and iterates it for many stages so the cloud forgets its
%   start:
%
%     P = jointstar.defaultPriors('HierKappa', true)   (d=79, matches
%       jointstar.production's spec; mutateIdx excludes sme_pieobs, the
%       one 'fixed' calibrated parameter -- CLAUDE.md "Owner rulings")
%     N = 2000 particles ~ jointstar.priorSample(P, N), equal weights
%     60 repetitions of: recompute the (transformed- or raw-scale)
%       weighted cloud covariance on the mutateIdx columns -> a random
%       2-block partition -> per-block chol((2.38^2/db)*Sig_b) proposal
%       (cScale fixed at 1, no adaptation) -> jointstar.mhMutate per
%       particle, M=2 sweeps, phi=0, logLik = @(tv) 0.
%     No resampling, no tempering: with phi=0 and equal weights the
%     mutation kernel's own invariant law is exactly the prior, so after
%     enough stages the cloud IS a (correlated, finite-N) prior sample.
%
%   TWO ARMS, same harness:
%     transformed arm:  MutationTransform-equivalent.  Covariance built
%       in eta = T.toEta(theta) coordinates (jointstar.paramTransform);
%       blocks from jointstar.blockAtoms + jointstar.blockPartitionAtoms
%       (StructuredBlocks-equivalent: ridge/relation atoms never split).
%     control arm:      the ORIGINAL untransformed kernel -- raw-theta
%       covariance, jointstar.blockPartition's plain per-parameter random
%       partition, mut.transform = [] in jointstar.mhMutate.  This also
%       targets the prior exactly (mhMutate's un-transformed branch is a
%       standard block RWM), so it is the control: if a parameter fails
%       in BOTH arms by a similar amount, the failure is harness/MC-noise
%       / tolerance, not a transform bug.
%
%   After the final stage, every one of the 79 parameters' particle-cloud
%   mean/sd/q05/q50/q95 (2000 correlated draws) is compared against a
%   fresh 200,000-draw jointstar.priorSample reference.  PASS per param:
%     |mean - refMean| < 0.05 * refSd
%     |sd / refSd - 1|  < 0.10
%     max(|q05-refQ05|, |q50-refQ50|, |q95-refQ95|) < 0.08 * refSd
%   (refSd == 0 only for the fixed sme_pieobs column; that column is
%   handled by an exact-value check instead -- see fixed-column note
%   below.)
%
%   Prints a full 79-row param table (ref / transformed / control
%   mean+sd + pass flags), failure counts per arm, the worst 10 params by
%   standardized mean discrepancy per arm, and the per-stage-averaged MH
%   acceptance rate per arm.
%
%   See also jointstar.runSMC, jointstar.mhMutate, jointstar.paramTransform,
%   jointstar.blockAtoms, benchmarks/verifyTransformInvariance.m.

rng(20260714);

N       = 2000;
NREF    = 200000;
NSTAGES = 60;
M       = 2;
NB      = 2;
CSCALE  = 1;
SEED_T  = 20260714;   % transformed arm
SEED_C  = 20260715;   % control arm

P  = jointstar.defaultPriors('HierKappa', true);
d  = P.d;
mi = P.mutateIdx;
covCols = find(mi);
dm = numel(covCols);

fprintf('verifyTransformInvarianceJoint: d=%d, mutated cols=%d, N=%d, stages=%d, M=%d, nB=%d\n', ...
    d, dm, N, NSTAGES, M, NB);
assert(d == 79, 'expected d=79 (HierKappa priors), got %d', d);
fixedCol = find(~mi);
assert(isscalar(fixedCol) && strcmp(P.names{fixedCol}, 'sme_pieobs'), ...
    'expected exactly one fixed column (sme_pieobs), got %s', ...
    strjoin(P.names(~mi), ','));

logPriorFn = @(tv) jointstar.priorLogPdf(P, tv);
logLikNull = @(tv) 0;

fprintf('\n--- transformed arm (MutationTransform + StructuredBlocks) ---\n');
[cloudT, accT] = runArm(P, mi, covCols, dm, N, NSTAGES, M, NB, CSCALE, SEED_T, ...
    logPriorFn, logLikNull, true);

fprintf('\n--- control arm (original raw-scale kernel, plain blockPartition) ---\n');
[cloudC, accC] = runArm(P, mi, covCols, dm, N, NSTAGES, M, NB, CSCALE, SEED_C, ...
    logPriorFn, logLikNull, false);

fprintf('\n--- reference: fresh %d-draw jointstar.priorSample ---\n', NREF);
rng(999);
ref = jointstar.priorSample(P, NREF);

names = P.names(:);
q = @(X, p) quantile(X, p, 1)';

refMean = mean(ref, 1)'; refSd = std(ref, 0, 1)';
refQ05  = q(ref, 0.05);  refQ50 = q(ref, 0.50);  refQ95 = q(ref, 0.95);

TMean = mean(cloudT, 1)'; TSd = std(cloudT, 0, 1)';
TQ05  = q(cloudT, 0.05);  TQ50 = q(cloudT, 0.50); TQ95 = q(cloudT, 0.95);

CMean = mean(cloudC, 1)'; CSd = std(cloudC, 0, 1)';
CQ05  = q(cloudC, 0.05);  CQ50 = q(cloudC, 0.50); CQ95 = q(cloudC, 0.95);

[TpassMean, TpassSd, TpassQ, Tpass, TstdMeanDiff] = checkArm(TMean, TSd, TQ05, TQ50, TQ95, ...
    refMean, refSd, refQ05, refQ50, refQ95, fixedCol);
[CpassMean, CpassSd, CpassQ, Cpass, CstdMeanDiff] = checkArm(CMean, CSd, CQ05, CQ50, CQ95, ...
    refMean, refSd, refQ05, refQ50, refQ95, fixedCol);

results = table(names, refMean, refSd, TMean, TSd, Tpass, CMean, CSd, Cpass, ...
    'VariableNames', {'param', 'refMean', 'refSd', 'transMean', 'transSd', 'transPass', ...
    'ctrlMean', 'ctrlSd', 'ctrlPass'});
results.transStdMeanDiff = TstdMeanDiff;
results.ctrlStdMeanDiff  = CstdMeanDiff;

fprintf('\n%-16s %10s %10s | %10s %10s %5s | %10s %10s %5s\n', ...
    'param', 'refMean', 'refSd', 'transMean', 'transSd', 'T-ok', 'ctrlMean', 'ctrlSd', 'C-ok');
for j = 1:d
    fprintf('%-16s %10.4g %10.4g | %10.4g %10.4g %5s | %10.4g %10.4g %5s\n', ...
        names{j}, refMean(j), refSd(j), TMean(j), TSd(j), tf(Tpass(j)), ...
        CMean(j), CSd(j), tf(Cpass(j)));
end

nFailT = nnz(~Tpass); nFailC = nnz(~Cpass);
fprintf('\nfailures: transformed arm %d/%d, control arm %d/%d\n', nFailT, d, nFailC, d);
fprintf('mean-only sub-check failures: transformed %d, control %d\n', ...
    nnz(~TpassMean), nnz(~CpassMean));
fprintf('sd-ratio sub-check failures:  transformed %d, control %d\n', ...
    nnz(~TpassSd), nnz(~CpassSd));
fprintf('quantile sub-check failures:  transformed %d, control %d\n', ...
    nnz(~TpassQ), nnz(~CpassQ));

fprintf('\nworst 10 params, transformed arm, by |standardized mean diff|:\n');
printWorst(names, TstdMeanDiff, TMean, refMean, TSd, refSd, Tpass);
fprintf('\nworst 10 params, control arm, by |standardized mean diff|:\n');
printWorst(names, CstdMeanDiff, CMean, refMean, CSd, refSd, Cpass);

fprintf('\naverage MH acceptance rate over %d stages: transformed=%.3f, control=%.3f\n', ...
    NSTAGES, mean(accT), mean(accC));

results.Properties.UserData = struct('accRateTransformed', accT, 'accRateControl', accC, ...
    'nStages', NSTAGES, 'N', N, 'M', M, 'nBlocks', NB);
end

% ==========================================================================
function [cloud, accRateStage] = runArm(P, mi, covCols, dm, N, nStages, M, nB, ...
    cScale, seed, logPriorFn, logLikNull, useTransform)
rng(seed);
d = P.d;

Ptc = jointstar.priorSample(P, N);
lp = zeros(N, 1);
for i = 1:N
    lp(i) = logPriorFn(Ptc(i, :));
end
if any(~isfinite(lp))
    error('jointstar:badInit', 'initial prior draw has non-finite log prior');
end
ll = zeros(N, 1);   % null likelihood, constant 0 for every particle
W = ones(N, 1) / N;

kernelT = [];
if useTransform
    kernelT = jointstar.paramTransform(P, mi);
end

accRateStage = zeros(nStages, 1);
tStart = tic;
for s = 1:nStages
    Pm = Ptc(:, covCols);
    if useTransform
        Em = kernelT.toEta(Ptc);   % FULL-WIDTH transform then restrict
        Em = Em(:, covCols);
        mu = W' * Em;
        Pc = Em - mu;
    else
        mu = W' * Pm;
        Pc = Pm - mu;
    end
    Sig = (Pc .* W)' * Pc;
    Sig = (Sig + Sig') / 2 + 1e-10 * eye(dm);

    if useTransform
        atoms = jointstar.blockAtoms(P, covCols);
        [bperm, edgesB] = jointstar.blockPartitionAtoms(atoms, nB);
    else
        [bperm, edgesB] = jointstar.blockPartition(dm, covCols, nB);
    end

    mut = struct();
    mut.M = M;
    mut.transform = kernelT;
    mut.covBlocks = cell(1, nB); mut.covLprops = cell(1, nB); mut.covEpsRows = cell(1, nB);
    for b = 1:nB
        sel = bperm(edgesB(b) + 1:edgesB(b + 1));
        mut.covBlocks{b} = covCols(sel);
        db = numel(sel);
        Sb = Sig(sel, sel) + 1e-12 * eye(db);
        mut.covLprops{b} = chol((cScale^2 * 2.38^2 / db) * Sb, 'lower');
        mut.covEpsRows{b} = sel;
    end
    nProp = M * nB;

    eps3 = randn(dm, M, N);   % fresh proposal noise every sweep/stage
    logu = log(rand(N, nProp));
    accN = zeros(N, 1);
    for i = 1:N
        [Ptc(i, :), lp(i), ll(i), accN(i)] = jointstar.mhMutate( ...
            Ptc(i, :), lp(i), ll(i), 0, mut, eps3(:, :, i), logu(i, :), ...
            logPriorFn, logLikNull);
    end
    accRateStage(s) = mean(accN) / nProp;

    if mod(s, 10) == 0 || s == nStages
        fprintf('  stage %3d/%d  acc=%.3f  (%.1fs elapsed)\n', ...
            s, nStages, accRateStage(s), toc(tStart));
    end
end
cloud = Ptc;
assert(all(Ptc(:, ~mi) == P.params(find(~mi, 1)).init), ...
    'fixed column moved -- mutateIdx wiring bug');
end

% ==========================================================================
function [passMean, passSd, passQ, pass, stdMeanDiff] = checkArm(mn, sd, q05, q50, q95, ...
    refMean, refSd, refQ05, refQ50, refQ95, fixedCol)
d = numel(mn);
passMean = false(d, 1); passSd = false(d, 1); passQ = false(d, 1);
stdMeanDiff = zeros(d, 1);
for j = 1:d
    if j == fixedCol
        % refSd==0 by construction (constant column): exact-value check
        passMean(j) = abs(mn(j) - refMean(j)) < 1e-9;
        passSd(j)   = sd(j) < 1e-9;
        passQ(j)    = abs(q50(j) - refMean(j)) < 1e-9;
        stdMeanDiff(j) = abs(mn(j) - refMean(j)) / max(abs(refMean(j)), 1e-9);
        continue
    end
    rs = refSd(j);
    stdMeanDiff(j) = abs(mn(j) - refMean(j)) / rs;
    passMean(j) = stdMeanDiff(j) < 0.05;
    passSd(j)   = abs(sd(j) / rs - 1) < 0.10;
    qDiff = max([abs(q05(j) - refQ05(j)), abs(q50(j) - refQ50(j)), abs(q95(j) - refQ95(j))]);
    passQ(j) = qDiff < 0.08 * rs;
end
pass = passMean & passSd & passQ;
end

% ==========================================================================
function printWorst(names, stdMeanDiff, mn, refMean, sd, refSd, pass)
[sorted, ord] = sort(stdMeanDiff, 'descend');
k = min(10, numel(ord));
for r = 1:k
    j = ord(r);
    fprintf('  %-16s stdMeanDiff=%7.4f  mean=%9.4g (ref %9.4g)  sd=%9.4g (ref %9.4g)  %s\n', ...
        names{j}, sorted(r), mn(j), refMean(j), sd(j), refSd(j), tf(pass(j)));
end
end

% ==========================================================================
function s = tf(b)
if b, s = 'PASS'; else, s = 'FAIL'; end
end
