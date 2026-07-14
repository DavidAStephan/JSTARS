function results = verifyTransformInvariance()
%VERIFYTRANSFORMINVARIANCE Prior-only invariance smoke test for the
%'MutationTransform' transformed-scale MH kernel
%(DESIGN_transformed_kernel.md, ADDENDUM "Invariance smoke test",
%required before any real A/B run).
%
%   results = benchmarks.verifyTransformInvariance()
%
%   For one representative mutated parameter per transform kind in
%   jointstar.paramTransform (identity; shifted-log at two different
%   shifts -- igsd/logn at lo=0 and tgamma/hkap at lo=1; logit-affine at
%   three different [lo,hi] pairs -- beta/negbeta/unif/tnorm-both-
%   finite; reflected-log for the one-sided tnorm), this:
%
%     1. Fixes every OTHER coordinate at its jointstar.defaultPriors
%        'init' value (a valid interior point that already satisfies
%        the AR(2) stationarity and hkap-truncation joint constraints).
%     2. Runs a long (2e5-draw) single-site random-walk MH chain in
%        eta-space for the one varying coordinate, targeting phi=0 (the
%        prior alone): log-target = logPrior(theta) + logJacElem(eta)_j.
%        This is exactly the prior's pushforward through T with no
%        likelihood involved, so it validates the transform + Jacobian
%        (R1-R3) in isolation, the same thing an A/B run cannot cleanly
%        isolate.
%     3. Compares the resulting theta_j sample's mean/sd/5%/95% against
%        a direct reference sample of theta_j: jointstar.priorSample's
%        column j for ordinary (independent) parameters.  For the hkap
%        case, priorSample's raw column additionally marginalises over
%        the hierarchy's own RANDOM hyperparameters, which is a
%        materially more dispersed, non-degenerate-but-different target
%        than the single-site chain (whose other coordinates, including
%        those hyperparameters, are held fixed) -- exactly the
%        "hyperparameter-degenerate type" case the design doc flags, so
%        the reference there is instead generated with those two
%        hyperparameter columns pinned 'fixed' at their init values
%        (see hkapConditionalReference below), which is the documented
%        fallback: "fix the hyperparameters at their prior means and
%        target the conditional."
%
%   Prints and returns a table of per-case (chain mean/sd/q05/q95),
%   (reference mean/sd/q05/q95), and acceptance rate.  All cases are
%   expected to agree to within ~2-3% at this chain length; a bigger
%   mismatch on any type is a loud signal of an R1-R3 bug (wrong
%   Jacobian, wrong accept ratio, or a T/T^-1 mismatch), to be resolved
%   BEFORE any likelihood-bearing MutationTransform A/B run.
%
%   See also jointstar.paramTransform, tests/testParamTransform.m.

rng(20260714);
nDraws = 200000;
nBurn = 5000;
nPilot = 3000;
nRef = 200000;

% {name, HierKappa, hkap-group-index-or-[]}
cases = { ...
    'gzbar',     false, []; ...  % norm                          -> identity
    'sig_c',     false, []; ...  % igsd,   (0,Inf)                -> shifted-log
    'm84_y',     false, []; ...  % logn,   (0,Inf)                -> shifted-log
    'kapc_2021', false, []; ...  % tgamma, [1,Inf)                -> shifted-log
    'kapy_20',   true,   1; ...  % hkap,   [1,Inf)                -> shifted-log (hyper-fixed conditional)
    'rhoU',      false, []; ...  % beta,   (0,1)                  -> logit-affine
    'gamma2',    false, []; ...  % negbeta,(-1,0)                 -> logit-affine
    'alphapi',   false, []; ...  % unif,   [.01,.99]              -> logit-affine
    'alpha',     false, []; ...  % tnorm both-finite, [.15,.55]   -> logit-affine
    'phiu',      false, [] ...   % tnorm one-sided,   (-Inf,0]    -> reflected-log
    };

rows = cell(size(cases, 1), 1);
for r = 1:size(cases, 1)
    nm = cases{r, 1}; hierK = cases{r, 2}; grp = cases{r, 3};
    rows{r} = runOneCase(nm, hierK, grp, nDraws, nBurn, nPilot, nRef);
    row = rows{r};
    fprintf(['%-10s kind=%d  chain(mean=%9.5g sd=%9.5g q05=%9.5g q95=%9.5g)  ' ...
        'ref(mean=%9.5g sd=%9.5g q05=%9.5g q95=%9.5g)  acc=%.2f\n'], ...
        row.param{1}, row.kind, row.chainMean, row.chainSd, row.chainQ05, row.chainQ95, ...
        row.refMean, row.refSd, row.refQ05, row.refQ95, row.accRate);
end
results = vertcat(rows{:});

relMean = abs(results.chainMean - results.refMean) ./ max(abs(results.refMean), 1e-6);
relSd = abs(results.chainSd - results.refSd) ./ max(results.refSd, 1e-6);
results.relMeanDiff = relMean;
results.relSdDiff = relSd;

fprintf('\nmax relative mean discrepancy: %.4f, max relative sd discrepancy: %.4f\n', ...
    max(relMean), max(relSd));
disp(results);
end

% ==========================================================================
function row = runOneCase(nm, hierK, grp, nDraws, nBurn, nPilot, nRef)
P = jointstar.defaultPriors('HierKappa', hierK);
T = jointstar.paramTransform(P, P.mutateIdx);
theta0 = [P.params.init];
j = find(strcmp(P.names, nm), 1);
assert(~isempty(j), 'param %s not found', nm);

logPrior = @(tv) jointstar.priorLogPdf(P, tv);
etaBase = T.toEta(theta0);

% ---- pilot: tune the RW step for a reasonable acceptance rate ---------
step = 1.0;
etaCur = etaBase(j);
thetaRow = theta0;
curVal = logPrior(thetaRow) + jacAt(T, etaBase, j);
for p = 1:5
    acc = 0;
    for it = 1:nPilot
        [etaCur, thetaRow, curVal, a] = mhStep(T, logPrior, etaBase, j, etaCur, thetaRow, curVal, step);
        acc = acc + a;
    end
    rate = acc / nPilot;
    if rate < 0.15
        step = step * 0.6;
    elseif rate > 0.45
        step = step * 1.6;
    else
        break
    end
end

% ---- main chain ---------------------------------------------------------
chain = zeros(nDraws, 1);
accN = 0;
for it = 1:(nBurn + nDraws)
    [etaCur, thetaRow, curVal, a] = mhStep(T, logPrior, etaBase, j, etaCur, thetaRow, curVal, step);
    if it > nBurn
        chain(it - nBurn) = thetaRow(j);
        accN = accN + a;
    end
end
accRate = accN / nDraws;

% ---- reference sample -----------------------------------------------
if isempty(grp)
    Theta = jointstar.priorSample(P, nRef);
    ref = Theta(:, j);
else
    P2 = P;
    P2.params(P.kap.laCols(grp)).type = 'fixed';
    P2.params(P.kap.lmCols(grp)).type = 'fixed';
    Theta2 = jointstar.priorSample(P2, nRef);
    ref = Theta2(:, j);
end

q = @(x, p) quantile(x, p);
row = table({nm}, double(T.kind(j)), mean(chain), std(chain), q(chain, 0.05), q(chain, 0.95), ...
    mean(ref), std(ref), q(ref, 0.05), q(ref, 0.95), accRate, ...
    'VariableNames', {'param', 'kind', 'chainMean', 'chainSd', 'chainQ05', 'chainQ95', ...
    'refMean', 'refSd', 'refQ05', 'refQ95', 'accRate'});
end

% ==========================================================================
function jv = jacAt(T, eta, j)
je = T.logJacElem(eta);
jv = je(j);
end

function [etaCur, thetaRow, curVal, accepted] = mhStep(T, logPrior, etaBase, j, etaCur, thetaRow, curVal, step)
etaProp = etaCur + step * randn();
etaFull = etaBase;
etaFull(j) = etaProp;
propRow = T.toTheta(etaFull);
jacProp = jacAt(T, etaFull, j);
accepted = 0;
lpProp = logPrior(propRow);
if isfinite(lpProp)
    propVal = lpProp + jacProp;
    if log(rand()) < (propVal - curVal)
        etaCur = etaProp; thetaRow = propRow; curVal = propVal;
        accepted = 1;
    end
end
end
