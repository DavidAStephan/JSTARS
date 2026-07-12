function out = verifyCacheEquivalence()
%VERIFYCACHEEQUIVALENCE Bitwise-equivalence check: eval cache vs no cache.
%
%   Mandatory verification for the eval-cache performance pass (see
%   jointstar.buildEvalCache).  For 100 prior-drawn thetas (fixed seed)
%   from horseshoePriors('HierKappa', true) plus 50 from defaultPriors,
%   compares jointstar.computeLogLik's old (no-cache) and new (cache)
%   code paths and reports the max absolute log-likelihood difference.
%   Also compares a state draw (fixed randn stream) old vs new for 5
%   thetas from each family, and repeats the whole thing under
%   'PieObs', false (the 7-observable configuration) to confirm the
%   cache correctly keys off the run's actual data/config rather than
%   being hardcoded to the 8-observable production spec.
%
%   Target: exactly 0 everywhere.  Run from repo root:
%     /Applications/MATLAB_R2026a.app/bin/matlab -batch ...
%       "cd(pwd); addpath('benchmarks'); verifyCacheEquivalence"

fprintf('=== eval-cache bitwise equivalence: PieObs=true (8 obs) ===\n');
dat8 = jointstar.loadData('data.csv', 'PieObs', true);
r1 = checkFamily(dat8, jointstar.horseshoePriors('HierKappa', true), 100, 1, 'horseshoe+HierKappa');
r2 = checkFamily(dat8, jointstar.defaultPriors(), 50, 2, 'default');

fprintf('\n=== eval-cache bitwise equivalence: PieObs=false (7 obs) ===\n');
dat7 = jointstar.loadData('data.csv', 'PieObs', false);
r3 = checkFamily(dat7, jointstar.horseshoePriors('HierKappa', true), 30, 3, 'horseshoe+HierKappa, 7obs');

allMax = max([r1.maxAbsDiff, r2.maxAbsDiff, r3.maxAbsDiff]);
allStateMax = max([r1.maxStateDiff, r2.maxStateDiff, r3.maxStateDiff]);
fprintf('\n=== SUMMARY ===\n');
fprintf('max |loglik diff| over all draws/families = %.6g\n', allMax);
fprintf('max |state draw diff| over all draws/families = %.6g\n', allStateMax);
if allMax == 0 && allStateMax == 0
    fprintf('RESULT: exact bitwise equivalence confirmed.\n');
else
    fprintf(['RESULT: NOT exactly 0 -- investigate before relying on the ' ...
        'cache (see file header of jointstar.computeLogLik for known, ' ...
        'already-mitigated sources).\n']);
end

out = struct('horseshoeHierKappa', r1, 'defaultPriors', r2, ...
    'horseshoeHierKappa7obs', r3, 'maxAbsDiff', allMax, 'maxStateDiff', allStateMax);
end

% ==========================================================================
function res = checkFamily(dat, P, nDraws, seed, label)
cache = jointstar.buildEvalCache(dat, P);
rng(seed);
maxAbsDiff = 0;
maxStateDiff = 0;
nFinite = 0;
nChecked = 0;
for i = 1:nDraws
    tv = jointstar.priorSample(P, 1);
    th = jointstar.thetaStruct(P, tv);
    if isfield(P, 'hs')
        [Lq, Lr] = jointstar.hsUnpack(P, tv);
        cf = struct('Lq', Lq, 'Lr', Lr);
    else
        cf = [];
    end

    spec0 = jointstar.ModelSpec.jointstar(th, dat, cf);
    [ll0, aux0] = jointstar.computeLogLik(spec0.system(), dat.y);

    spec1 = jointstar.ModelSpec.jointstar(th, dat, cf, cache);
    [ll1, aux1] = jointstar.computeLogLik(spec1.system(), dat.y, cache);

    nChecked = nChecked + 1;
    if isfinite(ll0) || isfinite(ll1)
        nFinite = nFinite + 1;
        d = abs(ll0 - ll1);
        if isnan(d), d = Inf; end
        maxAbsDiff = max(maxAbsDiff, d);
    end

    if i <= 5 && isfinite(ll0) && isfinite(ll1)
        rng(9000 + i);
        a0 = jointstar.drawStates(aux0, 2);
        rng(9000 + i);
        a1 = jointstar.drawStates(aux1, 2);
        maxStateDiff = max(maxStateDiff, max(abs(a0(:) - a1(:))));
    end
end
fprintf('  [%s] n=%d, nFinite(either)=%d, max |loglik diff| = %.6g, max |state diff| (5 draws) = %.6g\n', ...
    label, nChecked, nFinite, maxAbsDiff, maxStateDiff);
res = struct('nDraws', nChecked, 'nFinite', nFinite, ...
    'maxAbsDiff', maxAbsDiff, 'maxStateDiff', maxStateDiff);
end
