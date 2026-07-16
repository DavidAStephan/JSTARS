function out = profileConditional(dataFile, anchorSpec, paramNames, outDir, nGrid)
%PROFILECONDITIONAL Exact, noise-free conditional-likelihood identification
%   profiles -- the deterministic replacement for the single-seed
%   marginal-LML "profile" (SMC's log-marginal-likelihood has a ~10
%   log-point noise floor across seeds at N=2000; see CLAUDE.md
%   "Instrumentation").  This tool holds theta fixed at an anchor point
%   and varies ONE coordinate on a grid, re-evaluating the EXACT
%   Chan-Jeliazkov precision-based likelihood (jointstar.computeLogLik)
%   at each point.  No sampling, no RNG, no noise: two calls at the same
%   theta always return bit-identical numbers (verified below).
%
%   out = jointstar.benchmarks.profileConditional(dataFile, anchorSpec, ...
%             paramNames, outDir, nGrid)
%   (run as a plain script/function from repo root with 'benchmarks' on
%   the path -- it is a benchmarks/ tool, not a +jointstar/ package
%   function, so it is called as profileConditional(...), not
%   jointstar.profileConditional(...))
%
%   Arguments (all optional, defaults shown):
%     dataFile    'data.csv'
%     anchorSpec  path to EITHER
%                   (a) a pooled_posterior.csv (param,mean,sd,q05,q50,q95)
%                       -- anchor theta-hat = its q50 column, and the
%                       grid bounds below use that file's own sd/q05/q95
%                       columns directly (no need to touch the raw
%                       particle cloud); OR
%                   (b) a seed OutDir containing particles_stage_*.mat
%                       -- anchor = the highest-log-posterior (ll+lp)
%                       particle in the LAST (highest-numbered) snapshot
%                       in that directory if per-particle ll/lp are
%                       stored (they are, in every snapshot written by
%                       jointstar.runSMC's snapshot() helper -- see
%                       'P','logw','ll','lp','phi','seed' fields), else
%                       falls back to the logw-weighted mean.  Grid
%                       bounds are then the logw-weighted sd/q05/q95 of
%                       that same cloud.
%                 The choice actually used, and which anchor mode
%                 (pooled-csv vs seed-cloud, MAP-particle vs weighted-
%                 mean fallback), is recorded in out.anchorSource.
%     paramNames  cellstr of scalar parameters to profile individually
%                 {'phiy','phiu','kapHyp_lm_w2021','kapHyp_lm_w2023tot'}
%     outDir      'results/identification_profiles'
%     nGrid       41
%
%   RATIONALE for supporting >=2 anchors (pooled csv AND a seed cloud):
%   this project's structural posterior lives on a connected likelihood
%   RIDGE (CHECKPOINT_12), not independent modes, but the pooled q50 is
%   an equal-weight-mixture summary across seeds whose clouds sit at
%   different points along that ridge -- coordinatewise q50 of a ridge-
%   shaped cloud need not itself be a point ON the ridge (a "Frankenstein
%   theta"), which could distort what looks like a flat or curved
%   direction.  A single seed's own MAP particle IS guaranteed to be a
%   real, feasible, high-posterior point.  Comparing profiles built at
%   both should be checked before reading anything into a "FLAT" verdict.
%
%   METHOD (replicates jointstar.estimate's exact plumbing):
%     dat   = jointstar.loadData(dataFile, 'PieObs', true)        (CP7+)
%     P     = jointstar.defaultPriors('HierKappa', true)          (CP6+)
%     cache = jointstar.buildEvalCache(dat, P)
%     ll(theta) = jointstar.computeLogLik( ...
%                     jointstar.ModelSpec.jointstar(thetaStruct(P,theta), ...
%                         dat, [], cache).system(), dat.y, cache)
%     lp(theta) = jointstar.priorLogPdf(P, theta)      (full joint prior;
%                 kept joint, not just the marginal for the one moved
%                 coordinate, because of the joint AR(2) stationarity
%                 constraint on (phi1,phi2) and the hierarchical-kappa
%                 block -- moving one coordinate can still interact with
%                 those joint checks)
%     lpost(theta) = ll(theta) + lp(theta)
%   These are the same three calls jointstar.estimate's logLikTheta and
%   the SMC mutation kernel make -- this file duplicates them exactly
%   (read-only; it does not touch +jointstar/**) rather than reusing a
%   private nested function, since estimate.m's logLikTheta is a local
%   function not exported by the package.
%
%   GRID CONSTRUCTION per named parameter j:
%     raw span   = [qhat - 2*sd, qhat + 2*sd]   (qhat/sd = that param's
%                  posterior q05/q50/q95/sd from the anchor source: the
%                  pooled csv's own columns, or the seed cloud's logw-
%                  weighted quantiles/sd)
%     actually    [q05 - 2*sd, q95 + 2*sd]     (a little wider than +-2sd
%                  around the mean alone, since q05/q95 need not be
%                  symmetric -- igsd/beta/tgamma priors are skewed)
%     intersected with that parameter's PRIOR SUPPORT [lo, hi] from
%     jointstar.defaultPriors (e.g. phiy, phiu are tnorm truncated to
%     (-Inf, 0] -- sign-restricted per the CP7 owner ruling in CLAUDE.md,
%     so the grid never crosses 0); nGrid equally spaced points.
%     A grid point exactly at a +-Inf bound is impossible (both sides of
%     the intersection use the finite qhat+-k*sd span whenever the prior
%     bound is infinite), so linspace is always over a finite interval.
%
%   CLASSIFICATION (heuristic thresholds, not a formal test):
%     total ll range   = max(ll) - min(ll) over the FULL nGrid points
%                         (reported for reference: includes tail
%                         behaviour, e.g. hard rejection at a bound)
%     central-90% range = max(ll)-min(ll) restricted to the central 90%
%                         of the GRID INDEX range (drops the outer ~5%
%                         of points on each side, i.e. indices
%                         round(0.05*(nGrid-1))+1 : nGrid-that+1) -- the
%                         classification is intentionally based on ll
%                         (the likelihood alone), not lpost, since the
%                         whole point is to see what the DATA say net of
%                         the prior's own curvature.
%       range <  2  log points -> 'FLAT (weakly identified)'
%       range >  10 log points -> 'CURVED (data-identified)'
%       otherwise               -> 'INTERMEDIATE'
%     These cutoffs are arbitrary heuristics (an order-of-magnitude read
%     of "does the exact likelihood move at all across a +-2sd-ish band"),
%     not a likelihood-ratio-test calibration -- treat them as a quick
%     triage, not a p-value.
%
%   2-D JOINT PROFILE: phiy x sme_y on a 15x15 grid (same recipe, same
%   anchor, all other coordinates held fixed), because the COVID level
%   shift (phiy, a 2020-era negative GDP-Phillips loading) and the GDP
%   measurement-error sd (sme_y) can trade off against each other --
%   sme_y can absorb the same 2020 anomaly a level-shift can, so their
%   MARGINAL profiles alone can each look flat/curved while the JOINT
%   surface is a ridge running diagonally between them.
%
%   SANITY GATE (run automatically, first, before any grid): evaluate
%   ll at the anchor theta twice, from two INDEPENDENTLY built eval
%   caches (jointstar.buildEvalCache called twice). Errors loudly
%   (assert) if either value is non-finite or if they are not bit-
%   identical -- this is the determinism claim the whole tool rests on.
%
%   OUTPUT FILES (written to outDir):
%     profile_<param>.csv   columns: x, ll, lp, lpost   (one per param)
%     profile_2d_phiy_sme_y.csv   columns: phiy, sme_y, ll, lp, lpost
%
%   Returns a struct:
%     out.anchorSource   string describing what was used
%     out.anchorTheta    1 x P.d anchor vector
%     out.sanityGate      struct('ll1',...,'ll2',...,'ok',logical)
%     out.profiles(k)    struct('name',...,'x',...,'ll',...,'lp',...,
%                        'lpost',...,'rangeFull',...,'rangeCentral90',...,
%                        'verdict',...)
%     out.joint2d        struct('phiy',...,'sme_y',...,'ll',...,'lp',...,
%                        'lpost',...)
%
%   See also jointstar.estimate, jointstar.computeLogLik,
%   jointstar.buildEvalCache, jointstar.priorLogPdf.

if nargin < 1 || isempty(dataFile), dataFile = 'data.csv'; end
if nargin < 2 || isempty(anchorSpec)
    error('profileConditional:noAnchor', ...
        'anchorSpec is required (pooled_posterior.csv path or seed OutDir).');
end
if nargin < 3 || isempty(paramNames)
    paramNames = {'phiy', 'phiu', 'kapHyp_lm_w2021', 'kapHyp_lm_w2023tot'};
end
if nargin < 4 || isempty(outDir), outDir = 'results/identification_profiles'; end
if nargin < 5 || isempty(nGrid), nGrid = 41; end

if ~isfolder(outDir), mkdir(outDir); end

fprintf('profileConditional: dataFile=%s anchorSpec=%s nGrid=%d\n', ...
    dataFile, anchorSpec, nGrid);
fprintf('  params: %s\n', strjoin(paramNames, ', '));

% ---- build the exact production-config problem -------------------------
dat = jointstar.loadData(dataFile, 'PieObs', true);
P = jointstar.defaultPriors('HierKappa', true);
fprintf('  dat.T=%d, P.d=%d\n', dat.T, P.d);

% ---- resolve anchor + per-parameter (qhat,sd,q05,q95) source -----------
[anchorTheta, anchorSource, bounds] = resolveAnchor(anchorSpec, P, paramNames);
fprintf('  anchor source: %s\n', anchorSource);

% ---- sanity gate: determinism + finiteness at the anchor ---------------
cacheA = jointstar.buildEvalCache(dat, P);
cacheB = jointstar.buildEvalCache(dat, P);
ll1 = logLikThetaLocal(P, anchorTheta, dat, cacheA);
ll2 = logLikThetaLocal(P, anchorTheta, dat, cacheB);
sanityOk = isfinite(ll1) && (ll1 == ll2);
fprintf('  sanity gate: ll(anchor) = %.6f (rebuild) / %.6f (rebuild2) -> %s\n', ...
    ll1, ll2, mat2str(sanityOk));
assert(isfinite(ll1), 'profileConditional:nonFiniteAnchor', ...
    'll at the anchor theta is not finite (%.6g) -- anchor is infeasible.', ll1);
assert(ll1 == ll2, 'profileConditional:nondeterministic', ...
    'll(anchor) differs across two independently built eval caches (%.10f vs %.10f) -- not deterministic.', ...
    ll1, ll2);
out.sanityGate = struct('ll1', ll1, 'll2', ll2, 'ok', sanityOk);

cache = cacheA;  % reuse for the grid sweep

% ---- per-parameter 1-D profiles ----------------------------------------
nParams = numel(paramNames);
profiles = struct('name', cell(1, nParams), 'x', cell(1, nParams), ...
    'll', cell(1, nParams), 'lp', cell(1, nParams), 'lpost', cell(1, nParams), ...
    'rangeFull', cell(1, nParams), 'rangeCentral90', cell(1, nParams), ...
    'verdict', cell(1, nParams));

for k = 1:nParams
    name = paramNames{k};
    j = paramIndex(P, name);
    grid = buildGrid(P.params(j), bounds.(name), nGrid);

    llv = nan(nGrid, 1); lpv = nan(nGrid, 1);
    for g = 1:nGrid
        theta = anchorTheta;
        theta(j) = grid(g);
        llv(g) = logLikThetaLocal(P, theta, dat, cache);
        lpv(g) = jointstar.priorLogPdf(P, theta);
    end
    lpostv = llv + lpv;

    [rangeFull, rangeC90] = deal(rangeStat(llv, 1, nGrid), []);
    [ixLo, ixHi] = centralIdx(nGrid);
    rangeC90 = rangeStat(llv, ixLo, ixHi);
    verdict = classify(rangeC90);

    fprintf('  [%s] total ll range = %.3f log pts, central-90%% range = %.3f log pts -> %s\n', ...
        name, rangeFull, rangeC90, verdict);

    tbl = table(grid(:), llv, lpv, lpostv, 'VariableNames', {'x', 'll', 'lp', 'lpost'});
    writetable(tbl, fullfile(outDir, sprintf('profile_%s.csv', name)));

    profiles(k) = struct('name', name, 'x', grid(:), 'll', llv, 'lp', lpv, ...
        'lpost', lpostv, 'rangeFull', rangeFull, 'rangeCentral90', rangeC90, ...
        'verdict', verdict);
end

% ---- 2-D joint profile: phiy x sme_y -----------------------------------
n2 = 15;
jx = paramIndex(P, 'phiy');
jy = paramIndex(P, 'sme_y');
boundsXY = struct();
boundsXY.phiy = getOrComputeBounds(anchorSpec, P, 'phiy', anchorTheta, bounds);
boundsXY.sme_y = getOrComputeBounds(anchorSpec, P, 'sme_y', anchorTheta, bounds);
gx = buildGrid(P.params(jx), boundsXY.phiy, n2);
gy = buildGrid(P.params(jy), boundsXY.sme_y, n2);

[GX, GY] = ndgrid(gx, gy);
llXY = nan(n2, n2); lpXY = nan(n2, n2);
for a = 1:n2
    for b = 1:n2
        theta = anchorTheta;
        theta(jx) = GX(a, b);
        theta(jy) = GY(a, b);
        llXY(a, b) = logLikThetaLocal(P, theta, dat, cache);
        lpXY(a, b) = jointstar.priorLogPdf(P, theta);
    end
end
lpostXY = llXY + lpXY;

tbl2 = table(GX(:), GY(:), llXY(:), lpXY(:), lpostXY(:), ...
    'VariableNames', {'phiy', 'sme_y', 'll', 'lp', 'lpost'});
writetable(tbl2, fullfile(outDir, 'profile_2d_phiy_sme_y.csv'));
fprintf('  [2D phiy x sme_y] ll range = %.3f log pts over %dx%d grid\n', ...
    max(llXY(:)) - min(llXY(isfinite(llXY(:)))), n2, n2);

out.anchorSource = anchorSource;
out.anchorTheta = anchorTheta;
out.profiles = profiles;
out.joint2d = struct('phiy', GX, 'sme_y', GY, 'll', llXY, 'lp', lpXY, 'lpost', lpostXY);
out.outDir = outDir;
fprintf('profileConditional: done, wrote %d 1-D profiles + 1 joint profile to %s\n', ...
    nParams, outDir);
end

% =========================================================================
function ll = logLikThetaLocal(P, tv, dat, cache)
% Exact replica of jointstar.estimate's local logLikTheta (that function
% is not exported by +jointstar/estimate.m, so it is duplicated here
% read-only rather than reused).
th = jointstar.thetaStruct(P, tv);
spec = jointstar.ModelSpec.jointstar(th, dat, [], cache);
try
    ll = jointstar.computeLogLik(spec.system(), dat.y, cache);
catch err %#ok<NASGU>
    % computeLogLik itself never errors for an infeasible theta (returns
    % -Inf) -- but ModelSpec.jointstar/spec.system() upstream could throw
    % on a genuinely pathological grid point (e.g. degenerate system
    % matrices); treat that the same way the SMC mutation kernel treats
    % any infeasible draw: reject with -Inf, don't crash the sweep.
    ll = -Inf;
end
end

% =========================================================================
function j = paramIndex(P, name)
j = find(strcmp(P.names, name), 1);
if isempty(j)
    error('profileConditional:unknownParam', ...
        'parameter "%s" not found in P.names (P.d=%d).', name, P.d);
end
end

% =========================================================================
function grid = buildGrid(prm, b, nGrid)
% b = struct('qhat',...,'sd',...,'q05',...,'q95',...)
lo = max(prm.lo, b.q05 - 2 * b.sd);
hi = min(prm.hi, b.q95 + 2 * b.sd);
if ~(hi > lo)
    error('profileConditional:degenerateGrid', ...
        'grid collapsed for %s: prior support [%.4g,%.4g] vs span [%.4g,%.4g].', ...
        prm.name, prm.lo, prm.hi, b.q05 - 2 * b.sd, b.q95 + 2 * b.sd);
end
grid = linspace(lo, hi, nGrid);
end

% =========================================================================
function [ixLo, ixHi] = centralIdx(nGrid)
trim = round(0.05 * (nGrid - 1));
ixLo = trim + 1;
ixHi = nGrid - trim;
end

% =========================================================================
function r = rangeStat(x, ixLo, ixHi)
seg = x(ixLo:ixHi);
seg = seg(isfinite(seg));
if isempty(seg)
    r = NaN;
else
    r = max(seg) - min(seg);
end
end

% =========================================================================
function v = classify(rangeC90)
if isnan(rangeC90)
    v = 'FLAT (weakly identified)';   % all -Inf/NaN in the central band: no signal at all
elseif rangeC90 < 2
    v = 'FLAT (weakly identified)';
elseif rangeC90 > 10
    v = 'CURVED (data-identified)';
else
    v = 'INTERMEDIATE';
end
end

% =========================================================================
function [anchorTheta, anchorSource, bounds] = resolveAnchor(anchorSpec, P, paramNames)
% bounds.(name) = struct('qhat','sd','q05','q95') for every requested
% paramName (plus phiy/sme_y are added on demand for the 2-D profile via
% getOrComputeBounds).
bounds = struct();
if isfolder(anchorSpec)
    [anchorTheta, anchorSource, cloud] = anchorFromSeedDir(anchorSpec, P);
    for k = 1:numel(paramNames)
        name = paramNames{k};
        j = paramIndex(P, name);
        bounds.(name) = weightedBoundsCol(cloud.P(:, j), cloud.w);
    end
    bounds.auxCloud_ = cloud;   %#ok<STRNU> stash for on-demand extra columns (2-D profile)
elseif isfile(anchorSpec)
    [anchorTheta, anchorSource, tbl] = anchorFromPooledCsv(anchorSpec, P);
    for k = 1:numel(paramNames)
        name = paramNames{k};
        bounds.(name) = pooledBoundsRow(tbl, name);
    end
    bounds.auxTable_ = tbl; %#ok<STRNU>
else
    error('profileConditional:badAnchorSpec', ...
        'anchorSpec "%s" is neither an existing file nor an existing folder.', anchorSpec);
end
end

% =========================================================================
function b = getOrComputeBounds(anchorSpec, P, name, anchorTheta, bounds) %#ok<INUSD>
if isfield(bounds, name)
    b = bounds.(name);
    return
end
if isfolder(anchorSpec) && isfield(bounds, 'auxCloud_')
    j = paramIndex(P, name);
    b = weightedBoundsCol(bounds.auxCloud_.P(:, j), bounds.auxCloud_.w);
elseif isfile(anchorSpec) && isfield(bounds, 'auxTable_')
    b = pooledBoundsRow(bounds.auxTable_, name);
else
    error('profileConditional:noBoundsSource', ...
        'no bounds source available for "%s".', name);
end
end

% =========================================================================
function [anchorTheta, anchorSource, cloud] = anchorFromSeedDir(seedDir, P)
files = dir(fullfile(seedDir, 'particles_stage_*.mat'));
if isempty(files)
    error('profileConditional:noSnapshots', ...
        'no particles_stage_*.mat found in "%s".', seedDir);
end
stageNums = nan(numel(files), 1);
for i = 1:numel(files)
    tok = regexp(files(i).name, 'particles_stage_(\d+)\.mat', 'tokens', 'once');
    stageNums(i) = str2double(tok{1});
end
[~, ord] = sort(stageNums, 'descend');
last = files(ord(1));
S = load(fullfile(last.folder, last.name));
if ~isfield(S, 'P') || ~isfield(S, 'logw')
    error('profileConditional:badSnapshot', ...
        '"%s" is missing expected fields P/logw.', last.name);
end
if abs(S.phi - 1) > 1e-8
    warning('profileConditional:notFinalPhi', ...
        '"%s" has phi=%.4f (not 1) -- using it anyway as the requested anchor cloud.', ...
        last.name, S.phi);
end
w = exp(S.logw - max(S.logw)); w = w / sum(w);
cloud = struct('P', S.P, 'w', w);

if isfield(S, 'll') && isfield(S, 'lp')
    lpost = S.ll + S.lp;
    [~, iMax] = max(lpost);
    anchorTheta = S.P(iMax, :);
    anchorSource = sprintf('seed-cloud MAP particle (%s, stage file %s, particle %d, lpost=%.3f)', ...
        seedDir, last.name, iMax, lpost(iMax));
else
    anchorTheta = (w' * S.P);
    anchorSource = sprintf('seed-cloud logw-weighted MEAN (%s, stage file %s -- no per-particle ll/lp stored)', ...
        seedDir, last.name);
end
end

% =========================================================================
function b = weightedBoundsCol(col, w)
qhat = w' * col;
sd = sqrt(max(w' * (col - qhat).^2, 0));
q05 = weightedQuantile(col, w, 0.05);
q95 = weightedQuantile(col, w, 0.95);
b = struct('qhat', qhat, 'sd', sd, 'q05', q05, 'q95', q95);
end

% =========================================================================
function q = weightedQuantile(x, w, p)
[xs, ord] = sort(x);
ws = w(ord);
cw = cumsum(ws) - 0.5 * ws;   % midpoint rule, standard weighted-quantile convention
cw = cw / sum(ws);
if p <= cw(1), q = xs(1); return; end
if p >= cw(end), q = xs(end); return; end
q = interp1(cw, xs, p, 'linear');
end

% =========================================================================
function [anchorTheta, anchorSource, tbl] = anchorFromPooledCsv(csvPath, P)
tbl = readtable(csvPath);
req = {'param', 'mean', 'sd', 'q05', 'q50', 'q95'};
if ~all(ismember(req, tbl.Properties.VariableNames))
    error('profileConditional:badPooledCsv', ...
        '"%s" is missing expected columns %s.', csvPath, strjoin(req, ','));
end
anchorTheta = nan(1, P.d);
for j = 1:P.d
    r = strcmp(tbl.param, P.names{j});
    if ~any(r)
        error('profileConditional:missingParam', ...
            'pooled csv "%s" has no row for parameter "%s".', csvPath, P.names{j});
    end
    anchorTheta(j) = tbl.q50(find(r, 1));
end
anchorSource = sprintf('pooled-posterior q50 (%s)', csvPath);
end

% =========================================================================
function b = pooledBoundsRow(tbl, name)
r = find(strcmp(tbl.param, name), 1);
if isempty(r)
    error('profileConditional:missingParam', 'pooled csv has no row for "%s".', name);
end
b = struct('qhat', tbl.q50(r), 'sd', tbl.sd(r), 'q05', tbl.q05(r), 'q95', tbl.q95(r));
end
