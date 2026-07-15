function bridge_main()
% BRIDGE_MAIN  Bridging log-posterior profile between seed42 and seed101
% posterior modes, to discriminate genuine multimodality from a connected
% ridge.  Writes bridge_profiles.csv, bridge_summary.txt, bridge_profiles.png
% into the scratchpad directory (cwd when this script is invoked from there
% is fine; we use absolute paths throughout).

scratchDir = '/private/tmp/claude-501/-Users-davidstephan-Documents-JSTARS/57afed17-706d-40d4-b1e7-95b72c937bac/scratchpad';
repoRoot   = '/Users/davidstephan/Documents/JSTARS';
resDir     = fullfile(repoRoot, 'results', 'production');

diaryFile = fullfile(scratchDir, 'bridge_run.log');
if isfile(diaryFile), delete(diaryFile); end
diary(diaryFile); diary on;

fprintf('=== bridge_main starting %s ===\n', datestr(now));

% ---- 1. build the exact production likelihood/prior evaluator ----------
dat = jointstar.loadData(fullfile(repoRoot, 'data.csv'), 'PieObs', true);
P   = jointstar.defaultPriors('HierKappa', true);
cache = jointstar.buildEvalCache(dat, P);

evalTheta = @(tv) evalOne(P, dat, cache, tv);

names = P.names(:);
d = P.d;
fprintf('d = %d params, T = %d\n', d, dat.T);

% 'fixed' (calibrated) coordinates (sme_pieobs) must be held EXACTLY at
% their init value on every path -- they are excluded from mutation
% (P.mutateIdx) and priorLogPdf enforces bit-exact equality (x ~= q.init
% => -Inf).  All particles carry this coordinate as the same bit-identical
% double, so mathematically (1-t)*0.3 + t*0.3 = 0.3 for every t, but IEEE
% floating-point interpolation is not guaranteed to round back to exactly
% that bit pattern -- a spurious -Inf artifact of interpolation, not a
% ridge feature. Force it exactly instead of interpolating.
fixedMask = strcmp({P.params.type}, 'fixed');
fixedIdx = find(fixedMask);
fixedInit = [P.params(fixedMask).init];
fprintf('fixed (calibrated) coords held exact on every path: %s = %s\n', ...
    strjoin(names(fixedIdx), ','), mat2str(fixedInit));

% ---- 2. load seed42 and seed101 final particle clouds -------------------
s42  = load(fullfile(resDir, 'seed42',  'particles_stage_29.mat'));
s101 = load(fullfile(resDir, 'seed101', 'particles_stage_29.mat'));

assert(isequal(size(s42.P), [2000, d]));
assert(isequal(size(s101.P), [2000, d]));
assert(s42.phi == 1 && s101.phi == 1, 'expected final (phi=1) stage snapshots');

% ---- 3. SANITY GATE: reproduce stored ll/lp for 3 particles each --------
fprintf('\n--- SANITY GATE ---\n');
checkRows42  = [1, 1000, 2000];
checkRows101 = [1, 1000, 2000];
maxAbsErrLL = 0; maxAbsErrLP = 0;
for r = checkRows42
    tv = s42.P(r, :);
    [ll, lp] = evalTheta(tv);
    eLL = abs(ll - s42.ll(r));
    eLP = abs(lp - s42.lp(r));
    maxAbsErrLL = max(maxAbsErrLL, eLL);
    maxAbsErrLP = max(maxAbsErrLP, eLP);
    fprintf('seed42  row %5d: stored ll=%.6f my ll=%.6f (diff %.3e) | stored lp=%.6f my lp=%.6f (diff %.3e)\n', ...
        r, s42.ll(r), ll, eLL, s42.lp(r), lp, eLP);
end
for r = checkRows101
    tv = s101.P(r, :);
    [ll, lp] = evalTheta(tv);
    eLL = abs(ll - s101.ll(r));
    eLP = abs(lp - s101.lp(r));
    maxAbsErrLL = max(maxAbsErrLL, eLL);
    maxAbsErrLP = max(maxAbsErrLP, eLP);
    fprintf('seed101 row %5d: stored ll=%.6f my ll=%.6f (diff %.3e) | stored lp=%.6f my lp=%.6f (diff %.3e)\n', ...
        r, s101.ll(r), ll, eLL, s101.lp(r), lp, eLP);
end
fprintf('max abs err: ll %.3e, lp %.3e\n', maxAbsErrLL, maxAbsErrLP);
if maxAbsErrLL > 1e-6 || maxAbsErrLP > 1e-6
    diary off;
    error('bridge_main:sanityGateFailed', ...
        'Sanity gate FAILED: evaluator does not reproduce stored ll/lp to 1e-6 (max err ll=%.3e, lp=%.3e). STOPPING.', ...
        maxAbsErrLL, maxAbsErrLP);
end
fprintf('SANITY GATE PASSED (tol 1e-6)\n');

% ---- 4. endpoints --------------------------------------------------------
lpost42  = s42.ll  + s42.lp;
lpost101 = s101.ll + s101.lp;

[~, iMap42]  = max(lpost42);
[~, iMap101] = max(lpost101);
mapTheta42  = s42.P(iMap42, :);
mapTheta101 = s101.P(iMap101, :);
fprintf('\nseed42  MAP: row %d, ll=%.4f lp=%.4f lpost=%.4f\n', iMap42, s42.ll(iMap42), s42.lp(iMap42), lpost42(iMap42));
fprintf('seed101 MAP: row %d, ll=%.4f lp=%.4f lpost=%.4f\n', iMap101, s101.ll(iMap101), s101.lp(iMap101), lpost101(iMap101));

w42  = exp(s42.logw  - max(s42.logw));  w42  = w42  / sum(w42);
w101 = exp(s101.logw - max(s101.logw)); w101 = w101 / sum(w101);
meanTheta42  = clampFixed((w42'  * s42.P),  fixedIdx, fixedInit);
meanTheta101 = clampFixed((w101' * s101.P), fixedIdx, fixedInit);
% (the weighted sum over 2000 particles that all share the SAME
% bit-identical 'fixed' coordinate value is not guaranteed to round back
% to exactly that bit pattern either -- same floating-point artifact as
% the interpolation paths, clamped here for the same reason)

[llMean42, lpMean42]   = evalTheta(meanTheta42);
[llMean101, lpMean101] = evalTheta(meanTheta101);
fprintf('seed42  weighted-mean theta: ll=%.4f lp=%.4f lpost=%.4f\n', llMean42, lpMean42, llMean42+lpMean42);
fprintf('seed101 weighted-mean theta: ll=%.4f lp=%.4f lpost=%.4f\n', llMean101, lpMean101, llMean101+lpMean101);

% ---- restricted-block coordinate indices --------------------------------
blockNames = {'gwbar', 'gzbar', 'sig_xi'};
blockIdx = zeros(1, numel(blockNames));
for k = 1:numel(blockNames)
    blockIdx(k) = P.idx.(blockNames{k});
end
fprintf('\nblock coordinate indices: %s\n', mat2str(blockIdx));
for k = 1:numel(blockNames)
    fprintf('  %-8s idx=%d  seed42-MAP=%.4f  seed101-MAP=%.4f\n', ...
        blockNames{k}, blockIdx(k), mapTheta42(blockIdx(k)), mapTheta101(blockIdx(k)));
end

% ---- 5. build the four paths ---------------------------------------------
nPts = 40;
tgrid = linspace(0, 1, nPts)';

paths = struct('id', {}, 'label', {}, 'thetaFun', {});
paths(1).id = 1; paths(1).label = 'MAP-to-MAP full line';
paths(1).thetaFun = @(t) clampFixed((1 - t) * mapTheta42 + t * mapTheta101, fixedIdx, fixedInit);

paths(2).id = 2; paths(2).label = 'mean-to-mean full line';
paths(2).thetaFun = @(t) clampFixed((1 - t) * meanTheta42 + t * meanTheta101, fixedIdx, fixedInit);

paths(3).id = 3; paths(3).label = 'restricted block: seed42-MAP -> seed101-MAP on [gwbar,gzbar,sig_xi]';
paths(3).thetaFun = @(t) clampFixed(blockPath(mapTheta42, mapTheta101, blockIdx, t), fixedIdx, fixedInit);

paths(4).id = 4; paths(4).label = 'restricted block: seed101-MAP -> seed42-MAP on [gwbar,gzbar,sig_xi]';
paths(4).thetaFun = @(t) clampFixed(blockPath(mapTheta101, mapTheta42, blockIdx, t), fixedIdx, fixedInit);

% ---- 6. evaluate each path ------------------------------------------------
allRows = {};
summaryLines = {};
summaryLines{end+1} = sprintf('BRIDGING LOG-POSTERIOR PROFILE -- %s', datestr(now));
summaryLines{end+1} = sprintf('Sanity gate: max abs err ll=%.3e, lp=%.3e (tol 1e-6) -- PASSED', maxAbsErrLL, maxAbsErrLP);
summaryLines{end+1} = sprintf('seed42  MAP lpost = %.4f (ll=%.4f, lp=%.4f)', lpost42(iMap42), s42.ll(iMap42), s42.lp(iMap42));
summaryLines{end+1} = sprintf('seed101 MAP lpost = %.4f (ll=%.4f, lp=%.4f)', lpost101(iMap101), s101.ll(iMap101), s101.lp(iMap101));
summaryLines{end+1} = sprintf('seed42  weighted-mean lpost = %.4f (ll=%.4f, lp=%.4f)', llMean42+lpMean42, llMean42, lpMean42);
summaryLines{end+1} = sprintf('seed101 weighted-mean lpost = %.4f (ll=%.4f, lp=%.4f)', llMean101+lpMean101, llMean101, lpMean101);
summaryLines{end+1} = sprintf('block coords [gwbar,gzbar,sig_xi] idx = %s', mat2str(blockIdx));
summaryLines{end+1} = sprintf('  seed42-MAP  values: %.4f, %.4f, %.4f', mapTheta42(blockIdx));
summaryLines{end+1} = sprintf('  seed101-MAP values: %.4f, %.4f, %.4f', mapTheta101(blockIdx));
summaryLines{end+1} = '';

figure('Visible', 'off'); hold on;
colors = lines(numel(paths));
legendLabels = cell(numel(paths), 1);

for pk = 1:numel(paths)
    pth = paths(pk);
    ll_v = nan(nPts, 1); lp_v = nan(nPts, 1); lpost_v = nan(nPts, 1);
    infHits = {};
    for k = 1:nPts
        t = tgrid(k);
        tv = pth.thetaFun(t);
        [ll, lp] = evalTheta(tv);
        ll_v(k) = ll; lp_v(k) = lp;
        if lp == -Inf
            lpost_v(k) = -Inf;    % prior-infeasible; ll not computed (NaN)
        else
            lpost_v(k) = ll + lp;
        end
        if ~isfinite(lp) && lp == -Inf
            reasonStr = diagnoseConstraint(P, tv);
            infHits{end+1} = sprintf('t=%.4f: lp=-Inf (%s)', t, reasonStr); %#ok<AGROW>
        elseif ~isfinite(ll) && ll == -Inf
            infHits{end+1} = sprintf('t=%.4f: ll=-Inf (Htilde/Rt non-SPD at this theta)', t); %#ok<AGROW>
        end
        allRows(end+1, :) = {pth.id, t, ll, lp, lpost_v(k)}; %#ok<AGROW>
    end

    % endpoint / dip analysis on FINITE points only
    finiteMask = isfinite(lpost_v);
    hEnd0 = lpost_v(1); hEnd1 = lpost_v(end);
    lowerEnd = min(hEnd0, hEnd1);
    if any(finiteMask)
        minVal = min(lpost_v(finiteMask));
        [~, minIdxLocal] = min(lpost_v(finiteMask));
        finiteIdx = find(finiteMask);
        minIdx = finiteIdx(minIdxLocal);
    else
        minVal = NaN; minIdx = NaN;
    end
    dipBelowLower = lowerEnd - minVal;   % positive = dip below lower endpoint
    dipBelowBoth  = min(hEnd0, hEnd1) - minVal;
    upperEnd = max(hEnd0, hEnd1);
    if any(finiteMask)
        [maxVal, maxIdxLocal] = max(lpost_v(finiteMask));
        finiteIdx2 = find(finiteMask);
        maxIdx = finiteIdx2(maxIdxLocal);
        humpAboveUpper = maxVal - upperEnd;  % positive = interior point beats BOTH endpoints
    else
        maxVal = NaN; maxIdx = NaN; humpAboveUpper = NaN;
    end

    if dipBelowLower > 10
        verdict = 'SEPARATED-MODES';
    elseif dipBelowLower < 3
        verdict = 'CONNECTED-RIDGE';
    else
        verdict = 'SHALLOW-SADDLE';
    end

    fprintf('\n=== Path %d: %s ===\n', pth.id, pth.label);
    fprintf('  endpoint t=0: lpost=%.4f (ll=%.4f, lp=%.4f)\n', hEnd0, ll_v(1), lp_v(1));
    fprintf('  endpoint t=1: lpost=%.4f (ll=%.4f, lp=%.4f)\n', hEnd1, ll_v(end), lp_v(end));
    if isnan(minVal)
        fprintf('  ALL POINTS -Inf (fully infeasible path)\n');
    else
        fprintf('  min lpost=%.4f at t=%.4f (idx %d)\n', minVal, tgrid(minIdx), minIdx);
    end
    fprintf('  dip below lower endpoint: %.4f log points\n', dipBelowLower);
    if humpAboveUpper > 0.5
        fprintf('  NOTE: interior hump ABOVE both endpoints: max lpost=%.4f at t=%.4f (+%.4f above higher endpoint)\n', maxVal, tgrid(maxIdx), humpAboveUpper);
    end
    fprintf('  n finite points: %d / %d\n', nnz(finiteMask), nPts);
    if ~isempty(infHits)
        fprintf('  -Inf hits (%d):\n', numel(infHits));
        for ih = 1:min(numel(infHits), 10)
            fprintf('    %s\n', infHits{ih});
        end
        if numel(infHits) > 10
            fprintf('    ... and %d more\n', numel(infHits) - 10);
        end
    end
    fprintf('  VERDICT: %s\n', verdict);

    summaryLines{end+1} = sprintf('--- Path %d: %s ---', pth.id, pth.label); %#ok<AGROW>
    summaryLines{end+1} = sprintf('  endpoint t=0 (start): lpost=%.4f (ll=%.4f, lp=%.4f)', hEnd0, ll_v(1), lp_v(1)); %#ok<AGROW>
    summaryLines{end+1} = sprintf('  endpoint t=1 (end):   lpost=%.4f (ll=%.4f, lp=%.4f)', hEnd1, ll_v(end), lp_v(end)); %#ok<AGROW>
    if isnan(minVal)
        summaryLines{end+1} = '  ALL POINTS -Inf (fully infeasible path)'; %#ok<AGROW>
    else
        summaryLines{end+1} = sprintf('  minimum: lpost=%.4f at t=%.4f', minVal, tgrid(minIdx)); %#ok<AGROW>
    end
    summaryLines{end+1} = sprintf('  dip below lower endpoint (log points): %.4f', dipBelowLower); %#ok<AGROW>
    if humpAboveUpper > 0.5
        summaryLines{end+1} = sprintf('  NOTE: interior hump ABOVE both endpoints: max lpost=%.4f at t=%.4f (%.4f log points above the HIGHER endpoint) -- evidence FOR connectivity, not a barrier', maxVal, tgrid(maxIdx), humpAboveUpper); %#ok<AGROW>
    end
    summaryLines{end+1} = sprintf('  n finite / total points: %d / %d', nnz(finiteMask), nPts); %#ok<AGROW>
    if ~isempty(infHits)
        summaryLines{end+1} = sprintf('  -Inf hits (%d total):', numel(infHits)); %#ok<AGROW>
        for ih = 1:numel(infHits)
            summaryLines{end+1} = ['    ' infHits{ih}]; %#ok<AGROW>
        end
    end
    summaryLines{end+1} = sprintf('  VERDICT: %s', verdict); %#ok<AGROW>
    summaryLines{end+1} = ''; %#ok<AGROW>

    plotVals = lpost_v; plotVals(~isfinite(plotVals)) = NaN;
    plot(tgrid, plotVals, '-o', 'Color', colors(pk, :), 'MarkerSize', 3, 'LineWidth', 1.5);
    legendLabels{pk} = sprintf('Path %d: %s', pth.id, pth.label);
end

xlabel('t (0 = start, 1 = end)');
ylabel('log-posterior (ll + lp)');
title('Bridging log-posterior profiles: seed42 <-> seed101');
legend(legendLabels, 'Location', 'best', 'Interpreter', 'none');
grid on;
saveas(gcf, fullfile(scratchDir, 'bridge_profiles.png'));
fprintf('\nWrote %s\n', fullfile(scratchDir, 'bridge_profiles.png'));

% ---- 7. write outputs ------------------------------------------------------
T = cell2table(allRows, 'VariableNames', {'path_id', 't', 'll', 'lp', 'lpost'});
writetable(T, fullfile(scratchDir, 'bridge_profiles.csv'));
fprintf('Wrote %s\n', fullfile(scratchDir, 'bridge_profiles.csv'));

fid = fopen(fullfile(scratchDir, 'bridge_summary.txt'), 'w');
for k = 1:numel(summaryLines)
    fprintf(fid, '%s\n', summaryLines{k});
end
fclose(fid);
fprintf('Wrote %s\n', fullfile(scratchDir, 'bridge_summary.txt'));

fprintf('\n=== bridge_main done %s ===\n', datestr(now));
diary off;
end

% ==========================================================================
function [ll, lp] = evalOne(P, dat, cache, tv)
th = jointstar.thetaStruct(P, tv);
lp = jointstar.priorLogPdf(P, tv);
if ~isfinite(lp)
    ll = NaN;   % prior-infeasible: don't bother building the system;
                % lpost is -Inf regardless of what ll would have been
    return
end
spec = jointstar.ModelSpec.jointstar(th, dat, [], cache);
ll = jointstar.computeLogLik(spec.system(), dat.y, cache);
end

% ==========================================================================
function tv = clampFixed(tv, fixedIdx, fixedInit)
% Hold 'fixed' (calibrated, non-mutated) coordinates exactly at their init
% value -- see the note at the call sites for why this is necessary
% (floating-point interpolation is not guaranteed bit-exact even when both
% endpoints share the same value).
tv(fixedIdx) = fixedInit;
end

% ==========================================================================
function tv = blockPath(thetaStart, thetaEnd, blockIdx, t)
tv = thetaStart;
tv(blockIdx) = (1 - t) * thetaStart(blockIdx) + t * thetaEnd(blockIdx);
end

% ==========================================================================
function reasonStr = diagnoseConstraint(P, tv)
% Identify which prior constraint is violated at tv (best-effort, mirrors
% jointstar.priorLogPdf's checks).
prm = P.params;
reasonStr = 'unknown';
for j = 1:P.d
    x = tv(j); q = prm(j);
    switch q.type
        case 'tnorm'
            if x < q.lo || x > q.hi
                reasonStr = sprintf('%s (tnorm) out of [%.4g,%.4g]: x=%.4g', q.name, q.lo, q.hi, x);
                return
            end
        case 'unif'
            if x < q.lo || x > q.hi
                reasonStr = sprintf('%s (unif) out of [%.4g,%.4g]: x=%.4g', q.name, q.lo, q.hi, x);
                return
            end
        case 'beta'
            if x <= 0 || x >= 1
                reasonStr = sprintf('%s (beta) out of (0,1): x=%.4g', q.name, x);
                return
            end
        case 'negbeta'
            if x <= -1 || x >= 0
                reasonStr = sprintf('%s (negbeta) out of (-1,0): x=%.4g', q.name, x);
                return
            end
        case 'igsd'
            if x <= 0
                reasonStr = sprintf('%s (igsd, shock/meas sd) <= 0: x=%.4g', q.name, x);
                return
            end
        case 'logn'
            if x <= 0
                reasonStr = sprintf('%s (logn) <= 0: x=%.4g', q.name, x);
                return
            end
        case 'tgamma'
            if x < q.lo
                reasonStr = sprintf('%s (tgamma) < %.4g: x=%.4g', q.name, q.lo, x);
                return
            end
        case 'fixed'
            if x ~= q.init
                reasonStr = sprintf('%s (fixed) moved off %.4g: x=%.4g', q.name, q.init, x);
                return
            end
    end
end
if isfield(P, 'kap')
    kp = P.kap;
    kv = tv(kp.kapCols)';
    if any(kv < 1)
        reasonStr = 'hierarchical kappa < 1 (truncation violated)';
        return
    end
end
f2 = tv(P.idx.phi2); f1 = tv(P.idx.phisum) - f2;
if ~(abs(f2) < 1 && f1 + f2 < 1 && f2 - f1 < 1)
    reasonStr = sprintf('gap AR(2) non-stationary: phi1=%.4g phi2=%.4g (need |phi2|<1, phi1+phi2<1, phi2-phi1<1)', f1, f2);
    return
end
end
