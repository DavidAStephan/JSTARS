function out = profileLikelihood()
%PROFILELIKELIHOOD Profiling-only script (does not modify +jointstar).
%
%   Builds the full-spec JointSTAR problem (loadData with PieObs,
%   horseshoePriors('HierKappa',true), ModelSpec.jointstar) at a
%   prior-drawn feasible theta, then:
%     1. times jointstar.computeLogLik end-to-end (median over 200 calls)
%     2. breaks down the internal stages by timing a COPIED fragment of
%        computeLogLik.m's code (read-only profiling; the package file
%        itself is never touched)
%     3. runs the MATLAB profiler over 100 evals and reports the top-15
%        functions by self-time
%
%   Run from repo root:
%     /Applications/MATLAB_R2026a.app/bin/matlab -batch "cd(pwd); addpath('benchmarks'); profileLikelihood"

rng(1);
maxNumCompThreads(1);   % single-threaded per the brief

fprintf('=== building full-spec problem ===\n');
dat = jointstar.loadData('data.csv', 'PieObs', true);
P = jointstar.horseshoePriors('HierKappa', true);
fprintf('dat.T=%d, p=%d obs, P.d=%d params\n', dat.T, numel(dat.obsNames), P.d);

% ---- draw a feasible theta from the prior (rejection until finite ll) --
theta = [];
ll0 = -Inf;
for attempt = 1:500
    cand = jointstar.priorSample(P, 1);
    ll0 = logLikTheta(P, cand, dat);
    if isfinite(ll0)
        theta = cand;
        break
    end
end
if isempty(theta)
    error('profileLikelihood:noFeasibleDraw', ...
        'no feasible prior draw found in 500 attempts');
end
fprintf('feasible prior-drawn theta found on attempt %d, loglik = %.3f\n', ...
    attempt, ll0);

th = jointstar.thetaStruct(P, theta);
[Lq, Lr] = jointstar.hsUnpack(P, theta);
spec = jointstar.ModelSpec.jointstar(th, dat, struct('Lq', Lq, 'Lr', Lr));
sys = spec.system();
y = dat.y;
[m, T, p] = deal(size(sys.A1, 1), sys.T, size(y, 1));
fprintf('system dims: m=%d, T=%d, p=%d, stacked m*T=%d\n', m, T, p, m * T);

% ==================================================================
fprintf('\n=== (1) computeLogLik end-to-end timing (n=200) ===\n');
nRep = 200;
times = zeros(nRep, 1);
for k = 1:nRep
    t0 = tic;
    jointstar.computeLogLik(sys, y);
    times(k) = toc(t0);
end
fprintf('computeLogLik: median %.3f ms, mean %.3f ms, min %.3f ms, max %.3f ms\n', ...
    1000 * median(times), 1000 * mean(times), 1000 * min(times), 1000 * max(times));

fprintf('\n=== full pipeline timing (theta row -> logL), n=200 ===\n');
nRep2 = 200;
times2 = zeros(nRep2, 1);
for k = 1:nRep2
    t0 = tic;
    logLikTheta(P, theta, dat);
    times2(k) = toc(t0);
end
fprintf('thetaStruct+hsUnpack+ModelSpec.jointstar+computeLogLik: median %.3f ms, mean %.3f ms\n', ...
    1000 * median(times2), 1000 * mean(times2));

nRep3 = 200;
times3 = zeros(nRep3, 1);
for k = 1:nRep3
    t0 = tic;
    spec3 = jointstar.ModelSpec.jointstar(th, dat, struct('Lq', Lq, 'Lr', Lr));
    spec3.system();
    times3(k) = toc(t0);
end
fprintf('ModelSpec.jointstar + system(): median %.3f ms\n', 1000 * median(times3));

nRep4 = 200;
times4 = zeros(nRep4, 1);
for k = 1:nRep4
    t0 = tic;
    jointstar.hsUnpack(P, theta);
    times4(k) = toc(t0);
end
fprintf('hsUnpack: median %.3f ms\n', 1000 * median(times4));

nRep5 = 200;
times5 = zeros(nRep5, 1);
for k = 1:nRep5
    t0 = tic;
    jointstar.thetaStruct(P, theta);
    times5(k) = toc(t0);
end
fprintf('thetaStruct: median %.4f ms\n', 1000 * median(times5));

% ==================================================================
fprintf('\n=== (internal breakdown, copied fragments, n=%d) ===\n', nRep);
br = internalBreakdown(sys, y, nRep);
fn = fieldnames(br);
totalMs = 0;
for i = 1:numel(fn)
    totalMs = totalMs + br.(fn{i});
end
for i = 1:numel(fn)
    fprintf('  %-28s %8.3f ms  (%5.1f%%)\n', fn{i}, br.(fn{i}), 100 * br.(fn{i}) / totalMs);
end
fprintf('  %-28s %8.3f ms\n', 'SUM of fragments', totalMs);

% ==================================================================
fprintf('\n=== (2) MATLAB profiler over 100 evals, single-threaded ===\n');
profile('on', '-history');
for k = 1:100
    jointstar.computeLogLik(sys, y);
end
pinfo = profile('info');
profile('off');
save(fullfile(fileparts(mfilename('fullpath')), 'profile_info.mat'), 'pinfo');
printTopFunctions(pinfo, 15);

% ==================================================================
% (3) CACHED PATH: jointstar.buildEvalCache + the cache-aware fast path
% in ModelSpec.jointstar / computeLogLik.  Before/after comparison for
% the eval-cache performance pass; see jointstar.buildEvalCache and
% benchmarks/verifyCacheEquivalence.m for the correctness verification
% (bitwise-identical to everything measured above).
% ==================================================================
fprintf('\n=== (3) CACHED PATH: computeLogLik end-to-end timing (n=200) ===\n');
cache = jointstar.buildEvalCache(dat, P);
specC = jointstar.ModelSpec.jointstar(th, dat, struct('Lq', Lq, 'Lr', Lr), cache);
sysC = specC.system();
timesC = zeros(nRep, 1);
for k = 1:nRep
    t0 = tic;
    jointstar.computeLogLik(sysC, y, cache);
    timesC(k) = toc(t0);
end
fprintf('computeLogLik (cached): median %.3f ms, mean %.3f ms, min %.3f ms, max %.3f ms\n', ...
    1000 * median(timesC), 1000 * mean(timesC), 1000 * min(timesC), 1000 * max(timesC));
fprintf('speedup vs (1) no-cache computeLogLik: %.2fx\n', median(times) / median(timesC));

fprintf('\n=== (3) CACHED PATH: full pipeline timing (theta row -> logL), n=200 ===\n');
timesC2 = zeros(nRep2, 1);
for k = 1:nRep2
    t0 = tic;
    logLikThetaCached(P, theta, dat, cache);
    timesC2(k) = toc(t0);
end
fprintf('thetaStruct+hsUnpack+ModelSpec.jointstar+computeLogLik (cached): median %.3f ms, mean %.3f ms\n', ...
    1000 * median(timesC2), 1000 * mean(timesC2));
fprintf('speedup vs full-pipeline no-cache: %.2fx\n', median(times2) / median(timesC2));

fprintf('\n=== (3) CACHED PATH: MATLAB profiler over 100 evals, single-threaded ===\n');
profile('on', '-history');
for k = 1:100
    jointstar.computeLogLik(sysC, y, cache);
end
pinfoC = profile('info');
profile('off');
save(fullfile(fileparts(mfilename('fullpath')), 'profile_info_cached.mat'), 'pinfoC');
printTopFunctions(pinfoC, 15);

out = struct('ll0', ll0, 'medianMs', 1000 * median(times), 'breakdown', br, ...
    'medianMsCached', 1000 * median(timesC), ...
    'm', m, 'T', T, 'p', p, 'sys', sys, 'y', y, 'theta', theta, 'P', P, 'dat', dat, ...
    'cache', cache);
end

% ======================================================================
function ll = logLikThetaCached(P, tv, dat, cache)
th = jointstar.thetaStruct(P, tv);
if isfield(P, 'hs')
    [Lq, Lr] = jointstar.hsUnpack(P, tv);
    spec = jointstar.ModelSpec.jointstar(th, dat, struct('Lq', Lq, 'Lr', Lr), cache);
else
    spec = jointstar.ModelSpec.jointstar(th, dat, [], cache);
end
ll = jointstar.computeLogLik(spec.system(), dat.y, cache);
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

% ======================================================================
function br = internalBreakdown(sys, y, nRep)
%INTERNALBREAKDOWN Copied-and-instrumented fragment of computeLogLik.m
%   (read-only profiling probe -- the package file is never edited).
%   Times each documented stage, accumulated over nRep repeats, reported
%   as ms/eval.

acc = struct('unpackDims', 0, 'Sinv_assembly', 0, 'H_assembly', 0, ...
    'Halpha_matmul', 0, 'priorMean', 0, 'ZR_assembly', 0, ...
    'Htilde_assembly', 0, 'chol_Htilde', 0, 'triSolves', 0, ...
    'logdet_final', 0);

for rep = 1:nRep
    t0 = tic;
    [m, T, p] = sysDimsLocal(sys, y); %#ok<ASGLU>
    acc.unpackDims = acc.unpackDims + toc(t0);

    % ---- S^{-1} and log|S| ----
    t0 = tic;
    constQ = ismatrix(sys.Q);
    nBlk = T;
    iS = zeros(m * m * nBlk, 1); jS = iS; vS = iS;
    logdetS = 0;
    ptr = 0;
    [P1inv, ld] = invSPDLocal(sys.P1, 'P1'); %#ok<ASGLU>
    logdetS = logdetS + ld;
    [iS, jS, vS, ptr] = putBlockLocal(iS, jS, vS, ptr, P1inv, 0, 0);
    if constQ
        [Qinv, ldQ, okQ] = tryInvSPDLocal(sys.Q);
        if ~okQ, error('infeasible Q at this draw'); end
    end
    for t = 2:T
        if ~constQ
            [Qinv, ldQ, okQ] = tryInvSPDLocal(sys.Q(:, :, t));
            if ~okQ, error('infeasible Q at this draw'); end
        end
        logdetS = logdetS + ldQ;
        off = (t - 1) * m;
        [iS, jS, vS, ptr] = putBlockLocal(iS, jS, vS, ptr, Qinv, off, off);
    end
    Sinv = sparse(iS(1:ptr), jS(1:ptr), vS(1:ptr), m * T, m * T);
    logdetHalpha = -logdetS;
    acc.Sinv_assembly = acc.Sinv_assembly + toc(t0);

    % ---- H ----
    t0 = tic;
    nzH = m * T + m * m * (T - 1) + m * m * max(T - 2, 0) * ~isempty(sys.A2);
    iH = zeros(nzH, 1); jH = iH; vH = iH;
    iH(1:m * T) = (1:m * T)'; jH(1:m * T) = (1:m * T)'; vH(1:m * T) = 1;
    ptr = m * T;
    for t = 2:T
        A1t = slice3Local(sys.A1, t, m);
        [iH, jH, vH, ptr] = putBlockLocal(iH, jH, vH, ptr, -A1t, (t - 1) * m, (t - 2) * m);
    end
    if ~isempty(sys.A2)
        for t = 3:T
            A2t = slice3Local(sys.A2, t, m);
            [iH, jH, vH, ptr] = putBlockLocal(iH, jH, vH, ptr, -A2t, (t - 1) * m, (t - 3) * m);
        end
    end
    H = sparse(iH(1:ptr), jH(1:ptr), vH(1:ptr), m * T, m * T);
    acc.H_assembly = acc.H_assembly + toc(t0);

    % ---- Halpha = H' * Sinv * H ----
    t0 = tic;
    Halpha = H' * Sinv * H;
    acc.Halpha_matmul = acc.Halpha_matmul + toc(t0);

    % ---- prior mean contribution ----
    t0 = tic;
    cvec = zeros(m * T, 1);
    cvec(1:m) = sys.a1(:);
    if ~isempty(sys.c)
        if isvector(sys.c)
            cvec(m + 1:end) = repmat(sys.c(:), T - 1, 1);
        else
            cvec(m + 1:end) = reshape(sys.c(:, 2:T), [], 1);
        end
    end
    if any(cvec)
        Sc = Sinv * cvec;
        bPrior = H' * Sc;
        quadPrior = cvec' * Sc;
    else
        bPrior = zeros(m * T, 1);
        quadPrior = 0;
    end
    acc.priorMean = acc.priorMean + toc(t0);

    % ---- stacked measurement rows ----
    t0 = tic;
    obsMask = ~isnan(y);
    nObs = nnz(obsMask);
    constZ = ismatrix(sys.Z);
    hasZlag = isfield(sys, 'Zlag') && ~isempty(sys.Zlag);
    if hasZlag, constZL = ismatrix(sys.Zlag); end
    hasRfull = isfield(sys, 'Rfull') && ~isempty(sys.Rfull);
    if hasRfull, constR = ismatrix(sys.Rfull); end
    iZ = zeros(nObs * m * (1 + hasZlag), 1); jZ = iZ; vZ = iZ;
    iR = zeros(nObs * p, 1); jR = iR; vR = iR;
    ptrR = 0;
    halfLogDetRinv = 0;
    yv = zeros(nObs, 1);
    ptr = 0; row = 0;
    for t = 1:T
        idx = find(obsMask(:, t));
        if isempty(idx), continue; end
        if constZ, Zt = sys.Z; else, Zt = sys.Z(:, :, t); end
        yt = y(idx, t);
        if ~isempty(sys.d)
            dt = colSliceLocal(sys.d, t);
            yt = yt - dt(idx);
        end
        if hasZlag && t > 1
            if constZL, ZLt = sys.Zlag; else, ZLt = sys.Zlag(:, :, t); end
        else
            ZLt = [];
        end
        nIdx = numel(idx);
        if hasRfull
            if constR, Rt = sys.Rfull(idx, idx); else, Rt = sys.Rfull(idx, idx, t); end
            [LR, flagR] = chol((Rt + Rt') / 2, 'lower');
            if flagR ~= 0, error('infeasible Rfull at this draw'); end
            Rinv_t = LR' \ (LR \ eye(nIdx));
            Rinv_t = (Rinv_t + Rinv_t') / 2;
            halfLogDetRinv = halfLogDetRinv - sum(log(diag(LR)));
            [rr, cc, vv] = find(Rinv_t);
            nR = numel(vv);
            iR(ptrR + 1:ptrR + nR) = row + rr;
            jR(ptrR + 1:ptrR + nR) = row + cc;
            vR(ptrR + 1:ptrR + nR) = vv;
            ptrR = ptrR + nR;
        else
            Rt = colSliceLocal(sys.Rdiag, t);
            rinv_t = 1 ./ Rt(idx);
            halfLogDetRinv = halfLogDetRinv + 0.5 * sum(log(rinv_t));
            iR(ptrR + 1:ptrR + nIdx) = row + (1:nIdx)';
            jR(ptrR + 1:ptrR + nIdx) = row + (1:nIdx)';
            vR(ptrR + 1:ptrR + nIdx) = rinv_t;
            ptrR = ptrR + nIdx;
        end
        for kk = 1:nIdx
            row = row + 1;
            zrow = Zt(idx(kk), :);
            nz = find(zrow);
            nnz_r = numel(nz);
            iZ(ptr + 1:ptr + nnz_r) = row;
            jZ(ptr + 1:ptr + nnz_r) = (t - 1) * m + nz;
            vZ(ptr + 1:ptr + nnz_r) = zrow(nz);
            ptr = ptr + nnz_r;
            if ~isempty(ZLt)
                zlrow = ZLt(idx(kk), :);
                nz = find(zlrow);
                nnz_r = numel(nz);
                iZ(ptr + 1:ptr + nnz_r) = row;
                jZ(ptr + 1:ptr + nnz_r) = (t - 2) * m + nz;
                vZ(ptr + 1:ptr + nnz_r) = zlrow(nz);
                ptr = ptr + nnz_r;
            end
            yv(row) = yt(kk);
        end
    end
    Zs = sparse(iZ(1:ptr), jZ(1:ptr), vZ(1:ptr), nObs, m * T);
    RinvSp = sparse(iR(1:ptrR), jR(1:ptrR), vR(1:ptrR), nObs, nObs);
    acc.ZR_assembly = acc.ZR_assembly + toc(t0);

    % ---- Htilde, b ----
    t0 = tic;
    Ry = RinvSp * yv;
    Htilde = Halpha + Zs' * RinvSp * Zs;
    b = Zs' * Ry + bPrior;
    acc.Htilde_assembly = acc.Htilde_assembly + toc(t0);

    % ---- chol ----
    t0 = tic;
    [L, flag, perm] = chol(Htilde, 'lower', 'vector');
    if flag ~= 0, error('non-SPD Htilde at this draw'); end
    acc.chol_Htilde = acc.chol_Htilde + toc(t0);

    % ---- triangular solves ----
    t0 = tic;
    v = L \ b(perm);
    muTp = L' \ v; %#ok<NASGU>
    acc.triSolves = acc.triSolves + toc(t0);

    % ---- logdet + final scalar assembly ----
    t0 = tic;
    logdetHtilde = 2 * sum(log(full(diag(L))));
    yRy = yv' * Ry;
    logL = -0.5 * nObs * log(2 * pi) + halfLogDetRinv ...
        + 0.5 * logdetHalpha - 0.5 * logdetHtilde ...
        - 0.5 * (yRy + quadPrior - v' * v); %#ok<NASGU>
    acc.logdet_final = acc.logdet_final + toc(t0);
end

fn = fieldnames(acc);
br = struct();
for i = 1:numel(fn)
    br.(fn{i}) = 1000 * acc.(fn{i}) / nRep;   % ms/eval
end
end

function [m, T, p] = sysDimsLocal(sys, y)
m = size(sys.A1, 1);
T = sys.T;
p = size(y, 1);
end

function [Ainv, logdetA] = invSPDLocal(A, name)
[Ainv, logdetA, ok] = tryInvSPDLocal(A);
if ~ok
    error('jointstar:notSPD', '%s is not positive definite.', name);
end
end

function [Ainv, logdetA, ok] = tryInvSPDLocal(A)
if any(~isfinite(A(:)))
    Ainv = []; logdetA = NaN; ok = false; return;
end
[LA, fl] = chol((A + A') / 2, 'lower');
if fl ~= 0
    Ainv = []; logdetA = NaN; ok = false; return;
end
ok = true;
Ainv = LA' \ (LA \ eye(size(A, 1)));
Ainv = (Ainv + Ainv') / 2;
logdetA = 2 * sum(log(diag(LA)));
end

function B = slice3Local(A, t, m)
if ismatrix(A), B = A; else, B = A(:, :, t); end
if isscalar(B) && m > 1
    error('jointstar:dimMismatch', 'scalar system matrix with m > 1.');
end
end

function ct = colSliceLocal(C, t)
if isvector(C), ct = C(:); else, ct = C(:, t); end
end

function [ii, jj, vv, ptr] = putBlockLocal(ii, jj, vv, ptr, B, rowOff, colOff)
[r, c, v] = find(B);
n = numel(v);
ii(ptr + 1:ptr + n) = rowOff + r;
jj(ptr + 1:ptr + n) = colOff + c;
vv(ptr + 1:ptr + n) = v;
ptr = ptr + n;
end

% ======================================================================
function printTopFunctions(pinfo, topN)
FT = pinfo.FunctionTable;
n = numel(FT);
selfTime = zeros(n, 1);
for i = 1:n
    childTotal = 0;
    for c = 1:numel(FT(i).Children)
        childTotal = childTotal + FT(i).Children(c).TotalTime;
    end
    selfTime(i) = FT(i).TotalTime - childTotal;
end
[sorted, order] = sort(selfTime, 'descend');
topN = min(topN, n);
fprintf('%-4s %-45s %10s %10s %8s %s\n', 'rank', 'function', 'self(ms)', 'total(ms)', 'ncalls', 'file');
for k = 1:topN
    i = order(k);
    [~, fname, ext] = fileparts(FT(i).FileName);
    fprintf('%-4d %-45s %10.2f %10.2f %8d %s%s\n', k, FT(i).FunctionName, ...
        1000 * sorted(k), 1000 * FT(i).TotalTime, FT(i).NumCalls, fname, ext);
end
end
