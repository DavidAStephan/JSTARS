function [logL, aux] = computeLogLik(sys, y, cache)
%COMPUTELOGLIK Marginal log-likelihood via the sparse precision sampler.
%
%   [logL, aux] = jointstar.computeLogLik(sys, y)
%   [logL, aux] = jointstar.computeLogLik(sys, y, cache)
%
%   Chan-Jeliazkov (2009) style: stack all latent states across time into
%   one long vector alpha = (alpha_1', ..., alpha_T')' and build the joint
%   prior precision H_alpha = H' * S^{-1} * H, where H is the (unit lower
%   triangular, block-banded) transition-difference operator and
%   S = blkdiag(P1, Q_2, ..., Q_T).  H_alpha is block-tridiagonal for
%   first-order transitions and block-pentadiagonal where a second lag
%   (AR(2) cycle) enters.  Everything is assembled with MATLAB native
%   sparse matrices; nothing is ever densified.
%
%   The posterior state precision is Htilde = H_alpha + Z'*Rinv*Z and
%
%     log p(y|theta) = -n/2*log(2*pi) + 1/2*log|Rinv| + 1/2*log|H_alpha|
%                      - 1/2*log|Htilde|
%                      - 1/2*(y'*Rinv*y + mu_a'*H_alpha*mu_a - b'*mu_tilde)
%
%   with b = Z'*Rinv*y + H_alpha*mu_a and Htilde*mu_tilde = b.  Sparse
%   Cholesky with a fill-reducing permutation (chol(..., 'lower',
%   'vector')) supplies both log-determinants and the solve.
%
%   Missing observations (NaN in y) are handled by masking their rows out
%   of Z and Rinv -- never by imputation.
%
%   Inputs:
%     sys  struct from ModelSpec.system() (fields A1, A2, c, Q, a1, P1,
%          Z, d, Rdiag, T; slices constant or per-t).  Optional field
%          Zlag (p x m, or p x m x T): measurement loadings on the LAGGED
%          state, y_t = d_t + Z_t*alpha_t + Zlag_t*alpha_{t-1} + eps_t,
%          giving Z'R^{-1}Z its off-diagonal blocks.  Zlag is ignored at
%          t = 1; rows that need a lag the data cannot supply must be
%          masked upstream (NaN in y).  Optional field Rfull (p x p, or
%          p x p x T): full measurement-error covariance, used instead of
%          Rdiag when present; missing observations are handled by taking
%          the observed submatrix of Rfull at each t (exact Gaussian
%          marginalisation).  A non-PD Rfull submatrix returns -Inf like
%          the chol guard on Htilde.  Optional fields Quniq/Runiq (see
%          jointstar.ModelSpec.jointstar): the cache-aware path's compact
%          unique-slice representation of Q/Rfull, consumed only when
%          cache is given.
%     y    p x T data matrix, NaN = missing.
%     cache  optional struct from jointstar.buildEvalCache.  When given,
%          computeLogLik assembles the SAME triplets in a way that avoids
%          recomputation that is redundant for a FIXED run (data +
%          configuration): the regime-grouped Q/Rfull inverses are
%          computed once per distinct slice instead of once per t
%          (typically ~4-12 inversions instead of ~T), and the repeated
%          find() calls the assembly loop makes on bit-identical rows/
%          blocks (Z, Zlag are theta-dependent in VALUE but constant in
%          time; A1/A2/Q/Rfull are piecewise-constant in time) are done
%          once per distinct group via a setup pass and the per-(t,kk)
%          hot loop only does plain numeric-array reads (no cell arrays,
%          no branching -- both measurably matter; see
%          benchmarks/profileLikelihood.m).  Every one of these is an
%          EXACT memoization: the cached value is provably bit-identical
%          to what the naive per-t recomputation would produce, because
%          the underlying array slices are bit-identical copies by
%          construction (see jointstar.ModelSpec.jointstar).  The sparse
%          Cholesky of Htilde is deliberately NOT given this treatment --
%          an earlier version reused a cached fill-reducing permutation
%          to skip 'vector' mode's own AMD step, which measurably failed
%          bitwise equivalence (~1e-9 relative in logL; see the note at
%          the chol() call in evalWithCache below) and was reverted.
%          Every remaining piece of the cache-aware path is verified
%          bit-for-bit against the no-cache path over ~350 prior draws
%          across both prior families and both PieObs configurations;
%          see benchmarks/verifyCacheEquivalence.m and
%          tests/testEvalCache.m.  Omitted or empty => identical to the
%          pre-existing behaviour; existing callers are unaffected.
%
%   Outputs:
%     logL  scalar marginal log-likelihood.  -Inf (never an error) if
%           Htilde is numerically non-SPD at this theta: an infeasible
%           parameter draw is a rejection, not a crash.
%     aux   struct with the pieces needed to draw states without
%           refactorising: L, perm, muTilde, plus m, T.  Pass to
%           jointstar.drawStates.
%
%   See also jointstar.drawStates, jointstar.ModelSpec,
%   jointstar.buildEvalCache.

if nargin < 3, cache = []; end
if isempty(cache)
    if nargout > 1
        [logL, aux] = evalNoCache(sys, y);
    else
        logL = evalNoCache(sys, y);
    end
else
    if nargout > 1
        [logL, aux] = evalWithCache(sys, y, cache);
    else
        logL = evalWithCache(sys, y, cache);
    end
end
end

% ======================================================================
% ORIGINAL (no-cache) implementation.  Verbatim behaviour preserved so
% every existing caller (tests, toy models, benchmarks) is unaffected.
% ======================================================================
function [logL, aux] = evalNoCache(sys, y)

[m, T, p] = sysDims(sys, y);

% ---- S^{-1} and log|S| from the small (m x m) innovation blocks -------
% Blocks are tiny (m <= ~16), so dense chol per block is cheap and gives
% log-determinants directly; results go straight into sparse triplets.
constQ = ismatrix(sys.Q);
nBlk = T;
iS = zeros(m * m * nBlk, 1); jS = iS; vS = iS;
logdetS = 0;
ptr = 0;

[P1inv, ld] = invSPD(sys.P1, 'P1');   % P1 is fixed spec, not theta: throw
logdetS = logdetS + ld;
[iS, jS, vS, ptr] = putBlock(iS, jS, vS, ptr, P1inv, 0, 0);

% Q depends on theta: a chol failure means an infeasible draw (e.g.
% overflowing horseshoe tails) -> reject with -Inf, never crash
if constQ
    [Qinv, ldQ, okQ] = tryInvSPD(sys.Q);
    if ~okQ, logL = -Inf; aux = []; return; end
end
for t = 2:T
    if ~constQ
        [Qinv, ldQ, okQ] = tryInvSPD(sys.Q(:, :, t));
        if ~okQ, logL = -Inf; aux = []; return; end
    end
    logdetS = logdetS + ldQ;
    off = (t - 1) * m;
    [iS, jS, vS, ptr] = putBlock(iS, jS, vS, ptr, Qinv, off, off);
end
Sinv = sparse(iS(1:ptr), jS(1:ptr), vS(1:ptr), m * T, m * T);
logdetHalpha = -logdetS;                    % |H| = 1 (unit triangular)

% ---- H: unit lower block-banded transition operator -------------------
nzH = m * T + m * m * (T - 1) + m * m * max(T - 2, 0) * ~isempty(sys.A2);
iH = zeros(nzH, 1); jH = iH; vH = iH;
iH(1:m * T) = (1:m * T)'; jH(1:m * T) = (1:m * T)'; vH(1:m * T) = 1;
ptr = m * T;
for t = 2:T
    A1t = slice3(sys.A1, t, m);
    [iH, jH, vH, ptr] = putBlock(iH, jH, vH, ptr, -A1t, (t - 1) * m, (t - 2) * m);
end
if ~isempty(sys.A2)
    for t = 3:T
        A2t = slice3(sys.A2, t, m);
        [iH, jH, vH, ptr] = putBlock(iH, jH, vH, ptr, -A2t, (t - 1) * m, (t - 3) * m);
    end
end
H = sparse(iH(1:ptr), jH(1:ptr), vH(1:ptr), m * T, m * T);

Halpha = H' * Sinv * H;

% ---- prior mean contribution ------------------------------------------
% cvec stacks (a1; c_2; ...; c_T); the prior state mean solves
% H*mu_a = cvec, and H_alpha*mu_a = H'*Sinv*cvec,
% mu_a'*H_alpha*mu_a = cvec'*Sinv*cvec.
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

% ---- stacked measurement rows, masked for missing obs -----------------
obsMask = ~isnan(y);
nObs = nnz(obsMask);
if nObs == 0
    error('jointstar:noData', 'y contains no observed values.');
end

constZ = ismatrix(sys.Z);
hasZlag = isfield(sys, 'Zlag') && ~isempty(sys.Zlag);
if hasZlag, constZL = ismatrix(sys.Zlag); end
hasRfull = isfield(sys, 'Rfull') && ~isempty(sys.Rfull);
if hasRfull, constR = ismatrix(sys.Rfull); end
iZ = zeros(nObs * m * (1 + hasZlag), 1); jZ = iZ; vZ = iZ;
iR = zeros(nObs * p, 1); jR = iR; vR = iR;   % block-diag R^{-1} triplets
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
        dt = colSlice(sys.d, t);
        yt = yt - dt(idx);
    end
    if hasZlag && t > 1
        if constZL, ZLt = sys.Zlag; else, ZLt = sys.Zlag(:, :, t); end
    else
        ZLt = [];
    end
    % measurement precision block for the observed subvector at t
    nIdx = numel(idx);
    if hasRfull
        if constR, Rt = sys.Rfull(idx, idx); else, Rt = sys.Rfull(idx, idx, t); end
        [LR, flagR] = chol((Rt + Rt') / 2, 'lower');
        if flagR ~= 0
            logL = -Inf; aux = [];      % infeasible theta, not an error
            return
        end
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
        Rt = colSlice(sys.Rdiag, t);
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

% ---- posterior precision, sparse Cholesky, likelihood -----------------
Ry = RinvSp * yv;
Htilde = Halpha + Zs' * RinvSp * Zs;
b = Zs' * Ry + bPrior;

[L, flag, perm] = chol(Htilde, 'lower', 'vector');
if flag ~= 0
    logL = -Inf;                 % infeasible theta: reject, don't crash
    aux = [];
    return
end

v = L \ b(perm);
muTp = L' \ v;
muTilde = zeros(m * T, 1);
muTilde(perm) = muTp;

logdetHtilde = 2 * sum(log(full(diag(L))));
yRy = yv' * Ry;

logL = -0.5 * nObs * log(2 * pi) + halfLogDetRinv ...
    + 0.5 * logdetHalpha - 0.5 * logdetHtilde ...
    - 0.5 * (yRy + quadPrior - v' * v);

if ~isfinite(logL) && logL ~= -Inf
    error('jointstar:badLogLik', ...
        'log-likelihood is NaN/Inf (logdetHalpha=%.3g, logdetHtilde=%.3g).', ...
        logdetHalpha, logdetHtilde);
end

if nargout > 1
    aux = struct('L', L, 'perm', perm, 'muTilde', muTilde, 'm', m, 'T', T);
end
end

% ======================================================================
% CACHE-AWARE fast path.  Builds the IDENTICAL sparse matrices (Sinv, H,
% Halpha, Zs, RinvSp, Htilde) as evalNoCache above -- every value that
% goes into them is either taken straight from sys (unchanged) or is an
% EXACT memoization (bit-identical repeated sub-computation, not an
% approximation): Q/A1/A2/Rfull are piecewise-constant in t by
% construction (jointstar.ModelSpec.jointstar repmat()s/copies identical
% slices across each regime, and Z/Zlag are constant in t outright), so
% inverting/find()-ing ONE representative member of a cache-identified
% group (a setup loop over the handful of distinct groups, done with
% plain numeric arrays -- NOT cell arrays -- flattened via an
% offset/count index; an earlier version of this pass used cell arrays
% with a lazy isempty() check inside the O(T)/O(nObs) hot loops and
% measured NO net speedup, because MATLAB's cell-array access overhead
% ate the savings from skipping find()/chol(); see
% benchmarks/profileLikelihood.m) and reusing it for the rest of that
% group reproduces exactly what re-deriving it per t would have given.
% The sparse Cholesky of Htilde is NOT cached this way -- see the note
% at the chol() call below for why that specific piece was reverted
% after failing the bitwise-equivalence check.
% ======================================================================
function [logL, aux] = evalWithCache(sys, y, cache)

[m, T, p] = sysDims(sys, y);
if m ~= cache.m || T ~= cache.T || p ~= cache.p
    error('jointstar:cacheMismatch', ...
        ['eval cache dims (m=%d,T=%d,p=%d) do not match sys/y ' ...
         '(m=%d,T=%d,p=%d); rebuild the cache for this run configuration.'], ...
        cache.m, cache.T, cache.p, m, T, p);
end
obsMaskY = ~isnan(y);
if ~isequal(obsMaskY, cache.obsMask)
    error('jointstar:cacheMismatch', ...
        'y''s missing-data pattern does not match the cache''s; rebuild the cache for this run''s data.');
end
nObs = nnz(obsMaskY);
if nObs == 0
    error('jointstar:noData', 'y contains no observed values.');
end

% ---- P1^{-1}: hardcoded model constant, never a function of theta -----
if isequal(sys.P1, cache.P1)
    P1inv = cache.P1inv; ldP1 = cache.logdetP1;
else
    [P1inv, ldP1, okP1] = tryInvSPD(sys.P1);
    if ~okP1, logL = -Inf; aux = []; return; end
end

% ---- Q: invert each unique regime slice ONCE (setup loop over the ~4
% groups, not over T), flatten each group's (i,j,v) pattern into plain
% numeric arrays with an offset/count index -- the T-loop below then does
% pure numeric-array indexing (no cell arrays, no isempty branching) to
% stay off MATLAB's cell-array-access overhead, which measurably erased
% the gain from skipping find()/chol() in an earlier version of this pass
% (see the file's revision history / benchmarks/profileLikelihood.m).
hasQuniq = isfield(sys, 'Quniq') && ~isempty(sys.Quniq);
if hasQuniq
    Qgroup = cache.Qgroup;
    nGQ = size(sys.Quniq, 3);
    ldQvec = zeros(nGQ, 1);
    rowsC = cell(nGQ, 1); colsC = cell(nGQ, 1); valsC = cell(nGQ, 1);
    cntQ = zeros(nGQ, 1);
    for g = 1:nGQ
        [Qinv_g, ldQvec(g), okQ] = tryInvSPD(sys.Quniq(:, :, g));
        if ~okQ, logL = -Inf; aux = []; return; end
        [rowsC{g}, colsC{g}, valsC{g}] = find(Qinv_g);
        cntQ(g) = numel(valsC{g});
    end
    offQ = [0; cumsum(cntQ(1:end - 1))];
    QflatRow = vertcat(rowsC{:}); QflatCol = vertcat(colsC{:}); QflatVal = vertcat(valsC{:});
else
    constQ = ismatrix(sys.Q);
    if constQ
        [Qinv0, ldQ0, okQ] = tryInvSPD(sys.Q);
        if ~okQ, logL = -Inf; aux = []; return; end
        [rr0, cc0, vv0] = find(Qinv0);
    end
end

iS = zeros(m * m * T, 1); jS = iS; vS = iS;
logdetS = ldP1;
ptr = 0;
[iS, jS, vS, ptr] = putBlock(iS, jS, vS, ptr, P1inv, 0, 0);
for t = 2:T
    off = (t - 1) * m;
    if hasQuniq
        g = Qgroup(t);
        logdetS = logdetS + ldQvec(g);
        s = offQ(g); n = cntQ(g);
        iS(ptr + 1:ptr + n) = off + QflatRow(s + 1:s + n);
        jS(ptr + 1:ptr + n) = off + QflatCol(s + 1:s + n);
        vS(ptr + 1:ptr + n) = QflatVal(s + 1:s + n);
    elseif constQ
        logdetS = logdetS + ldQ0;
        n = numel(vv0);
        iS(ptr + 1:ptr + n) = off + rr0;
        jS(ptr + 1:ptr + n) = off + cc0;
        vS(ptr + 1:ptr + n) = vv0;
    else
        [Qinv, ldQ, okQ] = tryInvSPD(sys.Q(:, :, t));
        if ~okQ, logL = -Inf; aux = []; return; end
        logdetS = logdetS + ldQ;
        [iS, jS, vS, ptr] = putBlock(iS, jS, vS, ptr, Qinv, off, off);
        continue
    end
    ptr = ptr + n;
end
Sinv = sparse(iS(1:ptr), jS(1:ptr), vS(1:ptr), m * T, m * T);
logdetHalpha = -logdetS;

% ---- H: same flatten-once-per-group scheme for the handful of distinct
% A1/A2 slices (COVID/IS-availability windows) ---------------------------
nzH = m * T + m * m * (T - 1) + m * m * max(T - 2, 0) * ~isempty(sys.A2);
iH = zeros(nzH, 1); jH = iH; vH = iH;
iH(1:m * T) = (1:m * T)'; jH(1:m * T) = (1:m * T)'; vH(1:m * T) = 1;
ptr = m * T;

A1Group = cache.A1Group;
nGA1 = cache.nGA1;
rowsC = cell(nGA1, 1); colsC = cell(nGA1, 1); valsC = cell(nGA1, 1); cntA1 = zeros(nGA1, 1);
for g = 1:nGA1
    A1g = slice3(sys.A1, cache.A1GroupRep(g), m);
    [rowsC{g}, colsC{g}, valsC{g}] = find(-A1g);
    cntA1(g) = numel(valsC{g});
end
offA1 = [0; cumsum(cntA1(1:end - 1))];
A1flatRow = vertcat(rowsC{:}); A1flatCol = vertcat(colsC{:}); A1flatVal = vertcat(valsC{:});
for t = 2:T
    g = A1Group(t);
    s = offA1(g); n = cntA1(g);
    iH(ptr + 1:ptr + n) = (t - 1) * m + A1flatRow(s + 1:s + n);
    jH(ptr + 1:ptr + n) = (t - 2) * m + A1flatCol(s + 1:s + n);
    vH(ptr + 1:ptr + n) = A1flatVal(s + 1:s + n);
    ptr = ptr + n;
end
if ~isempty(sys.A2)
    A2Group = cache.A2Group;
    nGA2 = cache.nGA2;
    rowsC = cell(nGA2, 1); colsC = cell(nGA2, 1); valsC = cell(nGA2, 1); cntA2 = zeros(nGA2, 1);
    for g = 1:nGA2
        A2g = slice3(sys.A2, cache.A2GroupRep(g), m);
        [rowsC{g}, colsC{g}, valsC{g}] = find(-A2g);
        cntA2(g) = numel(valsC{g});
    end
    offA2 = [0; cumsum(cntA2(1:end - 1))];
    A2flatRow = vertcat(rowsC{:}); A2flatCol = vertcat(colsC{:}); A2flatVal = vertcat(valsC{:});
    for t = 3:T
        g = A2Group(t);
        s = offA2(g); n = cntA2(g);
        iH(ptr + 1:ptr + n) = (t - 1) * m + A2flatRow(s + 1:s + n);
        jH(ptr + 1:ptr + n) = (t - 3) * m + A2flatCol(s + 1:s + n);
        vH(ptr + 1:ptr + n) = A2flatVal(s + 1:s + n);
        ptr = ptr + n;
    end
end
H = sparse(iH(1:ptr), jH(1:ptr), vH(1:ptr), m * T, m * T);

Halpha = H' * Sinv * H;

% ---- prior mean contribution (cheap; unchanged from the no-cache path) -
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

% ---- stacked measurement rows: cached idx; Z/Zlag (constant across t)
% and Rfull's combined regime/idx-pattern groups are each flattened ONCE
% into plain numeric arrays with an offset/count index, same rationale as
% the H section above -- the per-(t,kk) loop, which runs O(nObs) times
% (the actual profiled bottleneck, ~43% of a full eval), must stay to
% pure numeric-array reads with NO cell-array access and NO branching to
% avoid re-introducing overhead that cancels the savings from skipping
% find()/chol().
constZ = ismatrix(sys.Z);
hasZlag = isfield(sys, 'Zlag') && ~isempty(sys.Zlag);
if hasZlag, constZL = ismatrix(sys.Zlag); end
hasRfull = isfield(sys, 'Rfull') && ~isempty(sys.Rfull);
hasRuniq = isfield(sys, 'Runiq') && ~isempty(sys.Runiq);
useRgroups = hasRfull || hasRuniq;
if hasRfull, constR = ismatrix(sys.Rfull); end

if constZ
    rowsC = cell(p, 1); valsC = cell(p, 1); cntZ = zeros(p, 1);
    for r = 1:p
        zrow = sys.Z(r, :);
        nz = find(zrow);
        rowsC{r} = nz(:); valsC{r} = zrow(nz)';
        cntZ(r) = numel(nz);
    end
    offZ = [0; cumsum(cntZ(1:end - 1))];
    ZflatCol = vertcat(rowsC{:}); ZflatVal = vertcat(valsC{:});
end
if hasZlag && constZL
    rowsC = cell(p, 1); valsC = cell(p, 1); cntZL = zeros(p, 1);
    for r = 1:p
        zlrow = sys.Zlag(r, :);
        nzl = find(zlrow);
        rowsC{r} = nzl(:); valsC{r} = zlrow(nzl)';
        cntZL(r) = numel(nzl);
    end
    offZL = [0; cumsum(cntZL(1:end - 1))];
    ZLflatCol = vertcat(rowsC{:}); ZLflatVal = vertcat(valsC{:});
end
if useRgroups
    nGRC = cache.nGRC;
    rowsC = cell(nGRC, 1); colsC = cell(nGRC, 1); valsC = cell(nGRC, 1);
    ldRvec = zeros(nGRC, 1); cntR = zeros(nGRC, 1);
    for g = 1:nGRC
        repT = cache.RcombRepT(g);
        repIdx = cache.idxByT{repT};
        if hasRuniq
            Rt = sys.Runiq(repIdx, repIdx, cache.RcombRgroupOf(g));
        elseif constR
            Rt = sys.Rfull(repIdx, repIdx);
        else
            Rt = sys.Rfull(repIdx, repIdx, repT);
        end
        [LR, flagR] = chol((Rt + Rt') / 2, 'lower');
        if flagR ~= 0, logL = -Inf; aux = []; return; end
        Rinv_rep = LR' \ (LR \ eye(numel(repIdx)));
        Rinv_rep = (Rinv_rep + Rinv_rep') / 2;
        ldRvec(g) = -sum(log(diag(LR)));
        [rowsC{g}, colsC{g}, valsC{g}] = find(Rinv_rep);
        cntR(g) = numel(valsC{g});
    end
    offR = [0; cumsum(cntR(1:end - 1))];
    RflatRow = vertcat(rowsC{:}); RflatCol = vertcat(colsC{:}); RflatVal = vertcat(valsC{:});
end

% ---- ROUND 2: the per-(t,kk) loop above (kept below as canVectorize's
% ELSE branch) is the actual profiled bottleneck (~43% of a full eval,
% ~O(nObs) interpreted iterations even though its per-iteration body was
% already cut to plain numeric-array reads in round 1).  Since Z/Zlag are
% ALWAYS constant p x m matrices in jointstar.ModelSpec.jointstar (never
% 3-D; constZ/constZL are always true for every cache-path call in this
% project -- verified: no test or benchmark ever builds a time-varying
% Z/Zlag AND passes a cache to computeLogLik), the value assigned at
% EVERY occurrence of equation-row r is the SAME small slice
% (ZflatVal/ZflatCol at offZ(r)), just shifted by the (t-1)*m / (t-2)*m
% column block for that occurrence's t -- so the whole per-(t,kk) loop
% reduces to a single indexed (gather-then-scatter) assignment per
% triplet array, using cache.obsRowOf/obsTOf (buildEvalCache: which row/
% time each of the nObs occurrences is, in the SAME column-major order
% find(obsMask) -- and hence the old loop -- visits them) and a standard
% ragged-array "repelem + cumsum" flatten (variable-length per-row/per-t
% blocks concatenated in occurrence order). This is a PURE SCATTER (no
% arithmetic reduction over multiple terms), so it reproduces the OLD
% loop's triplet arrays element-for-element, not merely value-for-value:
% verified for exact (isequal) elementwise equality of iZ/jZ/vZ/iR/jR/vR/
% yv against the old loop on 20 draws before this was relied on (see the
% round-2 report). halfLogDetRinv, by contrast, IS an order-sensitive
% scalar accumulation (sequential += across t in the old loop) -- summing
% it via a vectorized sum()/cumsum() is NOT guaranteed bit-identical to
% that (floating-point addition is non-associative and MATLAB's
% internal reduction algorithm is not specified to match a manual
% left-to-right loop), so it is deliberately LEFT AS AN EXPLICIT
% sequential loop over t (cheap: T ~ 200 scalar adds, not the O(nObs)
% cost being eliminated here) -- this is the one sub-piece of this pass
% that is a genuine, documented exception to "no loops remain."
canVectorize = constZ && (~hasZlag || constZL);

if canVectorize
    obsRowOf = cache.obsRowOf; obsTOf = cache.obsTOf;

    yv = y(cache.obsMask);
    if ~isempty(sys.d)
        if isvector(sys.d)
            dMat = repmat(sys.d(:), 1, T);
        else
            dMat = sys.d;
        end
        yv = yv - dMat(cache.obsMask);
    end

    % ---- Z (+ Zlag) triplets: ragged-flatten across all nObs occurrences
    nnzZ = cntZ(obsRowOf);
    totZ = sum(nnzZ);
    cumZ = [0; cumsum(nnzZ(1:end - 1))];
    localZ = (0:totZ - 1)' - repelem(cumZ, nnzZ);
    srcZ = repelem(offZ(obsRowOf), nnzZ) + localZ + 1;

    if hasZlag
        zlMask = obsTOf > 1;
        nnzZL = zeros(nObs, 1);
        nnzZL(zlMask) = cntZL(obsRowOf(zlMask));
        offZLsrc = zeros(nObs, 1);
        offZLsrc(zlMask) = offZL(obsRowOf(zlMask));
    else
        nnzZL = zeros(nObs, 1);
        offZLsrc = zeros(nObs, 1);
    end
    totZL = sum(nnzZL);
    cumZL = [0; cumsum(nnzZL(1:end - 1))];
    localZL = (0:totZL - 1)' - repelem(cumZL, nnzZL);
    srcZL = repelem(offZLsrc, nnzZL) + localZL + 1;

    nnzTot = nnzZ + nnzZL;
    destStart = [0; cumsum(nnzTot(1:end - 1))];
    ptr = totZ + totZL;
    iZ = zeros(ptr, 1); jZ = iZ; vZ = iZ;

    destZ = repelem(destStart, nnzZ) + localZ + 1;
    iZ(destZ) = repelem((1:nObs)', nnzZ);
    jZ(destZ) = ZflatCol(srcZ) + repelem((obsTOf - 1) * m, nnzZ);
    vZ(destZ) = ZflatVal(srcZ);

    if totZL > 0
        destZL = repelem(destStart + nnzZ, nnzZL) + localZL + 1;
        iZ(destZL) = repelem((1:nObs)', nnzZL);
        jZ(destZL) = ZLflatCol(srcZL) + repelem((obsTOf - 2) * m, nnzZL);
        vZ(destZL) = ZLflatVal(srcZL);
    end

    % ---- R^{-1} triplets: ragged-flatten across the T time periods -----
    hasDataT = cache.nIdxT > 0;
    if useRgroups
        nRperT = cntR(cache.RcombGroup);
        nRperT(~hasDataT) = 0;
        totR = sum(nRperT);
        cumR = [0; cumsum(nRperT(1:end - 1))];
        localR = (0:totR - 1)' - repelem(cumR, nRperT);
        srcR = repelem(offR(cache.RcombGroup), nRperT) + localR + 1;
        iR = repelem(cache.rowStartT, nRperT) + RflatRow(srcR);
        jR = repelem(cache.rowStartT, nRperT) + RflatCol(srcR);
        vR = RflatVal(srcR);
        ptrR = totR;

        halfLogDetRinv = 0;   % order-sensitive: explicit sequential loop, see note above
        for t = 1:T
            if ~hasDataT(t), continue; end
            halfLogDetRinv = halfLogDetRinv + ldRvec(cache.RcombGroup(t));
        end
    else
        if isvector(sys.Rdiag)
            RdiagMat = repmat(sys.Rdiag(:), 1, T);
        else
            RdiagMat = sys.Rdiag;
        end
        rinvAll = 1 ./ RdiagMat(cache.obsMask);
        iR = (1:nObs)'; jR = iR; vR = rinvAll;
        ptrR = nObs;

        halfLogDetRinv = 0;   % order-sensitive: explicit sequential loop, see note above
        for t = 1:T
            if ~hasDataT(t), continue; end
            so = cache.rowStartT(t); nT = cache.nIdxT(t);
            halfLogDetRinv = halfLogDetRinv + 0.5 * sum(log(rinvAll(so + 1:so + nT)));
        end
    end
else
    % Fallback: time-varying Z/Zlag (never produced by
    % jointstar.ModelSpec.jointstar -- constZ/constZL are always true on
    % every cache-path call in this project; kept only so a hand-rolled
    % sys with genuinely time-varying loadings, if ever combined with a
    % cache by a future caller, still gets a CORRECT answer rather than a
    % silently wrong fast path).
    iZ = zeros(nObs * m * (1 + hasZlag), 1); jZ = iZ; vZ = iZ;
    iR = zeros(nObs * p, 1); jR = iR; vR = iR;
    ptrR = 0;
    halfLogDetRinv = 0;
    yv = zeros(nObs, 1);
    ptr = 0; row = 0;

    for t = 1:T
        idx = cache.idxByT{t};
        if isempty(idx), continue; end
        yt = y(idx, t);
        if ~isempty(sys.d)
            dt = colSlice(sys.d, t);
            yt = yt - dt(idx);
        end
        hasZLt = hasZlag && t > 1;
        nIdx = numel(idx);

        if useRgroups
            g = cache.RcombGroup(t);
            halfLogDetRinv = halfLogDetRinv + ldRvec(g);
            so = offR(g); nR = cntR(g);
            iR(ptrR + 1:ptrR + nR) = row + RflatRow(so + 1:so + nR);
            jR(ptrR + 1:ptrR + nR) = row + RflatCol(so + 1:so + nR);
            vR(ptrR + 1:ptrR + nR) = RflatVal(so + 1:so + nR);
            ptrR = ptrR + nR;
        else
            Rt = colSlice(sys.Rdiag, t);
            rinv_t = 1 ./ Rt(idx);
            halfLogDetRinv = halfLogDetRinv + 0.5 * sum(log(rinv_t));
            iR(ptrR + 1:ptrR + nIdx) = row + (1:nIdx)';
            jR(ptrR + 1:ptrR + nIdx) = row + (1:nIdx)';
            vR(ptrR + 1:ptrR + nIdx) = rinv_t;
            ptrR = ptrR + nIdx;
        end

        for kk = 1:nIdx
            row = row + 1;
            r = idx(kk);

            if constZ
                so = offZ(r); nnz_r = cntZ(r);
                iZ(ptr + 1:ptr + nnz_r) = row;
                jZ(ptr + 1:ptr + nnz_r) = (t - 1) * m + ZflatCol(so + 1:so + nnz_r);
                vZ(ptr + 1:ptr + nnz_r) = ZflatVal(so + 1:so + nnz_r);
            else
                zrow = sys.Z(r, :, t);
                nz = find(zrow); zv = zrow(nz); nnz_r = numel(nz);
                iZ(ptr + 1:ptr + nnz_r) = row;
                jZ(ptr + 1:ptr + nnz_r) = (t - 1) * m + nz;
                vZ(ptr + 1:ptr + nnz_r) = zv;
            end
            ptr = ptr + nnz_r;

            if hasZLt
                if constZL
                    so = offZL(r); nnz_rl = cntZL(r);
                    iZ(ptr + 1:ptr + nnz_rl) = row;
                    jZ(ptr + 1:ptr + nnz_rl) = (t - 2) * m + ZLflatCol(so + 1:so + nnz_rl);
                    vZ(ptr + 1:ptr + nnz_rl) = ZLflatVal(so + 1:so + nnz_rl);
                else
                    zlrow = sys.Zlag(r, :, t);
                    nzl = find(zlrow); zlv = zlrow(nzl); nnz_rl = numel(nzl);
                    iZ(ptr + 1:ptr + nnz_rl) = row;
                    jZ(ptr + 1:ptr + nnz_rl) = (t - 2) * m + nzl;
                    vZ(ptr + 1:ptr + nnz_rl) = zlv;
                end
                ptr = ptr + nnz_rl;
            end
            yv(row) = yt(kk);
        end
    end
end
Zs = sparse(iZ(1:ptr), jZ(1:ptr), vZ(1:ptr), nObs, m * T);
RinvSp = sparse(iR(1:ptrR), jR(1:ptrR), vR(1:ptrR), nObs, nObs);

% ---- posterior precision, sparse Cholesky, likelihood ------------------
% NOTE on cache.perm: buildEvalCache also discovers a fill-reducing
% permutation from the probe draw, and this was ORIGINALLY meant to be
% reused here via chol(Htilde(perm,perm),'lower') to skip re-deriving it
% via 'vector' mode every eval (~0.48 ms/eval per the profiling pass).
% Verification (benchmarks/verifyCacheEquivalence.m) caught that this is
% NOT bitwise safe: even when chol(Htilde,'lower','vector') is confirmed
% to return exactly cache.perm for a given theta, feeding the
% already-permuted matrix to plain chol('lower') gives an L that differs
% from 'vector' mode's own L at the ~1e-12 level (confirmed against an
% independent dense chol() cross-check: both are equally valid
% factorisations of the same matrix, just arrived at via different
% internal supernode/blocking choices in the two call forms) -- and that
% amplifies through the triangular solve to an O(1e-9) RELATIVE
% difference in logL, above the 1e-10 bar this project holds bitwise
% equivalence to.  So this path recomputes the permutation via 'vector'
% mode every call, same as the no-cache path -- the ~0.48 ms/eval AMD
% saving is deliberately NOT taken; see the final report for the
% quantified deviation.  cache.perm is retained (unused here) only for
% the diagnostic comparison in tests/testEvalCache.m.
Ry = RinvSp * yv;
Htilde = Halpha + Zs' * RinvSp * Zs;
b = Zs' * Ry + bPrior;

[L, flag, perm] = chol(Htilde, 'lower', 'vector');
if flag ~= 0
    logL = -Inf;                 % infeasible theta: reject, don't crash
    aux = [];
    return
end

v = L \ b(perm);
muTp = L' \ v;
muTilde = zeros(m * T, 1);
muTilde(perm) = muTp;

logdetHtilde = 2 * sum(log(full(diag(L))));
yRy = yv' * Ry;

logL = -0.5 * nObs * log(2 * pi) + halfLogDetRinv ...
    + 0.5 * logdetHalpha - 0.5 * logdetHtilde ...
    - 0.5 * (yRy + quadPrior - v' * v);

if ~isfinite(logL) && logL ~= -Inf
    error('jointstar:badLogLik', ...
        'log-likelihood is NaN/Inf (logdetHalpha=%.3g, logdetHtilde=%.3g).', ...
        logdetHalpha, logdetHtilde);
end

if nargout > 1
    aux = struct('L', L, 'perm', perm, 'muTilde', muTilde, 'm', m, 'T', T);
end
end

% ======================================================================
function [m, T, p] = sysDims(sys, y)
m = size(sys.A1, 1);
T = sys.T;
p = size(y, 1);
if size(y, 2) ~= T
    error('jointstar:dimMismatch', 'y is %d x %d but sys.T = %d.', ...
        size(y, 1), size(y, 2), T);
end
end

function [Ainv, logdetA] = invSPD(A, name)
[Ainv, logdetA, ok] = tryInvSPD(A);
if ~ok
    error('jointstar:notSPD', '%s is not positive definite.', name);
end
end

function [Ainv, logdetA, ok] = tryInvSPD(A)
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

function B = slice3(A, t, m)
if ismatrix(A), B = A; else, B = A(:, :, t); end
if isscalar(B) && m > 1
    error('jointstar:dimMismatch', 'scalar system matrix with m > 1.');
end
end

function ct = colSlice(C, t)
if isvector(C), ct = C(:); else, ct = C(:, t); end
end

function [ii, jj, vv, ptr] = putBlock(ii, jj, vv, ptr, B, rowOff, colOff)
[r, c, v] = find(B);
n = numel(v);
ii(ptr + 1:ptr + n) = rowOff + r;
jj(ptr + 1:ptr + n) = colOff + c;
vv(ptr + 1:ptr + n) = v;
ptr = ptr + n;
end
