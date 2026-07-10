function [logL, aux] = computeLogLik(sys, y)
%COMPUTELOGLIK Marginal log-likelihood via the sparse precision sampler.
%
%   [logL, aux] = jointstar.computeLogLik(sys, y)
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
%          the chol guard on Htilde.
%     y    p x T data matrix, NaN = missing.
%
%   Outputs:
%     logL  scalar marginal log-likelihood.  -Inf (never an error) if
%           Htilde is numerically non-SPD at this theta: an infeasible
%           parameter draw is a rejection, not a crash.
%     aux   struct with the pieces needed to draw states without
%           refactorising: L, perm, muTilde, plus m, T.  Pass to
%           jointstar.drawStates.
%
%   See also jointstar.drawStates, jointstar.ModelSpec.

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
