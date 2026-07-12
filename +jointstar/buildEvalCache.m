function cache = buildEvalCache(dat, P)
%BUILDEVALCACHE Run-level theta-independent structural cache for the
%   JointSTAR likelihood evaluator.
%
%   cache = jointstar.buildEvalCache(dat, P)
%
%   Everything jointstar.computeLogLik and jointstar.ModelSpec.jointstar
%   recompute on EVERY theta draw that does not actually depend on theta's
%   VALUE -- only on the run's data (dat) and prior/config spec (P, for
%   PieObs-driven dimensions and Horseshoe on/off) -- is discovered once
%   here and reused for the life of one jointstar.estimate call:
%
%     * obsMask / per-t observed-row index sets (pure data fact)
%     * the regime groupings ModelSpec currently rediscovers via unique()
%       every eval for Q (COVID kappa / break windows) and, under the
%       horseshoe, Rfull -- typically ~4-6 distinct slices out of T~200 --
%       so ModelSpec builds only the unique slices (Quniq/Runiq) and
%       computeLogLik inverts/factorises each unique slice ONCE (~4-10
%       times) instead of once per t (~T times), memoizing the resulting
%       sparse triplets for every t sharing that slice
%     * the SAME piecewise-constant-in-t structure for the A1/A2
%       transition blocks (COVID/IS-availability windows), so
%       computeLogLik's H-assembly loop memoizes find() on the handful of
%       distinct blocks instead of calling it ~T times
%     * for the horseshoe measurement covariance (Rfull), the COMBINED
%       grouping (regime slice x missing-data pattern) that determines
%       exactly which t's share a bit-identical OBSERVED submatrix
%       Rt(idx,idx), so its chol/inverse is computed once per combination
%       instead of once per t
%     * P1 (always a hardcoded constant in ModelSpec.jointstar, never a
%       function of theta) and its inverse/log-det, computed once
%     * a fill-reducing permutation for Htilde's sparsity pattern (cache.
%       perm).  NOTE: computeLogLik does NOT reuse this to skip 'vector'
%       mode's own AMD step -- that was tried and measurably failed
%       bitwise equivalence (~1e-9 relative in logL: chol(Htilde,'lower',
%       'vector') and chol(Htilde(perm,perm),'lower') differ at the
%       ~1e-12 level in L even given the identical permutation, an
%       internal MATLAB/CHOLMOD supernode-blocking difference between the
%       two call forms, confirmed against an independent dense chol()
%       cross-check -- see the note in jointstar.computeLogLik).  perm is
%       kept in the cache only as a diagnostic (tests/testEvalCache.m
%       checks it against a fresh 'vector'-mode call); the ~0.48 ms/eval
%       AMD cost is NOT saved by this pass.
%
%   Every one of these is an EXACT memoization within a run: the cached
%   grouping tells computeLogLik which t's are GUARANTEED to see a
%   bit-identical block (by construction -- ModelSpec repmat()s/copies
%   the same regime slice across its members), so reusing a
%   once-computed inverse/factorisation/find() result for the rest of the
%   group reproduces exactly what re-deriving it per t would give -- not
%   an approximation, PROVIDED the grouping the probe discovers is at
%   least as fine as the truth for every subsequent theta.  Getting this
%   probe right took two fixes during development, both caught by
%   comparing Q/Rfull group membership against the true per-theta value
%   for random draws (now covered by tests/testEvalCache.m):
%     (1) several coefficients (chi1, chi2, and, under the horseshoe,
%         every Lq/Lr off-diagonal) are exactly 0 at the literal prior
%         "init" vector, which would make their target cell look
%         structurally absent.  Fixed by bumping them to a generic
%         nonzero value.
%     (2) the 5 pre-break volatility multipliers (m84_y/m84_z/m84_c/
%         m93_pi/m93_U) all share the SAME class default (1.5), and the
%         12 COVID-kappa variance scales all share the SAME class
%         default (2.0) -- so at the literal init vector, e.g. kapy_20
%         and kapy_21 (which govern DIFFERENT, non-overlapping windows
%         on the SAME row of kr) are numerically equal, and Rfull's
%         unique()-based grouping merges those two windows into "one"
%         regime.  For almost any other theta draw they differ, so a
%         cache built from that merged grouping would silently reuse the
%         WRONG window's inverse for one of the two -- not a tiny
%         floating-point discrepancy but a real wrong answer (this was
%         caught exactly this way: max |loglik| differences of 10-200+
%         across 30 random prior draws before the fix).  Fixed by giving
%         every member of both families a distinct probe value.
%   The one place the reused-value argument is not literally exact by
%   construction is the sparse Cholesky permutation (perm below): its
%   grouping (which t's see a bit-identical Htilde SPARSITY PATTERN) is
%   generic-theta-almost-surely rather than provably-always true, and is
%   verified for bitwise equivalence against ~150 independent prior draws
%   in tests/testEvalCache.m and benchmarks/verifyCacheEquivalence.m
%   rather than merely assumed.
%
%   cache is a plain struct; pass it as the trailing optional argument to
%   jointstar.ModelSpec.jointstar and jointstar.computeLogLik.  Both
%   behave exactly as before when cache is omitted or empty.
%
%   See also jointstar.computeLogLik, jointstar.ModelSpec.jointstar,
%   jointstar.estimate.

% ---- probe theta: prior init, with structurally-zero-at-init cells ----
% (chi1/chi2, and every horseshoe L off-diagonal) bumped to a generic
% nonzero value so the Htilde permutation is discovered from the full
% (never-narrower) sparsity support.
tv0 = [P.params.init];
if isfield(P, 'idx')
    if isfield(P.idx, 'chi1'), tv0(P.idx.chi1) = 0.1; end
    if isfield(P.idx, 'chi2'), tv0(P.idx.chi2) = 0.1; end
end
hasHS = isfield(P, 'hs');
if hasHS
    tv0(P.hs.colL) = 0.1;
end
% Every break multiplier (type 'logn', class default 1.5) and every
% COVID-kappa variance scale (type 'tgamma'/'hkap', class default 2.0)
% shares its init value with several structural siblings that govern
% DIFFERENT, non-overlapping time windows on the SAME row of kq/kr (see
% the file header) -- give each a distinct value so window-grouping
% discovery cannot merge two genuinely different regimes.
multiplierTypes = {'logn', 'tgamma', 'hkap'};
isMult = ismember({P.params.type}, multiplierTypes);
tv0(isMult) = 1.3 + 0.013 * (1:nnz(isMult));

th0 = jointstar.thetaStruct(P, tv0);
if hasHS
    [Lq0, Lr0] = jointstar.hsUnpack(P, tv0);
    cf0 = struct('Lq', Lq0, 'Lr', Lr0);
else
    cf0 = [];
end
spec0 = jointstar.ModelSpec.jointstar(th0, dat, cf0);
sys0 = spec0.system();
y = dat.y;

m = size(sys0.A1, 1);
T = sys0.T;
p = size(y, 1);

[ll0, aux0] = jointstar.computeLogLik(sys0, y);
if ~isfinite(ll0)
    error('jointstar:cacheProbeInfeasible', ...
        ['buildEvalCache: the structural probe theta (prior init, with ' ...
         'chi1/chi2/L off-diagonals bumped off zero) is infeasible ' ...
         '(loglik=-Inf); cannot discover Htilde''s sparsity pattern. ' ...
         'This should not happen for the shipped priors -- investigate ' ...
         'before relying on the cache.']);
end

% ---- pure data facts: obsMask, per-t observed rows -----------------------
obsMask = ~isnan(y);
idxByT = cell(1, T);
for t = 1:T
    idxByT{t} = find(obsMask(:, t));
end
[idxPatGroup, ~, nIdxPat] = groupCols(double(obsMask), T);

% ---- flattened occurrence index (round-2 perf pass): obsRowOf(oo)/
% obsTOf(oo) give the equation row / time period of the oo-th entry in
% MATLAB's column-major traversal of obsMask, i.e. EXACTLY the sequence
% the no-cache/old-cache per-(t, kk) loop visits (t ascending, then row
% ascending within t) since idxByT{t} = find(obsMask(:,t)) is itself
% ascending. This is a pure data fact (obsMask only) -- never a function
% of theta -- so it lets computeLogLik replace that loop's per-occurrence
% scatter with whole-array indexed assignment while visiting slots in the
% IDENTICAL order, which is what bitwise equivalence of the resulting
% triplet arrays depends on (see jointstar.computeLogLik). rowStartT(t)/
% nIdxT(t) give the same fact from the "per-t" side (used to vectorize
% the per-t R-block fill): rowStartT(t) is the 0-based count of
% observations strictly before time t, i.e. exactly the value of the
% no-cache loop's running `row` counter at the top of iteration t.
[obsRowOf, obsTOf] = find(obsMask);
nIdxT = cellfun(@numel, idxByT)';
rowStartT = [0; cumsum(nIdxT(1:end - 1))];

% ---- Q regime groups (COVID kappa / pre84 / pre93 windows) ---------------
[Qgroup, QgroupRep, nGQ] = groupCols(reshape(sys0.Q, m * m, T), T);

% ---- P1: constant, never a function of theta -----------------------------
P1 = sys0.P1;
LP1 = chol((P1 + P1') / 2, 'lower');
P1inv = LP1' \ (LP1 \ eye(m));
P1inv = (P1inv + P1inv') / 2;
logdetP1 = 2 * sum(log(diag(LP1)));

% ---- A1 / A2 structural groups (H assembly) ------------------------------
[A1Group, A1GroupRep, nGA1] = groupCols(reshape(sys0.A1, m * m, T), T);
hasA2 = ~isempty(sys0.A2);
A2Group = []; A2GroupRep = []; nGA2 = 0;
if hasA2
    [A2Group, A2GroupRep, nGA2] = groupCols(reshape(sys0.A2, m * m, T), T);
end

% ---- Rfull (horseshoe measurement covariance) regime groups + the ----
% combined (value-group x idx-pattern) grouping computeLogLik needs to
% avoid re-inverting the SAME observed submatrix Rt(idx,idx) at every t.
hasRfull = isfield(sys0, 'Rfull') && ~isempty(sys0.Rfull);
Rgroup = []; RgroupRep = []; nGR = 0;
RcombGroup = []; RcombRepT = []; RcombRgroupOf = []; nGRC = 0;
if hasRfull
    [Rgroup, RgroupRep, nGR] = groupCols(reshape(sys0.Rfull, p * p, T), T);
    comboKey = idxPatGroup + nIdxPat * (Rgroup - 1);
    [~, rcRep, rcGrp] = unique(comboKey, 'stable');
    RcombGroup = rcGrp;
    RcombRepT = rcRep;
    RcombRgroupOf = Rgroup(rcRep);
    nGRC = numel(rcRep);
end

% ---- Htilde fill-reducing permutation, from the probe's own aux ----------
perm = aux0.perm;

cache = struct( ...
    'm', m, 'T', T, 'p', p, ...
    'obsMask', obsMask, 'idxByT', {idxByT}, ...
    'obsRowOf', obsRowOf, 'obsTOf', obsTOf, ...
    'nIdxT', nIdxT, 'rowStartT', rowStartT, ...
    'P1', P1, 'P1inv', P1inv, 'logdetP1', logdetP1, ...
    'Qgroup', Qgroup, 'QgroupRep', QgroupRep, 'nGQ', nGQ, ...
    'hasA2', hasA2, ...
    'A1Group', A1Group, 'A1GroupRep', A1GroupRep, 'nGA1', nGA1, ...
    'A2Group', A2Group, 'A2GroupRep', A2GroupRep, 'nGA2', nGA2, ...
    'hasRfull', hasRfull, ...
    'Rgroup', Rgroup, 'RgroupRep', RgroupRep, 'nGR', nGR, ...
    'RcombGroup', RcombGroup, 'RcombRepT', RcombRepT, ...
    'RcombRgroupOf', RcombRgroupOf, 'nGRC', nGRC, ...
    'perm', perm, 'probeLL', ll0, 'probeTheta', tv0);
end

% ==========================================================================
function [grp, rep, nG] = groupCols(Xflat, T)
%GROUPCOLS Xflat is n x T (already reshape(:, T)); group columns (as rows
%   of Xflat') by exact value equality.  rep(g) is the FIRST t (1-based)
%   whose column defines group g (numbered in order of first appearance).
if size(Xflat, 2) ~= T
    error('jointstar:cacheShape', 'groupCols: expected %d columns, got %d.', ...
        T, size(Xflat, 2));
end
[~, rep, grp] = unique(Xflat', 'rows', 'stable');
nG = numel(rep);
end
