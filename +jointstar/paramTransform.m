function T = paramTransform(P, mutateIdx)
%PARAMTRANSFORM Elementwise bijection theta <-> eta for transformed-scale
%MH mutation (jointstar 'MutationTransform' option).
%
%   T = jointstar.paramTransform(P, mutateIdx)
%
%   Builds an elementwise transform eta = T(theta) dispatched purely by
%   each mutated coordinate's prior SUPPORT (prm.lo, prm.hi), read from
%   the priors struct P = jointstar.defaultPriors(...) -- never by
%   hardcoded parameter name or hardcoded bound (design doc
%   DESIGN_transformed_kernel.md, ADDENDUM requirement R2):
%
%     both lo,hi finite   ->  eta = logit((theta-lo)/(hi-lo))     "kind 1"
%     lo finite, hi=+Inf  ->  eta = log(theta-lo)                 "kind 2"
%     hi finite, lo=-Inf  ->  eta = log(hi-theta)                 "kind 3"
%     both infinite       ->  eta = theta (identity)               "kind 0"
%
%   This single bounds-driven dispatch reduces automatically to the
%   design table's per-type rows, since every prior type's [lo,hi] is
%   already recorded in prm(j) by jointstar.defaultPriors:
%     norm, kapHyp_lm/la, xi1/xi2, phi2, gzbar, gwbar, theta*, lam*,
%       chi*, nu, pistar          lo=-Inf,hi=Inf  -> identity (kind 0)
%     igsd (sigma's), logn        lo=0,   hi=Inf   -> log(theta)   (kind 2)
%     tgamma, hkap (COVID kappas) lo=1,   hi=Inf   -> log(theta-1) (kind 2)
%     beta (phisum,rhoU,...)      lo=0,   hi=1     -> logit(theta) (kind 1)
%     unif (alphapi)              lo=.01, hi=.99   -> logit affine (kind 1)
%     tnorm both-finite (alpha)   lo=.15, hi=.55   -> logit affine (kind 1)
%     tnorm one-sided (phiy,phiu) lo=-Inf,hi=0     -> log(-theta)  (kind 3)
%     negbeta (gamma2)            lo=-1,  hi=0     -> logit affine (kind 1)
%
%   negbeta note: the design table specifies eta = logit(-theta) for
%   gamma2.  The generic affine-logit here instead gives
%   theta = lo + (hi-lo)*s(eta) = s(eta) - 1, which equals design's
%   -s(eta') under the relabelling eta' = -eta (since s(-e) = 1 - s(e)).
%   A zero-mean symmetric Gaussian RW step in eta is exactly as symmetric
%   in -eta, so this is the identical family of kernels/pushforwards up
%   to that trivial sign relabelling of the auxiliary coordinate -- not
%   a different transform.  Using the one generic formula (vs. a
%   name-specific special case) is what keeps the dispatch bounds-driven
%   per R2.
%
%   Only mutateIdx columns are transformed (R4): 'fixed' params (e.g.
%   sme_pieobs) are excluded upstream via mutateIdx and are never
%   touched here; columns outside mutateIdx pass through toEta/toTheta
%   unchanged.
%
%   Fields of T:
%     .cols   1 x d logical, true at transformed (mutateIdx) columns
%     .kind   1 x d int8 (0/1/2/3 per above; only meaningful where cols)
%     .lo,.hi 1 x d double bounds, copied from prm(j).lo / prm(j).hi
%     .toEta(theta)   N x d (or 1 x d) theta -> eta.  Clamps only the
%                     numerically-degenerate case of theta sitting
%                     exactly on (or, from float round-off, minutely
%                     outside) its bound -- e.g. an exact-boundary
%                     prior draw -- to a large-but-finite |eta| <= 700
%                     (R5/C4).  A no-op for interior theta: for those,
%                     toTheta(toEta(theta)) == theta to machine
%                     precision (verified in tests/testParamTransform.m).
%     .toTheta(eta)   eta -> theta, exact inverse; proposals passed
%                     through here are NEVER clamped (R5).
%     .logJacElem(eta)  N x d, the PER-COORDINATE log|d theta_j/d eta_j|
%                     (0 outside mutateIdx columns) -- exposed mainly so
%                     unit tests can finite-difference-check one
%                     coordinate at a time (R3) without the other
%                     coordinates' terms mixed in.
%     .logJac(eta)    N x 1, sum(T.logJacElem(eta), 2) -- the aggregate
%                     Jacobian used in the MH accept ratio (R1), summed
%                     over the mutateIdx columns only, evaluated at the
%                     exact eta used (R3: exact log-derivative of this
%                     toTheta).  R7: this is ADDED to, never merged/
%                     deduplicated with, priorLogPdf's own igsd
%                     sigma->sigma^2 Jacobian term (a distinct,
%                     intra-prior term that stays where it is).
%
%   See also jointstar.defaultPriors, jointstar.priorLogPdf,
%   jointstar.mhMutate, jointstar.runSMC.

prm = P.params;
d = P.d;
cols = false(1, d);
if nargin > 1 && ~isempty(mutateIdx)
    cols(:) = mutateIdx;
else
    cols(:) = true;
end

kind = zeros(1, d, 'int8');
lo = -Inf(1, d);
hi = Inf(1, d);
for j = find(cols)
    lo(j) = prm(j).lo;
    hi(j) = prm(j).hi;
    finLo = isfinite(lo(j));
    finHi = isfinite(hi(j));
    if finLo && finHi
        kind(j) = 1;
    elseif finLo && ~finHi
        kind(j) = 2;
    elseif ~finLo && finHi
        kind(j) = 3;
    else
        kind(j) = 0;
    end
end

CLAMP = 700;   % |eta| cap: guards only literal +-Inf/NaN-producing
               % exact-boundary inputs (R5/C4); exp(+-700) is finite and
               % round-trips to (numerically) the bound itself, and no
               % legitimate interior prior/posterior draw approaches it.

T = struct('cols', cols, 'kind', kind, 'lo', lo, 'hi', hi, ...
    'toEta', @(th) toEtaImpl(th, cols, kind, lo, hi, CLAMP), ...
    'toTheta', @(et) toThetaImpl(et, cols, kind, lo, hi), ...
    'logJacElem', @(et) logJacElemImpl(et, cols, kind, lo, hi), ...
    'logJac', @(et) sum(logJacElemImpl(et, cols, kind, lo, hi), 2));
end

% ==========================================================================
function checkWidth(x, cols, who)
% Defensive: the masks (cols/kind/lo/hi) index FULL-WIDTH theta columns;
% a restricted (e.g. mutateIdx-subset) matrix would silently misalign
% every column past the first excluded parameter.  Make that loud.
if size(x, 2) ~= numel(cols)
    error('jointstar:transformWidth', ...
        ['paramTransform.%s: input has %d columns but the transform ' ...
         'was built for %d (full-width theta/eta required -- do not ' ...
         'pass a mutateIdx-restricted matrix).'], who, size(x, 2), numel(cols));
end
end

% ==========================================================================
function eta = toEtaImpl(theta, cols, kind, lo, hi, CLAMP)
checkWidth(theta, cols, 'toEta');
eta = theta;

k1 = cols & kind == 1;
if any(k1)
    y = (theta(:, k1) - lo(k1)) ./ (hi(k1) - lo(k1));
    y = min(max(y, 0), 1);                  % guard exact/near-boundary input
    e = log(y ./ (1 - y));                  % may be +-Inf at y in {0,1}
    eta(:, k1) = max(min(e, CLAMP), -CLAMP);
end

k2 = cols & kind == 2;
if any(k2)
    x = max(theta(:, k2) - lo(k2), realmin);
    eta(:, k2) = max(min(log(x), CLAMP), -CLAMP);
end

k3 = cols & kind == 3;
if any(k3)
    x = max(hi(k3) - theta(:, k3), realmin);
    eta(:, k3) = max(min(log(x), CLAMP), -CLAMP);
end
end

% ==========================================================================
function theta = toThetaImpl(eta, cols, kind, lo, hi)
checkWidth(eta, cols, 'toTheta');
theta = eta;

k1 = cols & kind == 1;
if any(k1)
    s = 1 ./ (1 + exp(-eta(:, k1)));
    theta(:, k1) = lo(k1) + (hi(k1) - lo(k1)) .* s;
end

k2 = cols & kind == 2;
if any(k2)
    theta(:, k2) = lo(k2) + exp(eta(:, k2));
end

k3 = cols & kind == 3;
if any(k3)
    theta(:, k3) = hi(k3) - exp(eta(:, k3));
end
end

% ==========================================================================
function j = logJacElemImpl(eta, cols, kind, lo, hi)
checkWidth(eta, cols, 'logJacElem');
n = size(eta, 1);
d = size(eta, 2);
j = zeros(n, d);

k1 = cols & kind == 1;
if any(k1)
    s = 1 ./ (1 + exp(-eta(:, k1)));
    j(:, k1) = log(hi(k1) - lo(k1)) + log(s) + log1p(-s);
end

k2 = cols & kind == 2;
if any(k2)
    j(:, k2) = eta(:, k2);
end

k3 = cols & kind == 3;
if any(k3)
    j(:, k3) = eta(:, k3);
end
end
