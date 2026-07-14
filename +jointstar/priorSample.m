function Theta = priorSample(P, N)
%PRIORSAMPLE Draw N parameter vectors from the prior (N x d matrix).
%
%   Theta = jointstar.priorSample(P, N)
%
%   Independent draws per parameter; truncated types by rejection; the
%   joint AR(2) stationarity constraint on (phi1, phi2) by rejection at
%   the vector level.  Uses only base-MATLAB randomness.
%
%   See also jointstar.defaultPriors, jointstar.priorLogPdf.

prm = P.params;
Theta = zeros(N, P.d);
for j = 1:P.d
    Theta(:, j) = drawOne(prm(j), N);
end

% hierarchical COVID kappas: hypers were drawn by the loop above (normal
% types); draw each kappa from its truncated Gamma(a_g, m_g/a_g) by
% rejection, falling back to just-above-1 if the hyper draw makes
% P(X >= 1) negligible (such draws are near-rejected by the prior anyway)
if isfield(P, 'kap')
    kp = P.kap;
    for j = 1:numel(kp.kapCols)
        g = kp.groups(j);
        a = exp(Theta(:, kp.laCols(g)));
        s = exp(Theta(:, kp.lmCols(g))) ./ a;
        x = s .* gammaDraw(a);
        for it = 1:200
            bad = x < 1;
            if ~any(bad), break; end
            x(bad) = s(bad) .* gammaDraw(a(bad));
        end
        x(x < 1) = 1 + 0.1 * rand(nnz(x < 1), 1);
        Theta(:, kp.kapCols(j)) = x;
    end
end

% AR(2) stationarity by rejection: redraw offending (phisum, phi2) pairs
% (phi1 = phisum - phi2)
i1 = P.idx.phisum; i2 = P.idx.phi2;
for it = 1:1000
    f2 = Theta(:, i2); f1 = Theta(:, i1) - f2;
    bad = ~(abs(f2) < 1 & f1 + f2 < 1 & f2 - f1 < 1);
    if ~any(bad), return; end
    Theta(bad, i1) = drawOne(prm(i1), nnz(bad));
    Theta(bad, i2) = drawOne(prm(i2), nnz(bad));
end
error('jointstar:priorRejection', 'AR(2) stationarity rejection did not converge.');
end

function x = drawOne(q, N)
switch q.type
    case 'norm'
        x = q.p1 + q.p2 * randn(N, 1);
    case 'tnorm'
        x = rejSample(@() q.p1 + q.p2 * randn(N, 1), ...
            @(v) v >= q.lo & v <= q.hi, N);
    case 'unif'
        x = q.lo + (q.hi - q.lo) * rand(N, 1);
    case 'beta'
        x = betaDraw(q.p1, q.p2, N);
    case 'negbeta'
        x = -betaDraw(q.p1, q.p2, N);
    case 'igsd'
        b = (q.p1 - 1) * q.p2^2;
        x = sqrt(b ./ gammaDraw(q.p1 * ones(N, 1)));
    case 'logn'
        x = exp(q.p1 + q.p2 * randn(N, 1));
    case 'tgamma'
        x = rejSample(@() q.p2 * gammaDraw(q.p1 * ones(N, 1)), ...
            @(v) v >= q.lo, N);
    case 'fixed'
        x = repmat(q.init, N, 1);   % calibrated constant
    case 'hkap'
        x = repmat(q.init, N, 1);   % overwritten by the hierarchical-kappa block
    otherwise
        error('jointstar:badPrior', 'unknown prior type %s', q.type);
end
end

function x = betaDraw(a, b, N)
ga = gammaDraw(a * ones(N, 1));
gb = gammaDraw(b * ones(N, 1));
x = ga ./ (ga + gb);
x = min(max(x, 1e-12), 1 - 1e-12);
end

function x = rejSample(gen, ok, N)
x = gen();
for it = 1:1000
    bad = ~ok(x);
    if ~any(bad), return; end
    y = gen();
    x(bad) = y(bad);
end
error('jointstar:priorRejection', 'truncated-prior rejection did not converge.');
end

function x = gammaDraw(a)
% Marsaglia-Tsang Gamma(a,1) with the a < 1 boost (Beta shapes < 1 occur)
sz = size(a);
a = a(:);               % row inputs break the masked indexing below
boost = a < 1;
ab = a + boost;
d = ab - 1 / 3;
c = 1 ./ sqrt(9 * d);
x = zeros(size(a));
todo = true(size(a));
while any(todo)
    n = nnz(todo);
    z = randn(n, 1);
    v = (1 + c(todo) .* z).^3;
    u = rand(n, 1);
    okk = v > 0 & log(u) < 0.5 * z.^2 + d(todo) .* (1 - v + log(max(v, realmin)));
    idx = find(todo);
    x(idx(okk)) = d(idx(okk)) .* v(okk);
    todo(idx(okk)) = false;
end
if any(boost)
    x(boost) = x(boost) .* rand(nnz(boost), 1).^(1 ./ a(boost));
end
x = reshape(max(x, realmin), sz);
end
