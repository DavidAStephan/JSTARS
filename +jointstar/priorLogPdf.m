function lp = priorLogPdf(P, tv)
%PRIORLOGPDF Log prior density of a parameter vector (row) under P.
%
%   lp = jointstar.priorLogPdf(P, tv)
%
%   Returns -Inf outside any parameter's support or the joint AR(2)
%   stationarity region for (phi1, phi2).  Truncation normalisers that
%   are constant in theta are omitted (harmless for SMC/MH; noted for
%   the logZ estimate).
%
%   See also jointstar.defaultPriors, jointstar.priorSample.

prm = P.params;
lp = 0;
for j = 1:P.d
    x = tv(j); q = prm(j);
    switch q.type
        case 'norm'
            lp = lp - 0.5 * ((x - q.p1) / q.p2)^2 - log(q.p2) - 0.5 * log(2 * pi);
        case 'tnorm'
            if x < q.lo || x > q.hi, lp = -Inf; return; end
            lp = lp - 0.5 * ((x - q.p1) / q.p2)^2 - log(q.p2) - 0.5 * log(2 * pi);
        case 'unif'
            if x < q.lo || x > q.hi, lp = -Inf; return; end
            lp = lp - log(q.hi - q.lo);
        case 'beta'
            if x <= 0 || x >= 1, lp = -Inf; return; end
            lp = lp + (q.p1 - 1) * log(x) + (q.p2 - 1) * log(1 - x) ...
                - betaln(q.p1, q.p2);
        case 'negbeta'
            if x <= -1 || x >= 0, lp = -Inf; return; end
            lp = lp + (q.p1 - 1) * log(-x) + (q.p2 - 1) * log(1 + x) ...
                - betaln(q.p1, q.p2);
        case 'igsd'
            if x <= 0, lp = -Inf; return; end
            a = q.p1; b = (q.p1 - 1) * q.p2^2;   % sigma^2 ~ IG(a, b)
            v = x^2;
            lp = lp + a * log(b) - gammaln(a) - (a + 1) * log(v) - b / v ...
                + log(2 * x);                    % Jacobian sigma -> sigma^2
        case 'logn'
            if x <= 0, lp = -Inf; return; end
            lp = lp - 0.5 * ((log(x) - q.p1) / q.p2)^2 - log(x * q.p2) ...
                - 0.5 * log(2 * pi);
        case 'tgamma'
            if x < q.lo, lp = -Inf; return; end
            lp = lp + (q.p1 - 1) * log(x) - x / q.p2 ...
                - q.p1 * log(q.p2) - gammaln(q.p1);
        case 'fixed'
            % calibrated constant: sampled at its value and excluded from
            % mutation via mutateIdx.  The support check makes any
            % mis-wiring loud (a moved constant would otherwise drift as
            % a flat-prior free parameter).
            if x ~= q.init, lp = -Inf; return; end
        case 'hkap'
            % handled jointly in the hierarchical-kappa block below
        otherwise
            error('jointstar:badPrior', 'unknown prior type %s', q.type);
    end
end

% ---- hierarchical COVID kappas (Checkpoint 6) ---------------------------
% kappa_i ~ Gamma(a_g, scale s_g) * 1[kappa >= 1], s_g = m_g / a_g, with
% hyperparameters carried in theta as logs (their own Normal priors are
% handled in the loop above).  The truncation normaliser P(X >= 1) depends
% on (a_g, m_g), so it must be included here.
if isfield(P, 'kap')
    kp = P.kap;
    kv = tv(kp.kapCols)';
    if any(kv < 1), lp = -Inf; return; end
    a = exp(tv(kp.laCols(kp.groups)))';
    m = exp(tv(kp.lmCols(kp.groups)))';
    s = m ./ a;
    normc = gammainc(1 ./ s, a, 'upper');     % P(Gamma(a, s) >= 1)
    if any(normc < 1e-290), lp = -Inf; return; end
    lp = lp + sum((a - 1) .* log(kv) - kv ./ s - a .* log(s) ...
        - gammaln(a) - log(normc));
end

% joint AR(2) stationarity of the output gap (phi1 = phisum - phi2)
f2 = tv(P.idx.phi2); f1 = tv(P.idx.phisum) - f2;
if ~(abs(f2) < 1 && f1 + f2 < 1 && f2 - f1 < 1)
    lp = -Inf;
end
if isnan(lp), lp = -Inf; end   % overflowed tails are rejections, not NaNs
end
