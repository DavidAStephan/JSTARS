function [logL, sm] = kalmanReference(sys, y)
%KALMANREFERENCE Textbook Kalman filter + RTS smoother (test oracle only).
%
%   [logL, sm] = kalmanReference(sys, y)
%
%   Deliberately plain, dense, first-order Kalman recursions used solely
%   to validate jointstar.computeLogLik / drawStates on small models
%   (Checkpoint 1).  Supports the same sys struct as computeLogLik except
%   A2 must be empty (put AR(2) states in companion form here if ever
%   needed).  Missing observations (NaN) are skipped, matching the
%   masking convention.
%
%   Outputs:
%     logL        marginal log-likelihood
%     sm.mean     m x T smoothed state means  E[alpha_t | y_{1:T}]
%     sm.var      m x m x T smoothed variances V[alpha_t | y_{1:T}]

if ~isempty(sys.A2)
    error('kalmanReference:ar2', 'Reference filter is first-order only.');
end

m = size(sys.A1, 1);
T = sys.T;

a = sys.a1(:);
P = sys.P1;
logL = 0;

aPred = zeros(m, T); PPred = zeros(m, m, T);
aFilt = zeros(m, T); PFilt = zeros(m, m, T);

for t = 1:T
    if t > 1
        A1t = tslice(sys.A1, t);
        ct = zeros(m, 1);
        if ~isempty(sys.c), ct = vslice(sys.c, t); end
        Qt = tslice(sys.Q, t);
        a = ct + A1t * a;
        P = A1t * P * A1t' + Qt;
    end
    aPred(:, t) = a; PPred(:, :, t) = P;

    idx = find(~isnan(y(:, t)));
    if ~isempty(idx)
        Zt = tslice(sys.Z, t);
        Zt = Zt(idx, :);
        if isfield(sys, 'Rfull') && ~isempty(sys.Rfull)
            Rf = tslice(sys.Rfull, t);
            Rt = Rf(idx, idx);
        else
            Rt = diag(vslicei(sys.Rdiag, t, idx));
        end
        yt = y(idx, t);
        if ~isempty(sys.d)
            dt = vslice(sys.d, t);
            yt = yt - dt(idx);
        end
        v = yt - Zt * a;
        F = Zt * P * Zt' + Rt;
        F = (F + F') / 2;
        K = P * Zt' / F;
        a = a + K * v;
        P = P - K * Zt * P;
        P = (P + P') / 2;
        logL = logL - 0.5 * (numel(v) * log(2 * pi) + log(det(F)) + v' * (F \ v));
    end
    aFilt(:, t) = a; PFilt(:, :, t) = P;
end

if nargout > 1
    % Rauch-Tung-Striebel backward pass
    sMean = zeros(m, T); sVar = zeros(m, m, T);
    sMean(:, T) = aFilt(:, T); sVar(:, :, T) = PFilt(:, :, T);
    for t = T - 1:-1:1
        A1n = tslice(sys.A1, t + 1);
        Pp = PPred(:, :, t + 1);
        J = PFilt(:, :, t) * A1n' / Pp;
        sMean(:, t) = aFilt(:, t) + J * (sMean(:, t + 1) - aPred(:, t + 1));
        V = PFilt(:, :, t) + J * (sVar(:, :, t + 1) - Pp) * J';
        sVar(:, :, t) = (V + V') / 2;
    end
    sm = struct('mean', sMean, 'var', sVar);
end
end

function B = tslice(A, t)
if ismatrix(A), B = A; else, B = A(:, :, t); end
end

function v = vslice(C, t)
if isvector(C), v = C(:); else, v = C(:, t); end
end

function v = vslicei(C, t, idx)
v = vslice(C, t);
v = v(idx);
end
