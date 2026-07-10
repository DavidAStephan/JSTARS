function hs = horseshoeSample(beta, hs)
%HORSESHOESAMPLE One Makalic-Schmidt Gibbs sweep of horseshoe scales.
%
%   hs = jointstar.horseshoeSample(beta, hs)
%
%   Grouped horseshoe prior for coefficients beta (k x 1):
%
%       beta_i | tau_g, lambda_i ~ N(0, tau_{g(i)}^2 * lambda_i^2)
%       lambda_i ~ C+(0, 1),   tau_g ~ C+(0, 1)
%
%   using the Makalic-Schmidt (2016) inverse-gamma auxiliary-variable
%   representation, which makes every conditional an inverse-gamma draw:
%
%       lambda_i^2 | .  ~ IG(1, 1/nu_i + beta_i^2 / (2 tau_g^2))
%       nu_i       | .  ~ IG(1, 1 + 1/lambda_i^2)
%       tau_g^2    | .  ~ IG((k_g+1)/2, 1/xi_g + sum_g beta_i^2/(2 lambda_i^2))
%       xi_g       | .  ~ IG(1, 1 + 1/tau_g^2)
%
%   This is intended as a Gibbs-within-SMC mutation (the ExtraMutate hook
%   of jointstar.runSMC): the scales enter only the prior on beta, not
%   the likelihood, so the particle's log-likelihood is unchanged by this
%   move.
%
%   hs struct fields (state, all updated in place):
%     .groups   k x 1 integer group index g(i) in 1..G  (set once)
%     .lambda2  k x 1 local scales^2
%     .nu       k x 1 auxiliaries
%     .tau2     G x 1 global scales^2 (one per group)
%     .xi       G x 1 auxiliaries
%
%   Initialise with jointstar.horseshoeSample('init', groups).
%
%   Uses only base MATLAB randomness (no Statistics Toolbox): IG(a,b) is
%   drawn as b / gamrnd(a,1) with a local Marsaglia-Tsang gamma sampler.
%
%   See also jointstar.runSMC.

if ischar(beta) || isstring(beta)          % hs = horseshoeSample('init', groups)
    assert(strcmpi(beta, 'init'), 'jointstar:badInit', ...
        'first argument must be a beta vector or ''init''.');
    groups = hs(:);
    G = max(groups);
    hs = struct('groups', groups, 'lambda2', ones(numel(groups), 1), ...
        'nu', ones(numel(groups), 1), 'tau2', ones(G, 1), 'xi', ones(G, 1));
    return
end

beta = beta(:);
g = hs.groups;
k = numel(beta);
assert(numel(g) == k, 'jointstar:dimMismatch', ...
    'beta has %d entries but groups has %d.', k, numel(g));
G = numel(hs.tau2);
tau2i = hs.tau2(g);                        % per-coefficient global scale

% lambda_i^2 | . ~ IG(1, 1/nu_i + beta_i^2/(2 tau^2))
% lambda^2 / tau^2 are clamped to [2.5e-3, 1e2] (lambda, tau in
% [0.05, 10]), matching the truncated half-Cauchy priors used for the
% covariance factorisations: the far upper tail destroys conditioning,
% and the spike below 0.05 creates razor-sharp conditionals that freeze
% SMC mutation while adding nothing (an implied L sd of 2.5e-3 is
% already "zero" for a covariance entry).
clS = @(x) min(max(x, 2.5e-3), 1e2);
hs.lambda2 = clS(igDraw(ones(k, 1), 1 ./ hs.nu + beta.^2 ./ (2 * tau2i)));
% nu_i | . ~ IG(1, 1 + 1/lambda_i^2)
hs.nu = igDraw(ones(k, 1), 1 + 1 ./ hs.lambda2);

% tau_g^2 | . ~ IG((k_g+1)/2, 1/xi_g + 0.5 * sum_{i in g} beta_i^2/lambda_i^2)
s = accumarray(g, beta.^2 ./ hs.lambda2, [G, 1]);
kg = accumarray(g, 1, [G, 1]);
hs.tau2 = clS(igDraw((kg + 1) / 2, 1 ./ hs.xi + s / 2));
% xi_g | . ~ IG(1, 1 + 1/tau_g^2)
hs.xi = igDraw(ones(G, 1), 1 + 1 ./ hs.tau2);
end

% ======================================================================
function x = igDraw(a, b)
% inverse-gamma(a, b) draws: x = b / Gamma(a, 1), clamped so the
% heavy-tailed chain can never overflow to Inf (NaN log-densities)
x = min(max(b ./ gammaDraw(a), 1e-12), 1e12);
end

function x = gammaDraw(a)
% Marsaglia-Tsang (2000) Gamma(a, 1) sampler, vectorised, a >= 0.5 here.
% For a < 1 use the boost x = Gamma(a+1) * U^(1/a).
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
    ok = v > 0 & log(u) < 0.5 * z.^2 + d(todo) .* (1 - v + log(max(v, realmin)));
    idx = find(todo);
    x(idx(ok)) = d(idx(ok)) .* v(ok);
    todo(idx(ok)) = false;
end
if any(boost)
    x(boost) = x(boost) .* rand(nnz(boost), 1).^(1 ./ a(boost));
end
x = max(x, realmin);   % guard against exact zeros downstream
end
