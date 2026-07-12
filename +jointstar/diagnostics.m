function tbl = diagnostics(out, quiet)
%DIAGNOSTICS Posterior summary of an SMC particle cloud.
%
%   tbl = jointstar.diagnostics(out)
%
%   out is the result struct of jointstar.runSMC.  Returns (and prints,
%   unless quiet) a table with the weighted posterior mean, sd and
%   5% / 50% / 95% quantiles of each parameter, plus prints the tempering
%   summary (stage count, final ESS, mean MH acceptance, wall-clock).

if nargin < 2, quiet = false; end

P = out.particles; W = out.weights;
d = size(P, 2);
mu = (W' * P)';
sd = sqrt((W' * (P - mu').^2))';
q = zeros(d, 3);
for j = 1:d
    q(j, :) = wquantile(P(:, j), W, [0.05, 0.50, 0.95]);
end

tbl = table(string(out.paramNames(:)), mu, sd, q(:, 1), q(:, 2), q(:, 3), ...
    'VariableNames', {'param', 'mean', 'sd', 'q05', 'q50', 'q95'});

if ~quiet
    s = out.stages;
    fprintf('SMC: %d stages, final ESS %.0f/%d, mean acc %.2f, %.1f s wall-clock\n', ...
        height(s), essOf(out.logw), numel(W), mean(s.acc_rate), out.wallclock);
    disp(tbl);
end
end

function e = essOf(logw)
w = exp(logw - max(logw)); w = w / sum(w);
e = 1 / sum(w.^2);
end

function q = wquantile(x, w, ps)
[xs, i] = sort(x);
cw = cumsum(w(i));
cw = cw / cw(end);
q = zeros(size(ps));
for k = 1:numel(ps)
    j = find(cw >= ps(k), 1, 'first');
    q(k) = xs(j);
end
end
