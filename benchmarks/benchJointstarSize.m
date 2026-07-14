function out = benchJointstarSize()
%BENCHJOINTSTARSIZE Time computeLogLik on a JointSTAR-sized system.
%
%   The full model specification was still pending confirmation when this
%   benchmark was written, so it uses a synthetic-but-structurally-honest
%   system of the documented size:
%   m = 16 states x T = 260 quarters (stacked dimension 4160), p = 11
%   observables, an AR(2) block (second-lag transition), a diagonal
%   16 x 16 innovation covariance (matching the production diagonal-Q
%   model), and ~35% missing observations concentrated early-sample,
%   matching the data coverage pattern.
%   Target from the brief: 50-100 ms per likelihood evaluation, 1 core.

rng(1);
m = 16; T = 260; p = 11;

% stable random transition with AR(2) in one block
A1 = 0.6 * eye(m) + 0.05 * (rand(m) - 0.5) .* (rand(m) < 0.25);
A1 = A1 * (0.95 / max(abs(eig(A1))));
A2 = zeros(m); A2(2, 2) = -0.2;    % AR(2) cycle lag

% Q diagonal (production innovation covariance is diagonal, no horseshoe)
Q = diag(0.1 + rand(m, 1));

Z = zeros(p, m);
for i = 1:p
    cols = randperm(m, 3);
    Z(i, cols) = randn(1, 3);       % each observable loads on ~3 states
end

spec = jointstar.ModelSpec('A1', A1, 'A2', A2, 'Q', Q, ...
    'a1', zeros(m, 1), 'P1', 10 * eye(m), 'Z', Z, ...
    'Rdiag', 0.1 + rand(p, 1), 'T', T);
sys = spec.system();

y = randn(p, T);
y(6:11, 1:80) = NaN;               % late-starting series (r, inflation, surveys)
y(3, 200:210) = NaN;

lastwarn('');
maxNumCompThreads(1);              % single-core, per the brief's target

nRep = 50;
tic;
for k = 1:nRep
    ll = jointstar.computeLogLik(sys, y);
end
tLik = toc / nRep;

[ll, aux] = jointstar.computeLogLik(sys, y);
tic;
for k = 1:nRep
    a = jointstar.drawStates(aux);
end
tDraw = toc / nRep;

maxNumCompThreads('automatic');

fprintf('JointSTAR-sized benchmark (m=%d, T=%d, p=%d, single core):\n', m, T, p);
fprintf('  logLik = %.2f (finite: %d)\n', ll, isfinite(ll));
fprintf('  likelihood evaluation: %.1f ms   (target 50-100 ms)\n', 1000 * tLik);
fprintf('  state draw (reusing factor): %.1f ms\n', 1000 * tDraw);
fprintf('  Cholesky factor density: %.4f\n', nnz(aux.L) / numel(aux.L));

out = struct('msLik', 1000 * tLik, 'msDraw', 1000 * tDraw, 'logLik', ll);
end
