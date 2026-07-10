function alpha = drawStates(aux, nDraws)
%DRAWSTATES Sample state trajectories alpha ~ N(muTilde, Htilde^{-1}).
%
%   alpha = jointstar.drawStates(aux)          one m x T draw
%   alpha = jointstar.drawStates(aux, nDraws)  m x T x nDraws draws
%
%   aux is the second output of jointstar.computeLogLik and carries the
%   sparse Cholesky factor L, its fill-reducing permutation perm
%   (Htilde(perm, perm) = L*L') and the posterior mean muTilde.  Each
%   draw costs one sparse back-substitution:
%
%       w = L' \ z,  z ~ N(0, I),   alpha(perm) = muTilde(perm) + w.
%
%   Set the RNG at the call site for reproducibility.
%
%   See also jointstar.computeLogLik.

if nargin < 2, nDraws = 1; end
if isempty(aux)
    error('jointstar:infeasibleTheta', ...
        'aux is empty: computeLogLik returned -Inf at this theta.');
end

n = aux.m * aux.T;
alpha = zeros(aux.m, aux.T, nDraws);
muP = aux.muTilde(aux.perm);
for k = 1:nDraws
    w = aux.L' \ randn(n, 1);
    x = zeros(n, 1);
    x(aux.perm) = muP + w;
    alpha(:, :, k) = reshape(x, aux.m, aux.T);
end
end
