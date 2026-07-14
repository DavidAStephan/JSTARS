function [theta, lp, ll, nAcc] = mhMutate(theta, lp, ll, phi, mut, ...
    epsMat, logu, logPrior, logLik)
%MHMUTATE Block random-walk MH mutation of one SMC particle.
%
%   [theta, lp, ll, nAcc] = jointstar.mhMutate(theta, lp, ll, phi, ...
%       mut, epsMat, logu, logPrior, logLik)
%
%   Applies mut.M sweeps of block-wise Metropolis-Hastings targeting the
%   tempered posterior p(theta) * p(y|theta)^phi, using covariance blocks
%   (mut.covBlocks / covLprops / covEpsRows): joint Gaussian steps with
%   the lower Cholesky factor of the scaled weighted cloud covariance.
%
%   Proposal noise epsMat and acceptance uniforms logu are pre-generated
%   by the caller, so the mutation is a pure function: results do not
%   depend on parfor scheduling.  nAcc counts accepted proposals (max
%   mut.M * #cov blocks).
%
%   Proposals with -Inf log-prior are rejected without evaluating the
%   likelihood (the expensive call).
%
%   See also jointstar.runSMC.

nBC = numel(mut.covBlocks);
nAcc = 0;
u = 0;
for j = 1:mut.M
    for b = 1:nBC
        u = u + 1;
        prop = theta;
        cols = mut.covBlocks{b};
        prop(cols) = prop(cols) + (mut.covLprops{b} * epsMat(mut.covEpsRows{b}, j))';
        [theta, lp, ll, nAcc] = tryAccept(theta, lp, ll, nAcc, prop, ...
            phi, logu(u), logPrior, logLik);
    end
end
end

function [theta, lp, ll, nAcc] = tryAccept(theta, lp, ll, nAcc, prop, ...
    phi, lu, logPrior, logLik)
lpP = logPrior(prop);
if ~isfinite(lpP)
    return
end
llP = logLik(prop);
if isnan(llP), llP = -Inf; end
logAcc = (lpP + phi * llP) - (lp + phi * ll);
if lu < logAcc
    theta = prop; lp = lpP; ll = llP;
    nAcc = nAcc + 1;
end
end
