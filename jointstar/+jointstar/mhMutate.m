function [theta, lp, ll, nAcc] = mhMutate(theta, lp, ll, phi, mut, ...
    epsMat, logu, logPrior, logLik)
%MHMUTATE Block random-walk MH mutation of one SMC particle.
%
%   [theta, lp, ll, nAcc] = jointstar.mhMutate(theta, lp, ll, phi, ...
%       mut, epsMat, logu, logPrior, logLik)
%
%   Applies mut.M sweeps of block-wise Metropolis-Hastings targeting the
%   tempered posterior p(theta) * p(y|theta)^phi.  Two kinds of blocks:
%
%   * covariance blocks (mut.covBlocks / covLprops / covEpsRows): joint
%     Gaussian steps with the lower Cholesky factor of the scaled
%     weighted cloud covariance -- the right metric for ordinary
%     parameters;
%   * scaled blocks (mut.scBlocks / scPos / scEpsRows): PER-PARTICLE
%     diagonal steps with sd = mut.cScale * mut.localScaleFn(theta),
%     evaluated fresh before each proposal -- the right metric for
%     hierarchically-scaled coordinates (horseshoe L_ij), whose
%     conditional prior scale is particle-specific.  Cloud-scale steps
%     on a tight-prior particle are rejected with probability ~1.
%
%   Proposal noise epsMat and acceptance uniforms logu are pre-generated
%   by the caller, so the mutation is a pure function: results do not
%   depend on parfor scheduling.  nAcc counts accepted proposals (max
%   mut.M * (#cov blocks + #scaled blocks)).
%
%   Proposals with -Inf log-prior are rejected without evaluating the
%   likelihood (the expensive call).
%
%   See also jointstar.runSMC.

nBC = numel(mut.covBlocks);
nBS = numel(mut.scBlocks);
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
    for b = 1:nBS
        u = u + 1;
        prop = theta;
        cols = mut.scBlocks{b};
        db = numel(cols);
        sd = (2.38 / sqrt(db)) * mut.cScale * mut.localScaleFn(theta);
        prop(cols) = prop(cols) + ...
            sd(mut.scPos{b}) .* epsMat(mut.scEpsRows{b}, j)';
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
