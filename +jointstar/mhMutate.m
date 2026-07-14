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
%   Transformed-scale kernel ('MutationTransform' option, see
%   jointstar.paramTransform / jointstar.runSMC): if mut.transform is a
%   nonempty transform struct T, the entire sweep is done in
%   eta = T.toEta(theta) coordinates -- proposals are Gaussian steps on
%   eta, theta is recovered only via T.toTheta(propEta) to evaluate
%   logPrior/logLik, and the MH log-accept ratio adds the Jacobian
%   difference T.logJac(propEta) - T.logJac(eta) (DESIGN_transformed_
%   kernel.md ADDENDUM requirement R1: this Jacobian difference is the
%   full asymmetry correction; no separate theta-space Hastings term).
%   theta is only ever overwritten by a freshly-accepted propTheta
%   (always an exact T.toTheta of a finite, non-clamped eta) -- never by
%   round-tripping the (possibly boundary-clamped) entry eta back
%   through toTheta -- so a clamped value is never written into stored
%   theta (R5).  With mut.transform empty/absent this function is the
%   original untransformed raw-scale kernel, unchanged in code path and
%   RNG consumption (R6).
%
%   See also jointstar.runSMC, jointstar.paramTransform.

nBC = numel(mut.covBlocks);
nAcc = 0;
u = 0;

if isfield(mut, 'transform') && ~isempty(mut.transform)
    T = mut.transform;
    eta = T.toEta(theta);
    jCur = T.logJac(eta);
    for j = 1:mut.M
        for b = 1:nBC
            u = u + 1;
            cols = mut.covBlocks{b};
            propEta = eta;
            propEta(cols) = propEta(cols) + (mut.covLprops{b} * epsMat(mut.covEpsRows{b}, j))';
            propTheta = T.toTheta(propEta);
            jProp = T.logJac(propEta);
            [eta, theta, lp, ll, jCur, nAcc] = tryAcceptTransformed(eta, theta, ...
                lp, ll, jCur, nAcc, propEta, propTheta, jProp, phi, logu(u), ...
                logPrior, logLik);
        end
    end
else
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

function [eta, theta, lp, ll, jCur, nAcc] = tryAcceptTransformed(eta, theta, ...
    lp, ll, jCur, nAcc, propEta, propTheta, jProp, phi, lu, logPrior, logLik)
lpP = logPrior(propTheta);
if ~isfinite(lpP)
    return
end
llP = logLik(propTheta);
if isnan(llP), llP = -Inf; end
logAcc = (lpP + phi * llP + jProp) - (lp + phi * ll + jCur);
if lu < logAcc
    eta = propEta; theta = propTheta; lp = lpP; ll = llP; jCur = jProp;
    nAcc = nAcc + 1;
end
end
