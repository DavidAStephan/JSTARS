function [Pmat, lp, ll] = horseshoeMutate(Pmat, lp, ll, phi, P) %#ok<INUSD>
%HORSESHOEMUTATE Gibbs-within-SMC sweep of the horseshoe hyper-scales.
%
%   [Pmat, lp, ll] = jointstar.horseshoeMutate(Pmat, lp, ll, phi, P)
%
%   For every particle: one Makalic-Schmidt sweep updating
%   (lambda^2, nu, tau^2, xi) conditional on that particle's L values
%   (jointstar.horseshoeSample), then refresh the log-prior.  The
%   likelihood is untouched -- the scales enter only the prior -- so this
%   move is invariant for every tempered target regardless of phi.
%   Intended as the ExtraMutate hook of jointstar.runSMC, with
%   opts.MutateIdx = P.mutateIdx so MH never moves the hyper columns.
%
%   Runs serially on the client for exact reproducibility (the sweep is
%   O(#coefficients) per particle -- negligible next to the likelihood).
%
%   See also jointstar.horseshoePriors, jointstar.horseshoeSample,
%   jointstar.runSMC.

hs = P.hs;
N = size(Pmat, 1);
for i = 1:N
    st = struct('groups', hs.groups, ...
        'lambda2', Pmat(i, hs.colLam2)', 'nu', Pmat(i, hs.colNu)', ...
        'tau2', Pmat(i, hs.colTau2)', 'xi', Pmat(i, hs.colXi)');
    st = jointstar.horseshoeSample(Pmat(i, hs.colL)', st);
    Pmat(i, hs.colLam2) = st.lambda2';
    Pmat(i, hs.colNu) = st.nu';
    Pmat(i, hs.colTau2) = st.tau2';
    Pmat(i, hs.colXi) = st.xi';
    lp(i) = jointstar.priorLogPdf(P, Pmat(i, :));
end
end
