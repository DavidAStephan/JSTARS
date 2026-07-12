classdef testHorseshoeSMC < matlab.unittest.TestCase
    %TESTHORSESHOESMC Checkpoint 5 machinery: horseshoe-in-SMC end to end.
    %
    %   (1) Consistency of the extended parameter space (sample/logpdf,
    %       CP4 back-compatibility of the LDL' refactor);
    %   (2) integration: on a small state-space with one truly correlated
    %       shock pair and one null pair, SMC with the Gibbs hook puts
    %       posterior mass away from zero on the true L entry and shrinks
    %       the null one.

    methods (Test)

        function extendedPriorSpaceConsistent(tc)
            rng(31);
            % gap column of Lq excluded by owner ruling (13 pairs gone:
            % 91-13 = 78) + 8 observables incl. pie_obs (28 Lr pairs)
            P = jointstar.horseshoePriors();
            nL = 78 + 28;
            nBase = 67;                     % 66 + sme_pieobs
            tc.verifyEqual(P.d, nBase + 3 * nL + 4 + 4);
            % +1: sme_pieobs is a calibrated constant (owner ruling)
            tc.verifyEqual(nnz(~P.mutateIdx), 2 * nL + 4 + 4 + 1);
            % sign restrictions per ruling: phi_y, phi_u <= 0
            tv0 = jointstar.priorSample(P, 1);
            tv0(P.idx.phiu) = 0.02;
            tc.verifyEqual(jointstar.priorLogPdf(P, tv0), -Inf, ...
                'positive phi_u must be outside the prior support');
            tc.verifyFalse(isfield(P.idx, 'Lq_xi_c'), ...
                'gap correlations must be excluded per owner ruling');
            tc.verifyFalse(isfield(P.idx, 'Lq_Ustar_c'));
            Theta = jointstar.priorSample(P, 50);
            for i = 1:10
                lp = jointstar.priorLogPdf(P, Theta(i, :));
                tc.verifyTrue(isfinite(lp), 'prior logpdf not finite at prior draw');
            end
            % nonpositive hyper -> -Inf
            tv = Theta(1, :); tv(P.hs.colTau2(1)) = -1;
            tc.verifyEqual(jointstar.priorLogPdf(P, tv), -Inf);
            % unpack round-trip
            [Lq, Lr] = jointstar.hsUnpack(P, Theta(1, :));
            tc.verifyEqual(diag(Lq), ones(14, 1));
            tc.verifyEqual(diag(Lr), ones(8, 1));
            tc.verifyEqual(Lq(3, 2), Theta(1, P.idx.Lq_pie_Ustar));
            tc.verifyEqual(Lq(:, 1), [1; zeros(13, 1)], ...
                'gap column of Lq must stay identity');
        end

        function hierKappaPriorConsistent(tc)
            rng(32);
            % Checkpoint 6 space: hierarchical kappas on top of horseshoe
            P = jointstar.horseshoePriors('HierKappa', true);
            tc.verifyEqual(P.baseD, 67 + 12);   % + 6 log-means, 6 log-shapes
            Theta = jointstar.priorSample(P, 50);
            tc.verifyTrue(all(Theta(:, P.kap.kapCols) >= 1, 'all'), ...
                'kappas must respect the >= 1 truncation');
            for i = 1:10
                lp = jointstar.priorLogPdf(P, Theta(i, :));
                tc.verifyTrue(isfinite(lp), 'hier-kappa prior logpdf not finite');
            end
            % kappa below 1 -> -Inf
            tv = Theta(1, :); tv(P.kap.kapCols(3)) = 0.9;
            tc.verifyEqual(jointstar.priorLogPdf(P, tv), -Inf);
            % shared hypers: same group => same (a, m) enter the density
            tc.verifyEqual(P.kap.groups([2, 4, 6, 11]), ones(4, 1), ...
                '2020-window kappas must share group 1');
        end

        function ldlRefactorMatchesCheckpoint4(tc)
            % identity factors must reproduce the diagonal-model likelihood
            dat = jointstar.loadData(fullfile(fileparts(fileparts( ...
                mfilename('fullpath'))), 'data.csv'));
            P0 = jointstar.defaultPriors();
            th = jointstar.thetaStruct(P0, [P0.params.init]);
            ll0 = jointstar.computeLogLik( ...
                jointstar.ModelSpec.jointstar(th, dat).system(), dat.y);
            cf = struct('Lq', eye(14), 'Lr', eye(numel(dat.obsNames)));
            ll1 = jointstar.computeLogLik( ...
                jointstar.ModelSpec.jointstar(th, dat, cf).system(), dat.y);
            % the two R paths (diagonal fast path vs chol-inverse of a
            % diagonal Rfull) agree to floating-point path noise only
            tc.verifyEqual(ll1, ll0, 'RelTol', 1e-7, ...
                'identity LDL'' does not reproduce the diagonal model');
        end

        function findsTrueCorrelationShrinksNull(tc)
            rng(33);
            % 3 independent random walks, y = level + noise.  True shock
            % correlation only between states 1 and 2 (rho = 0.85).
            T = 300; m = 3;
            sd = [0.6; 0.6; 0.6];
            Ltrue = eye(m); Ltrue(2, 1) = 0.85;   % strong, identifiable
            Qtrue = Ltrue * diag(sd.^2) * Ltrue';
            LQ = chol(Qtrue, 'lower');
            a = zeros(m, T);
            for t = 2:T, a(:, t) = a(:, t - 1) + LQ * randn(m, 1); end
            y = a + 0.3 * randn(m, T);

            % theta: 3 log shock sds + 3 L off-diagonals (one group)
            nL = 3;
            groups = ones(nL, 1);
            makeSys = @(tv) struct('A1', eye(m), 'A2', [], 'c', [], ...
                'Q', unpack3(tv) * diag(exp(2 * tv(1:3))) * unpack3(tv)', ...
                'a1', zeros(m, 1), 'P1', 25 * eye(m), 'Z', eye(m), ...
                'd', [], 'Rdiag', 0.09 * ones(m, 1), 'T', T);
            logLik = @(tv) jointstar.computeLogLik(makeSys(tv), y);

            colL = 4:6; colLam2 = 7:9; colNu = 10:12; colTau2 = 13; colXi = 14;
            logPrior = @(tv) hsLogPrior(tv, colL, colLam2, colNu, colTau2, colXi, groups);
            samplePrior = @(N) hsSample(N, colXi);

            hook = @(Pm, lp, ll, phi) gibbsHook(Pm, lp, ll, ...
                colL, colLam2, colNu, colTau2, colXi, groups, logPrior);
            prob = struct('samplePrior', samplePrior, 'logPrior', logPrior, ...
                'logLik', logLik);
            opts = struct('NParticles', 400, 'MSteps', 4, 'Seed', 5, ...
                'Verbose', false, 'UseParallel', false, ...
                'MutateIdx', [true(1, 6), false(1, 8)], 'ExtraMutate', hook);
            out = jointstar.runSMC(prob, opts);

            W = out.weights;
            Lpost = W' * out.particles(:, colL);       % [L21, L31, L32]
            q05 = arrayfun(@(j) wq(out.particles(:, j), W, 0.05), colL);
            q95 = arrayfun(@(j) wq(out.particles(:, j), W, 0.95), colL);

            tc.verifyGreaterThan(Lpost(1), 0.5, 'true L21 underestimated');
            tc.verifyGreaterThan(q05(1), 0, 'true L21 band must exclude 0');
            tc.verifyLessThan(abs(Lpost(2)), 0.25, 'null L31 not shrunk');
            tc.verifyLessThan(abs(Lpost(3)), 0.25, 'null L32 not shrunk');
            tc.verifyTrue(q05(2) < 0 && q95(2) > 0, 'null L31 band should straddle 0');
        end

    end
end

% ======================================================================
function L = unpack3(tv)
L = eye(3); L(2, 1) = tv(4); L(3, 1) = tv(5); L(3, 2) = tv(6);
end

function lp = hsLogPrior(tv, colL, colLam2, colNu, colTau2, colXi, groups)
lam2 = tv(colLam2)'; nu = tv(colNu)'; tau2 = tv(colTau2)'; xi = tv(colXi)';
if any([lam2; nu; tau2; xi] <= 0), lp = -Inf; return; end
% log sds ~ N(log 0.5, 1)
lp = -0.5 * sum((tv(1:3) - log(0.5)).^2);
s2 = tau2(groups) .* lam2;
Lv = tv(colL)';
ig = @(x, a, b) a .* log(b) - gammaln(a) - (a + 1) .* log(x) - b ./ x;
lp = lp - 0.5 * sum(Lv.^2 ./ s2) - 0.5 * sum(log(2 * pi * s2)) ...
    + sum(ig(lam2, 0.5, 1 ./ nu)) + sum(ig(nu, 0.5, 1)) ...
    + sum(ig(tau2, 0.5, 1 ./ xi)) + sum(ig(xi, 0.5, 1));
end

function Theta = hsSample(N, dLast)
d = dLast;
Theta = zeros(N, d);
Theta(:, 1:3) = log(0.5) + randn(N, 3);
cl = @(x) min(max(x, 1e-10), 1e10);
xi = cl(1 ./ gd(0.5 * ones(N, 1))); tau2 = cl((1 ./ xi) ./ gd(0.5 * ones(N, 1)));
nu = cl(1 ./ gd(0.5 * ones(N, 3))); lam2 = cl((1 ./ nu) ./ gd(0.5 * ones(N, 3)));
Theta(:, 4:6) = sqrt(tau2 .* lam2) .* randn(N, 3);
Theta(:, 7:9) = lam2; Theta(:, 10:12) = nu;
Theta(:, 13) = tau2; Theta(:, 14) = xi;
end

function [Pm, lp, ll] = gibbsHook(Pm, lp, ll, colL, colLam2, colNu, ...
    colTau2, colXi, groups, logPrior)
for i = 1:size(Pm, 1)
    st = struct('groups', groups, 'lambda2', Pm(i, colLam2)', ...
        'nu', Pm(i, colNu)', 'tau2', Pm(i, colTau2)', 'xi', Pm(i, colXi)');
    st = jointstar.horseshoeSample(Pm(i, colL)', st);
    Pm(i, colLam2) = st.lambda2'; Pm(i, colNu) = st.nu';
    Pm(i, colTau2) = st.tau2'; Pm(i, colXi) = st.xi';
    lp(i) = logPrior(Pm(i, :));
end
end

function x = gd(a)
% Gamma(a,1), a possibly < 1 (boost)
boost = a < 1; ab = a + boost;
d = ab - 1 / 3; c = 1 ./ sqrt(9 * d);
x = zeros(size(a)); todo = true(size(a));
while any(todo)
    n = nnz(todo); z = randn(n, 1); v = (1 + c(todo) .* z).^3; u = rand(n, 1);
    ok = v > 0 & log(u) < 0.5 * z.^2 + d(todo) .* (1 - v + log(max(v, realmin)));
    idx = find(todo);
    x(idx(ok)) = d(idx(ok)) .* v(ok); todo(idx(ok)) = false;
end
if any(boost), x(boost) = x(boost) .* rand(nnz(boost), 1).^(1 ./ a(boost)); end
x = max(x, realmin);
end

function q = wq(x, w, p)
[xs, i] = sort(x); cw = cumsum(w(i)); cw = cw / cw(end);
q = xs(find(cw >= p, 1, 'first'));
end
