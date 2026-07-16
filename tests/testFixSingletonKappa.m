classdef testFixSingletonKappa < matlab.unittest.TestCase
    %TESTFIXSINGLETONKAPPA 'FixSingletonKappa' option unit tests.
    %
    %   Only meaningful when HierKappa=true.  The two single-member
    %   COVID-kappa window groups (g5 = w2022tot = {kaphpp_2022}, g6 =
    %   w2023tot = {kappi_2023}) have hierarchical hypers with ZERO
    %   likelihood curvature (CHECKPOINT_14: they enter only
    %   priorLogPdf, never ModelSpec).  This flag removes their 4 hyper
    %   dims (kapHyp_lm/la_w2022tot, kapHyp_lm/la_w2023tot) from theta
    %   and gives each singleton kappa a fixed truncated-Gamma(2.0,1.25)
    %   prior instead -- the CONDITIONAL prior at the hypers' own prior
    %   means.  DEFAULT FALSE (or HierKappa=false) must reproduce
    %   today's defaultPriors add() sequence byte-for-byte.

    properties (Constant)
        DataFile = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
            'data.csv')
    end

    methods (Test)

        % (a) default-off inertness -----------------------------------
        function flagOffInertWithExplicitFalse(tc)
            P0 = jointstar.defaultPriors('HierKappa', true);
            P1 = jointstar.defaultPriors('HierKappa', true, ...
                'FixSingletonKappa', false);
            tc.verifyTrue(isequaln(P0, P1), ...
                'FixSingletonKappa=false must be byte-identical to omitted');
        end

        function flagInertWhenHierKappaFalse(tc)
            % under HierKappa=false all 12 kappas are already plain
            % tgamma(2,2) with no hypers and defaultPriors returns before
            % the hierarchical branch -- the flag must be a no-op there
            P0 = jointstar.defaultPriors('HierKappa', false);
            P1 = jointstar.defaultPriors('HierKappa', false, ...
                'FixSingletonKappa', true);
            tc.verifyTrue(isequaln(P0, P1), ...
                'FixSingletonKappa must be inert when HierKappa=false');
        end

        % (b) flag-on structure -----------------------------------------
        function flagOnRemovesFourHyperDims(tc)
            P0 = jointstar.defaultPriors('HierKappa', true);
            P2 = jointstar.defaultPriors('HierKappa', true, ...
                'FixSingletonKappa', true);
            tc.verifyEqual(P2.d, P0.d - 4);

            dropped = {'kapHyp_lm_w2022tot', 'kapHyp_la_w2022tot', ...
                'kapHyp_lm_w2023tot', 'kapHyp_la_w2023tot'};
            for i = 1:numel(dropped)
                tc.verifyFalse(ismember(dropped{i}, P2.names), ...
                    sprintf('%s must be removed from theta', dropped{i}));
            end
        end

        function flagOnKeepsSingletonKappasAsFixedTruncatedGamma(tc)
            P2 = jointstar.defaultPriors('HierKappa', true, ...
                'FixSingletonKappa', true);

            jh = find(strcmp(P2.names, 'kaphpp_2022'), 1);
            jp = find(strcmp(P2.names, 'kappi_2023'), 1);
            tc.verifyNotEmpty(jh);
            tc.verifyNotEmpty(jp);

            % still free parameters, still in mutateIdx
            tc.verifyTrue(P2.mutateIdx(jh));
            tc.verifyTrue(P2.mutateIdx(jp));

            qh = P2.params(jh); qp = P2.params(jp);
            tc.verifyEqual(qh.type, 'tgamma');
            tc.verifyEqual(qp.type, 'tgamma');
            tc.verifyEqual(qh.p1, 2.0, 'AbsTol', 1e-12);
            tc.verifyEqual(qh.p2, 1.25, 'AbsTol', 1e-12);
            tc.verifyEqual(qh.lo, 1);
            tc.verifyEqual(qp.p1, 2.0, 'AbsTol', 1e-12);
            tc.verifyEqual(qp.p2, 1.25, 'AbsTol', 1e-12);
            tc.verifyEqual(qp.lo, 1);
        end

        function flagOnLeavesMultiMemberGroupsUntouched(tc)
            P0 = jointstar.defaultPriors('HierKappa', true);
            P2 = jointstar.defaultPriors('HierKappa', true, ...
                'FixSingletonKappa', true);

            multiMemberHypers = {'kapHyp_lm_w2020', 'kapHyp_la_w2020', ...
                'kapHyp_lm_w2021', 'kapHyp_la_w2021', ...
                'kapHyp_lm_w2122', 'kapHyp_la_w2122', ...
                'kapHyp_lm_w2021tot', 'kapHyp_la_w2021tot'};
            for i = 1:numel(multiMemberHypers)
                tc.verifyTrue(ismember(multiMemberHypers{i}, P2.names), ...
                    sprintf('%s must remain a free param', multiMemberHypers{i}));
                j0 = find(strcmp(P0.names, multiMemberHypers{i}), 1);
                j2 = find(strcmp(P2.names, multiMemberHypers{i}), 1);
                tc.verifyEqual(P2.params(j2).p1, P0.params(j0).p1);
                tc.verifyEqual(P2.params(j2).p2, P0.params(j0).p2);
            end

            multiMemberKappas = {'kapc_2021', 'kapy_20', 'kapy_21', ...
                'kapu_20', 'kapu_2122', 'kappr_20', 'kappr_2122', ...
                'kappop_2021', 'kapk_20', 'kapk_21'};
            for i = 1:numel(multiMemberKappas)
                j2 = find(strcmp(P2.names, multiMemberKappas{i}), 1);
                tc.verifyEqual(P2.params(j2).type, 'hkap');
            end
        end

        function flagOnKapStructIsSelfConsistent(tc)
            P2 = jointstar.defaultPriors('HierKappa', true, ...
                'FixSingletonKappa', true);
            tc.verifyEqual(P2.kap.G, 4);
            tc.verifyEqual(numel(P2.kap.kapCols), 10);
            tc.verifyEqual(numel(P2.kap.groups), 10);
            tc.verifyEqual(numel(P2.kap.lmCols), 4);
            tc.verifyEqual(numel(P2.kap.laCols), 4);
            tc.verifyEqual(unique(P2.kap.groups(:)'), 1:4, ...
                'kept groups must be remapped onto the compact range 1..4');
            tc.verifyTrue(all(P2.kap.groups >= 1 & P2.kap.groups <= 4));

            % every hkap-typed kappa column must appear in kapCols exactly
            % once, and no tgamma-typed (singleton) kappa may appear
            hkapCols = find(strcmp({P2.params.type}, 'hkap'));
            tc.verifyEqual(sort(P2.kap.kapCols(:)'), sort(hkapCols));
        end

        function flagOnComposesWithGTrendRotationAndRateGapAR(tc)
            % orthogonal flags -- all just append/modify independent rows
            P = jointstar.defaultPriors('HierKappa', true, ...
                'FixSingletonKappa', true, 'GTrendRotation', true, ...
                'RateGapAR', true);
            tc.verifyTrue(any(strcmp(P.names, 'gtrend_sum')));
            tc.verifyTrue(any(strcmp(P.names, 'rho_rg')));
            tc.verifyEqual(P.kap.G, 4);
        end

        % (d) priorLogPdf finiteness / truncation preserved ---------------
        function priorLogPdfFiniteAtSampledDrawAndInfBelowTruncation(tc)
            rng(21);
            P2 = jointstar.defaultPriors('HierKappa', true, ...
                'FixSingletonKappa', true);
            Theta = jointstar.priorSample(P2, 10);
            tc.verifyEqual(size(Theta), [10, P2.d]);

            jh = find(strcmp(P2.names, 'kaphpp_2022'), 1);
            for i = 1:10
                lp = jointstar.priorLogPdf(P2, Theta(i, :));
                tc.verifyTrue(isfinite(lp));
            end

            tv = Theta(1, :);
            tv(jh) = 0.9;   % below the kappa>=1 truncation floor
            lp2 = jointstar.priorLogPdf(P2, tv);
            tc.verifyEqual(lp2, -Inf, ...
                'kappa>=1 truncation must still be enforced for the fixed prior');
        end

        function blockAtomsAndParamTransformAcceptReducedKap(tc)
            P2 = jointstar.defaultPriors('HierKappa', true, ...
                'FixSingletonKappa', true);
            covCols = find(P2.mutateIdx);
            atoms = jointstar.blockAtoms(P2, covCols);
            allRel = sort(horzcat(atoms{:}));
            tc.verifyEqual(allRel, 1:numel(covCols));

            T = jointstar.paramTransform(P2, P2.mutateIdx);
            jh = find(strcmp(P2.names, 'kaphpp_2022'), 1);
            jp = find(strcmp(P2.names, 'kappi_2023'), 1);
            tc.verifyEqual(T.kind(jh), int8(2));  % lo=1,hi=Inf -> log(theta-1)
            tc.verifyEqual(T.kind(jp), int8(2));
        end

        % (c) tiny flag-on smoke on the real model ------------------------
        function tinyFlagOnSmokeReachesPhi1(tc)
            dat = jointstar.loadData(tc.DataFile);
            P = jointstar.defaultPriors('HierKappa', true, ...
                'FixSingletonKappa', true);
            cache = jointstar.buildEvalCache(dat, P);

            prob = struct( ...
                'samplePrior', @(N) jointstar.priorSample(P, N), ...
                'logPrior', @(tv) jointstar.priorLogPdf(P, tv), ...
                'logLik', @(tv) logLikTheta(P, tv, dat, cache), ...
                'paramNames', {P.names}, ...
                'priors', P);
            opts = struct('NParticles', 60, 'MSteps', 1, 'Seed', 11, ...
                'Verbose', false, 'UseParallel', false, ...
                'MutateIdx', P.mutateIdx, ...
                'MutationTransform', true, 'StructuredBlocks', true, ...
                'MStepsLadder', true, 'WasteFree', true);
            out = jointstar.runSMC(prob, opts);

            tc.verifyEqual(out.stages.phi(end), 1, 'AbsTol', 1e-9);
            tc.verifyTrue(all(isfinite(out.particles(:))));
            tc.verifyTrue(all(isfinite(out.loglik)));
            tc.verifyGreaterThan(mean(out.stages.acc_rate), 0);
        end

    end
end

function ll = logLikTheta(P, tv, dat, cache)
th = jointstar.thetaStruct(P, tv);
spec = jointstar.ModelSpec.jointstar(th, dat, [], cache);
ll = jointstar.computeLogLik(spec.system(), dat.y, cache);
end
