classdef testPoolAcuteOnly < matlab.unittest.TestCase
    %TESTPOOLACUTEONLY 'PoolAcuteOnly' option unit tests.
    %
    %   Only meaningful when HierKappa=true, INDEPENDENT of
    %   FixSingletonKappa.  Keeps the kappa hierarchy ONLY for g1 = w2020
    %   (the 4-member acute window {kapy_20, kapu_20, kappr_20, kapk_20})
    %   and collapses ALL other groups -- g2(w2021), g3(w2122),
    %   g4(w2021tot), g5(w2022tot), g6(w2023tot) -- to a fixed
    %   truncated-Gamma(2.0, 1.25) a-priori prior (the same conditional
    %   calibration FixSingletonKappa already uses for its two
    %   singletons).  This removes 10 hyperparameters (5 groups x
    %   {lm, la}) from theta.  DEFAULT FALSE (or HierKappa=false) must
    %   reproduce today's defaultPriors add() sequence byte-for-byte.
    %   PoolAcuteOnly DOMINATES FixSingletonKappa when both are set (it
    %   is a strict superset).

    properties (Constant)
        DataFile = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
            'data.csv')
    end

    methods (Test)

        % (a) default-off inertness -----------------------------------
        function flagOffInertWithExplicitFalse(tc)
            P0 = jointstar.defaultPriors('HierKappa', true);
            P1 = jointstar.defaultPriors('HierKappa', true, ...
                'PoolAcuteOnly', false);
            tc.verifyTrue(isequaln(P0, P1), ...
                'PoolAcuteOnly=false must be byte-identical to omitted');
        end

        function flagInertWhenHierKappaFalse(tc)
            % under HierKappa=false all 12 kappas are already plain
            % tgamma(2,2) with no hypers and defaultPriors returns before
            % the hierarchical branch -- the flag must be a no-op there
            P0 = jointstar.defaultPriors('HierKappa', false);
            P1 = jointstar.defaultPriors('HierKappa', false, ...
                'PoolAcuteOnly', true);
            tc.verifyTrue(isequaln(P0, P1), ...
                'PoolAcuteOnly must be inert when HierKappa=false');
        end

        function flagOffInertEvenWithFixSingletonKappaTrue(tc)
            % omitted PoolAcuteOnly (default false) must not perturb the
            % existing FixSingletonKappa path
            P0 = jointstar.defaultPriors('HierKappa', true, ...
                'FixSingletonKappa', true);
            P1 = jointstar.defaultPriors('HierKappa', true, ...
                'FixSingletonKappa', true, 'PoolAcuteOnly', false);
            tc.verifyTrue(isequaln(P0, P1), ...
                'PoolAcuteOnly=false must not alter the FixSingletonKappa path');
        end

        % (b) flag-on structure -----------------------------------------
        function flagOnRemovesTenHyperDims(tc)
            P0 = jointstar.defaultPriors('HierKappa', true);
            P2 = jointstar.defaultPriors('HierKappa', true, ...
                'PoolAcuteOnly', true);
            tc.verifyEqual(P2.d, P0.d - 10);

            dropped = {'kapHyp_lm_w2021', 'kapHyp_la_w2021', ...
                'kapHyp_lm_w2122', 'kapHyp_la_w2122', ...
                'kapHyp_lm_w2021tot', 'kapHyp_la_w2021tot', ...
                'kapHyp_lm_w2022tot', 'kapHyp_la_w2022tot', ...
                'kapHyp_lm_w2023tot', 'kapHyp_la_w2023tot'};
            for i = 1:numel(dropped)
                tc.verifyFalse(ismember(dropped{i}, P2.names), ...
                    sprintf('%s must be removed from theta', dropped{i}));
            end
        end

        function flagOnKeepsPooledKappasAsFixedTruncatedGamma(tc)
            P2 = jointstar.defaultPriors('HierKappa', true, ...
                'PoolAcuteOnly', true);

            pooled = {'kapy_21', 'kapk_21', 'kapu_2122', 'kappr_2122', ...
                'kapc_2021', 'kappop_2021', 'kaphpp_2022', 'kappi_2023'};
            for i = 1:numel(pooled)
                j = find(strcmp(P2.names, pooled{i}), 1);
                tc.verifyNotEmpty(j, sprintf('%s must be present', pooled{i}));

                % still a free parameter, still in mutateIdx
                tc.verifyTrue(P2.mutateIdx(j), ...
                    sprintf('%s must remain in mutateIdx', pooled{i}));

                q = P2.params(j);
                tc.verifyEqual(q.type, 'tgamma');
                tc.verifyEqual(q.p1, 2.0, 'AbsTol', 1e-12);
                tc.verifyEqual(q.p2, 1.25, 'AbsTol', 1e-12);
                tc.verifyEqual(q.lo, 1);
            end
        end

        function flagOnLeavesW2020GroupUntouched(tc)
            P0 = jointstar.defaultPriors('HierKappa', true);
            P2 = jointstar.defaultPriors('HierKappa', true, ...
                'PoolAcuteOnly', true);

            tc.verifyTrue(ismember('kapHyp_lm_w2020', P2.names));
            tc.verifyTrue(ismember('kapHyp_la_w2020', P2.names));
            j0 = find(strcmp(P0.names, 'kapHyp_lm_w2020'), 1);
            j2 = find(strcmp(P2.names, 'kapHyp_lm_w2020'), 1);
            tc.verifyEqual(P2.params(j2).p1, P0.params(j0).p1);
            tc.verifyEqual(P2.params(j2).p2, P0.params(j0).p2);

            w2020 = {'kapy_20', 'kapu_20', 'kappr_20', 'kapk_20'};
            for i = 1:numel(w2020)
                j2 = find(strcmp(P2.names, w2020{i}), 1);
                tc.verifyEqual(P2.params(j2).type, 'hkap');
            end
        end

        function flagOnKapStructIsSelfConsistent(tc)
            P2 = jointstar.defaultPriors('HierKappa', true, ...
                'PoolAcuteOnly', true);
            tc.verifyEqual(P2.kap.G, 1);
            tc.verifyEqual(numel(P2.kap.kapCols), 4);
            tc.verifyEqual(numel(P2.kap.groups), 4);
            tc.verifyEqual(numel(P2.kap.lmCols), 1);
            tc.verifyEqual(numel(P2.kap.laCols), 1);
            tc.verifyTrue(all(P2.kap.groups == 1));

            % every hkap-typed kappa column must appear in kapCols exactly
            % once, and no tgamma-typed (pooled) kappa may appear
            hkapCols = find(strcmp({P2.params.type}, 'hkap'));
            tc.verifyEqual(sort(P2.kap.kapCols(:)'), sort(hkapCols));
        end

        function poolAcuteOnlyDominatesFixSingletonKappa(tc)
            % both flags set: PoolAcuteOnly (superset) must win
            Pboth = jointstar.defaultPriors('HierKappa', true, ...
                'FixSingletonKappa', true, 'PoolAcuteOnly', true);
            Ppool = jointstar.defaultPriors('HierKappa', true, ...
                'PoolAcuteOnly', true);
            tc.verifyTrue(isequaln(Pboth, Ppool), ...
                'PoolAcuteOnly=true must dominate FixSingletonKappa=true');
        end

        function flagOnComposesWithGTrendRotationAndRateGapAR(tc)
            % orthogonal flags -- all just append/modify independent rows
            P = jointstar.defaultPriors('HierKappa', true, ...
                'PoolAcuteOnly', true, 'GTrendRotation', true, ...
                'RateGapAR', true);
            tc.verifyTrue(any(strcmp(P.names, 'gtrend_sum')));
            tc.verifyTrue(any(strcmp(P.names, 'rho_rg')));
            tc.verifyEqual(P.kap.G, 1);
        end

        % (d) priorLogPdf finiteness / truncation preserved ---------------
        function priorLogPdfFiniteAtSampledDrawAndInfBelowTruncation(tc)
            rng(21);
            P2 = jointstar.defaultPriors('HierKappa', true, ...
                'PoolAcuteOnly', true);
            Theta = jointstar.priorSample(P2, 10);
            tc.verifyEqual(size(Theta), [10, P2.d]);

            jp = find(strcmp(P2.names, 'kappi_2023'), 1);
            for i = 1:10
                lp = jointstar.priorLogPdf(P2, Theta(i, :));
                tc.verifyTrue(isfinite(lp));
            end

            tv = Theta(1, :);
            tv(jp) = 0.9;   % below the kappa>=1 truncation floor
            lp2 = jointstar.priorLogPdf(P2, tv);
            tc.verifyEqual(lp2, -Inf, ...
                'kappa>=1 truncation must still be enforced for the fixed prior');
        end

        function blockAtomsAndParamTransformAcceptReducedKap(tc)
            P2 = jointstar.defaultPriors('HierKappa', true, ...
                'PoolAcuteOnly', true);
            covCols = find(P2.mutateIdx);
            atoms = jointstar.blockAtoms(P2, covCols);
            allRel = sort(horzcat(atoms{:}));
            tc.verifyEqual(allRel, 1:numel(covCols));

            T = jointstar.paramTransform(P2, P2.mutateIdx);
            jp = find(strcmp(P2.names, 'kappi_2023'), 1);
            jh = find(strcmp(P2.names, 'kaphpp_2022'), 1);
            tc.verifyEqual(T.kind(jp), int8(2));  % lo=1,hi=Inf -> log(theta-1)
            tc.verifyEqual(T.kind(jh), int8(2));
        end

        % (c) tiny flag-on smoke on the real model ------------------------
        function tinyFlagOnSmokeReachesPhi1(tc)
            dat = jointstar.loadData(tc.DataFile);
            P = jointstar.defaultPriors('HierKappa', true, ...
                'PoolAcuteOnly', true);
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
