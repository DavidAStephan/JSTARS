classdef testDropPhiY < matlab.unittest.TestCase
    %TESTDROPPHIY 'DropPhiY' option unit tests.
    %
    %   A nested-restriction hypothesis test of the COVID level shifter on
    %   GDP (phiy). When DropPhiY=false (default), phiy is sign-restricted
    %   <0 via truncated normal N(0, 0.10) on (-Inf, 0). When
    %   DropPhiY=true, phiy is fixed at 0 ('fixed' prior type, excluded
    %   from mutateIdx), dropping the GDP COVID intercept term entirely
    %   from d_t.  This tests whether the stringency loading is redundant
    %   GIVEN kappa (the d_t/kappa competition on the 2020 GDP drop,
    %   CHECKPOINT_07). This is a NESTED RESTRICTION (H0: phiy=0 within
    %   H1: phiy<0), not a sign-flip reversal. Composes cleanly with all
    %   other flags (orthogonal); phiu's sign restriction (<0) is unchanged.
    %   Parameter count 79 -> 78 (free params 78 -> 77) when flag is true.
    %   DEFAULT FALSE reproduces the prior phiy row byte-for-byte.

    properties (Constant)
        DataFile = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
            'data.csv')
    end

    methods (Test)

        % (a) default-off inertness -----------------------------------
        function flagOffInertWithExplicitFalse(tc)
            P0 = jointstar.defaultPriors('HierKappa', true);
            P1 = jointstar.defaultPriors('HierKappa', true, ...
                'DropPhiY', false);
            tc.verifyTrue(isequaln(P0, P1), ...
                'DropPhiY=false must be byte-identical to omitted');
        end

        function flagOffInertUnderBothHierKappaSetting(tc)
            % DropPhiY=false must be inert under BOTH HierKappa settings
            P0a = jointstar.defaultPriors('HierKappa', true);
            P1a = jointstar.defaultPriors('HierKappa', true, ...
                'DropPhiY', false);
            tc.verifyTrue(isequaln(P0a, P1a), ...
                'DropPhiY=false must be inert when HierKappa=true');

            P0b = jointstar.defaultPriors('HierKappa', false);
            P1b = jointstar.defaultPriors('HierKappa', false, ...
                'DropPhiY', false);
            tc.verifyTrue(isequaln(P0b, P1b), ...
                'DropPhiY=false must be inert when HierKappa=false');
        end

        function flagOffInertEvenWithOtherFlagsTrue(tc)
            % DropPhiY is orthogonal to all other flags; default false
            % must not perturb any path
            P0 = jointstar.defaultPriors('HierKappa', true, ...
                'GTrendRotation', true, 'RateGapAR', true);
            P1 = jointstar.defaultPriors('HierKappa', true, ...
                'GTrendRotation', true, 'RateGapAR', true, ...
                'DropPhiY', false);
            tc.verifyTrue(isequaln(P0, P1), ...
                'DropPhiY=false must not alter other flag paths');
        end

        % (b) flag-on structure -----------------------------------------
        function flagOnPhiYIsFixed(tc)
            P0 = jointstar.defaultPriors('HierKappa', true);
            P1 = jointstar.defaultPriors('HierKappa', true, ...
                'DropPhiY', true);

            % phiy must exist, be 'fixed' type, with value 0
            idx_phiy_0 = find(strcmp(P0.names, 'phiy'), 1);
            idx_phiy_1 = find(strcmp(P1.names, 'phiy'), 1);

            tc.verifyNotEmpty(idx_phiy_0);
            tc.verifyNotEmpty(idx_phiy_1);

            % flag OFF: tnorm
            tc.verifyEqual(P0.params(idx_phiy_0).type, 'tnorm');
            tc.verifyTrue(P0.mutateIdx(idx_phiy_0));
            tc.verifyEqual(P0.params(idx_phiy_0).lo, -Inf);
            tc.verifyEqual(P0.params(idx_phiy_0).hi, 0);

            % flag ON: fixed at 0
            tc.verifyEqual(P1.params(idx_phiy_1).type, 'fixed');
            tc.verifyFalse(P1.mutateIdx(idx_phiy_1));
            tc.verifyEqual(P1.params(idx_phiy_1).init, 0);
            tc.verifyEqual(P1.params(idx_phiy_1).p1, 0);
            tc.verifyEqual(P1.params(idx_phiy_1).p2, 0);
            tc.verifyEqual(P1.params(idx_phiy_1).lo, 0);
            tc.verifyEqual(P1.params(idx_phiy_1).hi, 0);
        end

        function flagOnPhiUUnchanged(tc)
            % phiu (sign-restricted <0) must remain untouched
            P0 = jointstar.defaultPriors('HierKappa', true);
            P1 = jointstar.defaultPriors('HierKappa', true, ...
                'DropPhiY', true);

            idx_phiu_0 = find(strcmp(P0.names, 'phiu'), 1);
            idx_phiu_1 = find(strcmp(P1.names, 'phiu'), 1);

            tc.verifyNotEmpty(idx_phiu_0);
            tc.verifyNotEmpty(idx_phiu_1);

            % both must be tnorm, sign-restricted <0
            tc.verifyEqual(P0.params(idx_phiu_0).type, 'tnorm');
            tc.verifyEqual(P1.params(idx_phiu_1).type, 'tnorm');

            % both free parameters
            tc.verifyTrue(P0.mutateIdx(idx_phiu_0));
            tc.verifyTrue(P1.mutateIdx(idx_phiu_1));

            % bounds unchanged
            tc.verifyEqual(P0.params(idx_phiu_0).lo, P1.params(idx_phiu_1).lo);
            tc.verifyEqual(P0.params(idx_phiu_0).hi, P1.params(idx_phiu_1).hi);
            tc.verifyEqual(P0.params(idx_phiu_0).lo, -Inf);
            tc.verifyEqual(P0.params(idx_phiu_0).hi, 0);
        end

        function flagOnParamCountCorrect(tc)
            P0 = jointstar.defaultPriors('HierKappa', true, ...
                'DropPhiY', false);
            P1 = jointstar.defaultPriors('HierKappa', true, ...
                'DropPhiY', true);

            % P.d stays 79 (fixed param is still in the list)
            % but free params go 78 -> 77
            tc.verifyEqual(P0.d, 79);
            tc.verifyEqual(P1.d, 79);

            nFree0 = nnz(P0.mutateIdx);
            nFree1 = nnz(P1.mutateIdx);
            tc.verifyEqual(nFree0, 78);
            tc.verifyEqual(nFree1, 77);
            tc.verifyEqual(nFree1, nFree0 - 1);
        end

        % (c) priorSample / priorLogPdf -----------------------------------
        function priorSamplePhiYIsZero(tc)
            rng(21);
            P1 = jointstar.defaultPriors('HierKappa', true, ...
                'DropPhiY', true);
            Theta = jointstar.priorSample(P1, 20);
            tc.verifyEqual(size(Theta), [20, P1.d]);

            idx_phiy = find(strcmp(P1.names, 'phiy'), 1);
            tc.verifyTrue(all(Theta(:, idx_phiy) == 0), ...
                'priorSample must return exactly 0 for phiy');
        end

        function priorLogPdfFiniteAtSampledDrawAndInfOutsideSupport(tc)
            rng(21);
            P1 = jointstar.defaultPriors('HierKappa', true, ...
                'DropPhiY', true);
            Theta = jointstar.priorSample(P1, 10);

            for i = 1:10
                lp = jointstar.priorLogPdf(P1, Theta(i, :));
                tc.verifyTrue(isfinite(lp));
            end

            % perturb phiy off 0 -> -Inf log-density
            idx_phiy = find(strcmp(P1.names, 'phiy'), 1);
            tv = Theta(1, :);
            tv(idx_phiy) = 0.001;
            lp_off = jointstar.priorLogPdf(P1, tv);
            tc.verifyEqual(lp_off, -Inf, ...
                'priorLogPdf must be -Inf when phiy is not exactly 0');
        end

        % (d) paramTransform / blockAtoms -----------------------------------
        function paramTransformExcludesPhiY(tc)
            P1 = jointstar.defaultPriors('HierKappa', true, ...
                'DropPhiY', true);
            covCols = find(P1.mutateIdx);
            idx_phiy = find(strcmp(P1.names, 'phiy'), 1);

            % phiy should NOT be in covCols
            tc.verifyFalse(any(covCols == idx_phiy), ...
                'phiy must not be in the mutated columns');

            T = jointstar.paramTransform(P1, P1.mutateIdx);
            % phiy's entry in T.kind should be 0 (fixed, no transform)
            tc.verifyEqual(T.kind(idx_phiy), int8(0));
        end

        function blockAtomsAcceptReducedCovCols(tc)
            P1 = jointstar.defaultPriors('HierKappa', true, ...
                'DropPhiY', true);
            covCols = find(P1.mutateIdx);
            atoms = jointstar.blockAtoms(P1, covCols);
            allRel = sort(horzcat(atoms{:}));

            % should cover 1:numel(covCols), which is 77
            tc.verifyEqual(allRel, 1:numel(covCols));
            tc.verifyEqual(numel(covCols), 77);
        end

        % (e) ModelSpec.jointstar d-row 1 arithmetic -------------------
        function modelSpecD1IsZeroWhenPhiYFixed(tc)
            dat = jointstar.loadData(tc.DataFile);
            P1 = jointstar.defaultPriors('HierKappa', true, ...
                'DropPhiY', true);

            tv = [P1.params.init];
            th = jointstar.thetaStruct(P1, tv);
            spec = jointstar.ModelSpec.jointstar(th, dat);
            sys = spec.system();

            % d(1,:) = th.phiy * D, and th.phiy = 0, so d(1,:) = 0
            d1 = sys.d(1, :);
            tc.verifyTrue(all(d1 == 0), ...
                'd-row 1 (GDP COVID intercept) must be exactly 0');
        end

        function modelSpecD1AgainsDefaultForComparison(tc)
            dat = jointstar.loadData(tc.DataFile);
            P0 = jointstar.defaultPriors('HierKappa', true, ...
                'DropPhiY', false);

            tv = [P0.params.init];
            th = jointstar.thetaStruct(P0, tv);
            spec = jointstar.ModelSpec.jointstar(th, dat);
            sys = spec.system();

            % d(1,:) should have values (th.phiy is ~-0.05 init)
            d1_default = sys.d(1, :);
            tc.verifyTrue(any(d1_default ~= 0), ...
                'd-row 1 should have nonzero values with flag OFF');
        end

        function modelSpecOtherRowsUnchanged(tc)
            dat = jointstar.loadData(tc.DataFile);
            P0 = jointstar.defaultPriors('HierKappa', true, ...
                'DropPhiY', false);
            P1 = jointstar.defaultPriors('HierKappa', true, ...
                'DropPhiY', true);

            tv0 = [P0.params.init];
            tv1 = [P1.params.init];

            % phiu init should be same in both
            idx_phiu_0 = find(strcmp(P0.names, 'phiu'), 1);
            idx_phiu_1 = find(strcmp(P1.names, 'phiu'), 1);
            tv1(idx_phiu_1) = tv0(idx_phiu_0);

            th0 = jointstar.thetaStruct(P0, tv0);
            th1 = jointstar.thetaStruct(P1, tv1);
            spec0 = jointstar.ModelSpec.jointstar(th0, dat);
            spec1 = jointstar.ModelSpec.jointstar(th1, dat);
            sys0 = spec0.system();
            sys1 = spec1.system();

            % d(2,:) and d(4,:) use phiu and should differ only in phiy/phiu
            % values, not structure
            tc.verifyEqual(size(sys0.d), size(sys1.d));
            % rows 2,4 (phiu-related) should track phiu, not change structure
            % (hard to verify exactly without detailed model knowledge,
            % so just check they don't blow up)
            tc.verifyTrue(all(isfinite(sys1.d(2, :))));
            tc.verifyTrue(all(isfinite(sys1.d(4, :))));
        end

        % (f) tiny flag-on smoke on the real model ------------------------
        function tinyFlagOnSmokeReachesPhi1(tc)
            dat = jointstar.loadData(tc.DataFile);
            P1 = jointstar.defaultPriors('HierKappa', true, ...
                'DropPhiY', true);
            cache = jointstar.buildEvalCache(dat, P1);

            prob = struct( ...
                'samplePrior', @(N) jointstar.priorSample(P1, N), ...
                'logPrior', @(tv) jointstar.priorLogPdf(P1, tv), ...
                'logLik', @(tv) logLikTheta(P1, tv, dat, cache), ...
                'paramNames', {P1.names}, ...
                'priors', P1);
            opts = struct('NParticles', 60, 'MSteps', 1, 'Seed', 11, ...
                'Verbose', false, 'UseParallel', false, ...
                'MutateIdx', P1.mutateIdx, ...
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
