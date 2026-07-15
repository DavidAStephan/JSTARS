classdef testRGapAR < matlab.unittest.TestCase
    %TESTRGAPAR Design e4d: 'RateGapAR' option unit tests.
    %
    %   Switches the xi state (the non-trend-growth component of r*,
    %   r*_t = 4/(1-alpha)*gz_t + xi_t) from a driftless random walk
    %   (A0(xi,xi)=1, P1(xi,xi)=4 fixed) to a stationary mean-zero AR(1)
    %   (A0(xi,xi)=rho_rg, P1(xi,xi)=sig_xi^2/(1-rho_rg^2)).  DEFAULT
    %   FALSE must reproduce the original RW system exactly; DEFAULT-ON
    %   behaviour must give the designed stationary system with an extra
    %   free parameter rho_rg, co-blocked with {phisum, phi2, nu} (same
    %   output-gap IS equation) for the transformed/structured kernel.

    properties (Constant)
        DataFile = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
            'data.csv')
    end

    methods (Test)

        function flagOffEmitsOriginalRows(tc)
            P = jointstar.defaultPriors();
            tc.verifyFalse(any(strcmp(P.names, 'rho_rg')));

            Pexplicit = jointstar.defaultPriors('RateGapAR', false);
            tc.verifyEqual(Pexplicit.names, P.names);
            tc.verifyEqual([Pexplicit.params.p1], [P.params.p1]);
            tc.verifyEqual([Pexplicit.params.p2], [P.params.p2]);
        end

        function flagOnAddsRhoRgWithDesignedPrior(tc)
            Poff = jointstar.defaultPriors('RateGapAR', false);
            P = jointstar.defaultPriors('RateGapAR', true);
            tc.verifyTrue(any(strcmp(P.names, 'rho_rg')));

            % adds exactly one new free parameter (not a rename)
            tc.verifyEqual(P.d, Poff.d + 1);

            j = find(strcmp(P.names, 'rho_rg'), 1);
            q = P.params(j);
            tc.verifyEqual(q.type, 'beta');
            tc.verifyEqual(q.p1, 9.99, 'AbsTol', 1e-12);
            tc.verifyEqual(q.p2, 1.76, 'AbsTol', 1e-12);
            tc.verifyEqual(q.lo, 0); tc.verifyEqual(q.hi, 1);

            % mean 0.85, sd 0.10 as designed
            a = q.p1; b = q.p2;
            m = a / (a + b);
            s = sqrt(a * b / ((a + b)^2 * (a + b + 1)));
            tc.verifyEqual(m, 0.85, 'AbsTol', 1e-3);
            tc.verifyEqual(s, 0.10, 'AbsTol', 1e-3);

            tc.verifyTrue(P.mutateIdx(j));  % free, not fixed/excluded
        end

        function flagOnCombinesWithHierKappa(tc)
            % RateGapAR is orthogonal to HierKappa (both just append rows)
            P = jointstar.defaultPriors('RateGapAR', true, 'HierKappa', true);
            tc.verifyTrue(any(strcmp(P.names, 'rho_rg')));
            tc.verifyTrue(isfield(P, 'kap'));
        end

        function modelSpecFlagOffIsBitIdenticalToNoRhoRgField(tc)
            % No 'rho_rg' field in th (whether or not the caller ever
            % touched 'RateGapAR') must give A0(xi,xi)=1, P1(xi,xi)=4,
            % exactly as before this change existed.
            dat = jointstar.loadData(tc.DataFile);
            P = jointstar.defaultPriors();
            th = jointstar.thetaStruct(P, [P.params.init]);
            tc.verifyFalse(isfield(th, 'rho_rg'));

            [~, ix] = jointstar.ModelSpec.jointstarStates();
            spec = jointstar.ModelSpec.jointstar(th, dat);
            tc.verifyEqual(spec.A1(ix.xi, ix.xi, 1), 1);
            tc.verifyEqual(spec.P1(ix.xi, ix.xi), 4);
        end

        function modelSpecFlagOnGivesStationaryARWithMatchingP1(tc)
            dat = jointstar.loadData(tc.DataFile);
            P = jointstar.defaultPriors('RateGapAR', true);
            tv = [P.params.init];
            th = jointstar.thetaStruct(P, tv);
            tc.verifyTrue(isfield(th, 'rho_rg'));

            [~, ix] = jointstar.ModelSpec.jointstarStates();
            spec = jointstar.ModelSpec.jointstar(th, dat);

            tc.verifyEqual(spec.A1(ix.xi, ix.xi, 1), th.rho_rg);
            tc.verifyGreaterThan(th.rho_rg, 0);
            tc.verifyLessThan(th.rho_rg, 1);  % stationary (Beta(0,1) support)

            expectedP1xi = th.sig_xi^2 / (1 - th.rho_rg^2);
            tc.verifyEqual(spec.P1(ix.xi, ix.xi), expectedP1xi, 'AbsTol', 1e-12);

            % everything else about P1 is untouched
            tc.verifyEqual(spec.P1(1:13, 1:13), ...
                diag([25, 25, 25, 100, 0.25, 100, 0.25, 100, 0.25, ...
                100, 0.25, 100, 0.25]));

            % xi row/column of A1 has no other cross-loadings (still a
            % genuinely univariate AR(1); off-diagonal xi entries in the
            % time-invariant A0 block are zero both before and after)
            A0xi = squeeze(spec.A1(ix.xi, :, 1));
            A0xi(ix.xi) = 0;
            tc.verifyEqual(A0xi, zeros(1, 14));
            Axi0 = squeeze(spec.A1(:, ix.xi, 1));
            Axi0(ix.xi) = 0;
            tc.verifyEqual(Axi0, zeros(14, 1));

            % loglik still finite at the prior mean (smoke check the new
            % path doesn't break the evaluator)
            ll = jointstar.computeLogLik(spec.system(), dat.y);
            tc.verifyTrue(isfinite(ll));
        end

        function blockAtomsCoBlocksRhoRgWithGapARNu(tc)
            P = jointstar.defaultPriors('RateGapAR', true);
            covCols = find(P.mutateIdx);
            atoms = jointstar.blockAtoms(P, covCols);
            absToRel = zeros(1, P.d); absToRel(covCols) = 1:numel(covCols);

            allRel = sort(horzcat(atoms{:}));
            tc.verifyEqual(allRel, 1:numel(covCols));

            jRg = absToRel(find(strcmp(P.names, 'rho_rg'), 1));
            jPhisum = absToRel(find(strcmp(P.names, 'phisum'), 1));
            jPhi2 = absToRel(find(strcmp(P.names, 'phi2'), 1));
            jNu = absToRel(find(strcmp(P.names, 'nu'), 1));
            foundAtom = 0;
            for a = 1:numel(atoms)
                if all(ismember([jRg, jPhisum, jPhi2, jNu], atoms{a}))
                    foundAtom = a; break;
                end
            end
            tc.verifyGreaterThan(foundAtom, 0, ...
                'rho_rg not co-blocked with {phisum, phi2, nu}');
        end

        function paramTransformGivesLogitKindForRhoRg(tc)
            % Beta(0,1) support => paramTransform's bounds-driven dispatch
            % gives kind 1 (logit) automatically -- no code change needed.
            P = jointstar.defaultPriors('RateGapAR', true);
            T = jointstar.paramTransform(P, P.mutateIdx);
            j = find(strcmp(P.names, 'rho_rg'), 1);
            tc.verifyEqual(T.kind(j), int8(1));
            tc.verifyEqual(T.lo(j), 0);
            tc.verifyEqual(T.hi(j), 1);
        end

        function priorSampleAndLogPdfRunCleanlyWithRateGapAR(tc)
            rng(21);
            P = jointstar.defaultPriors('RateGapAR', true);
            Theta = jointstar.priorSample(P, 20);
            tc.verifyEqual(size(Theta), [20, P.d]);
            j = find(strcmp(P.names, 'rho_rg'), 1);
            tc.verifyTrue(all(Theta(:, j) > 0 & Theta(:, j) < 1));
            for i = 1:20
                lp = jointstar.priorLogPdf(P, Theta(i, :));
                tc.verifyTrue(isfinite(lp));
            end
        end

    end
end
