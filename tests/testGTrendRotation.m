classdef testGTrendRotation < matlab.unittest.TestCase
    %TESTGTRENDROTATION Checkpoint 13: 'GTrendRotation' option unit tests.
    %
    %   Reparameterises the r*-trend growth pair (gzbar, gwbar) as
    %   (gtrend_sum, gtrend_split) = (gzbar+gwbar, gzbar-gwbar), mirroring
    %   the existing (phisum, phi2) gap-AR rotation.  DEFAULT FALSE must
    %   reproduce the original gzbar/gwbar rows and system exactly;
    %   DEFAULT-ON behaviour must produce an IDENTICAL state-space system
    %   for theta related by the (exact) linear rotation.

    properties (Constant)
        DataFile = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
            'data.csv')
    end

    methods (Test)

        function flagOffEmitsOriginalRows(tc)
            P = jointstar.defaultPriors();
            tc.verifyTrue(any(strcmp(P.names, 'gzbar')));
            tc.verifyTrue(any(strcmp(P.names, 'gwbar')));
            tc.verifyFalse(any(strcmp(P.names, 'gtrend_sum')));
            tc.verifyFalse(any(strcmp(P.names, 'gtrend_split')));

            Pexplicit = jointstar.defaultPriors('GTrendRotation', false);
            tc.verifyEqual(Pexplicit.names, P.names);
            tc.verifyEqual([Pexplicit.params.p1], [P.params.p1]);
            tc.verifyEqual([Pexplicit.params.p2], [P.params.p2]);
        end

        function flagOnEmitsRotatedRowsWithExactImpliedPrior(tc)
            P = jointstar.defaultPriors('GTrendRotation', true);
            tc.verifyFalse(any(strcmp(P.names, 'gzbar')));
            tc.verifyFalse(any(strcmp(P.names, 'gwbar')));
            tc.verifyTrue(any(strcmp(P.names, 'gtrend_sum')));
            tc.verifyTrue(any(strcmp(P.names, 'gtrend_split')));

            % same total parameter count as the flag-off spec (rename, not
            % add/drop)
            Poff = jointstar.defaultPriors('GTrendRotation', false);
            tc.verifyEqual(P.d, Poff.d);

            jSum = find(strcmp(P.names, 'gtrend_sum'), 1);
            jSplit = find(strcmp(P.names, 'gtrend_split'), 1);
            qSum = P.params(jSum); qSplit = P.params(jSplit);

            % gzbar ~ N(.30,.15), gwbar ~ N(.40,.15) (equal sds) =>
            % sum = gzbar+gwbar ~ N(.70, sqrt(.15^2+.15^2))
            % split = gzbar-gwbar ~ N(-.10, sqrt(.15^2+.15^2))
            % and, since s1==s2, Cov(sum,split) = s1^2-s2^2 = 0, so the
            % independent-Normal (sum,split) prior used here is EXACT, not
            % an approximation.
            tc.verifyEqual(qSum.type, 'norm');
            tc.verifyEqual(qSplit.type, 'norm');
            tc.verifyEqual(qSum.p1, 0.70, 'AbsTol', 1e-12);
            tc.verifyEqual(qSum.p2, sqrt(2) * 0.15, 'AbsTol', 1e-12);
            tc.verifyEqual(qSplit.p1, -0.10, 'AbsTol', 1e-12);
            tc.verifyEqual(qSplit.p2, sqrt(2) * 0.15, 'AbsTol', 1e-12);
            tc.verifyEqual(qSum.lo, -Inf); tc.verifyEqual(qSum.hi, Inf);
            tc.verifyEqual(qSplit.lo, -Inf); tc.verifyEqual(qSplit.hi, Inf);
        end

        function modelSpecIdenticalUnderExactRotation(tc)
            % Build theta at (gzbar,gwbar) = (0.3,0.4) flag-off, and at
            % (gtrend_sum,gtrend_split) = (0.7,-0.1) flag-on (the same
            % point under the exact rotation); ModelSpec.jointstar must
            % produce elementwise-identical system matrices.
            dat = jointstar.loadData(tc.DataFile);
            P = jointstar.defaultPriors();
            th = jointstar.thetaStruct(P, [P.params.init]);
            th.gzbar = 0.3; th.gwbar = 0.4;

            thRot = rmfield(th, {'gzbar', 'gwbar'});
            thRot.gtrend_sum = 0.7; thRot.gtrend_split = -0.1;

            specOff = jointstar.ModelSpec.jointstar(th, dat);
            specOn = jointstar.ModelSpec.jointstar(thRot, dat);

            fields = {'A1', 'A2', 'c', 'Q', 'a1', 'P1', 'Z', 'Zlag', 'd', ...
                'Rdiag', 'Rfull', 'T'};
            for i = 1:numel(fields)
                f = fields{i};
                a = specOff.(f); b = specOn.(f);
                tc.verifyEqual(size(a), size(b), ...
                    sprintf('field %s: size mismatch', f));
                if ~isempty(a)
                    tc.verifyEqual(a, b, 'AbsTol', 1e-12, ...
                        sprintf('field %s differs under exact rotation', f));
                end
            end

            % same loglik too, as an end-to-end cross-check
            llOff = jointstar.computeLogLik(specOff.system(), dat.y);
            llOn = jointstar.computeLogLik(specOn.system(), dat.y);
            tc.verifyEqual(llOff, llOn, 'AbsTol', 1e-8);
        end

        function blockAtomsCoBlocksRotatedPair(tc)
            P = jointstar.defaultPriors('GTrendRotation', true, 'HierKappa', true);
            covCols = find(P.mutateIdx);
            atoms = jointstar.blockAtoms(P, covCols);
            absToRel = zeros(1, P.d); absToRel(covCols) = 1:numel(covCols);

            allRel = sort(horzcat(atoms{:}));
            tc.verifyEqual(allRel, 1:numel(covCols));

            jSum = absToRel(find(strcmp(P.names, 'gtrend_sum'), 1));
            jSplit = absToRel(find(strcmp(P.names, 'gtrend_split'), 1));
            foundAtom = 0;
            for a = 1:numel(atoms)
                if all(ismember([jSum, jSplit], atoms{a}))
                    foundAtom = a; break;
                end
            end
            tc.verifyGreaterThan(foundAtom, 0, ...
                'gtrend_sum/gtrend_split not co-blocked into one atom');
        end

        function paramTransformIsIdentityForRotatedParams(tc)
            % Both rotated params are unbounded (norm, lo=-Inf,hi=Inf) =>
            % paramTransform's bounds-driven dispatch gives kind 0
            % (identity) automatically, same as gzbar/gwbar before.
            P = jointstar.defaultPriors('GTrendRotation', true);
            T = jointstar.paramTransform(P, P.mutateIdx);
            jSum = find(strcmp(P.names, 'gtrend_sum'), 1);
            jSplit = find(strcmp(P.names, 'gtrend_split'), 1);
            tc.verifyEqual(T.kind(jSum), int8(0));
            tc.verifyEqual(T.kind(jSplit), int8(0));
        end

        function priorSampleAndLogPdfRunCleanlyRotated(tc)
            % priorSample/priorLogPdf are table-driven by prm.type; verify
            % they run without error and give finite densities under the
            % renamed rows (no hardcoded gzbar/gwbar references touched).
            rng(11);
            P = jointstar.defaultPriors('GTrendRotation', true);
            Theta = jointstar.priorSample(P, 20);
            tc.verifyEqual(size(Theta), [20, P.d]);
            for i = 1:20
                lp = jointstar.priorLogPdf(P, Theta(i, :));
                tc.verifyTrue(isfinite(lp));
            end
        end

    end
end
