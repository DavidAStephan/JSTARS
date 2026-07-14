classdef testParamTransform < matlab.unittest.TestCase
    %TESTPARAMTRANSFORM jointstar.paramTransform unit tests for the
    %'MutationTransform' option (DESIGN_transformed_kernel.md, ADDENDUM
    %requirements R2-R5): per-type finite-difference Jacobian check,
    %full-vector round-trip, and boundary-clamp behaviour.

    methods (Test)

        function jacobianMatchesFiniteDifference(tc)
            % One representative mutated parameter per transform kind
            % (0=identity, 1=logit-affine, 2=shifted-log,
            % 3=reflected-log), with bounds read straight from
            % jointstar.defaultPriors -- nothing here hardcodes a bound
            % (R2/R3).
            P = jointstar.defaultPriors('HierKappa', true);
            T = jointstar.paramTransform(P, P.mutateIdx);

            cases = { ...
                'gzbar',   0; ...  % norm,              R           -> identity
                'sig_c',   2; ...  % igsd,              (0,Inf)      -> shifted-log
                'm84_y',   2; ...  % logn,              (0,Inf)      -> shifted-log
                'kapy_20', 2; ...  % hkap,              [1,Inf)      -> shifted-log
                'phisum',  1; ...  % beta,              (0,1)        -> logit-affine
                'gamma2',  1; ...  % negbeta,           (-1,0)       -> logit-affine
                'alphapi', 1; ...  % unif,              [.01,.99]    -> logit-affine
                'alpha',   1; ...  % tnorm both-finite, [.15,.55]    -> logit-affine
                'phiu',    3 ...   % tnorm one-sided,   (-Inf,0]     -> reflected-log
                };

            etaGrid = [-3, -1, -0.2, 0.2, 1, 3];
            h = 1e-6;
            for r = 1:size(cases, 1)
                nm = cases{r, 1}; expKind = cases{r, 2};
                j = find(strcmp(P.names, nm), 1);
                tc.verifyNotEmpty(j, sprintf('param %s not found', nm));
                tc.verifyEqual(double(T.kind(j)), expKind, ...
                    sprintf('%s: expected transform kind %d, got %d', ...
                    nm, expKind, T.kind(j)));

                for e0 = etaGrid
                    eta = zeros(1, P.d); eta(j) = e0;
                    etaP = eta; etaP(j) = e0 + h;
                    etaM = eta; etaM(j) = e0 - h;
                    thP = T.toTheta(etaP); thM = T.toTheta(etaM);
                    numDeriv = (thP(j) - thM(j)) / (2 * h);
                    je = T.logJacElem(eta);
                    analytic = exp(je(j));
                    tc.verifyEqual(analytic, abs(numDeriv), 'RelTol', 1e-6, ...
                        sprintf(['%s @ eta=%.2f: exp(logJac)=%.10g vs ' ...
                        'numeric |dtheta/deta|=%.10g'], nm, e0, analytic, numDeriv));
                end
            end
        end

        function aggregateLogJacIsSumOfElementwise(tc)
            P = jointstar.defaultPriors('HierKappa', true);
            T = jointstar.paramTransform(P, P.mutateIdx);
            rng(1);
            Theta = jointstar.priorSample(P, 20);
            eta = T.toEta(Theta);
            tc.verifyEqual(T.logJac(eta), sum(T.logJacElem(eta), 2), 'AbsTol', 1e-12);
        end

        function roundTripAcrossFullVector(tc)
            rng(123);
            P = jointstar.defaultPriors('HierKappa', true);
            T = jointstar.paramTransform(P, P.mutateIdx);
            Theta = jointstar.priorSample(P, 500);

            eta = T.toEta(Theta);
            back = T.toTheta(eta);
            mi = P.mutateIdx;
            tc.verifyEqual(back(:, mi), Theta(:, mi), 'AbsTol', 1e-10, ...
                'toTheta(toEta(theta)) must recover theta on interior prior draws (R5)');

            % non-mutated ('fixed') column must pass through untouched (R4)
            tc.verifyEqual(back(:, ~mi), Theta(:, ~mi), 'AbsTol', 0);
        end

        function clampHandlesExactBoundaryInputsWithoutCorruptingOtherCoords(tc)
            P = jointstar.defaultPriors('HierKappa', true);
            T = jointstar.paramTransform(P, P.mutateIdx);

            % interior points round-trip cleanly, one per kind
            checks = {'gzbar', 0.3; 'sig_c', 0.8; 'phisum', 0.75; 'phiu', -0.02};
            for r = 1:size(checks, 1)
                nm = checks{r, 1}; v = checks{r, 2};
                j = find(strcmp(P.names, nm), 1);
                th = zeros(1, P.d); th(j) = v;
                back = T.toTheta(T.toEta(th));
                tc.verifyEqual(back(j), v, 'AbsTol', 1e-10, ...
                    sprintf('%s: interior round-trip failed (R5)', nm));
            end

            % exact-boundary inputs must produce a finite, real eta (not
            % Inf/NaN/complex), and must not be written back into theta
            % except via a fresh forward map (never during MH rejection)
            jb = find(strcmp(P.names, 'phisum'), 1);   % beta support (0,1)
            th = zeros(1, P.d); th(jb) = 0;              % exact lower bound
            eta = T.toEta(th);
            tc.verifyTrue(isfinite(eta(jb)) && isreal(eta(jb)), ...
                'clamp must produce a finite, real eta at an exact boundary');

            jhi = find(strcmp(P.names, 'phisum'), 1);
            th2 = zeros(1, P.d); th2(jhi) = 1;            % exact upper bound
            eta2 = T.toEta(th2);
            tc.verifyTrue(isfinite(eta2(jhi)) && isreal(eta2(jhi)));

            js = find(strcmp(P.names, 'sig_c'), 1);       % (0,Inf) support
            th3 = zeros(1, P.d); th3(js) = 0;              % exact lower bound
            eta3 = T.toEta(th3);
            tc.verifyTrue(isfinite(eta3(js)) && isreal(eta3(js)));
        end

        function fullWidthTransformThenRestrictKeepsColumnsAligned(tc)
            % Regression test for the restricted-matrix misalignment bug:
            % the transform's masks index FULL-WIDTH theta columns, so
            % the runSMC covariance step must transform the full N x d
            % particle matrix FIRST and restrict to covCols after.  With
            % the 'fixed' sme_pieobs at column 50 of 79, a restricted
            % call would shift every later column (all the COVID kappas)
            % onto its neighbour's transform.  Verify BY NAME (not
            % position) that after transform-then-restrict every kappa
            % column carries log(theta-1) of the SAME named column, and
            % that passing a restricted matrix now errors loudly.
            rng(9);
            P = jointstar.defaultPriors('HierKappa', true);
            T = jointstar.paramTransform(P, P.mutateIdx);
            Theta = jointstar.priorSample(P, 50);
            covCols = find(P.mutateIdx);

            Em = T.toEta(Theta);          % full-width, then restrict
            Em = Em(:, covCols);

            kapNames = {'kapc_2021', 'kapy_20', 'kapy_21', 'kapu_20', ...
                'kapu_2122', 'kappr_20', 'kappr_2122', 'kaphpp_2022', ...
                'kappop_2021', 'kappi_2023', 'kapk_20', 'kapk_21'};
            for i = 1:numel(kapNames)
                jAbs = find(strcmp(P.names, kapNames{i}), 1);
                tc.verifyNotEmpty(jAbs);
                jRel = find(covCols == jAbs, 1);
                tc.verifyNotEmpty(jRel, ...
                    sprintf('%s missing from covCols', kapNames{i}));
                expected = log(Theta(:, jAbs) - P.params(jAbs).lo);
                tc.verifyEqual(Em(:, jRel), expected, 'AbsTol', 1e-12, ...
                    sprintf(['%s: transformed value in restricted matrix ' ...
                    'does not equal log(theta-lo) of the same named ' ...
                    'column (misalignment)'], kapNames{i}));
            end

            % identity (norm) column past the fixed param, same check
            jAbs = find(strcmp(P.names, 'kapHyp_lm_w2023tot'), 1);
            jRel = find(covCols == jAbs, 1);
            tc.verifyEqual(Em(:, jRel), Theta(:, jAbs), 'AbsTol', 0, ...
                'identity-kind column misaligned after restrict');

            % a restricted-width input must now error, not silently shift
            tc.verifyError(@() T.toEta(Theta(:, covCols)), ...
                'jointstar:transformWidth');
            tc.verifyError(@() T.toTheta(Theta(:, covCols)), ...
                'jointstar:transformWidth');
            tc.verifyError(@() T.logJacElem(Theta(:, covCols)), ...
                'jointstar:transformWidth');
        end

        function fixedParamNeverTransformed(tc)
            % sme_pieobs is 'fixed' (excluded from mutateIdx); R4 requires
            % it is untouched by the transform regardless of value.
            P = jointstar.defaultPriors('HierKappa', true);
            T = jointstar.paramTransform(P, P.mutateIdx);
            j = find(strcmp(P.names, 'sme_pieobs'), 1);
            tc.verifyFalse(P.mutateIdx(j));
            tc.verifyEqual(double(T.kind(j)), 0);

            th = zeros(1, P.d); th(j) = 0.30;
            eta = T.toEta(th);
            tc.verifyEqual(eta(j), 0.30, ...
                'a non-mutated column must pass through toEta unchanged');
            back = T.toTheta(eta);
            tc.verifyEqual(back(j), 0.30);
        end

    end
end
