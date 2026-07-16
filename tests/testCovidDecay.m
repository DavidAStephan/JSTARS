classdef testCovidDecay < matlab.unittest.TestCase
    %TESTCOVIDDECAY 'CovidDecay' option unit tests.
    %
    %   Only meaningful when HierKappa=true, MUTUALLY EXCLUSIVE with
    %   PoolAcuteOnly/FixSingletonKappa (errors if combined -- all three
    %   are alternative kappa reparameterizations of the same window
    %   groups). Replaces the 8 COVID kappas of the acute two-step groups
    %   g1(w2020), g2(w2021), g3(w2122) -- kapy_20/kapy_21 (row1 GDP),
    %   kapu_20/kapu_2122 (row4 U), kappr_20/kappr_2122 (row5
    %   participation), kapk_20/kapk_21 (row7 capital) -- with a
    %   parametric geometric-decay SD multiplier (Lenza-Primiceri style):
    %   kr(row,t) = 1 + s0_row*rho_c^(t-t0), t0 = 2020Q2, over each row's
    %   original union window; kr = 1 outside. NEW PARAMS (5): s0_y,
    %   s0_u, s0_pr, s0_k ('logn') and a single shared decay rate rho_c
    %   ('beta'). This removes 8 kappas + 6 hyper dims (theta 79 -> 70
    %   under HierKappa). DEFAULT FALSE (or HierKappa=false) must
    %   reproduce today's defaultPriors add() sequence byte-for-byte.

    properties (Constant)
        DataFile = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
            'data.csv')
    end

    methods (Test)

        % (a) default-off inertness -----------------------------------
        function flagOffInertWithExplicitFalse(tc)
            P0 = jointstar.defaultPriors('HierKappa', true);
            P1 = jointstar.defaultPriors('HierKappa', true, ...
                'CovidDecay', false);
            tc.verifyTrue(isequaln(P0, P1), ...
                'CovidDecay=false must be byte-identical to omitted');
        end

        function flagInertWhenHierKappaFalse(tc)
            % under HierKappa=false all 12 kappas are already plain
            % tgamma(2,2) with no hypers and defaultPriors returns before
            % the hierarchical branch -- the flag must be a no-op there
            P0 = jointstar.defaultPriors('HierKappa', false);
            P1 = jointstar.defaultPriors('HierKappa', false, ...
                'CovidDecay', true);
            tc.verifyTrue(isequaln(P0, P1), ...
                'CovidDecay must be inert when HierKappa=false');
        end

        function flagOffInertEvenWithOtherKappaFlagsTrue(tc)
            % omitted CovidDecay (default false) must not perturb the
            % existing FixSingletonKappa / PoolAcuteOnly paths
            P0 = jointstar.defaultPriors('HierKappa', true, ...
                'FixSingletonKappa', true);
            P1 = jointstar.defaultPriors('HierKappa', true, ...
                'FixSingletonKappa', true, 'CovidDecay', false);
            tc.verifyTrue(isequaln(P0, P1), ...
                'CovidDecay=false must not alter the FixSingletonKappa path');

            Q0 = jointstar.defaultPriors('HierKappa', true, ...
                'PoolAcuteOnly', true);
            Q1 = jointstar.defaultPriors('HierKappa', true, ...
                'PoolAcuteOnly', true, 'CovidDecay', false);
            tc.verifyTrue(isequaln(Q0, Q1), ...
                'CovidDecay=false must not alter the PoolAcuteOnly path');
        end

        % (e) mutual exclusivity -----------------------------------------
        function errorsWhenCombinedWithPoolAcuteOnly(tc)
            tc.verifyError(@() jointstar.defaultPriors('HierKappa', true, ...
                'CovidDecay', true, 'PoolAcuteOnly', true), ...
                'jointstar:covidDecayConflict');
        end

        function errorsWhenCombinedWithFixSingletonKappa(tc)
            tc.verifyError(@() jointstar.defaultPriors('HierKappa', true, ...
                'CovidDecay', true, 'FixSingletonKappa', true), ...
                'jointstar:covidDecayConflict');
        end

        % (b) flag-on structure -----------------------------------------
        function flagOnParamCountIs70(tc)
            P0 = jointstar.defaultPriors('HierKappa', true);
            P2 = jointstar.defaultPriors('HierKappa', true, ...
                'CovidDecay', true);
            tc.verifyEqual(P0.d, 79);
            tc.verifyEqual(P2.d, 70);  % 79 - 8 kappas - 6 hypers + 5 decay
        end

        function flagOnAddsFiveDecayParams(tc)
            P2 = jointstar.defaultPriors('HierKappa', true, ...
                'CovidDecay', true);
            newNames = {'s0_y', 's0_u', 's0_pr', 's0_k', 'rho_c'};
            for i = 1:numel(newNames)
                j = find(strcmp(P2.names, newNames{i}), 1);
                tc.verifyNotEmpty(j, sprintf('%s must be present', newNames{i}));
                tc.verifyTrue(P2.mutateIdx(j), ...
                    sprintf('%s must be a free parameter', newNames{i}));
            end
            jsy = find(strcmp(P2.names, 's0_y'), 1);
            tc.verifyEqual(P2.params(jsy).type, 'logn');
            tc.verifyEqual(P2.params(jsy).lo, 0);
            tc.verifyEqual(P2.params(jsy).hi, Inf);
            jrc = find(strcmp(P2.names, 'rho_c'), 1);
            tc.verifyEqual(P2.params(jrc).type, 'beta');
            tc.verifyEqual(P2.params(jrc).lo, 0);
            tc.verifyEqual(P2.params(jrc).hi, 1);
        end

        function flagOnRemovesEightKappasAndSixHypers(tc)
            P2 = jointstar.defaultPriors('HierKappa', true, ...
                'CovidDecay', true);

            droppedKappas = {'kapy_20', 'kapy_21', 'kapu_20', 'kapu_2122', ...
                'kappr_20', 'kappr_2122', 'kapk_20', 'kapk_21'};
            for i = 1:numel(droppedKappas)
                tc.verifyFalse(ismember(droppedKappas{i}, P2.names), ...
                    sprintf('%s must be removed from theta', droppedKappas{i}));
            end

            droppedHypers = {'kapHyp_lm_w2020', 'kapHyp_la_w2020', ...
                'kapHyp_lm_w2021', 'kapHyp_la_w2021', ...
                'kapHyp_lm_w2122', 'kapHyp_la_w2122'};
            for i = 1:numel(droppedHypers)
                tc.verifyFalse(ismember(droppedHypers{i}, P2.names), ...
                    sprintf('%s must be removed from theta', droppedHypers{i}));
            end
        end

        function flagOnLeavesKeptGroupsUntouched(tc)
            P0 = jointstar.defaultPriors('HierKappa', true);
            P2 = jointstar.defaultPriors('HierKappa', true, ...
                'CovidDecay', true);

            keptKappas = {'kapc_2021', 'kappop_2021', 'kaphpp_2022', ...
                'kappi_2023'};
            for i = 1:numel(keptKappas)
                j2 = find(strcmp(P2.names, keptKappas{i}), 1);
                tc.verifyNotEmpty(j2);
                tc.verifyEqual(P2.params(j2).type, 'hkap');
                tc.verifyTrue(P2.mutateIdx(j2));
            end

            keptHypers = {'kapHyp_lm_w2021tot', 'kapHyp_la_w2021tot', ...
                'kapHyp_lm_w2022tot', 'kapHyp_la_w2022tot', ...
                'kapHyp_lm_w2023tot', 'kapHyp_la_w2023tot'};
            for i = 1:numel(keptHypers)
                tc.verifyTrue(ismember(keptHypers{i}, P2.names), ...
                    sprintf('%s must remain a free param', keptHypers{i}));
                j0 = find(strcmp(P0.names, keptHypers{i}), 1);
                j2 = find(strcmp(P2.names, keptHypers{i}), 1);
                tc.verifyEqual(P2.params(j2).p1, P0.params(j0).p1);
                tc.verifyEqual(P2.params(j2).p2, P0.params(j0).p2);
            end
        end

        function flagOnKapStructIsSelfConsistent(tc)
            P2 = jointstar.defaultPriors('HierKappa', true, ...
                'CovidDecay', true);
            tc.verifyEqual(P2.kap.G, 3);
            tc.verifyEqual(numel(P2.kap.kapCols), 4);
            tc.verifyEqual(numel(P2.kap.groups), 4);
            tc.verifyEqual(numel(P2.kap.lmCols), 3);
            tc.verifyEqual(numel(P2.kap.laCols), 3);
            tc.verifyEqual(unique(P2.kap.groups(:)'), 1:3, ...
                'kept groups must be remapped onto the compact range 1..3');

            hkapCols = find(strcmp({P2.params.type}, 'hkap'));
            tc.verifyEqual(sort(P2.kap.kapCols(:)'), sort(hkapCols));
        end

        function flagOnComposesWithGTrendRotationAndRateGapAR(tc)
            % orthogonal flags -- all just append/modify independent rows
            P = jointstar.defaultPriors('HierKappa', true, ...
                'CovidDecay', true, 'GTrendRotation', true, ...
                'RateGapAR', true);
            tc.verifyTrue(any(strcmp(P.names, 'gtrend_sum')));
            tc.verifyTrue(any(strcmp(P.names, 'rho_rg')));
            tc.verifyTrue(any(strcmp(P.names, 's0_y')));
            tc.verifyEqual(P.kap.G, 3);
        end

        % (d) priorLogPdf / paramTransform / blockAtoms -------------------
        function priorLogPdfFiniteAtSampledDrawAndInfOutsideSupport(tc)
            rng(21);
            P2 = jointstar.defaultPriors('HierKappa', true, ...
                'CovidDecay', true);
            Theta = jointstar.priorSample(P2, 10);
            tc.verifyEqual(size(Theta), [10, P2.d]);

            for i = 1:10
                lp = jointstar.priorLogPdf(P2, Theta(i, :));
                tc.verifyTrue(isfinite(lp));
            end

            jrc = find(strcmp(P2.names, 'rho_c'), 1);
            jsy = find(strcmp(P2.names, 's0_y'), 1);

            tv = Theta(1, :);
            tv(jrc) = 1.5;   % outside (0,1) support
            lp2 = jointstar.priorLogPdf(P2, tv);
            tc.verifyEqual(lp2, -Inf, ...
                'rho_c support (0,1) must still be enforced');

            tv2 = Theta(1, :);
            tv2(jsy) = -0.5;   % outside (0,Inf) support
            lp3 = jointstar.priorLogPdf(P2, tv2);
            tc.verifyEqual(lp3, -Inf, ...
                's0_y support (0,Inf) must still be enforced');

            % kept-group kappa truncation (>=1) still enforced
            jp = find(strcmp(P2.names, 'kappi_2023'), 1);
            tv3 = Theta(1, :);
            tv3(jp) = 0.9;
            lp4 = jointstar.priorLogPdf(P2, tv3);
            tc.verifyEqual(lp4, -Inf, ...
                'kappa>=1 truncation on the kept groups must still be enforced');
        end

        function blockAtomsAndParamTransformAcceptReducedKap(tc)
            P2 = jointstar.defaultPriors('HierKappa', true, ...
                'CovidDecay', true);
            covCols = find(P2.mutateIdx);
            atoms = jointstar.blockAtoms(P2, covCols);
            allRel = sort(horzcat(atoms{:}));
            tc.verifyEqual(allRel, 1:numel(covCols));

            % s0_y/s0_u/s0_pr/s0_k/rho_c must be co-blocked in one atom
            jrc = find(strcmp(P2.names, 'rho_c'), 1);
            relRc = find(covCols == jrc);
            found = false;
            for a = 1:numel(atoms)
                if ismember(relRc, atoms{a})
                    tc.verifyEqual(numel(atoms{a}), 5, ...
                        'the CovidDecay atom must contain all 5 decay params');
                    found = true;
                end
            end
            tc.verifyTrue(found);

            T = jointstar.paramTransform(P2, P2.mutateIdx);
            jsy = find(strcmp(P2.names, 's0_y'), 1);
            tc.verifyEqual(T.kind(jsy), int8(2));  % lo=0,hi=Inf -> log(theta)
            tc.verifyEqual(T.kind(jrc), int8(1));  % lo=0,hi=1 -> logit(theta)
        end

        % (f) ModelSpec.jointstar decay-path arithmetic -------------------
        function modelSpecKrMatchesDecayFormula(tc)
            dat = jointstar.loadData(tc.DataFile);
            P = jointstar.defaultPriors('HierKappa', true, ...
                'CovidDecay', true);
            tv = [P.params.init];
            tv(P.idx.s0_y) = 1.2; tv(P.idx.s0_u) = 0.8;
            tv(P.idx.s0_pr) = 0.5; tv(P.idx.s0_k) = 1.0;
            tv(P.idx.rho_c) = 0.6;
            th = jointstar.thetaStruct(P, tv);
            spec = jointstar.ModelSpec.jointstar(th, dat);
            sys = spec.system();
            Rd = sys.Rdiag;

            w = dat.win;
            idx21 = find(w.w2020_21);
            idx22 = find(w.w2020_22);

            % row 1 (y): kr(1,t)=1+s0_y*rho_c^(t-t0); Rd(1,t)=sme_y^2*kr^2
            e21 = 0:(numel(idx21) - 1);
            expectKr1 = 1 + th.s0_y * th.rho_c .^ e21;
            actualKr1 = sqrt(Rd(1, idx21)) / th.sme_y;
            tc.verifyEqual(actualKr1, expectKr1, 'AbsTol', 1e-10);

            % row 7 (k)
            expectKr7 = 1 + th.s0_k * th.rho_c .^ e21;
            actualKr7 = sqrt(Rd(7, idx21)) / th.sme_k;
            tc.verifyEqual(actualKr7, expectKr7, 'AbsTol', 1e-10);

            % row 4 (U)
            e22 = 0:(numel(idx22) - 1);
            expectKr4 = 1 + th.s0_u * th.rho_c .^ e22;
            actualKr4 = sqrt(Rd(4, idx22)) / th.sme_U;
            tc.verifyEqual(actualKr4, expectKr4, 'AbsTol', 1e-10);

            % row 5 (pr)
            expectKr5 = 1 + th.s0_pr * th.rho_c .^ e22;
            actualKr5 = sqrt(Rd(5, idx22)) / th.sme_pr;
            tc.verifyEqual(actualKr5, expectKr5, 'AbsTol', 1e-10);

            % first period of the window: kappa = 1 + s0 (peak)
            tc.verifyEqual(actualKr1(1), 1 + th.s0_y, 'AbsTol', 1e-10);

            % kr == 1 exactly outside the windows (row 1 as representative,
            % excluding the unrelated pre-1984 break-multiplier window
            % which also fills row 1 via m84_y)
            outside = true(1, dat.T);
            outside(idx21) = false;
            outside(dat.pre84) = false;
            tc.verifyEqual(sqrt(Rd(1, outside)) / th.sme_y, ones(1, nnz(outside)), ...
                'AbsTol', 1e-10);
        end

        % (c) tiny flag-on smoke on the real model ------------------------
        function tinyFlagOnSmokeReachesPhi1(tc)
            dat = jointstar.loadData(tc.DataFile);
            P = jointstar.defaultPriors('HierKappa', true, ...
                'CovidDecay', true);
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
