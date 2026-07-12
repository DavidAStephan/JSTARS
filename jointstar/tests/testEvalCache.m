classdef testEvalCache < matlab.unittest.TestCase
    %TESTEVALCACHE Eval-cache performance pass: cache vs no-cache equality.
    %
    %   jointstar.buildEvalCache captures everything jointstar.
    %   ModelSpec.jointstar / jointstar.computeLogLik would otherwise
    %   recompute on every theta draw despite it not depending on theta's
    %   value (regime groupings, obsMask, P1^{-1}, ...).  These tests
    %   confirm the cache-aware fast path is bit-for-bit identical to the
    %   pre-existing (uncached) code path across both prior families and
    %   both PieObs configurations -- the cache must key off the actual
    %   run configuration (7 vs 8 observables), not assume the production
    %   default.

    properties (Constant)
        DataFile = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
            '..', 'data.csv')
    end

    methods (Test)

        function cacheMatchesNoCache_horseshoeHierKappa(tc)
            rng(41);
            dat = jointstar.loadData(tc.DataFile, 'PieObs', true);
            P = jointstar.horseshoePriors('HierKappa', true);
            cache = jointstar.buildEvalCache(dat, P);

            Theta = jointstar.priorSample(P, 12);
            for i = 1:size(Theta, 1)
                tc.verifyLoglikMatches(P, dat, cache, Theta(i, :));
            end
        end

        function cacheMatchesNoCache_defaultPriors(tc)
            rng(42);
            dat = jointstar.loadData(tc.DataFile, 'PieObs', true);
            P = jointstar.defaultPriors();
            cache = jointstar.buildEvalCache(dat, P);

            Theta = jointstar.priorSample(P, 8);
            for i = 1:size(Theta, 1)
                tc.verifyLoglikMatches(P, dat, cache, Theta(i, :));
            end
        end

        function cacheMatchesNoCache_pieObsFalse(tc)
            % 7-observable configuration: the cache must NOT silently
            % assume the 8-observable production shape.
            rng(43);
            dat = jointstar.loadData(tc.DataFile, 'PieObs', false);
            P = jointstar.horseshoePriors('HierKappa', true);
            cache = jointstar.buildEvalCache(dat, P);

            tc.verifyEqual(cache.p, 7);

            Theta = jointstar.priorSample(P, 8);
            for i = 1:size(Theta, 1)
                tc.verifyLoglikMatches(P, dat, cache, Theta(i, :));
            end
        end

        function cacheRejectsMismatchedConfig(tc)
            % A cache built for one PieObs configuration must not be
            % silently reused against data built with the other.
            dat8 = jointstar.loadData(tc.DataFile, 'PieObs', true);
            dat7 = jointstar.loadData(tc.DataFile, 'PieObs', false);
            P = jointstar.horseshoePriors('HierKappa', true);
            cache8 = jointstar.buildEvalCache(dat8, P);

            tv = [P.params.init];
            th = jointstar.thetaStruct(P, tv);
            [Lq, Lr] = jointstar.hsUnpack(P, tv);
            cf = struct('Lq', Lq, 'Lr', Lr);
            spec7 = jointstar.ModelSpec.jointstar(th, dat7, cf, cache8);
            tc.verifyError(@() jointstar.computeLogLik(spec7.system(), dat7.y, cache8), ...
                'jointstar:cacheMismatch');
        end

        function stateDrawsMatch(tc)
            rng(44);
            dat = jointstar.loadData(tc.DataFile, 'PieObs', true);
            P = jointstar.horseshoePriors('HierKappa', true);
            cache = jointstar.buildEvalCache(dat, P);

            tv = jointstar.priorSample(P, 1);
            th = jointstar.thetaStruct(P, tv);
            [Lq, Lr] = jointstar.hsUnpack(P, tv);
            cf = struct('Lq', Lq, 'Lr', Lr);

            spec0 = jointstar.ModelSpec.jointstar(th, dat, cf);
            [ll0, aux0] = jointstar.computeLogLik(spec0.system(), dat.y);
            spec1 = jointstar.ModelSpec.jointstar(th, dat, cf, cache);
            [ll1, aux1] = jointstar.computeLogLik(spec1.system(), dat.y, cache);
            tc.verifyTrue(isfinite(ll0));
            tc.verifyEqual(ll0, ll1);

            rng(123);
            a0 = jointstar.drawStates(aux0, 3);
            rng(123);
            a1 = jointstar.drawStates(aux1, 3);
            tc.verifyEqual(a0, a1);
        end

    end

    methods

        function verifyLoglikMatches(tc, P, dat, cache, tv)
            th = jointstar.thetaStruct(P, tv);
            if isfield(P, 'hs')
                [Lq, Lr] = jointstar.hsUnpack(P, tv);
                cf = struct('Lq', Lq, 'Lr', Lr);
            else
                cf = [];
            end
            spec0 = jointstar.ModelSpec.jointstar(th, dat, cf);
            ll0 = jointstar.computeLogLik(spec0.system(), dat.y);
            spec1 = jointstar.ModelSpec.jointstar(th, dat, cf, cache);
            ll1 = jointstar.computeLogLik(spec1.system(), dat.y, cache);
            if isnan(ll0) || isnan(ll1)
                tc.verifyEqual(ll0, ll1);
            elseif isfinite(ll0) || isfinite(ll1)
                tc.verifyEqual(ll1, ll0, ...
                    'cache-aware loglik must exactly match the no-cache path');
            end
        end

    end
end
