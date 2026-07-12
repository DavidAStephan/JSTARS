classdef testJointstarSystem < matlab.unittest.TestCase
    %TESTJOINTSTARSYSTEM Checkpoint 2: full JointSTAR likelihood on real data.
    %
    %   Builds the 14-state system from the model specification on the
    %   actual data file, checks the likelihood is finite and plausible at
    %   the prior-init parameter vector and across prior draws, that state
    %   draws behave, and that a single evaluation meets the 50-100 ms
    %   single-core target.  (The posterior-mode cross-check against the
    %   baseline model's published estimates lives in jointstar.validate.)

    properties (Constant)
        DataFile = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
            'data.csv')
    end

    methods (Test)

        function likelihoodFiniteAndPlausibleAtInit(tc)
            dat = jointstar.loadData(tc.DataFile);
            P = jointstar.defaultPriors();
            tv = [P.params.init];
            th = jointstar.thetaStruct(P, tv);
            spec = jointstar.ModelSpec.jointstar(th, dat);
            [ll, aux] = jointstar.computeLogLik(spec.system(), dat.y);

            tc.verifyTrue(isfinite(ll), 'loglik not finite at init');
            nObs = nnz(~isnan(dat.y));
            % plausible range: average log-density per observation within
            % (-10, 0) -- far outside signals a units or masking bug
            tc.verifyGreaterThan(ll / nObs, -10);
            tc.verifyLessThan(ll / nObs, 0);

            % smoothed states must be finite and trends must track data
            % scale: potential output near observed 100*log GDP
            alpha = jointstar.drawStates(aux, 3);
            tc.verifyTrue(all(isfinite(alpha(:))));
            [~, ix] = jointstar.ModelSpec.jointstarStates();
            a = th.alpha;
            tauStar = squeeze(alpha(ix.zstar, :, 1) + a * alpha(ix.kstar, :, 1) ...
                + (1 - a) * (alpha(ix.wstar, :, 1) + alpha(ix.prstar, :, 1) ...
                + alpha(ix.hppstar, :, 1) - alpha(ix.Ustar, :, 1)));
            yObs = dat.y(1, :);
            gap = tauStar(~isnan(yObs)) - yObs(~isnan(yObs));
            tc.verifyLessThan(max(abs(gap)), 25, ...
                'potential output drifts implausibly far from observed GDP');
        end

        function likelihoodMostlyFiniteOverPriorDraws(tc)
            rng(21);
            dat = jointstar.loadData(tc.DataFile);
            P = jointstar.defaultPriors();
            Theta = jointstar.priorSample(P, 30);
            nFinite = 0;
            for i = 1:30
                th = jointstar.thetaStruct(P, Theta(i, :));
                spec = jointstar.ModelSpec.jointstar(th, dat);
                ll = jointstar.computeLogLik(spec.system(), dat.y);
                nFinite = nFinite + isfinite(ll);
                % prior support must imply a valid system: -Inf only via
                % the chol guard, never NaN
                tc.verifyFalse(isnan(ll));
            end
            tc.verifyGreaterThan(nFinite, 25, sprintf( ...
                'only %d/30 prior draws gave finite loglik', nFinite));
        end

        function singleCoreTimingMeetsTarget(tc)
            dat = jointstar.loadData(tc.DataFile);
            P = jointstar.defaultPriors();
            th = jointstar.thetaStruct(P, [P.params.init]);
            maxNumCompThreads(1);
            cleanup = onCleanup(@() maxNumCompThreads('automatic'));
            nRep = 20;
            tic;
            for k = 1:nRep
                spec = jointstar.ModelSpec.jointstar(th, dat);
                jointstar.computeLogLik(spec.system(), dat.y);
            end
            ms = 1000 * toc / nRep;
            fprintf('JointSTAR eval (build + loglik, 1 core): %.1f ms\n', ms);
            tc.verifyLessThan(ms, 100, 'misses the 50-100 ms target');
        end

        function priorLogPdfFiniteAtDraws(tc)
            rng(22);
            P = jointstar.defaultPriors();
            Theta = jointstar.priorSample(P, 200);
            for i = 1:20
                tc.verifyTrue(isfinite(jointstar.priorLogPdf(P, Theta(i, :))));
            end
            % stationarity constraint enforced (|phi2| >= 1 violates it)
            tv = Theta(1, :);
            tv(P.idx.phi2) = -1.5;
            tc.verifyEqual(jointstar.priorLogPdf(P, tv), -Inf);
        end

    end
end
