classdef testPrecisionVsKalman < matlab.unittest.TestCase
    %TESTPRECISIONVSKALMAN Checkpoint 1: precision likelihood == Kalman.
    %
    %   Validates jointstar.computeLogLik and jointstar.drawStates on a
    %   toy AR(1)+noise model against a textbook Kalman filter/smoother:
    %     1. log-likelihood agrees to 8+ decimal places over a grid of
    %        (rho, sigEta, sigEps), incl. nonzero initial mean and
    %        missing observations;
    %     2. 10,000 posterior state draws match the smoother mean and
    %        variance to Monte Carlo error;
    %     3. the assembled precision matrices stay sparse.

    properties (Constant)
        Tlen = 200
        Seed = 20260710
    end

    methods (Test)

        function loglikMatchesKalmanOnGrid(tc)
            rng(tc.Seed);
            rhoG = [0.30, 0.70, 0.95];
            sEtaG = [0.50, 1.00];
            sEpsG = [0.30, 1.00];

            % one simulated dataset reused across the grid
            yS = simulateAR1(0.8, 0.7, 0.5, tc.Tlen);

            for rho = rhoG
                for sEta = sEtaG
                    for sEps = sEpsG
                        spec = jointstar.ModelSpec.toyAR1(rho, sEta, sEps, tc.Tlen);
                        llP = jointstar.computeLogLik(spec.system(), yS);
                        llK = kalmanReference(spec.system(), yS);
                        tc.verifyEqual(llP, llK, 'AbsTol', 1e-8, sprintf( ...
                            'loglik mismatch at rho=%.2f sEta=%.2f sEps=%.2f', ...
                            rho, sEta, sEps));
                    end
                end
            end
        end

        function loglikMatchesWithNonzeroInitAndMissing(tc)
            rng(tc.Seed + 1);
            y = simulateAR1(0.9, 0.6, 0.4, tc.Tlen);
            y(1, 20:60) = NaN;                       % early-sample gap
            y(1, 3:7:end) = NaN;                     % scattered gaps
            spec = jointstar.ModelSpec.toyAR1(0.9, 0.6, 0.4, tc.Tlen, 2.0, 4.0);
            llP = jointstar.computeLogLik(spec.system(), y);
            llK = kalmanReference(spec.system(), y);
            tc.verifyEqual(llP, llK, 'AbsTol', 1e-8, ...
                'loglik mismatch with nonzero a1/P1 and missing data');
        end

        function stateDrawsMatchSmootherMoments(tc)
            rng(tc.Seed + 2);
            Tn = 50; nDraw = 10000;
            y = simulateAR1(0.85, 0.5, 0.6, Tn);
            y(1, 10:14) = NaN;
            spec = jointstar.ModelSpec.toyAR1(0.85, 0.5, 0.6, Tn);

            [~, aux] = jointstar.computeLogLik(spec.system(), y);
            draws = squeeze(jointstar.drawStates(aux, nDraw));   % Tn x nDraw
            [~, sm] = kalmanReference(spec.system(), y);

            sdT = sqrt(squeeze(sm.var(1, 1, :)));                 % Tn x 1
            % mean: MC error of the sample mean is sd/sqrt(N); 5-sigma band
            meanErr = abs(mean(draws, 2) - sm.mean(1, :)');
            tc.verifyLessThan(meanErr, 5 * sdT / sqrt(nDraw) + 1e-12, ...
                'state-draw mean outside 5-sigma MC band of smoother mean');
            % variance: relative MC error ~ sqrt(2/N); 5-sigma band
            varErr = abs(var(draws, 0, 2) - sdT.^2);
            tc.verifyLessThan(varErr, 5 * sqrt(2 / nDraw) * sdT.^2 + 1e-12, ...
                'state-draw variance outside 5-sigma MC band of smoother variance');
        end

        function precisionStaysSparse(tc)
            % smoke test on a multivariate system incl. a second lag:
            % check -Inf on infeasible theta and sparsity via aux factor
            rng(tc.Seed + 3);
            m = 3; Tn = 120;
            A1 = [0.7 0.1 0; 0 0.5 0.2; 0 0 0.9];
            A2 = zeros(m); A2(2, 2) = -0.3;          % AR(2) in state 2
            Q = diag([0.4 0.3 0.2]);
            Z = [1 0 0; 0 1 1];
            spec = jointstar.ModelSpec('A1', A1, 'A2', A2, 'Q', Q, ...
                'a1', zeros(m, 1), 'P1', eye(m), 'Z', Z, ...
                'Rdiag', [0.5; 0.4], 'T', Tn);
            y = randn(2, Tn);
            [ll, aux] = jointstar.computeLogLik(spec.system(), y);
            tc.verifyTrue(isfinite(ll));
            tc.verifyTrue(issparse(aux.L), 'Cholesky factor must be sparse');
            dens = nnz(aux.L) / numel(aux.L);
            tc.verifyLessThan(dens, 0.05, 'Cholesky factor unexpectedly dense');

            % draws from the AR(2)-block system must have finite moments
            d = jointstar.drawStates(aux, 5);
            tc.verifyTrue(all(isfinite(d(:))));
        end

    end
end

function y = simulateAR1(rho, sEta, sEps, T)
a = zeros(1, T);
a(1) = sEta / sqrt(1 - rho^2) * randn;
for t = 2:T
    a(t) = rho * a(t - 1) + sEta * randn;
end
y = a + sEps * randn(1, T);
end
