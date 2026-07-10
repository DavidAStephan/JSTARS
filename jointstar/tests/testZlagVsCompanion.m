classdef testZlagVsCompanion < matlab.unittest.TestCase
    %TESTZLAGVSCOMPANION Zlag (lagged-state measurement loadings) is exact.
    %
    %   Model: alpha_t = rho*alpha_{t-1} + eta,  y_t = alpha_t +
    %   b*alpha_{t-1} + eps.  Precision version: 1 state + Zlag.  Oracle:
    %   companion form with states (alpha_t, alpha_{t-1}), singular Q --
    %   fine for the Kalman filter, impossible for the precision sampler,
    %   which is exactly why Zlag exists.  JointSTAR uses this for the
    %   rho*(x_{t-1} - x*_{t-1}) error-correction terms and the c*_{t-1}
    %   cycle loadings.

    methods (Test)

        function loglikMatchesCompanionKalman(tc)
            rng(5);
            T = 180; rho = 0.85; b = -0.6; sEta = 0.7; sEps = 0.5;
            a = zeros(1, T);
            a(1) = sEta / sqrt(1 - rho^2) * randn;
            for t = 2:T, a(t) = rho * a(t - 1) + sEta * randn; end
            y = a + b * [0, a(1:end - 1)] + sEps * randn(1, T);
            y(1) = NaN;                    % first obs needs the lag: masked
            y(40:45) = NaN;                % plus an interior gap

            % precision version: single state, Zlag loading
            spec = jointstar.ModelSpec('A1', rho, 'Q', sEta^2, ...
                'a1', 0, 'P1', sEta^2 / (1 - rho^2), 'Z', 1, 'Zlag', b, ...
                'Rdiag', sEps^2, 'T', T);
            llP = jointstar.computeLogLik(spec.system(), y);

            % oracle: companion form (alpha_t, alpha_{t-1})
            v0 = sEta^2 / (1 - rho^2);
            P1c = [v0, rho * v0; rho * v0, v0];
            sysC = struct('A1', [rho 0; 1 0], 'A2', [], 'c', [], ...
                'Q', [sEta^2 0; 0 0], 'a1', [0; 0], 'P1', P1c, ...
                'Z', [1 b], 'd', [], 'Rdiag', sEps^2, 'T', T);
            llK = kalmanReference(sysC, y);

            tc.verifyEqual(llP, llK, 'AbsTol', 1e-8, ...
                'Zlag likelihood does not match companion-form Kalman');
        end

    end
end
