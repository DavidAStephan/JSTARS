classdef testFullRVsKalman < matlab.unittest.TestCase
    %TESTFULLRVSKALMAN Full (correlated) measurement covariance is exact.
    %
    %   Bivariate local-level model with correlated measurement errors
    %   (and correlated state shocks): precision likelihood with
    %   sys.Rfull must match the Kalman filter to 1e-8, including
    %   partially-missing observation vectors, where masking must produce
    %   the observed submatrix of R (exact Gaussian marginalisation).

    methods (Test)

        function loglikMatchesKalman(tc)
            rng(17);
            T = 150; m = 2; p = 2;
            Q = [0.30, 0.12; 0.12, 0.20];
            Rf = [0.50, -0.25; -0.25, 0.40];
            A1 = eye(m);

            % simulate
            a = zeros(m, T); LQ = chol(Q, 'lower'); LRf = chol(Rf, 'lower');
            for t = 2:T, a(:, t) = a(:, t - 1) + LQ * randn(m, 1); end
            y = a + LRf * randn(m, T);
            y(1, 30:40) = NaN;             % series 1 gap: 2x2 -> 1x1 R
            y(:, 70:75) = NaN;             % full gap
            y(2, 100) = NaN;

            sys = struct('A1', A1, 'A2', [], 'c', [], 'Q', Q, ...
                'a1', zeros(m, 1), 'P1', 10 * eye(m), 'Z', eye(p), ...
                'd', [], 'Rdiag', [], 'Rfull', Rf, 'T', T);

            llP = jointstar.computeLogLik(sys, y);
            llK = kalmanReference(sys, y);
            tc.verifyEqual(llP, llK, 'AbsTol', 1e-8, ...
                'full-R likelihood does not match Kalman');
        end

        function nonPDRfullRejectsNotCrashes(tc)
            T = 20;
            Rbad = [1, 2; 2, 1];           % indefinite
            sys = struct('A1', eye(2), 'A2', [], 'c', [], 'Q', eye(2), ...
                'a1', zeros(2, 1), 'P1', eye(2), 'Z', eye(2), ...
                'd', [], 'Rdiag', [], 'Rfull', Rbad, 'T', T);
            ll = jointstar.computeLogLik(sys, randn(2, T));
            tc.verifyEqual(ll, -Inf);
        end

    end
end
