classdef testHorseshoeShrinkage < matlab.unittest.TestCase
    %TESTHORSESHOESHRINKAGE Horseshoe Gibbs step shrinks nulls, keeps signals.
    %
    %   Normal-means problem  y_i = beta_i + e_i,  e ~ N(0, 1), with a
    %   sparse truth: most beta_i = 0, a few large.  Full Gibbs alternates
    %   the conjugate beta update with jointstar.horseshoeSample.  The
    %   horseshoe should shrink the null coefficients hard toward zero
    %   while leaving the large signals nearly unshrunk -- exactly the
    %   behaviour we later rely on for the innovation-covariance
    %   off-diagonals L_ij.  Also checks the grouped variant: a group
    %   whose coefficients are all null should end up with a much smaller
    %   global scale tau_g than a signal-dense group.

    methods (Test)

        function shrinksNullsKeepsSignals(tc)
            rng(11);
            k = 60; nSig = 6; bTrue = zeros(k, 1);
            bTrue(1:nSig) = [5; -5; 4; -4; 6; 5];
            y = bTrue + randn(k, 1);

            hs = jointstar.horseshoeSample('init', ones(k, 1));
            beta = y;
            nIter = 3000; burn = 500;
            bSum = zeros(k, 1);
            for it = 1:nIter
                % beta_i | . ~ N(m, v),  v = (1 + 1/(tau^2 lam_i^2))^{-1}
                pv = 1 ./ (hs.tau2(hs.groups) .* hs.lambda2);
                v = 1 ./ (1 + pv);
                beta = v .* y + sqrt(v) .* randn(k, 1);
                hs = jointstar.horseshoeSample(beta, hs);
                if it > burn, bSum = bSum + beta; end
            end
            bHat = bSum / (nIter - burn);

            nulls = bHat(nSig + 1:end); signals = bHat(1:nSig);
            % Thresholds calibrated against a 1-D quadrature evaluation of
            % the exact horseshoe posterior mean at the Gibbs posterior-mean
            % tau (~0.59): shrink factor ~0.22 at |y|=1, ~0.36 at |y|=2,
            % ~0.91 at |y|=5.  Median is used for the nulls because a
            % single 3-sigma noise draw legitimately escapes shrinkage.
            tc.verifyLessThan(median(abs(nulls)), 0.3, ...
                'null coefficients insufficiently shrunk');
            % raw |y| for nulls averages ~1; horseshoe must beat it clearly
            tc.verifyLessThan(mean(abs(nulls)) / mean(abs(y(nSig + 1:end))), 0.5);
            % signals barely shrunk: within noise of the truth
            tc.verifyLessThan(max(abs(signals - bTrue(1:nSig))), 3 * 1.0, ...
                'signal coefficients over-shrunk');
        end

        function groupedGlobalScalesSeparate(tc)
            rng(12);
            % group 1: 20 nulls; group 2: 20 strong signals
            k = 40; groups = [ones(20, 1); 2 * ones(20, 1)];
            bTrue = [zeros(20, 1); 4 + randn(20, 1)];
            y = bTrue + randn(k, 1);

            hs = jointstar.horseshoeSample('init', groups);
            beta = y;
            nIter = 3000; burn = 500;
            tSum = zeros(2, 1);
            for it = 1:nIter
                pv = 1 ./ (hs.tau2(hs.groups) .* hs.lambda2);
                v = 1 ./ (1 + pv);
                beta = v .* y + sqrt(v) .* randn(k, 1);
                hs = jointstar.horseshoeSample(beta, hs);
                if it > burn, tSum = tSum + sqrt(hs.tau2); end
            end
            tHat = tSum / (nIter - burn);
            tc.verifyLessThan(tHat(1), 0.5 * tHat(2), sprintf( ...
                'null-group tau (%.3f) not clearly below signal-group tau (%.3f)', ...
                tHat(1), tHat(2)));
        end

    end
end
