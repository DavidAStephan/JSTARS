classdef testSMCToyBimodal < matlab.unittest.TestCase
    %TESTSMCTOYBIMODAL Checkpoint 3: SMC recovers a known bimodal target.
    %
    %   The "likelihood" is a 2-D two-component Gaussian mixture with
    %   asymmetric weights and well-separated modes; the prior is a wide
    %   zero-mean Gaussian.  The posterior is then itself a two-component
    %   Gaussian mixture whose mass split is available in closed form, so
    %   we can check that adaptive-tempering SMC (i) finds both modes and
    %   (ii) recovers the mass split to Monte Carlo error.  A plain MH
    %   chain started in one mode essentially never crosses; tempering is
    %   what makes this work.

    properties (Constant)
        w1 = 0.7
        mu1 = [-2, -1]
        mu2 = [3, 3]
        sigLik = 0.6
        sigPrior = 4.0
    end

    methods (Test)

        function recoversMassSplitAndBothModes(tc)
            d = 2;
            s2 = tc.sigLik^2; p2 = tc.sigPrior^2;

            logLik = @(th) logMix(th, tc.w1, tc.mu1, tc.mu2, s2);
            logPrior = @(th) -0.5 * (th * th') / p2 - d / 2 * log(2 * pi * p2);
            samplePrior = @(N) tc.sigPrior * randn(N, d);

            % closed-form posterior mass split: each mixture component k
            % contributes w_k * N(mu_k; 0, (s2+p2) I) marginal mass.
            v = s2 + p2;
            m1 = tc.w1 * exp(-0.5 * sum(tc.mu1.^2) / v);
            m2 = (1 - tc.w1) * exp(-0.5 * sum(tc.mu2.^2) / v);
            splitTrue = m1 / (m1 + m2);
            % posterior component means (product-of-Gaussians shrinkage)
            shrink = p2 / (s2 + p2);
            post1 = shrink * tc.mu1; post2 = shrink * tc.mu2;

            prob = struct('samplePrior', samplePrior, 'logPrior', logPrior, ...
                'logLik', logLik, 'paramNames', {{'x1', 'x2'}});
            opts = struct('NParticles', 2000, 'MSteps', 5, 'Seed', 314, ...
                'Verbose', false, 'UseParallel', false);
            out = jointstar.runSMC(prob, opts);

            % assign particles to modes by nearest posterior mean
            P = out.particles; W = out.weights;
            d1 = sum((P - post1).^2, 2); d2 = sum((P - post2).^2, 2);
            in1 = d1 < d2;
            splitHat = sum(W(in1));

            tc.verifyGreaterThan(sum(in1), 50, 'mode 1 not populated');
            tc.verifyGreaterThan(sum(~in1), 50, 'mode 2 not populated');
            tc.verifyEqual(splitHat, splitTrue, 'AbsTol', 0.05, ...
                sprintf('mass split %.3f vs true %.3f', splitHat, splitTrue));

            % within-mode means should sit at the shrunk component means
            mHat1 = (W(in1)' * P(in1, :)) / sum(W(in1));
            mHat2 = (W(~in1)' * P(~in1, :)) / sum(W(~in1));
            tc.verifyEqual(mHat1, post1, 'AbsTol', 0.15);
            tc.verifyEqual(mHat2, post2, 'AbsTol', 0.15);

            % tempering must actually temper (no single-jump schedule)
            tc.verifyGreaterThanOrEqual(height(out.stages), 2);
            tc.verifyEqual(out.stages.phi(end), 1, 'AbsTol', 1e-12);
        end

        function smcIsReproducibleGivenSeed(tc)
            logLik = @(th) logMix(th, tc.w1, tc.mu1, tc.mu2, tc.sigLik^2);
            logPrior = @(th) -0.5 * (th * th') / tc.sigPrior^2;
            prob = struct('samplePrior', @(N) tc.sigPrior * randn(N, 2), ...
                'logPrior', logPrior, 'logLik', logLik);
            opts = struct('NParticles', 300, 'MSteps', 3, 'Seed', 99, ...
                'Verbose', false, 'UseParallel', false);
            o1 = jointstar.runSMC(prob, opts);
            o2 = jointstar.runSMC(prob, opts);
            tc.verifyEqual(o1.particles, o2.particles, ...
                'same seed must give identical particle clouds');
        end

    end
end

function ll = logMix(th, w1, mu1, mu2, s2)
q1 = sum((th - mu1).^2) / s2;
q2 = sum((th - mu2).^2) / s2;
c = -log(2 * pi * s2);
a = log(w1) + c - 0.5 * q1;
b = log(1 - w1) + c - 0.5 * q2;
mx = max(a, b);
ll = mx + log(exp(a - mx) + exp(b - mx));
end
