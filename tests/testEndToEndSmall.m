classdef testEndToEndSmall < matlab.unittest.TestCase
    %TESTENDTOENDSMALL SMC + precision likelihood on a small state-space.
    %
    %   Local-level model  y_t = mu_t + eps,  mu_t = mu_{t-1} + eta,
    %   parameters theta = (log sigEta, log sigEps).  Simulate data at
    %   known values, run the full pipeline (jointstar.computeLogLik
    %   inside jointstar.runSMC), and check the posterior concentrates
    %   near the truth.  This exercises exactly the plumbing the full
    %   JointSTAR estimation will use, at toy scale.

    methods (Test)

        function posteriorCoversTruth(tc)
            rng(2026);
            Tn = 200;
            sEtaTrue = 0.3; sEpsTrue = 1.0;
            mu = cumsum([0, sEtaTrue * randn(1, Tn - 1)]);
            y = mu + sEpsTrue * randn(1, Tn);

            makeSys = @(th) struct('A1', 1, 'A2', [], 'c', [], ...
                'Q', exp(2 * th(1)), 'a1', 0, 'P1', 100, ...
                'Z', 1, 'd', [], 'Rdiag', exp(2 * th(2)), 'T', Tn);
            logLik = @(th) jointstar.computeLogLik(makeSys(th), y);

            % prior: log-sd ~ N(0, 1.5^2) on both -- weakly informative
            logPrior = @(th) -0.5 * (th * th') / 1.5^2;
            samplePrior = @(N) 1.5 * randn(N, 2);

            prob = struct('samplePrior', samplePrior, 'logPrior', logPrior, ...
                'logLik', logLik, 'paramNames', {{'log_sig_eta', 'log_sig_eps'}});
            opts = struct('NParticles', 400, 'MSteps', 3, 'Seed', 7, ...
                'Verbose', false, 'UseParallel', false);
            out = jointstar.runSMC(prob, opts);

            W = out.weights; P = out.particles;
            pm = W' * P;
            psd = sqrt(W' * (P - pm).^2);

            thTrue = [log(sEtaTrue), log(sEpsTrue)];
            z = abs(pm - thTrue) ./ psd;
            tc.verifyLessThan(z, 4, sprintf( ...
                'posterior mean [%.2f %.2f] too far from truth [%.2f %.2f]', ...
                pm(1), pm(2), thTrue(1), thTrue(2)));

            % posterior should be much tighter than the prior
            tc.verifyLessThan(psd, 0.75);
        end

    end
end
