classdef testLmlRecording < matlab.unittest.TestCase
    %TESTLMLRECORDING Per-stage lml_inc logging and cumulative out.lml.
    %
    %   Reuses the small local-level state-space + precision-likelihood
    %   pipeline from testEndToEndSmall to check: out.lml is finite,
    %   equals out.logZ (same cumulative tempering-identity estimate,
    %   under its newly-documented name), equals the sum of the stages
    %   table's lml_inc column, is reproducible given a fixed seed, and
    %   -- via a LogFile -- that smc_log.csv actually carries an lml_inc
    %   column whose sum matches out.lml.

    methods (Test)

        function lmlFiniteReproducibleAndLogged(tc)
            rng(2026);
            Tn = 200;
            sEtaTrue = 0.3; sEpsTrue = 1.0;
            mu = cumsum([0, sEtaTrue * randn(1, Tn - 1)]);
            y = mu + sEpsTrue * randn(1, Tn);

            makeSys = @(th) struct('A1', 1, 'A2', [], 'c', [], ...
                'Q', exp(2 * th(1)), 'a1', 0, 'P1', 100, ...
                'Z', 1, 'd', [], 'Rdiag', exp(2 * th(2)), 'T', Tn);
            logLik = @(th) jointstar.computeLogLik(makeSys(th), y);
            logPrior = @(th) -0.5 * (th * th') / 1.5^2;
            samplePrior = @(N) 1.5 * randn(N, 2);

            prob = struct('samplePrior', samplePrior, 'logPrior', logPrior, ...
                'logLik', logLik, 'paramNames', {{'log_sig_eta', 'log_sig_eps'}});

            logFile = [tempname, '.csv'];
            tc.addTeardown(@() deleteIfExists(logFile));
            opts = struct('NParticles', 400, 'MSteps', 3, 'Seed', 7, ...
                'Verbose', false, 'UseParallel', false, 'LogFile', logFile);

            out1 = jointstar.runSMC(prob, opts);
            out2 = jointstar.runSMC(prob, opts);

            tc.verifyTrue(isfinite(out1.lml), 'out.lml must be finite');
            tc.verifyEqual(out1.lml, out1.logZ, ...
                'out.lml must equal the cumulative tempering-identity logZ');
            tc.verifyEqual(out1.lml, sum(out1.stages.lml_inc), 'AbsTol', 1e-9, ...
                'out.lml must equal the sum of the per-stage lml_inc column');
            tc.verifyEqual(out1.lml, out2.lml, ...
                'lml must be reproducible for a fixed seed');

            tc.verifyTrue(isfile(logFile));
            logged = readtable(logFile);
            tc.verifyTrue(ismember('lml_inc', logged.Properties.VariableNames), ...
                'smc_log.csv must carry an lml_inc column');
            tc.verifyEqual(sum(logged.lml_inc), out1.lml, 'AbsTol', 1e-9);
        end

    end
end

function deleteIfExists(f)
if isfile(f), delete(f); end
end
