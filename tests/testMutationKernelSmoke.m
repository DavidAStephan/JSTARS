classdef testMutationKernelSmoke < matlab.unittest.TestCase
    %TESTMUTATIONKERNELSMOKE Fast end-to-end smoke tests for the
    %'MutationTransform' / 'StructuredBlocks' jointstar.runSMC options:
    %flags on must run to completion without error on the real 79-param
    %prior structure and give finite results; flags off (explicit or
    %omitted) must be bit-identical (R6); missing prob.priors with a
    %flag on must error clearly rather than silently misbehave.

    methods (Test)

        function flagsOffMatchesOmittedOptions(tc)
            [prob, opts] = toyProblem();
            out1 = jointstar.runSMC(prob, opts);

            opts2 = opts;
            opts2.MutationTransform = false;
            opts2.StructuredBlocks = false;
            out2 = jointstar.runSMC(prob, opts2);

            tc.verifyEqual(out1.particles, out2.particles);
            tc.verifyEqual(out1.lml, out2.lml);
        end

        function mutationTransformRunsOnRealPriors(tc)
            [prob, opts] = realPriorsProblem();
            opts.MutationTransform = true;
            out = jointstar.runSMC(prob, opts);
            tc.verifyTrue(isfinite(out.lml));
            tc.verifyTrue(all(isfinite(out.loglik)));
            tc.verifyGreaterThan(mean(out.stages.acc_rate), 0, ...
                'transformed-kernel acceptance rate must be positive');
        end

        function structuredBlocksRunsOnRealPriors(tc)
            [prob, opts] = realPriorsProblem();
            opts.StructuredBlocks = true;
            out = jointstar.runSMC(prob, opts);
            tc.verifyTrue(isfinite(out.lml));
            tc.verifyGreaterThan(mean(out.stages.acc_rate), 0);
        end

        function bothOptionsTogetherRunOnRealPriors(tc)
            [prob, opts] = realPriorsProblem();
            opts.MutationTransform = true;
            opts.StructuredBlocks = true;
            out = jointstar.runSMC(prob, opts);
            tc.verifyTrue(isfinite(out.lml));
            tc.verifyGreaterThan(mean(out.stages.acc_rate), 0);
        end

        function bothFlagsRequirePriorsField(tc)
            [prob, opts] = toyProblem();   % prob has NO .priors field
            opts.MutationTransform = true;
            tc.verifyError(@() jointstar.runSMC(prob, opts), 'jointstar:missingPriors');
        end

    end
end

function [prob, opts] = toyProblem()
logLik = @(th) -0.5 * sum((th - [1, -1]).^2) / 0.5^2;
logPrior = @(th) -0.5 * (th * th') / 3^2;
samplePrior = @(N) 3 * randn(N, 2);
prob = struct('samplePrior', samplePrior, 'logPrior', logPrior, ...
    'logLik', logLik, 'paramNames', {{'x1', 'x2'}});
opts = struct('NParticles', 200, 'MSteps', 3, 'Seed', 42, ...
    'Verbose', false, 'UseParallel', false);
end

function [prob, opts] = realPriorsProblem()
% The real 79-param prior structure with a cheap synthetic
% (non-JointSTAR) likelihood -- exercises paramTransform/blockAtoms on
% the actual prior types/bounds without paying for computeLogLik.
P = jointstar.defaultPriors('HierKappa', true);
j1 = find(strcmp(P.names, 'gamma1'), 1);
j2 = find(strcmp(P.names, 'rhoU'), 1);
llfun = @(tv) -0.5 * ((tv(j1) - 0.5)^2 + (tv(j2) + 0.3)^2) / 0.05^2;
prob = struct( ...
    'samplePrior', @(N) jointstar.priorSample(P, N), ...
    'logPrior', @(tv) jointstar.priorLogPdf(P, tv), ...
    'logLik', llfun, ...
    'paramNames', {P.names}, ...
    'priors', P);
opts = struct('NParticles', 60, 'MSteps', 1, 'Seed', 11, ...
    'Verbose', false, 'UseParallel', false, 'MutateIdx', P.mutateIdx);
end
