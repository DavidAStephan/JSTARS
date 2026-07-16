classdef testWasteFree < matlab.unittest.TestCase
    %TESTWASTEFREE 'WasteFree' waste-free SMC mutation (Dau & Chopin 2022,
    %arXiv:2011.02328): flag off (explicit or omitted) must be
    %bit-identical to the pre-existing resample-then-mutate-keep-endpoint
    %code path; flag on must reach phi=1 with a sane, growing particle
    %cloud whose per-stage size follows the documented M/P eval-budget-
    %parity rule (wfM fixed, P_stage = round(M_stage/WF_MFRAC)+1,
    %n_cloud == wfM*P_stage).

    methods (Test)

        function wasteFreeOffMatchesOmittedOptions(tc)
            [prob, opts] = toyProblem();
            out1 = jointstar.runSMC(prob, opts);   % WasteFree omitted

            opts2 = opts;
            opts2.WasteFree = false;
            out2 = jointstar.runSMC(prob, opts2);  % WasteFree explicit false

            tc.verifyEqual(out1.particles, out2.particles, ...
                'omitted vs explicit-false WasteFree must be bit-identical');
            tc.verifyEqual(out1.logw, out2.logw);
            tc.verifyEqual(out1.lml, out2.lml);
            tc.verifyEqual(size(out1.particles, 1), opts.NParticles, ...
                'flag-off cloud size must stay fixed at NParticles');
            tc.verifyFalse( ...
                any(ismember({'n_cloud', 'wf_m', 'wf_p'}, ...
                out1.stages.Properties.VariableNames)), ...
                'flag-off stages table must not carry WasteFree diagnostic columns');
        end

        function wasteFreeOnReachesPhi1WithGrowingCloud(tc)
            [prob, opts] = ladderProblem();
            opts.WasteFree = true;
            out = jointstar.runSMC(prob, opts);

            tc.verifyEqual(out.stages.phi(end), 1, 'AbsTol', 1e-9);
            tc.verifyTrue(all(isfinite(out.particles(:))), ...
                'WasteFree particle cloud must be all-finite');
            tc.verifyTrue(all(isfinite(out.loglik)));
            tc.verifyGreaterThan(size(out.particles, 1), opts.NParticles, ...
                'WasteFree cloud must grow past NParticles (M*P_stage design)');

            tc.verifyTrue(all(ismember({'n_cloud', 'wf_m', 'wf_p'}, ...
                out.stages.Properties.VariableNames)), ...
                'WasteFree stages table must carry the n_cloud/wf_m/wf_p diagnostic columns');
            tc.verifyEqual(out.stages.n_cloud, ...
                out.stages.wf_m .* out.stages.wf_p, ...
                'n_cloud must equal wf_m*wf_p every stage (the M/P rule)');
            tc.verifyEqual(out.stages.wf_m, ...
                repmat(out.stages.wf_m(1), height(out.stages), 1), ...
                'wf_m (chain count) must stay fixed across the whole run');
            tc.verifyEqual(size(out.particles, 1), out.stages.n_cloud(end), ...
                'final particle cloud size must match the last logged n_cloud');
        end

        function wasteFreePStageFollowsEvalBudgetParityRule(tc)
            % No MStepsLadder here, so M_stage == MSteps every stage and
            % P_stage should therefore be CONSTANT at
            % round(MSteps/WF_MFRAC) + 1 (WF_MFRAC = 0.25, see
            % jointstar.runSMC 'WasteFree' doc) for every logged stage.
            [prob, opts] = ladderProblem();
            opts.WasteFree = true;
            out = jointstar.runSMC(prob, opts);

            expectedP = round(opts.MSteps / 0.25) + 1;
            tc.verifyEqual(out.stages.wf_p, ...
                repmat(expectedP, height(out.stages), 1), ...
                'P_stage must equal round(MSteps/0.25)+1 with MStepsLadder off');

            expectedM = max(2, round(0.25 * opts.NParticles));
            tc.verifyEqual(out.stages.wf_m(1), expectedM, ...
                'wf_m must equal round(0.25*NParticles)');
        end

        function wasteFreeCombinedWithTransformBlocksLadderOnRealPriors(tc)
            % Smoke test mirroring the production flag combination
            % (MutationTransform + StructuredBlocks + MStepsLadder), with
            % WasteFree layered on top, on the real 79-param prior
            % structure (cheap synthetic likelihood -- see
            % testMutationKernelSmoke's realPriorsProblem convention).
            [prob, opts] = realPriorsProblem();
            opts.MutationTransform = true;
            opts.StructuredBlocks = true;
            opts.MStepsLadder = true;
            opts.WasteFree = true;
            out = jointstar.runSMC(prob, opts);

            tc.verifyTrue(isfinite(out.lml));
            tc.verifyTrue(all(isfinite(out.particles(:))));
            tc.verifyEqual(out.stages.phi(end), 1, 'AbsTol', 1e-9);
            tc.verifyGreaterThan(mean(out.stages.acc_rate), 0);
        end

    end
end

function [prob, opts] = toyProblem()
% Small, fast problem shared with testMStepsLadder's convention.
logLik = @(th) -0.5 * sum((th - [1, -1]).^2) / 0.5^2;
logPrior = @(th) -0.5 * (th * th') / 3^2;
samplePrior = @(N) 3 * randn(N, 2);
prob = struct('samplePrior', samplePrior, 'logPrior', logPrior, ...
    'logLik', logLik, 'paramNames', {{'x1', 'x2'}});
opts = struct('NParticles', 200, 'MSteps', 3, 'Seed', 42, ...
    'Verbose', false, 'UseParallel', false);
end

function [prob, opts] = ladderProblem()
% A high ESS-target-fraction forces many small adaptive-tempering
% increments, so the phi schedule sweeps gradually through the full
% [0,1] range and exercises several waste-free mutation stages.
logLik = @(th) -0.5 * sum(th.^2) / 0.7^2;
logPrior = @(th) -0.5 * (th * th') / 3^2;
samplePrior = @(N) 3 * randn(N, 3);
prob = struct('samplePrior', samplePrior, 'logPrior', logPrior, ...
    'logLik', logLik, 'paramNames', {{'x1', 'x2', 'x3'}});
opts = struct('NParticles', 80, 'MSteps', 2, 'Seed', 123, ...
    'ESSTargetFrac', 0.9, 'Verbose', false, 'UseParallel', false);
end

function [prob, opts] = realPriorsProblem()
% The real 79-param prior structure with a cheap synthetic
% (non-JointSTAR) likelihood -- exercises paramTransform/blockAtoms on
% the actual prior types/bounds without paying for computeLogLik (see
% testMutationKernelSmoke.m).
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
