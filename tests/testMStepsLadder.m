classdef testMStepsLadder < matlab.unittest.TestCase
    %TESTMSTEPSLADDER 'MStepsLadder' late-stage mutation-effort ladder:
    %flag off (explicit or omitted) must leave the per-stage MH sweep
    %count M constant at o.MSteps and be bit-identical to the pre-
    %existing code path; flag on must map the per-stage tempering
    %exponent phi (reached BEFORE that stage's mutation, i.e. the phi
    %already logged for that very row) to M_stage exactly as specified:
    %MSteps for phi<0.7, 2*MSteps for 0.7<=phi<0.95, 3*MSteps for
    %phi>=0.95.

    methods (Test)

        function ladderOffLeavesMConstant(tc)
            [prob, opts] = toyProblem();
            out = jointstar.runSMC(prob, opts);

            tc.verifyTrue(ismember('m_stage', out.stages.Properties.VariableNames), ...
                'stages table must carry an m_stage column');
            tc.verifyEqual(out.stages.m_stage, ...
                repmat(opts.MSteps, height(out.stages), 1), ...
                'MStepsLadder=false must leave M_stage == MSteps every stage');
        end

        function ladderOffOmittedMatchesExplicitFalse(tc)
            [prob, opts] = toyProblem();
            out1 = jointstar.runSMC(prob, opts);   % MStepsLadder omitted

            opts2 = opts;
            opts2.MStepsLadder = false;
            out2 = jointstar.runSMC(prob, opts2);  % MStepsLadder explicit false

            tc.verifyEqual(out1.particles, out2.particles, ...
                'omitted vs explicit-false MStepsLadder must be bit-identical');
            tc.verifyEqual(out1.lml, out2.lml);
            tc.verifyEqual(out1.stages.m_stage, out2.stages.m_stage);
        end

        function ladderOnMapsPhiToMAsSpecified(tc)
            [prob, opts] = ladderProblem();
            opts.MStepsLadder = true;
            out = jointstar.runSMC(prob, opts);

            phi = out.stages.phi;
            mStage = out.stages.m_stage;
            expected = zeros(size(phi));
            expected(phi < 0.7) = opts.MSteps;
            expected(phi >= 0.7 & phi < 0.95) = 2 * opts.MSteps;
            expected(phi >= 0.95) = 3 * opts.MSteps;

            tc.verifyEqual(mStage, expected, ...
                'm_stage must equal MSteps/2*MSteps/3*MSteps for phi<0.7 / [0.7,0.95) / >=0.95');

            % the schedule must actually exercise all three regimes, or
            % this test is not a meaningful check of the ladder mapping
            tc.verifyGreaterThan(sum(phi < 0.7), 0, 'no stage below phi=0.7');
            tc.verifyGreaterThan(sum(phi >= 0.7 & phi < 0.95), 0, ...
                'no stage in [0.7, 0.95)');
            tc.verifyGreaterThan(sum(phi >= 0.95), 0, 'no stage at/above phi=0.95');
        end

        function ladderOnIncreasesTotalMutationWorkModestly(tc)
            % sanity check on the design intent ("keep total extra work
            % modest, ~+40% given typical schedules") -- not a precise
            % bound, just confirms the ladder does not blow up work by
            % some gross multiple on a realistic schedule.
            [prob, opts] = ladderProblem();
            outOff = jointstar.runSMC(prob, opts);

            opts.MStepsLadder = true;
            outOn = jointstar.runSMC(prob, opts);

            workOff = sum(outOff.stages.m_stage);
            workOn = sum(outOn.stages.m_stage);
            tc.verifyGreaterThan(workOn, workOff, ...
                'ladder-on must do at least as much mutation work as ladder-off');
            tc.verifyLessThan(workOn, 3 * workOff, ...
                'ladder-on must not blow up total mutation work to the max multiplier throughout');
        end

    end
end

function [prob, opts] = toyProblem()
% Small, fast problem shared with testMutationKernelSmoke's convention.
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
% [0,1] range and populates all three ladder regimes.
logLik = @(th) -0.5 * sum(th.^2) / 0.7^2;
logPrior = @(th) -0.5 * (th * th') / 3^2;
samplePrior = @(N) 3 * randn(N, 3);
prob = struct('samplePrior', samplePrior, 'logPrior', logPrior, ...
    'logLik', logLik, 'paramNames', {{'x1', 'x2', 'x3'}});
opts = struct('NParticles', 300, 'MSteps', 2, 'Seed', 123, ...
    'ESSTargetFrac', 0.97, 'Verbose', false, 'UseParallel', false);
end
