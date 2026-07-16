function runFixSingletonKappa(dataFile, outRoot, seeds)
%RUNFIXSINGLETONKAPPA Seed sweep for the 'FixSingletonKappa' A/B.
%
%   runFixSingletonKappa(dataFile, outRoot, seeds)
%
%   Runs the 'FixSingletonKappa' option (DEFAULT FALSE on
%   jointstar.estimate) ON TOP OF the full baseline kernel --
%   HierKappa + PieObs + the transformed-kernel trio (MutationTransform +
%   StructuredBlocks + MStepsLadder) + WasteFree -- across an independent
%   seed list, for the same multi-seed / cross-seed-Rhat convergence
%   check every quotable result in this project requires (see CLAUDE.md
%   "Convergence discipline"; pattern follows benchmarks/runNeffSweep.m).
%
%   'FixSingletonKappa' removes the 4 likelihood-flat hyperparameter
%   dimensions of the two SINGLE-member COVID-kappa window groups
%   (kaphpp_2022, kappi_2023 -- CHECKPOINT_14: zero likelihood curvature,
%   contraction ratios 0.91/1.34, structurally unidentified) and replaces
%   each singleton kappa's prior with a fixed truncated-Gamma(2.0, 1.25)
%   calibrated at the conditional prior implied by the hypers' own prior
%   means. See jointstar.defaultPriors and tests/testFixSingletonKappa.m.
%
%   Each seed s runs:
%       jointstar.estimate(dataFile, 'NParticles', 2000, 'MSteps', 2, ...
%           'Seed', s, 'HierKappa', true, 'PieObs', true, ...
%           'MutationTransform', true, 'StructuredBlocks', true, ...
%           'MStepsLadder', true, 'WasteFree', true, ...
%           'FixSingletonKappa', true, 'OutDir', od)
%   where od = fullfile(outRoot, sprintf('seed%d', s)).  Only Seed and
%   OutDir differ across invocations.
%
%   Idempotent/resumable: skips any seed whose OutDir already contains
%   posterior_summary.csv (same resumption pattern as production.m /
%   runNeffSweep.m).  Each seed is wrapped in try/catch so one failure
%   does not kill the sweep.
%
%   NOTE: this is the DRIVER only.  It is not invoked here -- running the
%   full seed sweep (10 seeds x N=2000) is a long estimation and is the
%   orchestrator's job, not a sub-agent smoke check (see repo hard rules).
%
%   Arguments (all optional):
%       dataFile  -- input CSV (default: 'data.csv')
%       outRoot   -- output root directory (default:
%                    fullfile('results','experiments','FixSingletonKappa'))
%       seeds     -- vector of seed values (default: [42 7 101 1 2 3 4 5
%                    6 8], 10 seeds: the 3 production seeds plus 7 more
%                    independent ones, matching this project's >=3-seed
%                    convergence-check convention)
%
%   This script prints a timestamped line before and after each seed with
%   elapsed wall-clock time (minutes), and a final summary of completed/
%   skipped/failed counts.
%
%   See also jointstar.estimate, jointstar.defaultPriors,
%   benchmarks/runNeffSweep, benchmarks/computeNeff,
%   tests/testFixSingletonKappa.

% ---- defaults -----------------------------------------------------------
if nargin < 1 || isempty(dataFile)
    dataFile = 'data.csv';
end
if nargin < 2 || isempty(outRoot)
    outRoot = fullfile('results', 'experiments', 'FixSingletonKappa');
end
if nargin < 3 || isempty(seeds)
    seeds = [42 7 101 1 2 3 4 5 6 8];
end

% Ensure outRoot exists
if ~isfolder(outRoot)
    mkdir(outRoot);
end

% ---- loop over seeds with timing and error handling --------------------
nSeeds = numel(seeds);
nCompleted = 0;
nSkipped = 0;
nFailed = 0;
failedSeeds = [];

for k = 1:nSeeds
    s = seeds(k);
    od = fullfile(outRoot, sprintf('seed%d', s));

    % Check if already done
    if isfile(fullfile(od, 'posterior_summary.csv'))
        fprintf('runFixSingletonKappa: seed %d already done (%s); skipping.\n', s, od);
        nSkipped = nSkipped + 1;
        continue
    end

    % Timestamped start
    tStart = datetime('now');
    fprintf('[%s] runFixSingletonKappa: starting seed %d -> %s\n', ...
        tStart, s, od);
    tic;

    % Run estimate with error handling
    try
        jointstar.estimate(dataFile, ...
            'NParticles', 2000, 'MSteps', 2, ...
            'Seed', s, 'HierKappa', true, ...
            'PieObs', true, ...
            'MutationTransform', true, 'StructuredBlocks', true, ...
            'MStepsLadder', true, 'WasteFree', true, ...
            'FixSingletonKappa', true, 'OutDir', od);

        elapsed = toc;
        elapsedMin = elapsed / 60;
        tEnd = datetime('now');
        fprintf('[%s] runFixSingletonKappa: seed %d completed (%.2f minutes)\n', ...
            tEnd, s, elapsedMin);
        nCompleted = nCompleted + 1;

    catch err
        elapsed = toc;
        elapsedMin = elapsed / 60;
        fprintf('runFixSingletonKappa: seed %d FAILED after %.2f minutes\n', s, elapsedMin);
        fprintf('Error report:\n%s\n', getReport(err));
        nFailed = nFailed + 1;
        failedSeeds = [failedSeeds, s]; %#ok<AGROW>
    end
end

% ---- final summary ------------------------------------------------------
fprintf('\n');
fprintf('runFixSingletonKappa summary:\n');
fprintf('  Completed: %d\n', nCompleted);
fprintf('  Skipped:   %d\n', nSkipped);
fprintf('  Failed:    %d\n', nFailed);
if nFailed > 0
    fprintf('  Failed seeds: %s\n', sprintf('%d ', failedSeeds));
end
fprintf('  Total seeds requested: %d\n', nSeeds);
end
