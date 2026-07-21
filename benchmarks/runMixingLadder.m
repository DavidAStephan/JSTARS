function runMixingLadder(stepName, extraOpts, seeds)
%RUNMIXINGLADDER  Extended mixing/ladder experiments with customizable options.
%
%   runMixingLadder(stepName)
%   runMixingLadder(stepName, extraOpts)
%   runMixingLadder(stepName, extraOpts, seeds)
%
%   Runs jointstar.estimate across a seed sweep (default: 10 seeds) with a
%   fixed production configuration and optional extra name-value pairs.
%   Results are stored in results/experiments/mixing_ladder/stepName/seedN.
%
%   Each run uses the EXACT production configuration:
%       jointstar.estimate(dataFile, 'NParticles', 2000, 'MSteps', 2, ...
%           'Seed', s, 'HierKappa', true, 'PieObs', true, ...
%           'MutationTransform', true, 'StructuredBlocks', true, ...
%           'MStepsLadder', true, 'WasteFree', true, 'PoolAcuteOnly', true, ...
%           'OutDir', od, extraOpts{:})
%   where extraOpts comes LAST so it can override the production options
%   (name-value pairs later in the arg list win in inputParser).
%
%   Idempotent/resumable: skips any seed whose OutDir already contains
%   posterior_summary.csv (same resumption pattern as production.m and
%   runNeffSweep.m).
%
%   Arguments (all optional):
%       stepName  -- descriptive label for this experiment (required for
%                    result folder hierarchy)
%       extraOpts -- cell array of name-value pairs to pass LAST to
%                    jointstar.estimate, allowing override of production
%                    defaults (e.g. {'ESSTargetFrac', 0.7, 'NParticles', 4000})
%                    (default: {} -- no overrides, run pure production config)
%       seeds     -- vector of seed values (default: [42 7 101 1 2 3 4 5 6 8])
%
%   This script prints a timestamped line before and after each seed with
%   elapsed wall-clock time (minutes), and a final summary of completed/
%   skipped/failed counts. If a seed fails, the error is caught and
%   printed; execution continues to the next seed (one failure does not
%   kill the sweep).
%
%   See also jointstar.production, jointstar.estimate, runNeffSweep.

% ---- defaults -----------------------------------------------------------
if nargin < 1 || isempty(stepName)
    error('stepName is required (descriptive label for results/experiments/mixing_ladder/stepName/)');
end
if nargin < 2 || isempty(extraOpts)
    extraOpts = {};
end
if nargin < 3 || isempty(seeds)
    seeds = [42, 7, 101, 1, 2, 3, 4, 5, 6, 8];
end

dataFile = 'data.csv';
expRoot = fullfile('results', 'experiments', 'mixing_ladder', stepName);

% Ensure expRoot exists
if ~isfolder(expRoot)
    mkdir(expRoot);
end

% ---- loop over seeds with timing and error handling --------------------
nSeeds = numel(seeds);
nCompleted = 0;
nSkipped = 0;
nFailed = 0;
failedSeeds = [];

for k = 1:nSeeds
    s = seeds(k);
    od = fullfile(expRoot, sprintf('seed%d', s));

    % Check if already done
    if isfile(fullfile(od, 'posterior_summary.csv'))
        fprintf('runMixingLadder: seed %d already done (%s); skipping.\n', s, od);
        nSkipped = nSkipped + 1;
        continue
    end

    % Timestamped start
    tStart = datetime('now');
    fprintf('[%s] runMixingLadder: starting seed %d -> %s\n', ...
        tStart, s, od);
    tic;

    % Run estimate with error handling
    try
        jointstar.estimate(dataFile, ...
            'NParticles', 2000, 'MSteps', 2, ...
            'Seed', s, 'HierKappa', true, ...
            'PieObs', true, ...
            'MutationTransform', true, 'StructuredBlocks', true, ...
            'MStepsLadder', true, 'WasteFree', true, 'PoolAcuteOnly', true, ...
            'OutDir', od, extraOpts{:});

        elapsed = toc;
        elapsedMin = elapsed / 60;
        tEnd = datetime('now');
        fprintf('[%s] runMixingLadder: seed %d completed (%.2f minutes)\n', ...
            tEnd, s, elapsedMin);
        nCompleted = nCompleted + 1;

    catch err
        elapsed = toc;
        elapsedMin = elapsed / 60;
        fprintf('runMixingLadder: seed %d FAILED after %.2f minutes\n', s, elapsedMin);
        fprintf('Error report:\n%s\n', getReport(err));
        nFailed = nFailed + 1;
        failedSeeds = [failedSeeds, s]; %#ok<AGROW>
    end
end

% ---- final summary ------------------------------------------------------
fprintf('\n');
fprintf('runMixingLadder summary (%s):\n', stepName);
fprintf('  Completed: %d\n', nCompleted);
fprintf('  Skipped:   %d\n', nSkipped);
fprintf('  Failed:    %d\n', nFailed);
if nFailed > 0
    fprintf('  Failed seeds: %s\n', sprintf('%d ', failedSeeds));
end
fprintf('  Total seeds requested: %d\n', nSeeds);
end
