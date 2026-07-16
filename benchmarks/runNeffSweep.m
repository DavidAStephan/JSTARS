function runNeffSweep(dataFile, outRoot, seeds)
%RUNNEFFWEEP Extended seed sweep for Herbst-Schorfheide G=20 convergence.
%
%   runNeffSweep(dataFile, outRoot, seeds)
%
%   This function runs 17 additional independent seeds (default: [1 2 3 4
%   5 6 8 9 10 11 12 13 14 15 16 17 18], deliberately excluding the three
%   production seeds 42, 7, 101) for the Herbst-Schorfheide-style G=20
%   N_eff / STD-across-runs convergence assessment (these 17 + production's
%   42/7/101 = G=20 total). Results are suitable for supplementary
%   diagnostics and extended cross-seed Rhat inspection.
%
%   Each seed uses the EXACT production configuration to ensure
%   comparability:
%       jointstar.estimate(dataFile, 'NParticles', 2000, 'MSteps', 2, ...
%           'Seed', s, 'HierKappa', true, 'PieObs', true, ...
%           'MutationTransform', true, 'StructuredBlocks', true, ...
%           'MStepsLadder', true, 'OutDir', od)
%   where od = fullfile(outRoot, sprintf('seed%d', s)). Only Seed and
%   OutDir differ across invocations.
%
%   Idempotent/resumable: skips any seed whose OutDir already contains
%   posterior_summary.csv (same resumption pattern as production.m).
%
%   Arguments (all optional):
%       dataFile  -- input CSV (default: 'data.csv')
%       outRoot   -- output root directory (default:
%                    fullfile('results','neff_sweep'))
%       seeds     -- vector of seed values (default: [1 2 3 4 5 6 8 9 10
%                    11 12 13 14 15 16 17 18])
%
%   This script prints a timestamped line before and after each seed with
%   elapsed wall-clock time (minutes), and a final summary of completed/
%   skipped/failed counts. If a seed fails, the error is caught and
%   printed; execution continues to the next seed (one failure does not
%   kill the sweep).
%
%   See also jointstar.production, jointstar.estimate.

% ---- defaults -----------------------------------------------------------
if nargin < 1 || isempty(dataFile)
    dataFile = 'data.csv';
end
if nargin < 2 || isempty(outRoot)
    outRoot = fullfile('results', 'neff_sweep');
end
if nargin < 3 || isempty(seeds)
    seeds = [1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18];
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
        fprintf('runNeffSweep: seed %d already done (%s); skipping.\n', s, od);
        nSkipped = nSkipped + 1;
        continue
    end

    % Timestamped start
    tStart = datetime('now');
    fprintf('[%s] runNeffSweep: starting seed %d -> %s\n', ...
        tStart, s, od);
    tic;

    % Run estimate with error handling
    try
        jointstar.estimate(dataFile, ...
            'NParticles', 2000, 'MSteps', 2, ...
            'Seed', s, 'HierKappa', true, ...
            'PieObs', true, ...
            'MutationTransform', true, 'StructuredBlocks', true, ...
            'MStepsLadder', true, 'OutDir', od);

        elapsed = toc;
        elapsedMin = elapsed / 60;
        tEnd = datetime('now');
        fprintf('[%s] runNeffSweep: seed %d completed (%.2f minutes)\n', ...
            tEnd, s, elapsedMin);
        nCompleted = nCompleted + 1;

    catch err
        elapsed = toc;
        elapsedMin = elapsed / 60;
        fprintf('runNeffSweep: seed %d FAILED after %.2f minutes\n', s, elapsedMin);
        fprintf('Error report:\n%s\n', getReport(err));
        nFailed = nFailed + 1;
        failedSeeds = [failedSeeds, s]; %#ok<AGROW>
    end
end

% ---- final summary ------------------------------------------------------
fprintf('\n');
fprintf('runNeffSweep summary:\n');
fprintf('  Completed: %d\n', nCompleted);
fprintf('  Skipped:   %d\n', nSkipped);
fprintf('  Failed:    %d\n', nFailed);
if nFailed > 0
    fprintf('  Failed seeds: %s\n', sprintf('%d ', failedSeeds));
end
fprintf('  Total seeds requested: %d\n', nSeeds);
end
