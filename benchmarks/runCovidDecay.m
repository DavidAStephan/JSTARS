function runCovidDecay(dataFile, outRoot, seeds)
%RUNCOVIDDECAY Seed sweep for the 'CovidDecay' kappa-prior flag.
%
%   runCovidDecay(dataFile, outRoot, seeds)
%
%   Runs the 'CovidDecay' option ON TOP OF the full baseline trio +
%   waste-free kernel (HierKappa, PieObs, MutationTransform,
%   StructuredBlocks, MStepsLadder, WasteFree) at production scale
%   (N=2000, MSteps=2), across a seed list, for the multi-seed
%   convergence check this project requires before any spec change is
%   credited as real (see CLAUDE.md "Convergence discipline"). Each
%   seed uses:
%
%       jointstar.estimate(dataFile, 'NParticles', 2000, 'MSteps', 2, ...
%           'Seed', s, 'HierKappa', true, 'PieObs', true, ...
%           'MutationTransform', true, 'StructuredBlocks', true, ...
%           'MStepsLadder', true, 'WasteFree', true, ...
%           'CovidDecay', true, 'OutDir', od)
%   where od = fullfile(outRoot, sprintf('seed%d', s)). Only Seed and
%   OutDir differ across invocations.
%
%   'CovidDecay' replaces the 8 COVID kappas belonging to the acute
%   two-step hierarchical groups (row1 GDP, row4 U, row5 participation,
%   row7 capital) with a parametric geometric-decay SD multiplier
%   (Lenza-Primiceri style), dropping 8 kappas + 6 hyper dims and adding
%   5 decay params (theta 79 -> 70 under HierKappa). See
%   jointstar.defaultPriors / jointstar.ModelSpec.jointstar /
%   tests/testCovidDecay.m for the full specification. Mutually
%   exclusive with PoolAcuteOnly/FixSingletonKappa (jointstar.estimate
%   errors if combined).
%
%   Idempotent/resumable: skips any seed whose OutDir already contains
%   posterior_summary.csv (same resumption pattern as production.m /
%   runNeffSweep.m / runPoolAcuteOnly.m). This is the DRIVER only -- it
%   is not invoked by this commit; the orchestrator runs it explicitly
%   when ready.
%
%   Arguments (all optional):
%       dataFile  -- input CSV (default: 'data.csv')
%       outRoot   -- output root directory (default:
%                    fullfile('results','experiments','CovidDecay'))
%       seeds     -- vector of seed values (default: [42 7 101 1 2 3 4 5
%                    6 8], 10 seeds: the 3 production seeds plus 7 more
%                    for a Gelman-style R-hat / N_eff check)
%
%   This script prints a timestamped line before and after each seed with
%   elapsed wall-clock time (minutes), and a final summary of completed/
%   skipped/failed counts. If a seed fails, the error is caught and
%   printed; execution continues to the next seed (one failure does not
%   kill the sweep).
%
%   See also jointstar.estimate, jointstar.production, runNeffSweep,
%   runPoolAcuteOnly, runFixSingletonKappa.

% ---- defaults -----------------------------------------------------------
if nargin < 1 || isempty(dataFile)
    dataFile = 'data.csv';
end
if nargin < 2 || isempty(outRoot)
    outRoot = fullfile('results', 'experiments', 'CovidDecay');
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
        fprintf('runCovidDecay: seed %d already done (%s); skipping.\n', s, od);
        nSkipped = nSkipped + 1;
        continue
    end

    % Timestamped start
    tStart = datetime('now');
    fprintf('[%s] runCovidDecay: starting seed %d -> %s\n', ...
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
            'CovidDecay', true, 'OutDir', od);

        elapsed = toc;
        elapsedMin = elapsed / 60;
        tEnd = datetime('now');
        fprintf('[%s] runCovidDecay: seed %d completed (%.2f minutes)\n', ...
            tEnd, s, elapsedMin);
        nCompleted = nCompleted + 1;

    catch err
        elapsed = toc;
        elapsedMin = elapsed / 60;
        fprintf('runCovidDecay: seed %d FAILED after %.2f minutes\n', s, elapsedMin);
        fprintf('Error report:\n%s\n', getReport(err));
        nFailed = nFailed + 1;
        failedSeeds = [failedSeeds, s]; %#ok<AGROW>
    end
end

% ---- final summary ------------------------------------------------------
fprintf('\n');
fprintf('runCovidDecay summary:\n');
fprintf('  Completed: %d\n', nCompleted);
fprintf('  Skipped:   %d\n', nSkipped);
fprintf('  Failed:    %d\n', nFailed);
if nFailed > 0
    fprintf('  Failed seeds: %s\n', sprintf('%d ', failedSeeds));
end
fprintf('  Total seeds requested: %d\n', nSeeds);
end
