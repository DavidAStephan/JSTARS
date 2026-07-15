function out = production(dataFile)
%PRODUCTION THE JointSTAR production entry point. No options.
%
%   out = jointstar.production(dataFile)
%
%   This IS the production recipe -- there are no name-value options,
%   because every choice below is already an owner ruling (see
%   CLAUDE.md "Final/production spec call" and "Owner rulings").  Per
%   CLAUDE.md's Convergence discipline, this model's structural
%   parameters are NOT seed-stable in a single SMC run: >=3 independent
%   seeds pooled into one equal-weight mixture (an inter-seed
%   uncertainty envelope; see CHECKPOINT_10) is what gets reported,
%   never a single seed.  jointstar.production is that pooled recipe,
%   run to completion, in one call.
%
%   HEADLINE OUTPUT: the pooled coefficient table (posterior
%   mean/sd/q05/q50/q95 per structural parameter, ~405 rows), returned
%   as out.coefficients and written to results/production/
%   pooled_posterior.csv.  This is accompanied by the cross-seed Rhat
%   convergence table and the pooled smoothed latent-state bands. This
%   function does NOT run Table-3 validation against the in-house
%   baseline -- jointstar.validate remains available as a dormant
%   standalone tool for that comparison, it is just not part of this
%   pipeline.
%
%   For each seed in [42, 7, 101] this runs
%       jointstar.estimate(dataFile, 'NParticles', 2000, 'MSteps', 2, ...
%           'Seed', s, 'HierKappa', true, 'PieObs', true, ...
%           'MutationTransform', true, 'StructuredBlocks', true, ...
%           'MStepsLadder', true, 'OutDir', <outRoot>/seed<s>)
%   skipping any seed whose OutDir already contains posterior_summary.csv
%   (idempotent/resumable: safe to re-run after a partial or interrupted
%   pass -- only the missing seeds are (re-)computed).
%
%   The transformed-kernel trio (MutationTransform + StructuredBlocks +
%   MStepsLadder) is the CHECKPOINT_13 convergence configuration: it
%   cut max cross-seed Rhat from 5.82 (raw kernel) to ~2.20 by mutating
%   in unconstrained coordinates, co-blocking the named ridge atoms, and
%   raising the late-stage MH-step count (x2 for phi>=0.7, x3 for
%   phi>=0.95). These change only the sampler's proposal, not the model
%   or its parameter set, so the pooling/Rhat machinery below is
%   unaffected. NOTE: existing results/production/ was produced by the
%   OLD raw kernel -- delete or move it before re-running so the pooled
%   table is regenerated under the new configuration (gamma2 and several
%   sigma/kappa params shift; the old values were partly sticky-kernel
%   artifacts, see CHECKPOINT_12/13).
%
%   outRoot defaults to 'results/production' relative to the current
%   directory, so the normal invocation, jointstar.production('data.csv')
%   run from the repo root, writes to results/production/.
%   (Deliberately NOT derived from fileparts(dataFile): 'results/production'
%   relative to cwd is the same convention jointstar.estimate's own
%   default 'OutDir' = 'results' already uses.)
%
%   Once all three seeds are estimated, this reuses:
%     - the exact pooling math in benchmarks/poolRuns.m (equal-weight,
%       1/3 each, mixture of the seeds' final phi=1 particle clouds),
%       replicated below as a local subfunction rather than depended on
%       as a script, writing
%         results/production/pooled_posterior.csv
%       (columns: param, mean, sd, q05, q50, q95) -- the reported
%       structural-parameter table. Per the CHECKPOINT_10 math audit,
%       read this as an INTER-SEED UNCERTAINTY ENVELOPE, not a
%       converged posterior: pooling averages seed noise but does not
%       remove the shared finite-N/mixing bias, and where all seeds
%       jointly miss posterior mass the pooled intervals can still be
%       too narrow. It is preferable to any single seed for reporting.
%     - the exact Gelman-style Rhat formula in
%       benchmarks/runConvergenceCheck.m (Rhat = sqrt(1 + B/W), B the
%       between-seed variance of the weighted posterior means, W the
%       mean of the seeds' weighted posterior variances), replicated
%       below, writing
%         results/production/convergence_rhat.csv
%       Expect several parameters with Rhat > 1.1: this model's
%       structural params are known seed-unstable (long likelihood
%       ridges, not a sampler bug) -- which is why the pooled envelope,
%       not any single seed, is what gets reported.
%
%   It also pools the smoothed latent-state bands into
%     results/production/smoothed_states.csv
%   (same columns as an individual run's smoothed_states.csv: date +
%   q05/med/q95 for rstar/nairu/taustar/gap/trend_inflation).  NOTE ON
%   METHOD: jointstar.estimate's smoothed_states.csv stores only the
%   per-run quantile bands, not the underlying per-particle state
%   draws, so there is no pooled draw sample to re-quantile here. This
%   function therefore takes the documented fallback: it averages the
%   three seeds' per-date quantile series column-by-column. Latent
%   states (r*, NAIRU, gap, ...) are far more robust across seeds than
%   the structural parameters (CLAUDE.md Convergence discipline), so
%   averaging already-close bands is a reasonable approximation of the
%   stratified pool -- it is NOT identical to re-quantiling a pooled
%   draw sample, which this codebase does not currently retain.
%
%   Returns a struct with fields: seeds, pooledFile, coefficients,
%   rhatFile, statesFile, maxRhat.  out.coefficients is pooled_posterior.csv
%   read back in as a MATLAB table (columns param, mean, sd, q05, q50,
%   q95) -- just type out.coefficients to view the pooled coefficient
%   table directly.
%
%   Cost: ~27 min/seed at MSteps=2, N=2000 -- about 80 minutes
%   wall-clock the first time all three seeds are missing.  Effectively
%   instant on any later call, since each seed is skipped independently
%   once results/production/seed<s>/posterior_summary.csv exists.
%
%   See also jointstar.estimate, jointstar.validate,
%   benchmarks/poolRuns.m, benchmarks/runConvergenceCheck.m.

seeds = [42, 7, 101];
outRoot = fullfile('results', 'production');
if ~isfolder(outRoot), mkdir(outRoot); end

seedDirs = cell(1, numel(seeds));
for k = 1:numel(seeds)
    s = seeds(k);
    od = fullfile(outRoot, sprintf('seed%d', s));
    seedDirs{k} = od;
    if isfile(fullfile(od, 'posterior_summary.csv'))
        fprintf('jointstar.production: seed %d already done (%s); skipping.\n', s, od);
        continue
    end
    fprintf('jointstar.production: running seed %d -> %s\n', s, od);
    jointstar.estimate(dataFile, 'NParticles', 2000, 'MSteps', 2, ...
        'Seed', s, 'HierKappa', true, ...
        'PieObs', true, ...
        'MutationTransform', true, 'StructuredBlocks', true, ...
        'MStepsLadder', true, 'OutDir', od);
end

P = jointstar.defaultPriors('HierKappa', true);

% ---- pool posterior (benchmarks/poolRuns.m logic, replicated) ---------
pooledFile = fullfile(outRoot, 'pooled_posterior.csv');
poolPosteriorLocal(seedDirs, P, pooledFile);

% ---- Rhat across seeds (benchmarks/runConvergenceCheck.m logic) ------
rhatFile = fullfile(outRoot, 'convergence_rhat.csv');
[maxRhat, nOver] = rhatLocal(seedDirs, P, rhatFile);
fprintf('jointstar.production: max Rhat = %.3f, %d params with Rhat > 1.1\n', ...
    maxRhat, nOver);

% ---- pooled smoothed states (average of per-seed quantile series) ----
statesFile = fullfile(outRoot, 'smoothed_states.csv');
poolStatesLocal(seedDirs, statesFile);

% ---- headline output: pooled coefficient table ------------------------
coefficients = readtable(pooledFile);

out = struct('seeds', seeds, 'pooledFile', pooledFile, 'coefficients', coefficients, ...
    'rhatFile', rhatFile, 'statesFile', statesFile, 'maxRhat', maxRhat);

fprintf('\njointstar.production done. Pooled coefficient table: %d parameters -> %s\n', ...
    height(coefficients), pooledFile);
disp(coefficients);
fprintf('Cross-seed Rhat: max %.3f, %d params with Rhat > 1.1 (%s)\n', ...
    maxRhat, nOver, rhatFile);
fprintf('Pooled smoothed states: %s\n', statesFile);
end

% ======================================================================
function poolPosteriorLocal(dirs, P, outFile)
% Exact pooling math of benchmarks/poolRuns.m: equal-weight (1/numel(dirs)
% each) mixture of the seeds' final phi=1 particle clouds.
allP = []; allW = [];
for r = 1:numel(dirs)
    S = loadFinalSnapshot(dirs{r});
    w = exp(S.logw - max(S.logw)); w = w / sum(w);
    allP = [allP; S.P]; %#ok<AGROW>
    allW = [allW; w / numel(dirs)]; %#ok<AGROW>
end
d = P.d;
mu = (allW' * allP)';
sd = sqrt(allW' * (allP - mu').^2)';
q = zeros(d, 3);
pset = [0.05, 0.5, 0.95];
for j = 1:d
    [xs, i] = sort(allP(:, j));
    cw = cumsum(allW(i)); cw = cw / cw(end);
    for kq = 1:3
        q(j, kq) = xs(find(cw >= pset(kq), 1, 'first'));
    end
end
tbl = table(string(P.names(:)), mu, sd, q(:, 1), q(:, 2), q(:, 3), ...
    'VariableNames', {'param', 'mean', 'sd', 'q05', 'q50', 'q95'});
writetable(tbl, outFile);
end

% ======================================================================
function [maxRhat, nOver] = rhatLocal(dirs, P, outFile)
% Exact Rhat formula of benchmarks/runConvergenceCheck.m: Rhat = sqrt(1 +
% B/W) with B the between-seed variance of the weighted posterior means
% and W the mean of the seeds' weighted posterior variances.
R = numel(dirs);
mu = zeros(R, P.d); v = zeros(R, P.d);
for r = 1:R
    S = loadFinalSnapshot(dirs{r});
    w = exp(S.logw - max(S.logw)); w = w / sum(w);
    mu(r, :) = w' * S.P;
    v(r, :) = w' * (S.P - mu(r, :)).^2;
end
B = var(mu, 0, 1);
W = mean(v, 1);
ok = W > 1e-12;   % skip the calibrated constant column(s)
rhat = ones(1, P.d);
rhat(ok) = sqrt(1 + B(ok) ./ W(ok));
tbl = table(string(P.names(:)), rhat(:), mu', 'VariableNames', ...
    {'param', 'rhat', 'means_by_seed'});
writetable(splitvars(tbl), outFile);
maxRhat = max(rhat(ok));
nOver = nnz(rhat > 1.1 & ok);
end

% ======================================================================
function S = loadFinalSnapshot(d)
snaps = dir(fullfile(d, 'particles_stage_*.mat'));
assert(~isempty(snaps), 'no particle snapshots in %s', d);
[~, iL] = max([snaps.datenum]);
S = load(fullfile(d, snaps(iL).name));
assert(abs(S.phi - 1) < 1e-9, 'last snapshot in %s is not the phi=1 cloud', d);
end

% ======================================================================
function poolStatesLocal(dirs, outFile)
% Pooled latent-state bands.  jointstar.estimate's smoothed_states.csv
% stores only the per-run quantile series (q05/med/q95), not the
% underlying per-particle state draws, so there is no pooled draw sample
% to re-quantile.  Per the production spec's documented fallback, this
% pools by averaging each seed's quantile series column-by-column --
% valid because latent states are far more robust across seeds than the
% structural parameters (CLAUDE.md Convergence discipline), so simple
% averaging of already-close bands is a reasonable stratified-pool
% approximation.
T = readtable(fullfile(dirs{1}, 'smoothed_states.csv'));
varNames = T.Properties.VariableNames;
numVars = varNames(~strcmp(varNames, 'date'));
acc = T(:, numVars);
for k = 2:numel(dirs)
    Tk = readtable(fullfile(dirs{k}, 'smoothed_states.csv'));
    assert(isequal(Tk.date, T.date), ...
        'smoothed_states.csv dates differ across seeds in %s', dirs{k});
    acc{:, :} = acc{:, :} + Tk{:, numVars};
end
acc{:, :} = acc{:, :} / numel(dirs);
tbl = [T(:, {'date'}), acc];
writetable(tbl, outFile);
end
