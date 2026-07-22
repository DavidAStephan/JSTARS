% jointstar.production('data.csv') is the canonical production path; this
% is a standalone diagnostic driver for the METHODOLOGY_NOTE.md Sec 7
% headline convergence figure (G=20 rank-folded R-hat) and does not
% touch results/production/.
%
% Regenerates the G=20 Herbst-Schorfheide-style field-native convergence
% report (benchmarks/computeNeff.m: N_eff + rank-normalized folded R-hat)
% in one command: runs the 3 production seeds [42 7 101] plus 17
% additional independent seeds under the EXACT current production
% configuration, then calls computeNeff over all 20 resulting particle
% clouds.
%
% SEED LIST: [42 7 101] (production) + [1 2 3 4 5 6 8 9 10 11 12 13 14 15
% 16 17 18] (17 extra, deliberately excluding 7 since it is already a
% production seed) = 20 total. This is a documented, FIXED default list,
% not a re-derivation of the original CHECKPOINT_14 sweep: that sweep's
% exact seed list lived in results/neff_sweep/ (removed from the working
% tree at the CP16 repo cleanup, commit c6ec0f5; recoverable via
% `git show ed86c7a:benchmarks/runNeffSweep.m` and
% `git log --all --diff-filter=A --name-only -- results/neff_sweep`,
% which is in fact where this seed list was taken from). Also, that
% sweep predates WasteFree/PoolAcuteOnly/ESSTargetFrac=0.7 (adopted at
% CHECKPOINT_14-16 after the sweep), so a rerun here uses the CURRENT
% production configuration throughout -- figures may therefore differ
% slightly from the originally reported max(bulk,folded) ~= 1.22-1.24.
%
% Each non-reused seed runs the EXACT production configuration (copied
% verbatim from +jointstar/production.m):
%   jointstar.estimate('data.csv', 'NParticles', 2000, 'MSteps', 2, ...
%       'Seed', s, 'HierKappa', true, 'PieObs', true, ...
%       'MutationTransform', true, 'StructuredBlocks', true, ...
%       'MStepsLadder', true, 'WasteFree', true, ...
%       'PoolAcuteOnly', true, 'ESSTargetFrac', 0.7, 'OutDir', od)
%
% Idempotent/resumable: skips any seed whose OutDir already contains
% posterior_summary.csv. The three production seeds are special-cased to
% reuse results/production/seed<s> when that directory already has a
% posterior_summary.csv, rather than re-running ~9 min/seed of duplicate
% compute under a second OutDir -- so the assembled 20-directory list is
% a MIX of results/production/seed<s> (for 42/7/101, when available) and
% results/neff_g20/seed<s> (for those three if results/production/ is
% absent, and always for the 17 extra seeds).
%
% Full cold run: ~9 min/seed x up to 20 seeds (fewer if the 3 production
% seeds are reused) -- do not run this casually; see the module docstring
% verification note for a cheap 3-seed smoke test instead.
%
% Output: results/neff_g20/neff_g20.csv (benchmarks/computeNeff.m's
% columns: param, neff, std_mean, post_sd, mcse_over_sd, rhat_classic,
% rhat_rank_bulk, rhat_rank_folded), written atomically via
% jointstar.writeTableAtomic.
%
% See also jointstar.production, benchmarks/computeNeff.m,
% benchmarks/runConvergenceCheck.m.

dataFile = 'data.csv';
outRoot = fullfile('results', 'neff_g20');
if ~isfolder(outRoot), mkdir(outRoot); end

prodSeeds = [42, 7, 101];
extraSeeds = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];
allSeeds = [prodSeeds, extraSeeds];

dirs = cell(1, numel(allSeeds));
for k = 1:numel(allSeeds)
    s = allSeeds(k);

    prodDir = fullfile('results', 'production', sprintf('seed%d', s));
    isProdSeed = any(s == prodSeeds);

    if isProdSeed && isfile(fullfile(prodDir, 'posterior_summary.csv'))
        fprintf('runNeffG20: seed %d reusing existing production run -> %s\n', s, prodDir);
        dirs{k} = prodDir;
        continue
    end

    od = fullfile(outRoot, sprintf('seed%d', s));
    dirs{k} = od;
    if isfile(fullfile(od, 'posterior_summary.csv'))
        fprintf('runNeffG20: seed %d already done (%s); skipping.\n', s, od);
        continue
    end

    fprintf('runNeffG20: running seed %d -> %s\n', s, od);
    jointstar.estimate(dataFile, 'NParticles', 2000, 'MSteps', 2, ...
        'Seed', s, 'HierKappa', true, 'PieObs', true, ...
        'MutationTransform', true, 'StructuredBlocks', true, ...
        'MStepsLadder', true, 'WasteFree', true, ...
        'PoolAcuteOnly', true, 'ESSTargetFrac', 0.7, 'OutDir', od);
end
fprintf('=== all 20 seeds ready ===\n');

% ---- computeNeff over all 20 dirs, written atomically ---------------------
rawFile = fullfile(outRoot, 'neff_g20_raw.csv');   % computeNeff's mandatory sink
T = computeNeff(dirs, rawFile);
if isfile(rawFile), delete(rawFile); end

finalFile = fullfile(outRoot, 'neff_g20.csv');
jointstar.writeTableAtomic(T, finalFile);

rhatRank = max([T.rhat_rank_bulk, T.rhat_rank_folded], [], 2, 'omitnan');
fprintf('\nrunNeffG20: G = %d seeds, %d parameters -> %s\n', ...
    numel(dirs), height(T), finalFile);
fprintf('  max rhat_rank (bulk/folded) = %.4f\n', max(rhatRank, [], 'omitnan'));
fprintf('  median N_eff = %.2f\n', median(T.neff));
