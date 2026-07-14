%% ARM B: Run 3 seeds with new MutationTransform and StructuredBlocks flags
% Writes to scratchpad/armB/seed{42,7,101}
% Sequential runs, ~6 min each

cd('/Users/davidstephan/Documents/JSTARS');
fprintf('=== ARM B CONVERGENCE A/B TEST ===\n');
fprintf('Working directory: %s\n', pwd);
fprintf('which jointstar.estimate: %s\n', which('jointstar.estimate'));
fprintf('\n');

outBase = '/private/tmp/claude-501/-Users-davidstephan-Documents-JSTARS/1883750c-1387-4d11-8beb-57d3da17699c/scratchpad/armB';
seeds = [42, 7, 101];

for k = 1:numel(seeds)
    seed = seeds(k);
    outDir = fullfile(outBase, sprintf('seed%d', seed));
    fprintf('\n--- SEED %d -> %s ---\n', seed, outDir);
    fprintf('Starting: %s\n', datetime('now'));

    tic
    out = jointstar.estimate('data.csv', ...
        'NParticles', 2000, 'MSteps', 2, ...
        'HierKappa', true, 'PieObs', true, ...
        'MutationTransform', true, 'StructuredBlocks', true, ...
        'Seed', seed, 'OutDir', outDir);
    elapsed = toc;

    fprintf('Completed: %s (%.1f sec)\n', datetime('now'), elapsed);
    fprintf('LML: %.3f\n', out.smc.lml);
end

fprintf('\n=== ARM B RUNS COMPLETE ===\n');
