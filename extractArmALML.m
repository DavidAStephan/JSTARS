cd('/Users/davidstephan/Documents/JSTARS');
seeds = [42, 7, 101];
for k = 1:3
    seed = seeds(k);
    log_tbl = readtable(sprintf('results/production/seed%d/smc_log.csv', seed));
    lml = sum(log_tbl.lml_inc);
    fprintf('ARM A seed %d: LML = %.2f\n', seed, lml);
end
