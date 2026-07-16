function compareCovidFlags()
%COMPARECOVIDFLAGS  Paired flag-vs-baseline comparison for the four COVID
%kappa-handling flags (FixSingletonKappa, PoolAcuteOnly, CovidDecay,
%DropPhiY), each run on the SAME 10 seeds as the baseline waste-free
%kernel (no covid flag): seeds = [42 7 101 1 2 3 4 5 6 8].
%
% Analysis-only: reads existing particles_stage_*.mat / posterior_summary
% / smoothed_states.csv snapshots under results/experiments/. Runs NO
% estimation.
%
% Param sets differ by flag (this is the key subtlety -- do NOT assume a
% fixed 79-param vector):
%   FixSingletonKappa : 75 params (drops 4 single-member-group hypers)
%   PoolAcuteOnly     : 69 params (hierarchy kept only for 2020 acute grp)
%   CovidDecay        : 70 params (per-window kappa -> s0_y/u/pr/k + rho_c)
%   DropPhiY          : 79 params (phiy present but fixed at 0)
%   baseline (WF, no covid flag) : 79 params
% Names are derived per-run from that run's OWN posterior_summary.csv
% 'param' column, whose order is verified to exactly match the S.P
% particle-matrix columns for that run.
%
% Convergence math reproduces benchmarks/computeNeff.m EXACTLY:
%   - rhat_classic = sqrt(1 + B/W) on per-seed weighted means/vars
%   - N_eff (Durham-Geweke/Herbst-Schorfheide) = V_post(pooled equal-
%     weight mixture) / std_across_seeds(weighted mean)^2
%   - rank-normalized, folded Rhat (Vehtari et al. 2021), each cloud
%     first resampled (systematic resampling) to a common length using a
%     LOCAL, fixed-seed RandStream (no global rng side effect); headline
%     rhat_rank(j) = max(bulk, folded).
% These are computed SEPARATELY for the flag's 10 clouds and for the
% baseline's SAME 10 seeds (paired), each using its own param count/names.
%
% Economics: pool each variant (equal-weight mixture over its 10 seeds,
% same recipe as poolRuns.m/production.m) to get per-param posterior
% mean/sd/q05/q50/q95, then join flag vs baseline on shared param NAMES.
%
% Latent states: arithmetic mean across the 10 seeds of each smoothed
% quantile-band column (q05/med/q95 for rstar/nairu/gap/trend_inflation),
% for both the flag and the baseline-on-the-same-seeds; end-of-sample row
% and the 2020Q2 gap trough are extracted from the averaged table.
%
% Outputs (results/experiments/covid_compare/):
%   baseline_stats.csv           per-param convergence + pooled posterior
%                                 for baseline on the 10 shared seeds
%   <Flag>_stats.csv              same, for the flag's 10 clouds
%   <Flag>_economics.csv          joined flag-vs-baseline pooled posterior
%                                 on shared param names (+ flag-only rows)
%   <Flag>_latent.csv             baseline vs flag averaged latent bands
%                                 (end-of-sample + 2020Q2 gap trough)
%   summary.csv                   one row per flag: convergence headline
%                                 + key econ deltas

outDir = 'results/experiments/covid_compare';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

seeds = [42, 7, 101, 1, 2, 3, 4, 5, 6, 8];

baselineDirs = cell(1, numel(seeds));
for i = 1:numel(seeds)
    s = seeds(i);
    if ismember(s, [42, 7, 101])
        baselineDirs{i} = sprintf('results/experiments/WF_wastefree/seed%d', s);
    else
        baselineDirs{i} = sprintf('results/experiments/WF_sweep/seed%d', s);
    end
end

fprintf('================================================================\n');
fprintf('BASELINE (waste-free kernel, no covid flag), same %d seeds\n', numel(seeds));
fprintf('================================================================\n');
baseStats = computeVariantStats(baselineDirs);
writetable(baseStats.tbl, fullfile(outDir, 'baseline_stats.csv'));
printVariantSummary('BASELINE', baseStats);
baseLatent = averageSmoothedStates(baselineDirs);
[baseEnd, baseTrough] = extractLatentPoints(baseLatent);
printLatentSummary('BASELINE', baseEnd, baseTrough);

flagsDef = struct('name', {}, 'dir', {});
flagsDef(1) = struct('name', 'FixSingletonKappa', 'dir', 'results/experiments/FixSingletonKappa');
flagsDef(2) = struct('name', 'PoolAcuteOnly',     'dir', 'results/experiments/PoolAcuteOnly');
flagsDef(3) = struct('name', 'CovidDecay',        'dir', 'results/experiments/CovidDecay');
flagsDef(4) = struct('name', 'DropPhiY',          'dir', 'results/experiments/DropPhiY');

econSel = ["gamma2", "nu", "alpha", "phisum", "phi2", "rhoU", "sig_k", "sig_Ustar"];

summaryRows = table();

for k = 1:numel(flagsDef)
    fname = flagsDef(k).name;
    fdir = flagsDef(k).dir;
    flagDirs = cell(1, numel(seeds));
    for i = 1:numel(seeds)
        flagDirs{i} = sprintf('%s/seed%d', fdir, seeds(i));
    end

    fprintf('\n================================================================\n');
    fprintf('FLAG: %s, same %d seeds\n', fname, numel(seeds));
    fprintf('================================================================\n');
    flagStats = computeVariantStats(flagDirs);
    writetable(flagStats.tbl, fullfile(outDir, sprintf('%s_stats.csv', fname)));
    printVariantSummary(fname, flagStats);

    % ---- economics join on shared param names --------------------------
    % NB: index against tbl.param (each tbl is sorted by neff ascending),
    % NOT against .names (original posterior_summary.csv order) -- the
    % two orders differ, so mixing them silently misaligns params.
    sharedNames = intersect(baseStats.names, flagStats.names, 'stable');
    sharedNames = sort(sharedNames);
    [~, ibShared] = ismember(sharedNames, baseStats.tbl.param);
    [~, ifShared] = ismember(sharedNames, flagStats.tbl.param);
    econTbl = table(sharedNames, ...
        baseStats.tbl.pooled_mean(ibShared), baseStats.tbl.post_sd(ibShared), baseStats.tbl.q50(ibShared), ...
        flagStats.tbl.pooled_mean(ifShared), flagStats.tbl.post_sd(ifShared), flagStats.tbl.q50(ifShared), ...
        flagStats.tbl.pooled_mean(ifShared) - baseStats.tbl.pooled_mean(ibShared), ...
        flagStats.tbl.q50(ifShared) - baseStats.tbl.q50(ibShared), ...
        'VariableNames', {'param', 'base_mean', 'base_sd', 'base_q50', ...
        'flag_mean', 'flag_sd', 'flag_q50', 'delta_mean', 'delta_q50'});

    % flag-only params (present in flag, absent from baseline -- e.g.
    % CovidDecay's s0_*/rho_c) get appended with base_* = NaN
    flagOnlyNames = sort(setdiff(flagStats.names, baseStats.names, 'stable'));
    if ~isempty(flagOnlyNames)
        [~, ifOnly] = ismember(flagOnlyNames, flagStats.tbl.param);
        nOnly = numel(flagOnlyNames);
        extra = table(flagOnlyNames, nan(nOnly, 1), nan(nOnly, 1), nan(nOnly, 1), ...
            flagStats.tbl.pooled_mean(ifOnly), flagStats.tbl.post_sd(ifOnly), flagStats.tbl.q50(ifOnly), ...
            nan(nOnly, 1), nan(nOnly, 1), ...
            'VariableNames', econTbl.Properties.VariableNames);
        econTbl = [econTbl; extra]; %#ok<AGROW>
    end
    writetable(econTbl, fullfile(outDir, sprintf('%s_economics.csv', fname)));

    % ---- latent states ---------------------------------------------------
    flagLatent = averageSmoothedStates(flagDirs);
    [flagEnd, flagTrough] = extractLatentPoints(flagLatent);
    printLatentSummary(fname, flagEnd, flagTrough);
    latTbl = buildLatentTable(baseEnd, baseTrough, flagEnd, flagTrough);
    writetable(latTbl, fullfile(outDir, sprintf('%s_latent.csv', fname)));

    % ---- console: key econ deltas ---------------------------------------
    fprintf('  -- key structural param deltas (flag - baseline, pooled q50) --\n');
    for j = 1:numel(econSel)
        row = econTbl(econTbl.param == econSel(j), :);
        if isempty(row)
            fprintf('    %-10s  not present in one of the two variants\n', econSel(j));
        else
            fprintf('    %-10s  base=%8.4f  flag=%8.4f  delta=%8.4f\n', ...
                econSel(j), row.base_q50, row.flag_q50, row.delta_q50);
        end
    end
    if ~isempty(flagOnlyNames)
        fprintf('  -- flag-only COVID params (q50) --\n');
        for j = 1:numel(flagOnlyNames)
            row = econTbl(econTbl.param == flagOnlyNames(j), :);
            fprintf('    %-20s  flag=%8.4f\n', flagOnlyNames(j), row.flag_q50);
        end
    end

    % ---- summary row ------------------------------------------------------
    econDeltaStr = string(econDeltaString(econTbl, econSel));
    latDeltaStr = string(latentDeltaString(baseEnd, baseTrough, flagEnd, flagTrough));
    summaryRows = [summaryRows; table(string(fname), flagStats.d, flagStats.maxRhatRank, ...
        flagStats.medianNeff, econDeltaStr, latDeltaStr, ...
        'VariableNames', {'flag', 'n_params', 'max_rhat_rank', 'median_neff', 'econ_deltas', 'latent_deltas'})]; %#ok<AGROW>
end

writetable(summaryRows, fullfile(outDir, 'summary.csv'));

fprintf('\n================================================================\n');
fprintf('BASELINE (recomputed, same 10 seeds): n=%d  max_rhat_rank=%.3f  median_neff=%.3f\n', ...
    baseStats.d, baseStats.maxRhatRank, baseStats.medianNeff);
fprintf('Outputs written under %s\n', outDir);

end

% ==========================================================================
function S = computeVariantStats(dirs)
% Convergence + pooled-posterior stats for one variant's G clouds,
% reproducing benchmarks/computeNeff.m's math exactly, but with names/d
% derived from the run's OWN posterior_summary.csv rather than assumed
% fixed (79-param) priors.
G = numel(dirs);
[names0, d] = readParamNames(dirs{1});

clouds = cell(G, 1);
clouds{1} = loadFinalSnapshot(dirs{1});
assert(size(clouds{1}.P, 2) == d, 'particle dim mismatch in %s', dirs{1});
for g = 2:G
    [namesG, dG] = readParamNames(dirs{g});
    assert(dG == d && all(strcmp(namesG, names0)), ...
        'param name/order mismatch between %s and %s', dirs{1}, dirs{g});
    clouds{g} = loadFinalSnapshot(dirs{g});
    assert(size(clouds{g}.P, 2) == d, 'particle dim mismatch in %s', dirs{g});
end

mu = zeros(G, d);
v = zeros(G, d);
for g = 1:G
    Sc = clouds{g};
    w = exp(Sc.logw - max(Sc.logw));
    w = w / sum(w);
    mu(g, :) = w' * Sc.P;
    v(g, :) = w' * (Sc.P - mu(g, :)).^2;
end

B_classic = var(mu, 0, 1);
W_classic = mean(v, 1);
okClassic = W_classic > 1e-12;
rhat_classic = ones(1, d);
rhat_classic(okClassic) = sqrt(1 + B_classic(okClassic) ./ W_classic(okClassic));

allP = [];
allW = [];
for g = 1:G
    Sc = clouds{g};
    w = exp(Sc.logw - max(Sc.logw));
    w = w / sum(w);
    allP = [allP; Sc.P]; %#ok<AGROW>
    allW = [allW; w / G]; %#ok<AGROW>
end
muPooled = (allW' * allP)';
Vpost = (allW' * (allP - muPooled').^2)';

std_mean = std(mu, 0, 1)';
neff = Vpost(:) ./ (std_mean.^2);
post_sd = sqrt(Vpost(:));

q = zeros(d, 3);
qset = [0.05, 0.5, 0.95];
for j = 1:d
    [xs, idxs] = sort(allP(:, j));
    cw = cumsum(allW(idxs));
    cw = cw / cw(end);
    for kq = 1:3
        f = find(cw >= qset(kq), 1, 'first');
        q(j, kq) = xs(f);
    end
end

% rank-normalized folded Rhat -- LOCAL rng stream, same recipe/seed as
% computeNeff.m so results are directly comparable/reproducible.
Ncommon = min(cellfun(@(Sc) size(Sc.P, 1), clouds));
stream = RandStream('mt19937ar', 'Seed', 20260716);
chainDraws = zeros(Ncommon, d, G);
for g = 1:G
    Sc = clouds{g};
    w = exp(Sc.logw - max(Sc.logw));
    w = w / sum(w);
    idxs = systematicResampleLocal(w, Ncommon, stream);
    chainDraws(:, :, g) = Sc.P(idxs, :);
end
Stot = Ncommon * G;
rhat_rank_bulk = nan(d, 1);
rhat_rank_folded = nan(d, 1);
pooledDraws = reshape(permute(chainDraws, [1 3 2]), Stot, d);
medPooled = median(pooledDraws, 1);
for j = 1:d
    theta = squeeze(chainDraws(:, j, :));
    zBulk = rankNormalizeLocal(theta(:));
    zBulk = reshape(zBulk, Ncommon, G);
    rhat_rank_bulk(j) = rhatOfChainsLocal(zBulk);

    zeta = abs(theta(:) - medPooled(j));
    zFold = rankNormalizeLocal(zeta);
    zFold = reshape(zFold, Ncommon, G);
    rhat_rank_folded(j) = rhatOfChainsLocal(zFold);
end
rhat_rank = max([rhat_rank_bulk, rhat_rank_folded], [], 2, 'omitnan');

tbl = table(string(names0), neff, post_sd, muPooled, q(:, 1), q(:, 2), q(:, 3), ...
    rhat_classic(:), rhat_rank_bulk, rhat_rank_folded, rhat_rank, ...
    'VariableNames', {'param', 'neff', 'post_sd', 'pooled_mean', 'q05', 'q50', 'q95', ...
    'rhat_classic', 'rhat_rank_bulk', 'rhat_rank_folded', 'rhat_rank'});
tbl = sortrows(tbl, 'neff', 'ascend');

S = struct();
S.names = string(names0);
S.d = d;
S.tbl = tbl;
S.maxRhatRank = max(rhat_rank, [], 'omitnan');
S.maxRhatClassic = max(rhat_classic(okClassic));
% omitnan: a genuinely fixed/degenerate param (exact-zero variance across
% ALL seeds, e.g. DropPhiY's phiy held at literal 0, sd=0 to machine
% precision) gives an unguarded 0/0 = NaN here (unlike e.g. sme_pieobs,
% whose near-fixed prior still carries ~1e-15 sd and so a huge-but-finite
% neff) -- a single such column must not silently NaN the whole variant's
% median via plain median().
S.medianNeff = median(neff, 'omitnan');
S.nDegenerateNeff = nnz(isnan(neff));
end

% ==========================================================================
function printVariantSummary(label, S)
fprintf('  %s: G=%d seeds, %d params\n', label, 10, S.d);
fprintf('    median N_eff = %.2f (omitnan)', S.medianNeff);
if S.nDegenerateNeff > 0
    fprintf('  [%d param(s) exact-degenerate across all seeds -> N_eff=NaN, excluded]', S.nDegenerateNeff);
end
fprintf('\n');
fprintf('    max rhat_classic = %.4f\n', S.maxRhatClassic);
fprintf('    max rhat_rank (bulk/folded) = %.4f  <-- headline\n', S.maxRhatRank);
end

% ==========================================================================
function [names, d] = readParamNames(dirPath)
T = readtable(fullfile(dirPath, 'posterior_summary.csv'));
names = string(T.param);
d = numel(names);
end

% ==========================================================================
function S = loadFinalSnapshot(d)
% Final snapshot = highest stage index parsed from the filename
% (particles_stage_NN.mat) -- copy-invariant (mtime is not, since cp
% resets it), identical convention to computeNeff.m.
snaps = dir(fullfile(d, 'particles_stage_*.mat'));
assert(~isempty(snaps), 'no particle snapshots in %s', d);
stageNo = zeros(numel(snaps), 1);
for kk = 1:numel(snaps)
    tok = regexp(snaps(kk).name, 'particles_stage_(\d+)\.mat', 'tokens', 'once');
    stageNo(kk) = str2double(tok{1});
end
[~, iL] = max(stageNo);
S = load(fullfile(d, snaps(iL).name));
assert(abs(S.phi - 1) < 1e-9, 'last snapshot in %s is not the phi=1 cloud', d);
end

% ==========================================================================
function idx = systematicResampleLocal(w, N, stream)
w = w(:);
positions = ((0:N - 1)' + rand(stream, 1)) / N;
cumW = cumsum(w);
cumW(end) = 1;
idx = zeros(N, 1);
i = 1;
for m = 1:N
    while positions(m) > cumW(i)
        i = i + 1;
    end
    idx(m) = i;
end
end

% ==========================================================================
function z = rankNormalizeLocal(x)
S = numel(x);
r = tiedrank(x);
z = norminv((r - 3/8) / (S + 1/4));
end

% ==========================================================================
function rhat = rhatOfChainsLocal(z)
chainMeans = mean(z, 1);
chainVars = var(z, 0, 1);
B = var(chainMeans, 0, 2);
W = mean(chainVars, 2);
if W < 1e-12
    rhat = NaN;
else
    rhat = sqrt(1 + B / W);
end
end

% ==========================================================================
function Tavg = averageSmoothedStates(dirs)
% Arithmetic mean across the G seeds' smoothed-state quantile bands
% (each seed's bands are already a within-seed posterior summary; simple
% cross-seed averaging is the inter-seed latent-state envelope this
% project uses elsewhere for latent states, which are far more
% seed-stable than the structural params).
G = numel(dirs);
T0 = readSmoothedStates(dirs{1});
nRows = height(T0);
nCols = width(T0) - 1;
acc = zeros(nRows, nCols);
for g = 1:G
    Tg = readSmoothedStates(dirs{g});
    assert(height(Tg) == nRows, 'row-count mismatch in %s', dirs{g});
    acc = acc + Tg{:, 2:end};
end
Tavg = T0;
Tavg{:, 2:end} = acc / G;
end

% ==========================================================================
function T = readSmoothedStates(dirPath)
% dates are dd/MM/uuuu (per CLAUDE.md / loadData convention); MATLAB's
% auto-detection flags this as ambiguous (many day<=12 rows also parse as
% MM/dd), so force the explicit format rather than trust auto-detect.
f = fullfile(dirPath, 'smoothed_states.csv');
opts = detectImportOptions(f);
opts = setvaropts(opts, 'date', 'InputFormat', 'dd/MM/uuuu');
T = readtable(f, opts);
end

% ==========================================================================
function [endRow, trough] = extractLatentPoints(Tavg)
% endRow: last row (end of sample). trough: the 2020Q2 row (gap COVID
% trough), matched by calendar year/quarter rather than string date, so
% it is robust to any date-format drift across runs.
endRow = Tavg(end, :);
yy = year(Tavg.date);
mm = month(Tavg.date);
idx = find(yy == 2020 & mm == 4, 1, 'first');
assert(~isempty(idx), '2020Q2 row not found');
trough = Tavg(idx, :);
end

% ==========================================================================
function printLatentSummary(label, endRow, trough)
fprintf('  %s latent (end of sample, %s):\n', label, string(endRow.date));
fprintf('    rstar            med=%7.3f  [%7.3f, %7.3f]\n', endRow.rstar_med, endRow.rstar_q05, endRow.rstar_q95);
fprintf('    nairu            med=%7.3f  [%7.3f, %7.3f]\n', endRow.nairu_med, endRow.nairu_q05, endRow.nairu_q95);
fprintf('    gap              med=%7.3f  [%7.3f, %7.3f]\n', endRow.gap_med, endRow.gap_q05, endRow.gap_q95);
fprintf('    trend_inflation  med=%7.3f  [%7.3f, %7.3f]\n', endRow.trend_inflation_med, endRow.trend_inflation_q05, endRow.trend_inflation_q95);
fprintf('  %s gap COVID trough (2020Q2): med=%7.3f  [%7.3f, %7.3f]\n', ...
    label, trough.gap_med, trough.gap_q05, trough.gap_q95);
end

% ==========================================================================
function T = buildLatentTable(baseEnd, baseTrough, flagEnd, flagTrough)
metrics = {'rstar_end', 'nairu_end', 'gap_end', 'trend_inflation_end', 'gap_2020Q2'};
baseMed = [baseEnd.rstar_med, baseEnd.nairu_med, baseEnd.gap_med, baseEnd.trend_inflation_med, baseTrough.gap_med];
baseQ05 = [baseEnd.rstar_q05, baseEnd.nairu_q05, baseEnd.gap_q05, baseEnd.trend_inflation_q05, baseTrough.gap_q05];
baseQ95 = [baseEnd.rstar_q95, baseEnd.nairu_q95, baseEnd.gap_q95, baseEnd.trend_inflation_q95, baseTrough.gap_q95];
flagMed = [flagEnd.rstar_med, flagEnd.nairu_med, flagEnd.gap_med, flagEnd.trend_inflation_med, flagTrough.gap_med];
flagQ05 = [flagEnd.rstar_q05, flagEnd.nairu_q05, flagEnd.gap_q05, flagEnd.trend_inflation_q05, flagTrough.gap_q05];
flagQ95 = [flagEnd.rstar_q95, flagEnd.nairu_q95, flagEnd.gap_q95, flagEnd.trend_inflation_q95, flagTrough.gap_q95];
delta = flagMed - baseMed;
T = table(string(metrics'), baseMed', baseQ05', baseQ95', flagMed', flagQ05', flagQ95', delta', ...
    'VariableNames', {'metric', 'base_med', 'base_q05', 'base_q95', 'flag_med', 'flag_q05', 'flag_q95', 'delta_med'});
end

% ==========================================================================
function s = econDeltaString(econTbl, econSel)
parts = strings(1, numel(econSel));
for j = 1:numel(econSel)
    row = econTbl(econTbl.param == econSel(j), :);
    if isempty(row)
        parts(j) = sprintf('%s=NA', econSel(j));
    else
        parts(j) = sprintf('%s: base=%.4f flag=%.4f d=%.4f', econSel(j), row.base_q50, row.flag_q50, row.delta_q50);
    end
end
s = strjoin(parts, '; ');
end

% ==========================================================================
function s = latentDeltaString(baseEnd, baseTrough, flagEnd, flagTrough)
s = sprintf(['rstar_end: base=%.3f flag=%.3f d=%.3f; nairu_end: base=%.3f flag=%.3f d=%.3f; ' ...
    'gap_end: base=%.3f flag=%.3f d=%.3f; trend_infl_end: base=%.3f flag=%.3f d=%.3f; ' ...
    'gap_2020Q2: base=%.3f flag=%.3f d=%.3f'], ...
    baseEnd.rstar_med, flagEnd.rstar_med, flagEnd.rstar_med - baseEnd.rstar_med, ...
    baseEnd.nairu_med, flagEnd.nairu_med, flagEnd.nairu_med - baseEnd.nairu_med, ...
    baseEnd.gap_med, flagEnd.gap_med, flagEnd.gap_med - baseEnd.gap_med, ...
    baseEnd.trend_inflation_med, flagEnd.trend_inflation_med, flagEnd.trend_inflation_med - baseEnd.trend_inflation_med, ...
    baseTrough.gap_med, flagTrough.gap_med, flagTrough.gap_med - baseTrough.gap_med);
end
