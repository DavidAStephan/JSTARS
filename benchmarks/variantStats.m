function S = variantStats(dirs)
%VARIANTSTATS  Convergence + pooled-posterior statistics for a variant set.
%
%   S = variantStats(dirs)
%
%   Computes convergence diagnostics (Rhat, N_eff) and pooled-posterior
%   statistics for a set of particle-cloud directories (typically from
%   multiple independent seeds of the same model specification).
%
%   This function extracts and generalizes the convergence-statistics logic
%   from benchmarks/compareCovidFlags.m's local computeVariantStats, making
%   it reusable for any variant set.
%
%   Input:
%       dirs -- cell array of directory paths, each containing:
%               - posterior_summary.csv (param names/order)
%               - particles_stage_*.mat (final stage clouds with P, logw, phi)
%
%   Output:
%       S -- struct with fields:
%           .names          (string array) parameter names
%           .d              (int) parameter count
%           .tbl            (table) per-param convergence + pooled posterior
%           .maxRhatRank    (double) max rank-normalized Rhat across params
%           .maxRhatClassic (double) max classical Rhat across params
%           .medianNeff     (double) median N_eff across params
%           .nDegenerateNeff (int) count of params with exactly-zero variance
%           .G              (int) number of clouds (seeds)
%
%   The computation reproduces benchmarks/compareCovidFlags.m's math exactly:
%     - rhat_classic = sqrt(1 + B/W) on per-seed weighted means/vars
%     - N_eff (Durham-Geweke/Herbst-Schorfheide) = V_post(pooled equal-
%       weight mixture) / std_across_seeds(weighted mean)^2
%     - rank-normalized, folded Rhat (Vehtari et al. 2021), each cloud
%       first resampled (systematic resampling) to a common length using a
%       LOCAL, fixed-seed RandStream (Seed=20260716); headline rhat_rank(j)
%       = max(bulk, folded).
%
%   See also compareCovidFlags, computeNeff.

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
S.nParams = d;
S.G = G;
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
