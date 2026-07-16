function T = computeNeff(seedDirs, outFile)
% COMPUTENEFF  Field-native SMC convergence report across independent seeds.
%
%   T = computeNeff(seedDirs, outFile)
%
% seedDirs : cellstr of run directories, each containing one or more
%            particles_stage_*.mat snapshots (S.P [N x d], S.logw [N x 1],
%            S.phi). The LAST snapshot (by file datenum, same convention
%            as +jointstar/production.m's loadFinalSnapshot) is taken from
%            each directory and must have phi == 1 (asserted).
% outFile  : path to write the CSV report to.
%
% Methodology
% -----------
% This is a cross-SEED agreement report for an SMC particle posterior, not
% an MCMC trace diagnostic -- there is no time ordering within a cloud.
% Two complementary families of statistic are computed per parameter j:
%
%  (1) Herbst-Schorfheide (2014, Sec 4.3) / Durham-Geweke (2012) N_eff:
%        mu_g(j)     = weighted posterior mean of seed g
%        std_mean(j) = STD across the G seeds of mu_g(j)   (Durham-Geweke
%                       across-run dispersion of the estimator)
%        V_post(j)   = weighted variance of the POOLED equal-weight (1/G
%                       each) mixture of all G clouds (identical pooling
%                       math to production.m's poolPosteriorLocal)
%        neff(j)     = V_post(j) / std_mean(j)^2
%      This measures how many independent-equivalent draws the G seeds'
%      combined evidence is worth for estimating the posterior mean of j.
%
%  (2) Rank-normalized, FOLDED R-hat (Vehtari, Gelman, Simpson, Carpenter,
%      Burkner 2021 "Rank-normalization, folding, and localization"),
%      treating each seed's cloud as one "chain":
%        - Each cloud is first resampled (systematic resampling, weights
%          w = exp(logw-max)/sum) to an equal-weight sample of common
%          length N (N = the minimum native particle count across the G
%          seeds), using a LOCAL, fixed-seed RandStream so the result is
%          reproducible and the caller's global rng state is left
%          untouched.
%        - bulk: pool the G*N resampled draws, rank-transform per
%          parameter with tiedrank (ties -> average/mid rank), map to
%          z = norminv((r - 3/8)/(S + 1/4)) with S = G*N, then compute the
%          classical Rhat = sqrt(1 + B/W) on z across the G chains (B =
%          between-chain variance of the chains' mean z, W = mean of the
%          chains' within-chain variance of z -- same formula family as
%          production.m's rhatLocal, just applied to z instead of theta).
%        - folded: same recipe applied to zeta = abs(theta - median(all
%          pooled resampled draws)) instead of theta.
%        - rhat_rank(j) = max(bulk, folded).
%      Near-constant columns (W < 1e-12 in the respective z/zeta space,
%      e.g. any calibrated/'fixed'-prior parameter) are reported as NaN
%      for that half; rhat_rank is NaN only if BOTH halves are NaN.
%
%      DEVIATION FROM VEHTARI ET AL. (documented, deliberate): the
%      standard recipe additionally SPLITS each chain in half before
%      rank-normalizing, to catch within-chain non-stationarity /
%      trends. That step is omitted here on purpose: an SMC particle
%      cloud at a fixed tempering stage phi is EXCHANGEABLE -- the
%      particles carry no time/iteration ordering, so splitting a cloud
%      in half yields two halves that are (in expectation) identically
%      distributed by construction and the split step can detect
%      nothing. The R-hat computed here is therefore a CROSS-SEED
%      agreement statistic (does the SMC posterior estimate itself
%      change with the seed?), not a proof of MCMC-style convergence to
%      a stationary distribution -- there is no stationary chain here to
%      converge to in the first place.
%
%      rhat_classic (also reported, for continuity with
%      convergence_rhat.csv) is the plain, non-rank formula sqrt(1+B/W)
%      applied directly to the per-seed weighted means/variances mu_g,
%      v_g -- i.e. exactly production.m's rhatLocal computation,
%      reproduced here so the two reports can be cross-checked.
%
% NOISE WARNING: with only G=3 seeds (as in the validation run below),
% std_mean(j) is the sample std of 3 numbers, i.e. effectively a
% chi-square statistic on 2 degrees of freedom -- extremely noisy, so
% neff at G=3 should be read as "order of magnitude" only, not a precise
% count. The real report is intended to run with G=20 seeds, where
% std_mean is far better behaved.
%
% Output CSV columns (sorted by neff ascending):
%   param, neff, std_mean, post_sd, mcse_over_sd, rhat_classic,
%   rhat_rank_bulk, rhat_rank_folded

P = jointstar.defaultPriors('HierKappa', true);
names = P.names(:);
d = P.d;
G = numel(seedDirs);

% ---- load final (phi==1) snapshot per seed -----------------------------
clouds = cell(G, 1);
for g = 1:G
    clouds{g} = loadFinalSnapshot(seedDirs{g});
end

% ---- per-seed weighted mean / variance (native particle counts) -------
mu = zeros(G, d);
v  = zeros(G, d);
for g = 1:G
    S = clouds{g};
    w = exp(S.logw - max(S.logw));
    w = w / sum(w);
    mu(g, :) = w' * S.P;
    v(g, :)  = w' * (S.P - mu(g, :)).^2;
end

% ---- rhat_classic: exact production.m / rhatLocal formula --------------
B_classic = var(mu, 0, 1);
W_classic = mean(v, 1);
okClassic = W_classic > 1e-12;
rhat_classic = ones(1, d);
rhat_classic(okClassic) = sqrt(1 + B_classic(okClassic) ./ W_classic(okClassic));

% ---- pooled equal-weight (1/G each) mixture -> V_post ------------------
allP = [];
allW = [];
for g = 1:G
    S = clouds{g};
    w = exp(S.logw - max(S.logw));
    w = w / sum(w);
    allP = [allP; S.P]; %#ok<AGROW>
    allW = [allW; w / G]; %#ok<AGROW>
end
muPooled = (allW' * allP)';
Vpost = (allW' * (allP - muPooled').^2)';

% ---- Durham-Geweke / Herbst-Schorfheide N_eff ---------------------------
std_mean = std(mu, 0, 1)';          % sample std across G seeds (ddof=1), column
neff = Vpost(:) ./ (std_mean.^2);
post_sd = sqrt(Vpost(:));
mcse_over_sd = std_mean ./ post_sd;

% ---- rank-normalized folded R-hat ---------------------------------------
% common resample length = min native particle count across seeds
Ncommon = min(cellfun(@(S) size(S.P, 1), clouds));
stream = RandStream('mt19937ar', 'Seed', 20260716);  % local, fixed, reproducible

chainDraws = zeros(Ncommon, d, G);
for g = 1:G
    S = clouds{g};
    w = exp(S.logw - max(S.logw));
    w = w / sum(w);
    idx = systematicResample(w, Ncommon, stream);
    chainDraws(:, :, g) = S.P(idx, :);
end

Stot = Ncommon * G;
rhat_rank_bulk = nan(d, 1);
rhat_rank_folded = nan(d, 1);

% pooled raw draws across all chains, per parameter, for the folding median
pooledDraws = reshape(permute(chainDraws, [1 3 2]), Stot, d);  % [Stot x d]
medPooled = median(pooledDraws, 1);

for j = 1:d
    theta = squeeze(chainDraws(:, j, :));   % [Ncommon x G]

    % bulk: rank-normalize theta across all Stot pooled draws
    zBulk = rankNormalize(theta(:));
    zBulk = reshape(zBulk, Ncommon, G);
    rhat_rank_bulk(j) = rhatOfChains(zBulk);

    % folded: rank-normalize |theta - global median| across pooled draws
    zeta = abs(theta(:) - medPooled(j));
    zFold = rankNormalize(zeta);
    zFold = reshape(zFold, Ncommon, G);
    rhat_rank_folded(j) = rhatOfChains(zFold);
end

% NaN only if BOTH halves are NaN (max(...,'omitnan') along dim 2 handles
% this directly: an all-NaN row stays NaN, a row with one NaN falls back
% to the other value).
rhat_rank = max([rhat_rank_bulk, rhat_rank_folded], [], 2, 'omitnan');

% ---- assemble output table ---------------------------------------------
T = table(string(names), neff, std_mean, post_sd, mcse_over_sd, ...
    rhat_classic(:), rhat_rank_bulk, rhat_rank_folded, ...
    'VariableNames', {'param', 'neff', 'std_mean', 'post_sd', ...
    'mcse_over_sd', 'rhat_classic', 'rhat_rank_bulk', 'rhat_rank_folded'});
T = sortrows(T, 'neff', 'ascend');
writetable(T, outFile);

% ---- console summary -----------------------------------------------------
maxRhatClassic = max(rhat_classic(okClassic));
maxRhatRank = max(rhat_rank, [], 'omitnan');
fprintf('computeNeff: G = %d seeds, %d parameters\n', G, d);
fprintf('  neff: min = %.2f, median = %.2f, count(neff<100) = %d\n', ...
    min(neff), median(neff), nnz(neff < 100));
fprintf('  max rhat_classic = %.4f\n', maxRhatClassic);
fprintf('  max rhat_rank (bulk/folded) = %.4f\n', maxRhatRank);
fprintf('  -> %s\n', outFile);

end

% ==========================================================================
function S = loadFinalSnapshot(d)
% Final snapshot = highest stage index parsed from the filename
% (particles_stage_NN.mat), then asserted phi == 1.  This is more robust
% than production.m's datenum-based pick, which mis-selects when a results
% directory has been copied (cp resets all mtimes) -- selecting by the
% stage number in the name is copy-invariant.
snaps = dir(fullfile(d, 'particles_stage_*.mat'));
assert(~isempty(snaps), 'no particle snapshots in %s', d);
stageNo = zeros(numel(snaps), 1);
for k = 1:numel(snaps)
    tok = regexp(snaps(k).name, 'particles_stage_(\d+)\.mat', 'tokens', 'once');
    stageNo(k) = str2double(tok{1});
end
[~, iL] = max(stageNo);
S = load(fullfile(d, snaps(iL).name));
assert(abs(S.phi - 1) < 1e-9, 'last snapshot in %s is not the phi=1 cloud', d);
end

% ==========================================================================
function idx = systematicResample(w, N, stream)
% Systematic resampling of an N-length index vector from weights w (must
% sum to 1), drawing its single random offset from the supplied LOCAL
% RandStream so the global rng is never touched.
w = w(:);
positions = ((0:N-1)' + rand(stream, 1)) / N;
cumW = cumsum(w);
cumW(end) = 1;  % guard against floating-point roundoff
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
function z = rankNormalize(x)
% Fractional-rank normal transform (Vehtari et al. 2021 eq. for rank
% normalization), with tiedrank handling ties via average/mid rank.
S = numel(x);
r = tiedrank(x);
z = norminv((r - 3/8) / (S + 1/4));
end

% ==========================================================================
function rhat = rhatOfChains(z)
% z is [N x G] (N draws per chain, G chains, equal weight within a chain).
% Classical Rhat = sqrt(1 + B/W), same formula family as production.m's
% rhatLocal; B = between-chain variance of chain means, W = mean
% within-chain variance. Near-constant columns (W < 1e-12) -> NaN.
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
