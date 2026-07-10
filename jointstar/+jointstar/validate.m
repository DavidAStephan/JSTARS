function tbl = validate(outDir, P)
%VALIDATE Compare an estimation run against the in-house baseline model.
%
%   tbl = jointstar.validate(outDir)      e.g. jointstar.validate('results/cp6')
%   tbl = jointstar.validate(outDir, P)   pass the prior spec if not default
%
%   Loads the final particle cloud from outDir and scores the posterior
%   against the baseline model's published POST-COVID estimates, mapping
%   parameter conventions: the baseline reports the SUM of the two
%   output-gap loadings per equation, and (phi1, phi2) separately with a
%   memo sum.  For the kappas the baseline's own notes warn the means are
%   unreliable (skewed, convergence problems) -- the MODE column is
%   compared instead.
%
%   Columns: baseline mean and 90% CI, new-posterior mean and 90% CI,
%   ratio of means, and two flags: overlap of the 90% intervals, and the
%   brief's "within factor of ~2" test on the point estimate.

if nargin < 2 || isempty(P)
    % detect the parameter space from the cloud width when possible
    P = [];
end

snaps = dir(fullfile(outDir, 'particles_stage_*.mat'));
assert(~isempty(snaps), 'no particle snapshots in %s', outDir);
[~, iLast] = max([snaps.datenum]);
S = load(fullfile(outDir, snaps(iLast).name));
assert(abs(S.phi - 1) < 1e-9, 'last snapshot is not the phi=1 cloud');
w = exp(S.logw - max(S.logw)); w = w / sum(w);

if isempty(P)
    for maker = {@() jointstar.horseshoePriors('HierKappa', true), ...
            @() jointstar.horseshoePriors(), @() jointstar.defaultPriors()}
        P = maker{1}();
        if P.d == size(S.P, 2), break; end
    end
    assert(P.d == size(S.P, 2), 'could not match a prior spec to the cloud');
end
ix = P.idx;
col = @(nm) S.P(:, ix.(nm));

% ---- baseline: published POST-COVID parameter estimates ---------------
% name, baseline mean (kappa: mode), lo, hi, new-run sample
B = {
 'U: gap loading sum (xi1+xi2)',   -0.52, -0.59, -0.42, col('xi1') + col('xi2');
 'U: gap AR (rhoU)',                0.20,  0.16,  0.25, col('rhoU');
 'pr: gap loading sum',             0.07,  0.03,  0.12, col('theta1') + col('theta2');
 'pr: gap AR (rhopr)',              0.88,  0.80,  0.94, col('rhopr');
 'hpp: gap loading sum',            0.16,  0.11,  0.23, col('lam1') + col('lam2');
 'hpp: gap AR (rhohpp)',            0.44,  0.34,  0.55, col('rhohpp');
 'Phillips: pi^e weight (gamma1)',  0.58,  0.45,  0.66, col('gamma1');
 'Phillips: slope (gamma2)',       -0.09, -0.14, -0.04, col('gamma2');
 'gap AR(1) (phi1)',                1.65,  1.57,  1.70, col('phisum') - col('phi2');
 'gap AR(2) (phi2)',               -0.69, -0.77, -0.60, col('phi2');
 'gap AR sum (phi1+phi2)',          0.96,  0.94,  0.98, col('phisum');
 'kappa c 2020-21 (vs mode)',       1.7,   1.2,   6.0,  col('kapc_2021');
 'kappa y 2020 (vs mode)',         10.7,   6.4,  38.0,  col('kapy_20');
 'kappa y 2021 (vs mode)',          2.1,   1.0,  16.0,  col('kapy_21');
 'kappa u 2020 (vs mode)',         16.0,  12.0,  29.0,  col('kapu_20');
 'kappa u 2021-22 (vs mode)',       4.3,   1.6,  10.0,  col('kapu_2122');
 'kappa pr 2020 (vs mode)',        10.0,   5.0,  16.0,  col('kappr_20');
 'kappa pr 2021-22 (vs mode)',      1.2,   1.1,   2.7,  col('kappr_2122');
 'kappa hpp 2020-22 (vs mode)',     3.3,   2.4,   5.8,  col('kaphpp_2022');
 'kappa pop 2020-21 (vs mode)',     1.3,   1.0,   4.4,  col('kappop_2021');
 'kappa pi 2020-23 (vs mode)',      2.3,   1.7,   3.8,  col('kappi_2023');
 'kappa k 2020 (vs mode)',          1.8,   1.3,  10.0,  col('kapk_20');
 'kappa k 2021 (vs mode)',          2.6,   1.7,  28.0,  col('kapk_21')};

n = size(B, 1);
newMean = zeros(n, 1); newLo = zeros(n, 1); newHi = zeros(n, 1);
for i = 1:n
    x = B{i, 5};
    newMean(i) = w' * x;
    q = wq(x, w, [0.05, 0.95]);
    newLo(i) = q(1); newHi(i) = q(2);
end
base = cell2mat(B(:, 2)); blo = cell2mat(B(:, 3)); bhi = cell2mat(B(:, 4));
ratio = newMean ./ base;
overlap = newLo <= bhi & blo <= newHi;
factor2 = abs(log(abs(newMean ./ base))) <= log(2) & sign(newMean) == sign(base);

tbl = table(string(B(:, 1)), base, blo, bhi, newMean, newLo, newHi, ...
    ratio, overlap, factor2, 'VariableNames', {'quantity', ...
    'base_mean', 'base_q05', 'base_q95', 'new_mean', 'new_q05', ...
    'new_q95', 'ratio', 'ci_overlap', 'within_factor2'});
end

function q = wq(x, w, ps)
[xs, i] = sort(x); cw = cumsum(w(i)); cw = cw / cw(end);
q = arrayfun(@(p) xs(find(cw >= p, 1, 'first')), ps);
end
