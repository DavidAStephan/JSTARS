function [tbl, tauTbl] = horseshoeDiag(out, P, outDir)
%HORSESHOEDIAG Posterior shrinkage diagnostic for the L off-diagonals.
%
%   [tbl, tauTbl] = jointstar.horseshoeDiag(out, P, outDir)
%
%   out from jointstar.runSMC (via estimate with 'Horseshoe'), P from
%   jointstar.horseshoePriors.  Reports, for every off-diagonal of Lq
%   (transition shocks) and Lr (measurement errors): posterior median,
%   5%/95% quantiles, group, and whether the 90% band excludes zero --
%   i.e. which cross-shock correlations the data actually identifies vs
%   which the horseshoe shrinks away.  Writes hs_shrinkage.csv and two
%   heatmaps (median L, identified cells boxed) to outDir/figures.
%
%   Also prints and returns tauTbl: the posterior median and 5%/95%
%   quantiles of each group's global scale tau_g = sqrt(tau2_g)
%   (weighted quantiles over the particle cloud, same convention as the
%   L quantiles above).  The tau2_g columns are located via the prior
%   spec's own hyperparameter bookkeeping (P.hs.colTau2) -- not hardcoded
%   indices.  Written as tau_group.csv next to hs_shrinkage.csv.

hs = P.hs;
W = out.weights;
nL = numel(hs.colL);
med = zeros(nL, 1); q05 = med; q95 = med;
for u = 1:nL
    x = out.particles(:, hs.colL(u));
    [xs, i] = sort(x); cw = cumsum(W(i)); cw = cw / cw(end);
    q05(u) = xs(find(cw >= 0.05, 1));
    med(u) = xs(find(cw >= 0.50, 1));
    q95(u) = xs(find(cw >= 0.95, 1));
end
names = P.names(hs.colL)';
identified = sign(q05) == sign(q95) & abs(med) > 1e-3;
tbl = table(string(names), hs.groups, med, q05, q95, identified, ...
    'VariableNames', {'L_entry', 'group', 'median', 'q05', 'q95', ...
    'band_excludes_zero'});

% ---- group-level tau_g diagnostic --------------------------------------
G = numel(hs.colTau2);
tauMed = zeros(G, 1); tauQ05 = tauMed; tauQ95 = tauMed;
for g = 1:G
    x = sqrt(out.particles(:, hs.colTau2(g)));
    [xs, i] = sort(x); cw = cumsum(W(i)); cw = cw / cw(end);
    tauQ05(g) = xs(find(cw >= 0.05, 1));
    tauMed(g) = xs(find(cw >= 0.50, 1));
    tauQ95(g) = xs(find(cw >= 0.95, 1));
end
tauTbl = table(groupLabels(G), tauMed, tauQ05, tauQ95, ...
    'VariableNames', {'group', 'tau_median', 'tau_q05', 'tau_q95'});
disp(tauTbl);

if nargin > 2
    figDir = fullfile(outDir, 'figures');
    if ~isfolder(figDir), mkdir(figDir); end
    writetable(tbl, fullfile(outDir, 'hs_shrinkage.csv'));
    writetable(tauTbl, fullfile(outDir, 'tau_group.csv'));

    stateNames = cellstr(jointstar.ModelSpec.jointstarStates());
    obsNames = {'y', 'pi', 'wapop', 'U', 'lpr', 'hpp', 'k', 'pie_obs'};
    obsNames = obsNames(1:hs.mr);
    Mq = zeros(hs.mq); Iq = false(hs.mq);
    Mq(sub2ind([hs.mq, hs.mq], hs.LqRow, hs.LqCol)) = med(1:hs.nLq);
    Iq(sub2ind([hs.mq, hs.mq], hs.LqRow, hs.LqCol)) = identified(1:hs.nLq);
    Mr = zeros(hs.mr); Ir = false(hs.mr);
    Mr(sub2ind([hs.mr, hs.mr], hs.LrRow, hs.LrCol)) = med(hs.nLq + 1:end);
    Ir(sub2ind([hs.mr, hs.mr], hs.LrRow, hs.LrCol)) = identified(hs.nLq + 1:end);

    drawHeat(Mq, Iq, stateNames, ...
        'Transition-shock L_{ij}: posterior median (boxes: 90% band excludes 0)', ...
        fullfile(figDir, 'hs_heatmap_Lq.png'));
    drawHeat(Mr, Ir, obsNames, ...
        'Measurement-error L_{ij}: posterior median (boxes: 90% band excludes 0)', ...
        fullfile(figDir, 'hs_heatmap_Lr.png'));
end
end

function drawHeat(M, I, labels, ttl, fname)
n = size(M, 1);
f = figure('Color', 'w', 'Visible', 'off', 'Position', [0 0 620 560]);
lim = max(abs(M(:))) + 1e-6;
imagesc(M, [-lim, lim]);
colormap(bluewhitered());
colorbar;
set(gca, 'XTick', 1:n, 'XTickLabel', labels, 'YTick', 1:n, ...
    'YTickLabel', labels, 'TickLabelInterpreter', 'none');
xtickangle(45);
hold on;
[ri, ci] = find(I);
for k = 1:numel(ri)
    rectangle('Position', [ci(k) - 0.5, ri(k) - 0.5, 1, 1], ...
        'EdgeColor', 'k', 'LineWidth', 1.6);
end
title(ttl, 'Interpreter', 'tex');
exportgraphics(f, fname, 'Resolution', 150);
close(f);
end

function lbl = groupLabels(G)
% Descriptive names for the horseshoe global-scale groups, per the
% grouping documented in jointstar.horseshoePriors (measurement,
% trend, drift, cross-block trend<->drift).  Falls back to numeric
% labels if the group count ever differs from that fixed layout.
known = {'measurement', 'trend', 'drift', 'cross'};
if G == numel(known)
    lbl = string(known(:));
else
    lbl = string(arrayfun(@(g) sprintf('group%d', g), (1:G)', 'UniformOutput', false));
end
end

function cmap = bluewhitered()
n = 128;
up = [linspace(0, 1, n)', linspace(0.35, 1, n)', ones(n, 1)];
dn = [ones(n, 1), linspace(1, 0.35, n)', linspace(1, 0, n)'];
cmap = [up; dn(2:end, :)];
end
