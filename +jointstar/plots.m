function figs = plots(results, outDir)
%PLOTS Fan charts for the headline latent series.
%
%   figs = jointstar.plots(results)          display only
%   figs = jointstar.plots(results, outDir)  also save PNGs to outDir
%
%   results is the output of jointstar.estimate (needs .states, .dat).
%   Produces r*, output gap, trend inflation, NAIRU (vs observed U), and
%   potential output (vs observed GDP) with 90% bands.

st = results.states;
dates = st.date;
figs = gobjects(0);

panels = { ...
    'rstar',           'Neutral real rate r* (% pa)',        [];
    'gap',             'Output gap c* (%)',                  [];
    'trend_inflation', 'Trend inflation \pi^e (% pa)',       [];
    'nairu',           'NAIRU U* (pp) and observed U',       results.dat.y(4, :);
    'taustar',         'Potential output (100 log) and GDP', results.dat.y(1, :)};

for i = 1:size(panels, 1)
    [nm, ttl, obs] = panels{i, :};
    f = figure('Name', nm, 'Color', 'w', 'Visible', 'off');
    lo = st.([nm '_q05']); md = st.([nm '_med']); hi = st.([nm '_q95']);
    ok = ~isnan(md);
    fill([dates(ok); flipud(dates(ok))], [lo(ok); flipud(hi(ok))], ...
        [0.75 0.85 1.0], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
    hold on;
    plot(dates, md, 'b-', 'LineWidth', 1.4);
    if ~isempty(obs)
        plot(dates, obs', 'k:', 'LineWidth', 0.9);
        legend({'90% band', 'median', 'observed'}, 'Location', 'best');
    else
        legend({'90% band', 'median'}, 'Location', 'best');
    end
    if strcmp(nm, 'gap'), yline(0, 'k-', 'Alpha', 0.3); end
    title(ttl); grid on; box off;
    figs(end + 1) = f; %#ok<AGROW>
    if nargin > 1
        if ~isfolder(outDir), mkdir(outDir); end
        exportgraphics(f, fullfile(outDir, [nm '.png']), 'Resolution', 150);
    end
end
set(figs, 'Visible', 'on');
end
