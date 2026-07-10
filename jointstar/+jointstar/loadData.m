function dat = loadData(csvPath, varargin)
%LOADDATA Read the quarterly macro CSV and build model-ready series.
%
%   dat = jointstar.loadData(csvPath)
%   dat = jointstar.loadData(csvPath, 'PieObs', false)
%
%   'PieObs' (default true, Checkpoint 7+): append the constructed
%   inflation-expectations series pi_e as an 8th observable -- a direct
%   noisy measurement of the trend-inflation state, pi_e_t = pie_t + eps.
%   This replaces the brief's Consensus long-run CPI survey, which is
%   proprietary and unavailable off-site.  (pi_e is also used to deflate
%   the cash rate; the real rate enters the gap transition as exogenous
%   data, so the two uses are distinct.)  Set false to reproduce the
%   7-observable model of Checkpoints 2-6.
%
%   Transformations (agreed with the model owner, 2026-07-10):
%     * wapop = lf / (part_rate/100)    (working-age population)
%     * inflation = 400 * dlog(trimmed-mean CPI index)  (quarterly
%       annualised) -- the `inf` column is the index level
%     * real cash rate r = cash_rate_pa - pi_e (expectations supplied)
%     * stringency: blank = 0
%   Units convention: "100 log" for quantities (so gaps are in per cent),
%   rates in percentage points, inflation and interest rates in annual
%   per cent.  The participation observable is lpr = 100*log(pr/100) and
%   the wapop observable 100*log(wapop), which makes the log labour-market
%   identities lf* = wapop* + pr*, er* = -U* internally consistent (U in
%   pp approximates -100*log(1-u/100)).
%
%   The sample is trimmed to start at the first y_nf observation
%   (1974Q3): before that only the capital stock is observed and carrying
%   15 years of one-series state evolution buys little.
%
%   Observable order (rows of dat.y):
%     1 y     100*log real non-farm GDP
%     2 pi    trimmed-mean inflation, quarterly annualised
%     3 wapop 100*log working-age population
%     4 U     unemployment rate, pp
%     5 lpr   100*log(participation rate/100)
%     6 hpp   100*log average hours per worker
%     7 k     100*log real capital stock
%
%   Rows are pre-masked (set to NaN) wherever a data ingredient of that
%   measurement equation is unavailable: each error-correction row needs
%   its own lag, and the Phillips-curve row needs pi_{t-1} and U_t.
%
%   Also returned: exogenous series (d, dlag, r1, r2, hasIS), the
%   intercept ingredients (lags with NaN->0 after masking), date vector,
%   regime indices (pre-1984Q1, pre-1993Q1, COVID kappa windows), and
%   anchor values for the initial state mean.

% Parse the header by hand: the file has trailing empty columns (Excel
% export) that defeat detectImportOptions, and the day-first dates are
% ambiguous to autodetection.
fid = fopen(csvPath, 'r');
hdr = strsplit(fgetl(fid), ',', 'CollapseDelimiters', false);
fclose(fid);
named = find(~cellfun(@isempty, strtrim(hdr)));
opts = delimitedTextImportOptions('NumVariables', numel(hdr), ...
    'DataLines', 2, 'ExtraColumnsRule', 'ignore');
vnames = matlab.lang.makeValidName( ...
    [strtrim(hdr(named)), compose('unused%d', 1:(numel(hdr) - numel(named)))]);
opts.VariableNames = vnames;
opts = setvartype(opts, vnames, 'double');
opts = setvartype(opts, vnames{1}, 'datetime');
opts = setvaropts(opts, vnames{1}, 'InputFormat', 'dd/MM/uuuu');
opts.SelectedVariableNames = vnames(1:numel(named));
tb = readtable(csvPath, opts);
tb.Properties.VariableNames{1} = 'date';
req = {'y_nf', 'avg_hours', 'part_rate', 'u', 'lf', 'cap', ...
    'cash_rate_pa', 'stringency', 'inf', 'pi_e'};
missingCols = setdiff(req, tb.Properties.VariableNames);
if ~isempty(missingCols)
    error('jointstar:badData', 'data file is missing column(s): %s', ...
        strjoin(missingCols, ', '));
end

% drop trailing junk rows (Excel export artifacts): no date or no data
bad = isnat(tb.date);
tb(bad, :) = [];

% quarterly annualised trimmed-mean inflation from the index level
pi_tm = [NaN; 400 * diff(log(tb.inf))];

% trim sample to first GDP observation
t0 = find(~isnan(tb.y_nf), 1, 'first');
tb = tb(t0:end, :);
pi_tm = pi_tm(t0:end);
T = height(tb);
dates = tb.date';

Y   = 100 * log(tb.y_nf)';
K   = 100 * log(tb.cap)';
HP  = 100 * log(tb.avg_hours)';
LPR = 100 * log(tb.part_rate / 100)';
W   = 100 * log(tb.lf ./ (tb.part_rate / 100))';
U   = tb.u';
PI  = pi_tm';
r   = (tb.cash_rate_pa - tb.pi_e)';

D = tb.stringency';
D(isnan(D)) = 0;

lag = @(x) [NaN, x(1:end - 1)];
PIlag = lag(PI); Wlag = lag(W); Ulag = lag(U);
LPRlag = lag(LPR); HPlag = lag(HP); Klag = lag(K);
r1 = lag(r); r2 = lag(r1);
Dlag = lag(D); Dlag(1) = 0;

ipd = inputParser;
ipd.addParameter('PieObs', true);
ipd.parse(varargin{:});

% ---- observation matrix with equation-level masking -------------------
y = [Y; PI; W; U; LPR; HP; K];
y(2, isnan(PIlag) | isnan(U)) = NaN;     % Phillips needs pi_{t-1}, U_t
y(3, isnan(Wlag)) = NaN;
y(4, isnan(Ulag)) = NaN;
y(5, isnan(LPRlag)) = NaN;
y(6, isnan(HPlag)) = NaN;
y(7, isnan(Klag)) = NaN;
obsNames = {'y', 'pi', 'wapop', 'U', 'lpr', 'hpp', 'k'};
if ipd.Results.PieObs
    y = [y; tb.pi_e'];                   % direct measurement of pie state
    obsNames{end + 1} = 'pie_obs';
end

% intercept ingredients: NaN -> 0 is safe because the corresponding row
% is already masked wherever an ingredient was missing
z0 = @(x) fillmissing(x, 'constant', 0);
ing = struct('PIlag', z0(PIlag), 'Ulev', z0(U), 'Wlag', z0(Wlag), ...
    'Ulag', z0(Ulag), 'LPRlag', z0(LPRlag), 'HPlag', z0(HPlag), ...
    'Klag', z0(Klag), 'Dlag', Dlag);

% IS term availability: both r_{t-1} and r_{t-2} observed
hasIS = ~isnan(r1) & ~isnan(r2);
r1 = z0(r1); r2 = z0(r2);

% ---- regime and COVID-kappa windows -----------------------------------
q = @(ystr) datetime(ystr, 'InputFormat', 'dd/MM/yyyy');
pre84 = dates < q('01/01/1984');
pre93 = dates < q('01/01/1993');
inw = @(d1, d2) dates >= q(d1) & dates <= q(d2);
% All windows start 2020Q2 where "2020" is named and are hard-capped at
% 2023Q3: kappa = 1 for every variable from 2023Q4 onward (brief override
% of the model specification's inflation window).
win = struct( ...
    'w2020',   inw('01/04/2020', '01/10/2020'), ...
    'w2021',   inw('01/01/2021', '01/10/2021'), ...
    'w2020_21', inw('01/04/2020', '01/10/2021'), ...
    'w2021_22', inw('01/01/2021', '01/10/2022'), ...
    'w2020_22', inw('01/04/2020', '01/10/2022'), ...
    'w2020_23', inw('01/04/2020', '01/07/2023'));

% ---- anchors for the initial state mean --------------------------------
firstv = @(x) x(find(~isnan(x), 1, 'first'));
anchor = struct('Y0', firstv(Y), 'K0', firstv(K), 'W0', firstv(W), ...
    'HP0', firstv(HP), 'LPR0', firstv(LPR), 'U0', firstv(U), ...
    'PIE0', 10);   % mid-70s expectations; wide P1 makes this weak

dat = struct('y', y, 'dates', dates, 'T', T, 'd', D, 'r1', r1, 'r2', r2, ...
    'hasIS', hasIS, 'ing', ing, 'pre84', pre84, 'pre93', pre93, ...
    'win', win, 'anchor', anchor, 'obsNames', {obsNames});
end
