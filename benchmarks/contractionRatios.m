function T = contractionRatios(pooledCsv, outFile)
%CONTRACTIONRATIOS Compute prior-vs-posterior contraction ratios (weak ID honesty stats).
%
%   T = contractionRatios()
%   T = contractionRatios(pooledCsv, outFile)
%
%   Implements Lenza-Primiceri-style contraction ratio = posterior sd / prior sd
%   for each parameter. Flags weak identification and prior-dominance:
%   - 'data-identified':   contraction < 0.3  (data strongly shapes posterior)
%   - 'partially-id':      contraction in [0.3, 0.7]  (data + prior compete)
%   - 'prior-dominated':   contraction > 0.7  (posterior close to prior)
%   - 'fixed':             prior sd ~ 0 (calibrated constant, contraction NaN)
%
%   Inputs:
%     pooledCsv   = path to pooled-posterior CSV from jointstar.production
%                   default: 'results/production/pooled_posterior.csv'
%     outFile     = path to output CSV (param, prior_sd, post_sd, contraction,
%                   prior_q05, prior_q95, post_q05, post_q95, flag)
%                   default: 'results/production/contraction_ratios.csv'
%
%   Output:
%     T           = table with all columns; sorted by contraction (descending)
%
%   Uses NS=1e5 prior draws via jointstar.priorSample with local rng stream.

if nargin < 1 || isempty(pooledCsv)
    pooledCsv = 'results/production/pooled_posterior.csv';
end
if nargin < 2 || isempty(outFile)
    outFile = 'results/production/contraction_ratios.csv';
end

% ---- Set up local rng stream ----
rng_stream = RandStream('mt19937ar', 'Seed', 12345);
old_stream = RandStream.setGlobalStream(rng_stream);
cleanup = onCleanup(@() RandStream.setGlobalStream(old_stream));

% ---- Get default priors (HierKappa=true) ----
P = jointstar.defaultPriors('HierKappa', true);
fprintf('Loaded %d parameters\n', P.d);

% ---- Draw prior samples ----
NS = 1e5;
fprintf('Drawing %d prior samples...\n', NS);
Theta_prior = jointstar.priorSample(P, NS);

% ---- Compute prior statistics per parameter ----
prior_sd = std(Theta_prior, [], 1)';
prior_q05 = quantile(Theta_prior, 0.05, 1)';
prior_q95 = quantile(Theta_prior, 0.95, 1)';

% ---- Read pooled posterior CSV ----
fprintf('Reading pooled posterior from %s...\n', pooledCsv);
post_table = readtable(pooledCsv);
% param column in CSV becomes cell array of strings when read
param_col = post_table.param;
if iscell(param_col)
    param_names = param_col;
elseif isstring(param_col)
    param_names = cellstr(param_col);
else
    param_names = cellstr(string(param_col));
end
post_sd = post_table.sd;
post_q05 = post_table.q05;
post_q95 = post_table.q95;

% ---- Match posterior to prior by parameter name ----
% Preallocate output arrays
n_post = height(post_table);
prior_sd_matched = zeros(n_post, 1);
prior_q05_matched = zeros(n_post, 1);
prior_q95_matched = zeros(n_post, 1);
contraction = zeros(n_post, 1);
flag = repmat(string(""), n_post, 1);

% Match each posterior param to prior
for i = 1:n_post
    pname = param_names{i};
    % Find this parameter's index in the prior struct array
    idx = [];
    for j = 1:numel(P.params)
        if strcmp(P.params(j).name, pname)
            idx = j;
            break;
        end
    end

    if ~isempty(idx)
        prior_sd_matched(i) = prior_sd(idx);
        prior_q05_matched(i) = prior_q05(idx);
        prior_q95_matched(i) = prior_q95(idx);

        % Compute contraction: posterior sd / prior sd
        % Handle fixed/near-zero prior sd specially
        if prior_sd_matched(i) < 1e-10
            contraction(i) = NaN;
            flag(i) = "fixed";
        else
            contraction(i) = post_sd(i) / prior_sd_matched(i);
            % Assign flags based on contraction thresholds
            if contraction(i) < 0.3
                flag(i) = "data-identified";
            elseif contraction(i) <= 0.7
                flag(i) = "partially-identified";
            else
                flag(i) = "prior-dominated";
            end
        end
    else
        % Parameter in posterior not found in prior (shouldn't happen)
        warning('contractionRatios:paramMismatch', ...
            'Parameter %s not found in prior', pname);
        prior_sd_matched(i) = NaN;
        prior_q05_matched(i) = NaN;
        prior_q95_matched(i) = NaN;
        contraction(i) = NaN;
        flag(i) = "unknown";
    end
end

% ---- Build output table ----
% Ensure param_names is a cell array of strings
if isstring(param_names)
    param_names = cellstr(param_names);
elseif ~iscell(param_names)
    param_names = cellstr(param_names);
end

T = table(param_names, ...
    prior_sd_matched, post_sd, ...
    contraction, ...
    prior_q05_matched, prior_q95_matched, ...
    post_q05, post_q95, ...
    flag, ...
    'VariableNames', ...
    {'param', 'prior_sd', 'post_sd', 'contraction', ...
     'prior_q05', 'prior_q95', 'post_q05', 'post_q95', 'flag'});

% ---- Sort by contraction (descending), NaN last ----
% Separate sortable rows from non-sortable (fixed or unknown flagged)
is_fixed = flag == "fixed";
is_unknown = flag == "unknown";
is_sortable = ~is_fixed & ~is_unknown;

sortable_idx = find(is_sortable);
[~, sorted_order] = sort(contraction(sortable_idx), 'descend');
final_order = [sortable_idx(sorted_order); find(is_fixed); find(is_unknown)];

T = T(final_order, :);

% ---- Write output CSV ----
fprintf('Writing output to %s...\n', outFile);
writetable(T, outFile);

% ---- Print summary ----
fprintf('\n=== Contraction Summary ===\n');
fprintf('Total params: %d\n', n_post);
fprintf('Data-identified (< 0.3):       %3d\n', nnz(flag == "data-identified"));
fprintf('Partially-identified (0.3-0.7): %3d\n', nnz(flag == "partially-identified"));
fprintf('Prior-dominated (> 0.7):        %3d\n', nnz(flag == "prior-dominated"));
fprintf('Fixed/constant:                 %3d\n', nnz(flag == "fixed"));
fprintf('Unknown/mismatch:               %3d\n', nnz(flag == "unknown"));
fprintf('\nOutput: %s\n', outFile);

end
