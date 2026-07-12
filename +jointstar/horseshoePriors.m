function P = horseshoePriors(varargin)
%HORSESHOEPRIORS Parameter space for Checkpoint 5+: base theta + LDL' off-
%   diagonals under grouped horseshoe priors.
%
%   P = jointstar.horseshoePriors()
%   P = jointstar.horseshoePriors('HierKappa', true)   % + Checkpoint 6
%
%   ALL pairs involving the output-gap shock (the entire c column of Lq)
%   are EXCLUDED from the estimated off-diagonals by owner ruling
%   (2026-07-10, see CHECKPOINT_05/06.md): free gap<->trend shock
%   correlations re-explain the cycle as correlated noise -- collapsing
%   the gap AR(2) persistence, the IS coefficient nu, and the Okun
%   loadings away from the work model's structure.  Correlations remain
%   estimable among trends, drifts, and measurement errors.  Edit
%   EXCLUDE_GAP / EXCLUDE below to revisit.
%
%   Extends jointstar.defaultPriors() with:
%     * 78 off-diagonals of Lq (unit-lower Cholesky of the 14 underlying
%       transition-shock covariances, state order; 91 pairs minus the 13
%       gap-column exclusions) and 28 off-diagonals of Lr (8 measurement
%       errors incl. pie_obs, observable order) — type 'hsL', prior
%       L_ij | lambda_ij, tau_g ~ N(0, tau_g^2 lambda_ij^2), with the
%       lambda/tau half-Cauchys truncated to [0.05, 10] (see
%       priorSample / horseshoeSample for why);
%     * per-coefficient local scales lambda^2 and auxiliaries nu, and
%       per-group global scales tau^2 with auxiliaries xi — type
%       'hsHyper', the Makalic-Schmidt inverse-gamma augmentation of
%       C+(0,1) priors.  These columns are Gibbs-updated between SMC
%       stages (jointstar.horseshoeMutate), never moved by MH:
%       P.mutateIdx excludes them.
%
%   Groups (own tau_g each):
%     1  measurement <-> measurement        (28 pairs)
%     2  trend shock <-> trend shock        (28 pairs: U*, pi^e, z*,
%                                            k*, wapop*, hpp*, pr*, xi)
%     3  drift shock <-> drift shock        (10 pairs: g^z..g^pr)
%     4  cross-block trend <-> drift        (40 pairs)
%
%   The existing sig_* / sme_* parameters become the D^(1/2) scales of
%   the factorisations (conditional, in-order scales; marginal variances
%   are the LDL' diagonal).  kappa / break multipliers keep scaling
%   MARGINAL variances via K*Sigma*K, preserving correlations.
%
%   P.hs carries the column bookkeeping; P.baseD the size of the base
%   block consumed by ModelSpec.jointstar.
%
%   See also jointstar.defaultPriors, jointstar.horseshoeMutate,
%   jointstar.hsUnpack.

ip = inputParser;
ip.addParameter('HierKappa', false);
ip.parse(varargin{:});

P = jointstar.defaultPriors('HierKappa', ip.Results.HierKappa);
P.baseD = P.d;
P.baseNames = P.names;

EXCLUDE = {};            % extra named exclusions, if any
EXCLUDE_GAP = true;      % owner ruling: no correlations with the gap shock

stateNames = cellstr(jointstar.ModelSpec.jointstarStates());
obsNames = {'y', 'pi', 'wapop', 'U', 'lpr', 'hpp', 'k', 'pie_obs'};
isDrift = ismember(stateNames, {'gz', 'gk', 'gw', 'ghpp', 'gpr'});

prm = P.params;
k = numel(prm);
    function add(name, type, p1, p2, lo, hi, init)
        k = k + 1;
        prm(k) = struct('name', name, 'type', type, 'p1', p1, 'p2', p2, ...
            'lo', lo, 'hi', hi, 'init', init);
    end

% ---- L off-diagonals ---------------------------------------------------
mq = numel(stateNames); mr = numel(obsNames);
groups = []; LqRow = []; LqCol = []; LrRow = []; LrCol = [];
u = 0;
for j = 1:mq
    for i = j + 1:mq
        nm = sprintf('Lq_%s_%s', stateNames{i}, stateNames{j});
        if ismember(nm, EXCLUDE), continue; end
        if EXCLUDE_GAP && (strcmp(stateNames{i}, 'c') || strcmp(stateNames{j}, 'c'))
            continue
        end
        u = u + 1;
        LqRow(u, 1) = i; LqCol(u, 1) = j;
        add(nm, 'hsL', 0, 0, -Inf, Inf, 0);
        if isDrift(i) && isDrift(j)
            groups(u, 1) = 3;
        elseif ~isDrift(i) && ~isDrift(j)
            groups(u, 1) = 2;
        else
            groups(u, 1) = 4;
        end
    end
end
nLq = u;
v = 0;
for j = 1:mr
    for i = j + 1:mr
        nm = sprintf('Lr_%s_%s', obsNames{i}, obsNames{j});
        if ismember(nm, EXCLUDE), continue; end
        v = v + 1;
        LrRow(v, 1) = i; LrCol(v, 1) = j;
        add(nm, 'hsL', 0, 0, -Inf, Inf, 0);
        groups(nLq + v, 1) = 1;
    end
end
nLr = v;
nL = nLq + nLr;
colL = P.baseD + (1:nL);

% ---- hyper columns (Gibbs-only) ----------------------------------------
for i = 1:nL, add(sprintf('hs_lam2_%03d', i), 'hsHyper', 0, 0, 0, Inf, 1); end
colLam2 = P.baseD + nL + (1:nL);
for i = 1:nL, add(sprintf('hs_nu_%03d', i), 'hsHyper', 0, 0, 0, Inf, 1); end
colNu = colLam2(end) + (1:nL);
G = 4;
for g = 1:G, add(sprintf('hs_tau2_g%d', g), 'hsHyper', 0, 0, 0, Inf, 0.01); end
colTau2 = colNu(end) + (1:G);
for g = 1:G, add(sprintf('hs_xi_g%d', g), 'hsHyper', 0, 0, 0, Inf, 1); end
colXi = colTau2(end) + (1:G);

P.params = prm;
P.names = {prm.name};
P.d = numel(prm);
P.idx = cell2struct(num2cell(1:P.d), P.names, 2);
P.mutateIdx = ~ismember({prm.type}, {'hsHyper', 'fixed'});
P.hs = struct('colL', colL, 'colLam2', colLam2, 'colNu', colNu, ...
    'colTau2', colTau2, 'colXi', colXi, 'groups', groups, 'G', G, ...
    'nLq', nLq, 'nLr', nLr, 'mq', mq, 'mr', mr, ...
    'LqRow', LqRow, 'LqCol', LqCol, 'LrRow', LrRow, 'LrCol', LrCol);
end
