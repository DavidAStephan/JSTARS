function atoms = blockAtoms(P, covCols)
%BLOCKATOMS Ridge/relation atom groups for structured MH blocking.
%
%   atoms = jointstar.blockAtoms(P, covCols)
%
%   Builds the atom list of DESIGN_transformed_kernel.md Section I2 (the
%   'StructuredBlocks' option): known ridge/relation parameter groups
%   that should never be split across MH blocks, plus a singleton atom
%   for every other mutated column.
%
%   Named dynamics groups are looked up BY NAME against P.names, keeping
%   only names that actually exist in this prior spec (silent drop --
%   e.g. a reduced spec missing a row's loading just contributes a
%   smaller/absent atom, never an error):
%     {phisum, phi2, nu, rho_rg} (rho_rg only present under 'RateGapAR' --
%     design e4d: it feeds the same output-gap IS equation as phi1/phi2/nu,
%     so it is co-blocked with them rather than left as its own singleton
%     atom on the raw kernel), {rhoU, xi1, xi2, phiu},
%     {rhopr, theta1, theta2, phipr}, {rhohpp, lam1, lam2, phihpp},
%     {rhok, chi1, chi2, phik}, {gzbar, gwbar} (or, under 'GTrendRotation',
%     {gtrend_sum, gtrend_split}), {sig_gz, sig_gw, sig_xi}, {s0_y, s0_u,
%     s0_pr, s0_k, rho_c} (present only under 'CovidDecay' -- co-blocks
%     the geometric-decay ridge, since rho_c is shared across all four
%     rows).
%
%   The hierarchical-COVID-kappa hyper groups (present only when P was
%   built with 'HierKappa', true) are built directly from P.kap, since
%   window-group membership is only recorded there (not recoverable from
%   parameter names alone): for each group g, one atom of
%   {members(g), kapHyp_lm_g, kapHyp_la_g}.
%
%   Every covCols column not claimed by a named/hierarchical atom is its
%   own singleton atom.  The union of all returned atoms, with no
%   overlaps, equals 1:numel(covCols).
%
%   covCols must be the same absolute-column list runSMC restricts MH to
%   (find(o.MutateIdx)).  Atoms are returned as RELATIVE indices into
%   covCols (1..numel(covCols)) -- the same convention
%   jointstar.blockPartition's (bperm, edgesB) output uses -- so
%   jointstar.blockPartitionAtoms can swap in for blockPartition with no
%   other code change.
%
%   See also jointstar.blockPartitionAtoms, jointstar.blockPartition,
%   jointstar.runSMC.

names = P.names;
d = P.d;
dm = numel(covCols);
absToRel = zeros(1, d);
absToRel(covCols) = 1:dm;

atomDefs = { ...
    {'phisum', 'phi2', 'nu', 'rho_rg'}, ... % rho_rg: 'RateGapAR' addition
    {'rhoU', 'xi1', 'xi2', 'phiu'}, ...
    {'rhopr', 'theta1', 'theta2', 'phipr'}, ...
    {'rhohpp', 'lam1', 'lam2', 'phihpp'}, ...
    {'rhok', 'chi1', 'chi2', 'phik'}, ...
    {'gzbar', 'gwbar'}, ...
    {'gtrend_sum', 'gtrend_split'}, ... % 'GTrendRotation' rename of the above
    {'sig_gz', 'sig_gw', 'sig_xi'}, ...
    {'s0_y', 's0_u', 's0_pr', 's0_k', 'rho_c'} ... % 'CovidDecay' addition
    };

atoms = {};
used = false(1, dm);
for a = 1:numel(atomDefs)
    rel = namesToRel(names, atomDefs{a}, absToRel);
    rel = rel(~used(rel));
    if ~isempty(rel)
        atoms{end + 1} = rel; %#ok<AGROW>
        used(rel) = true;
    end
end

if isfield(P, 'kap')
    kp = P.kap;
    for g = 1:kp.G
        memberAbs = kp.kapCols(kp.groups == g)';
        hyperAbs = [kp.lmCols(g), kp.laCols(g)];
        rel = absToRel([memberAbs, hyperAbs]);
        rel = rel(rel > 0);          % keep only columns actually in covCols
        rel = rel(~used(rel));
        if ~isempty(rel)
            atoms{end + 1} = rel; %#ok<AGROW>
            used(rel) = true;
        end
    end
end

rest = find(~used);
for k = 1:numel(rest)
    atoms{end + 1} = rest(k); %#ok<AGROW>
end
end

% ==========================================================================
function rel = namesToRel(names, wanted, absToRel)
rel = [];
for i = 1:numel(wanted)
    j = find(strcmp(names, wanted{i}), 1);
    if isempty(j), continue; end       % silently drop a name that doesn't exist
    r = absToRel(j);
    if r > 0, rel(end + 1) = r; end    %#ok<AGROW>  drop if not in covCols
end
end
