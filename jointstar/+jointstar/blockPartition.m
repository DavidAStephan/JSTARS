function [bperm, edgesB] = blockPartition(dm, covCols, nB, ridgeAtoms, atomGroups)
%BLOCKPARTITION Random block partition of MH covariance columns.
%
%   [bperm, edgesB] = jointstar.blockPartition(dm, covCols, nB)
%   [bperm, edgesB] = jointstar.blockPartition(dm, covCols, nB, ridgeAtoms, atomGroups)
%
%   Default (ridgeAtoms false, or omitted): identical to the classic
%   partition -- bperm = randperm(dm), edgesB = round(linspace(0, dm, nB+1)).
%   Block b covers columns covCols(bperm(edgesB(b)+1:edgesB(b+1))).
%
%   With ridgeAtoms true, each column-index group in atomGroups (absolute
%   theta-column indices, matching covCols) is kept indivisible: all of
%   its present members always land in the same MH block.  Members not
%   present in covCols are dropped; an atom left with fewer than 2
%   present members dissolves into ordinary singleton columns.
%
%   Implementation: build a permutation over "units" (atoms + singleton
%   columns), shuffle the units, unroll into bperm (atoms occupy a
%   contiguous run in bperm by construction), then compute the usual
%   evenly-spaced block edges and snap any edge that would fall inside an
%   atom's run to that atom's nearer boundary.  Block count and target
%   size (~40 columns) are unchanged from the default partition.
%
%   See also jointstar.runSMC.

if nargin < 4 || isempty(ridgeAtoms), ridgeAtoms = false; end
if nargin < 5, atomGroups = {}; end

if ridgeAtoms && ~isempty(atomGroups)
    [bperm, atomSpans] = unitPermutation(dm, covCols, atomGroups);
    edgesB = snapEdges(round(linspace(0, dm, nB + 1)), atomSpans);
else
    bperm = randperm(dm);
    edgesB = round(linspace(0, dm, nB + 1));
end
end

% ======================================================================
function [bperm, atomSpans] = unitPermutation(dm, covCols, atomGroups)
% Resolve atoms (absolute column ids) to positions within covCols, drop
% members absent from covCols, dissolve atoms with < 2 surviving members,
% and never let a column belong to more than one atom.
inAtom = false(1, dm);
atomPos = {};
for a = 1:numel(atomGroups)
    pos = find(ismember(covCols, atomGroups{a}));
    pos = pos(~inAtom(pos));
    if numel(pos) < 2
        continue
    end
    atomPos{end + 1} = pos; %#ok<AGROW>
    inAtom(pos) = true;
end
singles = num2cell(find(~inAtom));
units = [atomPos, singles];
nu = numel(units);
isAtomUnit = false(1, nu);
isAtomUnit(1:numel(atomPos)) = true;

uperm = randperm(nu);
bperm = zeros(1, dm);
atomSpans = zeros(numel(atomPos), 2);
ptr = 0;
for u = 1:nu
    idx = uperm(u);
    members = units{idx};
    len = numel(members);
    bperm(ptr + 1:ptr + len) = members;
    if isAtomUnit(idx)
        atomSpans(idx, :) = [ptr + 1, ptr + len];
    end
    ptr = ptr + len;
end
end

function edges = snapEdges(edges, atomSpans)
% Snap interior edges that fall inside an atom's contiguous run (in
% bperm-index space) to that atom's nearer boundary, then clamp for
% monotonicity so edges never decrease left to right.
for k = 2:numel(edges) - 1
    e = edges(k);
    for a = 1:size(atomSpans, 1)
        s = atomSpans(a, 1) - 1;   % edge value just before the atom's run
        f = atomSpans(a, 2);       % edge value just after the atom's run
        if e > s && e < f
            if (e - s) <= (f - e)
                e = s;
            else
                e = f;
            end
        end
    end
    edges(k) = max(e, edges(k - 1));
end
end
