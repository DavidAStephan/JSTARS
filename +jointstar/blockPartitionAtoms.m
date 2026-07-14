function [bperm, edgesB] = blockPartitionAtoms(atoms, nB)
%BLOCKPARTITIONATOMS Random atom-based block partition of MH covariance
%columns ('StructuredBlocks' option), preserving named ridge/relation
%atoms (jointstar.blockAtoms) as indivisible units.
%
%   [bperm, edgesB] = jointstar.blockPartitionAtoms(atoms, nB)
%
%   atoms: 1xK cell array of RELATIVE column-index row vectors (as
%   returned by jointstar.blockAtoms), partitioning 1:dm exactly once
%   each (dm = the total column count across all atoms).
%
%   Atoms are randomly permuted (randperm(K)) then greedily packed into
%   nB blocks, each atom going whole into whichever block currently has
%   the fewest columns -- balances block size by parameter count while
%   never splitting an atom across blocks.
%
%   Returns (bperm, edgesB) in the SAME contract as
%   jointstar.blockPartition: block b covers columns
%   bperm(edgesB(b)+1:edgesB(b+1)).  This lets callers (jointstar.runSMC)
%   select between the two partition strategies with no other code
%   change downstream of the partition call.
%
%   See also jointstar.blockAtoms, jointstar.blockPartition,
%   jointstar.runSMC.

K = numel(atoms);
ord = randperm(K);
sizes = zeros(1, nB);
bucket = cell(1, nB);
for b = 1:nB, bucket{b} = []; end
for t = 1:K
    a = atoms{ord(t)};
    [~, b] = min(sizes);
    bucket{b} = [bucket{b}, a];
    sizes(b) = sizes(b) + numel(a);
end
bperm = [bucket{:}];
edgesB = [0, cumsum(sizes)];
end
