function [bperm, edgesB] = blockPartition(dm, covCols, nB)
%BLOCKPARTITION Random block partition of MH covariance columns.
%
%   [bperm, edgesB] = jointstar.blockPartition(dm, covCols, nB)
%
%   bperm = randperm(dm), edgesB = round(linspace(0, dm, nB+1)).
%   Block b covers columns covCols(bperm(edgesB(b)+1:edgesB(b+1))).
%
%   See also jointstar.runSMC.

bperm = randperm(dm);
edgesB = round(linspace(0, dm, nB + 1));
end
