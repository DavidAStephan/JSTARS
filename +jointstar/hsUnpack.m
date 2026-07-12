function [Lq, Lr] = hsUnpack(P, tv)
%HSUNPACK Assemble the unit-lower Cholesky factors from a particle row.
%
%   [Lq, Lr] = jointstar.hsUnpack(P, tv)  with P from horseshoePriors.
%
%   Lq is mq x mq (transition shocks, state order), Lr is mr x mr
%   (measurement errors, observable order), both unit lower triangular.
%
%   See also jointstar.horseshoePriors, jointstar.ModelSpec.jointstar.

hs = P.hs;
Lq = eye(hs.mq);
Lq(sub2ind([hs.mq, hs.mq], hs.LqRow, hs.LqCol)) = tv(hs.colL(1:hs.nLq));
Lr = eye(hs.mr);
Lr(sub2ind([hs.mr, hs.mr], hs.LrRow, hs.LrCol)) = tv(hs.colL(hs.nLq + 1:end));
end
