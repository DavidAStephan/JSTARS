function th = thetaStruct(P, tv)
%THETASTRUCT Convert a parameter row vector to a named struct.
%
%   th = jointstar.thetaStruct(P, tv)  with P from defaultPriors.
%
%   See also jointstar.defaultPriors, jointstar.ModelSpec.jointstar.

th = cell2struct(num2cell(tv(:)), P.names(:), 1);
end
