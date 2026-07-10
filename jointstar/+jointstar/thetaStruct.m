function th = thetaStruct(P, tv)
%THETASTRUCT Convert a parameter row vector to a named struct.
%
%   th = jointstar.thetaStruct(P, tv)  with P from defaultPriors.
%
%   See also jointstar.defaultPriors, jointstar.ModelSpec.jointstar.

if isfield(P, 'baseD')
    % horseshoe-extended space: only the base block becomes named fields
    % (L factors are unpacked separately via jointstar.hsUnpack)
    th = cell2struct(num2cell(reshape(tv(1:P.baseD), [], 1)), ...
        P.baseNames(:), 1);
else
    th = cell2struct(num2cell(tv(:)), P.names(:), 1);
end
end
