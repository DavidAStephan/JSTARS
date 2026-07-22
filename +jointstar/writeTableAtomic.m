function writeTableAtomic(tbl, outFile, varargin)
%WRITETABLEATOMIC writetable with an atomic same-directory rename.
%
%   jointstar.writeTableAtomic(tbl, outFile)
%   jointstar.writeTableAtomic(tbl, outFile, ...)  % extra writetable args
%
%   Writes tbl to a uniquely-named temp file in the SAME directory as
%   outFile (same filesystem is required for movefile's rename to be
%   atomic), then movefile(..., 'f') to the final name.  A reader can
%   never observe a partially-written outFile.  On any error the temp
%   file is deleted (if it exists) and the error is rethrown.
%
%   See also writetable, movefile.

targetDir = fileparts(outFile);
if isempty(targetDir), targetDir = pwd; end
tmpFile = [tempname(targetDir) '.csv'];

try
    writetable(tbl, tmpFile, varargin{:});
    movefile(tmpFile, outFile, 'f');
catch err
    if isfile(tmpFile)
        delete(tmpFile);
    end
    rethrow(err);
end
end
