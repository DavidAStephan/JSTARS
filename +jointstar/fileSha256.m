function h = fileSha256(path)
%FILESHA256 Lowercase hex SHA-256 digest of a file's bytes.
%
%   h = jointstar.fileSha256(path)
%
%   Primary path: java.security.MessageDigest over the file's raw bytes
%   (no text-mode decoding, so line-ending differences never matter).
%   Falls back to shelling out to `shasum -a 256` if the Java path errors
%   for any reason, and finally to the literal string 'unavailable' if
%   even that fails -- this helper must NEVER error, since it is used for
%   provenance stamping that must not be able to abort a run.
%
%   See also jointstar.estimate, jointstar.production.

try
    fid = fopen(path, 'r');
    if fid == -1
        error('jointstar:fileSha256:openFailed', 'could not open %s', path);
    end
    bytes = fread(fid, Inf, '*uint8');
    fclose(fid);
    md = java.security.MessageDigest.getInstance('SHA-256');
    md.update(bytes);
    digest = typecast(md.digest(), 'uint8');
    h = lower(sprintf('%02x', digest));
    return
catch
    % fall through to the shasum fallback below
end

try
    [status, cmdout] = system(sprintf('shasum -a 256 %s', escapeForShell(path)));
    if status == 0
        tok = strsplit(strtrim(cmdout));
        h = lower(tok{1});
        return
    end
catch
    % fall through to the final fallback below
end

h = 'unavailable';
end

function s = escapeForShell(path)
% Single-quote the path for a POSIX shell, escaping any embedded quotes.
s = ['''' strrep(path, '''', '''\''''') ''''];
end
