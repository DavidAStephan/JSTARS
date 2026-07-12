function startup(nWorkers)
%STARTUP Add toolbox paths, set default RNG, optionally launch a parpool.
%
%   startup          paths + rng only
%   startup(8)       also opens a local parpool with 8 workers (requires
%                    Parallel Computing Toolbox; silently skipped if absent)
%
%   Run from the repo root.  Tests: runtests('tests').

root = fileparts(mfilename('fullpath'));
addpath(root);
addpath(fullfile(root, 'tests'));

rng(42);   % default seed; every entry point re-seeds explicitly

if nargin >= 1 && nWorkers > 0
    if license('test', 'Distrib_Computing_Toolbox')
        p = gcp('nocreate');
        if isempty(p) || p.NumWorkers ~= nWorkers
            delete(p);
            parpool('local', nWorkers);
        end
    else
        warning('jointstar:noPCT', ...
            'Parallel Computing Toolbox not licensed; running single-threaded.');
    end
end
end
