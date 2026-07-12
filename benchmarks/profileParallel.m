function out = profileParallel()
%PROFILEPARALLEL Time ~2000 full-spec likelihood evals in a parfor
%   (Threads pool, 6 workers) vs a plain for loop, mimicking runSMC's
%   per-particle broadcast pattern (one theta ROW per iteration, full
%   dat/P/spec closures broadcast). Read-only profiling; does not touch
%   +jointstar.
%
%   Run from repo root.

rng(1);

dat = jointstar.loadData('data.csv', 'PieObs', true);
P = jointstar.horseshoePriors('HierKappa', true);

N = 2000;
fprintf('drawing %d particles from the prior (rejecting infeasible draws)...\n', N);
Theta = zeros(N, P.d);
nBad = 0;
for i = 1:N
    ok = false;
    for attempt = 1:50
        cand = jointstar.priorSample(P, 1);
        ll = logLikTheta(P, cand, dat);
        if isfinite(ll)
            Theta(i, :) = cand;
            ok = true;
            break
        end
    end
    if ~ok
        Theta(i, :) = cand;   % keep even if -Inf; timing does not need finiteness
        nBad = nBad + 1;
    end
end
fprintf('done (%d rows fell back to a possibly-infeasible draw after 50 attempts)\n', nBad);

fprintf('\n=== broadcast-data sizes (per runSMC-style parfor iteration) ===\n');
info = whos('dat'); fprintf('  dat struct : %.2f MB\n', info.bytes / 1e6);
info = whos('P'); fprintf('  P struct   : %.2f MB\n', info.bytes / 1e6);
info = whos('Theta'); fprintf('  Theta (all %d rows) : %.2f MB (%.3f KB/row)\n', ...
    N, info.bytes / 1e6, info.bytes / N / 1e3);

% ---- serial for loop, single-threaded BLAS (matches estimate.m's
%      Threads-pool convention of capping to 1 thread per worker) ----
maxNumCompThreads(1);
ll_for = zeros(N, 1);
tic;
for i = 1:N
    ll_for(i) = logLikTheta(P, Theta(i, :), dat);
end
tFor = toc;
fprintf('\nserial for, 1 thread : %.2f s total, %.3f ms/eval\n', tFor, 1000 * tFor / N);
maxNumCompThreads('automatic');

% ---- parfor on a Threads pool sized to 6 workers -----------------------
pool = gcp('nocreate');
if isempty(pool) || ~isa(pool, 'parallel.ThreadPool') || pool.NumWorkers ~= 6
    if ~isempty(pool), delete(pool); end
    pool = parpool('Threads', 6);
end
wait(parfevalOnAll(pool, @maxNumCompThreads, 0, 1));   % 1 BLAS thread/worker

ll_par = zeros(N, 1);
tic;
parfor i = 1:N
    ll_par(i) = logLikTheta(P, Theta(i, :), dat); %#ok<PFBNS>
end
tPar = toc;
fprintf('parfor, Threads pool (6 workers, 1 BLAS thread each) : %.2f s total, %.3f ms/eval\n', ...
    tPar, 1000 * tPar / N);

speedup = tFor / tPar;
fprintf('\nspeedup (serial/parfor) = %.2fx   (ideal = 6x on 6 cores)\n', speedup);
fprintf('parallel efficiency = %.1f%%\n', 100 * speedup / 6);

maxDiff = max(abs(ll_for - ll_par));
fprintf('max |ll_for - ll_par| = %.3g (sanity: should be ~0, same draws)\n', maxDiff);

% ==========================================================================
% CACHED PATH: same 2000 draws, same pool, jointstar.buildEvalCache threaded
% through every eval.  Before/after comparison for the eval-cache
% performance pass (see jointstar.buildEvalCache); correctness already
% verified bitwise-identical in benchmarks/verifyCacheEquivalence.m.
% ==========================================================================
fprintf('\n=== cached path ===\n');
cache = jointstar.buildEvalCache(dat, P);

maxNumCompThreads(1);
ll_for_c = zeros(N, 1);
tic;
for i = 1:N
    ll_for_c(i) = logLikThetaCached(P, Theta(i, :), dat, cache);
end
tForC = toc;
fprintf('serial for, 1 thread, cached : %.2f s total, %.3f ms/eval\n', tForC, 1000 * tForC / N);
maxNumCompThreads('automatic');

wait(parfevalOnAll(pool, @maxNumCompThreads, 0, 1));
ll_par_c = zeros(N, 1);
tic;
parfor i = 1:N
    ll_par_c(i) = logLikThetaCached(P, Theta(i, :), dat, cache); %#ok<PFBNS>
end
tParC = toc;
fprintf('parfor, Threads pool (6 workers, 1 BLAS thread each), cached : %.2f s total, %.3f ms/eval\n', ...
    tParC, 1000 * tParC / N);

speedupC = tForC / tParC;
fprintf('\nspeedup (serial/parfor), cached = %.2fx   (ideal = 6x on 6 cores)\n', speedupC);
fprintf('parallel efficiency, cached = %.1f%%\n', 100 * speedupC / 6);

maxDiffC = max(abs(ll_for_c - ll_par_c));
fprintf('max |ll_for_c - ll_par_c| = %.3g (sanity: should be ~0, same draws)\n', maxDiffC);
maxDiffVsUncached = max(abs(ll_for - ll_for_c));
fprintf('max |ll_for - ll_for_c| (cached vs uncached, same draws) = %.3g\n', maxDiffVsUncached);

fprintf('\n=== before/after summary ===\n');
fprintf('  no-cache : serial %.3f ms/eval, parfor %.3f ms/eval, efficiency %.1f%%\n', ...
    1000 * tFor / N, 1000 * tPar / N, 100 * speedup / 6);
fprintf('  cached   : serial %.3f ms/eval, parfor %.3f ms/eval, efficiency %.1f%%\n', ...
    1000 * tForC / N, 1000 * tParC / N, 100 * speedupC / 6);

% ==========================================================================
% ROUND 2 TASK 2: pool-type / scheduling-overhead experiment.  Same N=2000
% cached-path draws as above, on (c) a Processes pool (6 workers; the
% one-time worker-side data transfer is measured SEPARATELY via
% parallel.pool.Constant -- constructing a Constant is exactly the
% one-shot broadcast a Processes pool needs to pay once per pool lifetime,
% so timing it apart from the steady-state parfor isolates "startup tax"
% from "per-eval throughput") and (d) a Threads pool with a CHUNKED parfor
% (parfor over a handful of chunks, inner plain for over each chunk's
% share of the 2000 draws) as a scheduling-overhead control -- if
% per-iteration parfor scheduling overhead is a meaningful share of the
% ~ms-scale eval cost, coarser chunking should recover some of it even on
% the SAME Threads pool with no process/broadcast cost at all.
% ==========================================================================
fprintf('\n=== pool-type / scheduling-overhead experiment (cached path, same %d draws) ===\n', N);

% ---- (c) Processes pool: 6 workers, delete/recreate; broadcast timed
% separately from steady-state via parallel.pool.Constant --------------
pool = gcp('nocreate');
if ~isempty(pool), delete(pool); end
tPoolStart = tic;
poolP = parpool('Processes', 6);
tPoolStartS = toc(tPoolStart);
wait(parfevalOnAll(poolP, @maxNumCompThreads, 0, 1));   % 1 BLAS thread/worker
% workers are separate processes: make sure +jointstar is on their path
% (Threads-pool workers share the client process so this was never an
% issue above; local Processes-pool workers normally inherit the client's
% path automatically, but this call is cheap insurance and is itself
% folded into the "pool start" time, not steady-state).
wait(parfevalOnAll(poolP, @() addpath(fileparts(pwd)), 0));

tBroadcast = tic;
datC = parallel.pool.Constant(dat);
PconstC = parallel.pool.Constant(P);
cacheC = parallel.pool.Constant(cache);
tBroadcastS = toc(tBroadcast);

ll_par_proc = zeros(N, 1);
tic;
parfor i = 1:N
    ll_par_proc(i) = logLikThetaCached(PconstC.Value, Theta(i, :), datC.Value, cacheC.Value); %#ok<PFBNS>
end
tParProc = toc;
speedupProc = tForC / tParProc;
fprintf('Processes pool (6 workers): pool-start %.2f s, one-time Constant broadcast %.2f s (excluded below), steady-state %.2f s total = %.3f ms/eval\n', ...
    tPoolStartS, tBroadcastS, tParProc, 1000 * tParProc / N);
fprintf('  speedup vs 1-thread serial cached = %.2fx, efficiency = %.1f%%\n', speedupProc, 100 * speedupProc / 6);
maxDiffProc = max(abs(ll_for_c - ll_par_proc));
fprintf('  max |ll_for_c - ll_par_proc| = %.3g (sanity: should be ~0, same draws)\n', maxDiffProc);

delete(poolP);

% ---- (d) Threads pool, CHUNKED parfor: parfor over nChunks chunks, plain
% for inside each chunk (RNG-free workload here -- this is a pure
% scheduling-overhead control, not the runSMC mutation loop) ------------
poolT = parpool('Threads', 6);
wait(parfevalOnAll(poolT, @maxNumCompThreads, 0, 1));

chunkCounts = [6, 12];
chunkMs = zeros(size(chunkCounts));
chunkSpeedup = zeros(size(chunkCounts));
chunkEff = zeros(size(chunkCounts));
chunkMaxDiff = zeros(size(chunkCounts));
for cc = 1:numel(chunkCounts)
    nChunks = chunkCounts(cc);
    edges = round(linspace(0, N, nChunks + 1));
    outCell = cell(nChunks, 1);
    tic;
    parfor c = 1:nChunks
        idxRange = edges(c) + 1:edges(c + 1); %#ok<PFBNS>
        localRes = zeros(numel(idxRange), 1);
        for k = 1:numel(idxRange)
            ii = idxRange(k);
            localRes(k) = logLikThetaCached(P, Theta(ii, :), dat, cache);
        end
        outCell{c} = localRes;
    end
    tChunk = toc;
    ll_chunk = vertcat(outCell{:});
    spd = tForC / tChunk;
    chunkMs(cc) = 1000 * tChunk / N;
    chunkSpeedup(cc) = spd;
    chunkEff(cc) = 100 * spd / 6;
    chunkMaxDiff(cc) = max(abs(ll_for_c - ll_chunk));
    fprintf('Threads pool, chunked (%2d chunks): %.2f s total = %.3f ms/eval, speedup=%.2fx, efficiency=%.1f%%, max|diff|=%.3g\n', ...
        nChunks, tChunk, chunkMs(cc), spd, chunkEff(cc), chunkMaxDiff(cc));
end

fprintf('\n=== 4-way pool comparison summary (cached path, N=%d) ===\n', N);
fprintf('  (a) serial, 1 thread              : %.3f ms/eval\n', 1000 * tForC / N);
fprintf('  (b) Threads pool (plain parfor)    : %.3f ms/eval, efficiency %.1f%%\n', ...
    1000 * tParC / N, 100 * speedupC / 6);
fprintf('  (c) Processes pool (steady-state)  : %.3f ms/eval, efficiency %.1f%% (+ %.2f s one-time broadcast)\n', ...
    1000 * tParProc / N, 100 * speedupProc / 6, tBroadcastS);
for cc = 1:numel(chunkCounts)
    fprintf('  (d) Threads pool, %2d-chunk parfor  : %.3f ms/eval, efficiency %.1f%%\n', ...
        chunkCounts(cc), chunkMs(cc), chunkEff(cc));
end

out = struct('N', N, 'tFor', tFor, 'tPar', tPar, 'speedup', speedup, ...
    'msPerEvalFor', 1000 * tFor / N, 'msPerEvalPar', 1000 * tPar / N, ...
    'tForCached', tForC, 'tParCached', tParC, 'speedupCached', speedupC, ...
    'msPerEvalForCached', 1000 * tForC / N, 'msPerEvalParCached', 1000 * tParC / N, ...
    'maxDiffCachedVsUncached', maxDiffVsUncached, ...
    'tPoolStartProcesses', tPoolStartS, 'tBroadcastProcesses', tBroadcastS, ...
    'tParProcesses', tParProc, 'msPerEvalParProcesses', 1000 * tParProc / N, ...
    'speedupProcesses', speedupProc, 'maxDiffProcesses', maxDiffProc, ...
    'chunkCounts', chunkCounts, 'msPerEvalChunked', chunkMs, ...
    'speedupChunked', chunkSpeedup, 'efficiencyChunked', chunkEff, ...
    'maxDiffChunked', chunkMaxDiff);
end

function ll = logLikTheta(P, tv, dat)
th = jointstar.thetaStruct(P, tv);
if isfield(P, 'hs')
    [Lq, Lr] = jointstar.hsUnpack(P, tv);
    spec = jointstar.ModelSpec.jointstar(th, dat, struct('Lq', Lq, 'Lr', Lr));
else
    spec = jointstar.ModelSpec.jointstar(th, dat);
end
ll = jointstar.computeLogLik(spec.system(), dat.y);
end

function ll = logLikThetaCached(P, tv, dat, cache)
th = jointstar.thetaStruct(P, tv);
if isfield(P, 'hs')
    [Lq, Lr] = jointstar.hsUnpack(P, tv);
    spec = jointstar.ModelSpec.jointstar(th, dat, struct('Lq', Lq, 'Lr', Lr), cache);
else
    spec = jointstar.ModelSpec.jointstar(th, dat, [], cache);
end
ll = jointstar.computeLogLik(spec.system(), dat.y, cache);
end
