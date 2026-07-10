# Running on a different computer

Short version: **you almost never need to edit `estimate.m`.** The toolbox
adapts to the machine on its own, and the few things that do change per
machine are set *when you call* `estimate`, not inside it.

## What adapts automatically (no edits)

- **Core count / pool size.** `openPool()` calls `parpool('Threads')` with
  no size, which defaults to the machine's physical cores. A 32-core box
  gets 32 workers with no change.
- **BLAS thread oversubscription.** `estimate` caps each worker to 1 BLAS
  thread for the duration of the run and restores your session afterward
  (measured ~2.4x speedup; the benefit grows with core count). Automatic.
- **Parallel Computing Toolbox subscription.** If the PCT is not licensed
  (`estimate.m:153`), it falls back to serial with a warning instead of
  erroring. Automatic.

## What you set per machine — at the call site, not in the file

```matlab
cd jointstar; startup            % add paths, set RNG

% (optional) size the pool yourself before calling, e.g. on a 32-core box:
parpool('Threads', 32);           % estimate reuses an existing pool

results = jointstar.estimate('C:\path\to\your_data.csv', ...
    'NParticles', 2000, 'Seed', 42, ...
    'Horseshoe', true, 'HierKappa', true);
```

- **Data path** — first argument. That's the only "input" that changes.
- **Pool size** — open your own `parpool(...)` first if the default
  under- or over-counts on a shared machine; `estimate` detects and reuses
  it (`gcp('nocreate')`).
- **`NParticles` / `MSteps`** — leave these at the defaults. They control
  statistical quality, not just speed; lowering them to "fit" a weaker
  machine silently degrades the posterior. If a run is too slow, add
  workers or use a faster machine, don't cut particles.

## The one edit you might make INSIDE estimate.m

Only if you want a **Processes** pool instead of the default **Threads**
pool. Past ~16 workers a Processes pool sometimes scales better (separate
processes, separate memory bandwidth); below that, Threads has lower
overhead. To switch, change the one line in `openPool()`:

```matlab
% estimate.m, ~line 157
pool = parpool('Threads');        % <- default
pool = parpool('Processes');      % <- alternative; try on 16+ workers
```

The thread-cap logic already handles both pool types correctly, so this is
the *only* change needed. Benchmark both on your machine with a short run
(`'NParticles', 200`) and keep whichever is faster; nothing else differs.

## Recommended production run (any machine)

Because the posterior sits on ridges (see `checkpoints/CHECKPOINT_08.md`),
run **≥3 independent seeds and pool them** rather than trusting one run:

```matlab
parpool('Threads', 32);                  % open once, reused across calls
for s = [42 7 101]
    jointstar.estimate('your_data.csv', 'Seed', s, ...
        'OutDir', sprintf('results/seed%d', s), ...
        'Horseshoe', true, 'HierKappa', true);
end
% then pool + diagnose:
%   benchmarks/runConvergenceCheck.m  (R-hat across seeds)
%   benchmarks/poolRuns.m             (results/pooled_posterior.csv)
```

Seeds are embarrassingly parallel, so three full runs still fit the
30-minute budget on 32 workers.

## Sanity check after moving

1. `runtests('tests')` — fast, should be all green regardless of machine.
2. A small smoke run (`'NParticles', 60`) to confirm your data file loads
   and the output magnitudes look sane, before the full estimation.
