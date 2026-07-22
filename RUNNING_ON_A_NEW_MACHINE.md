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
  (`estimate.m`'s `openPool()`, ~line 337), it falls back to serial with a
  warning instead of erroring. Automatic.

## What you set per machine — at the call site, not in the file

```matlab
startup                          % run from the repo root; adds paths, sets RNG

% (optional) size the pool yourself before calling, e.g. on a 32-core box:
parpool('Threads', 32);           % estimate reuses an existing pool

% the production command (3 seeds pooled, all settings baked in):
out = jointstar.production('C:\path\to\your_data.csv');

% or a single low-level run for A/B / diagnostics:
results = jointstar.estimate('C:\path\to\your_data.csv', ...
    'NParticles', 2000, 'Seed', 42, ...
    'HierKappa', true);
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
% estimate.m, openPool(), ~line 341
pool = parpool('Threads');        % <- default
pool = parpool('Processes');      % <- alternative; try on 16+ workers
```

The thread-cap logic already handles both pool types correctly, so this is
the *only* change needed. Benchmark both on your machine with a short run
(`'NParticles', 200`) and keep whichever is faster; nothing else differs.

## Recommended production run (any machine)

Because the posterior sits on ridges (see `checkpoints/CHECKPOINT_08.md`),
run **≥3 independent seeds and pool them** rather than trusting one run —
but don't hand-loop this yourself.  `jointstar.production` already bakes in
the exact current recipe (seeds, flags, `ESSTargetFrac`, pooling, and
R-hat), so it is the only call you need:

```matlab
parpool('Threads', 32);                  % optional: open once, reused
out = jointstar.production('your_data.csv');
```

It is idempotent/resumable (skips any seed whose `OutDir` already has
`posterior_summary.csv`) and writes the pooled coefficient table, the
cross-seed R-hat table, the pooled smoothed states, and `provenance.json`
to `results/production/`.

**Do not** hand-loop `jointstar.estimate` over the seeds yourself with a
partial flag set (e.g. `'HierKappa', true` alone) and then call
`benchmarks/runConvergenceCheck.m` / `benchmarks/poolRuns.m` on the
result — those old standalone scripts pool whatever is sitting in
`results/production/seed<s>`, so a manually-run, differently-configured
seed silently poisons the exact directories `jointstar.production` trusts.
If `results/production/` already contains seed directories from an
earlier configuration (fewer flags, different `ESSTargetFrac`, a
different parameter count), delete or move them before calling
`jointstar.production` fresh: a seed directory carrying a `provenance.json`
from a mismatched configuration makes the new provenance check at pooling
time **error** (data-vintage/config mismatch across seeds); an older,
pre-provenance seed directory (no `provenance.json`) only gets a
**warning**, not an error, so a stale mix can still slip through silently
unless you check first.

## Sanity check after moving

1. `runtests('tests')` — fast, should be all green regardless of machine.
2. A small smoke run (`'NParticles', 60`) to confirm your data file loads
   and the output magnitudes look sane, before the full estimation.
