# Orchestration setup for JSTARS — Claude Code + MATLAB

Already in place in this repo:

```
JSTARS/
├── CLAUDE.md
└── .claude/
    └── agents/
        ├── code-auditor.md
        ├── uc-estimator.md
        ├── convergence-runner.md
        ├── diagnostics-runner.md
        ├── spec-comparator.md
        ├── spec-improver.md
        └── matlab-debugger.md
```

These files are untracked (gitignored-by-omission, not committed) —
they were dropped in as a generic template and have since been rewritten
to match this project specifically: a precision-based Bayesian SMC
estimator for the JointSTAR unobserved-components model
(`jointstar/+jointstar/`), not a classical Kalman-filter/MLE toolbox.
`CLAUDE.md`'s "Project context," "Convergence discipline," "Owner
rulings," and "Known open issues" sections carry the project-specific
facts sub-agents need — read those before changing any sub-agent file.

## Before first use

1. `CLAUDE.md`'s "Project context" section is filled in for this repo
   already. If the toolbox, data, or MATLAB setup changes, update it
   there — sub-agents rely on it for consistency across runs.
2. Run the main session on Fable:
   ```
   claude --model claude-fable-5
   ```
   or `/model claude-fable-5` inside a running session. Make sure your
   Claude Code build is current — Fable access was suspended June 12
   and restored July 1, 2026, so an older CLI may not recognize the
   model string.
3. In VS Code (Claude Code extension), the same `.claude/agents/` and
   `CLAUDE.md` are read automatically — no separate setup needed.
   `/model claude-fable-5` works the same way in the extension's chat
   panel.
4. `matlab` is confirmed **not** on `PATH` on this machine — `uc-
   estimator`, `convergence-runner`, and `matlab-debugger` are already
   written to call the full path,
   `/Applications/MATLAB_R2026a.app/bin/matlab -batch "..."`. If you
   move to a different machine, update that path in those three files
   (and in `CLAUDE.md`'s "Project context") rather than assuming
   `matlab` resolves.

## Notes

- Model names used here: `haiku` (claude-haiku-4-5-20251001) for cheap
  workers, `sonnet` (claude-sonnet-5) for judgment-heavy sub-agents,
  Fable as the orchestrator. Escalate a specific sub-agent to
  claude-opus-4-8 only if it fails verification twice on the same
  subtask — don't set it as a default.
- Keep Fable's own reasoning effort at "high" as a standing default;
  raise it only for the initial planning turn or an escalated
  adjudication, per the guidance in CLAUDE.md.
