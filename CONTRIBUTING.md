# Contributing (Local Discipline)

This codebase evolves quickly. The goal of this doc is not process for process'
sake; it's to make work resilient to interruptions, context loss, and partial
refactors.

## Two Modes Of Work

### 1) GSD Mode (Structured)

Use the existing `.planning/` framework when the work is multi-step, touches
multiple subsystems (ingestion + graph + UI), or changes invariants.

### 2) Casual Mode (Lightweight)

For small experiments and quick iteration, keep a minimal, durable trail:

- Update `.planning/NOTES.md` at the start/end of the session.
- Prefer small checkpoint commits over long uncommitted diffs.

## Checkpoint Commits

Checkpoint commits are allowed on feature branches. They are how you avoid
"mystery state" after interruptions.

Guidelines:

- Commit early and often when making multi-file changes.
- Keep commits reversible (one idea per commit when possible).
- Use direct prefixes: `wip:`, `feat:`, `fix:`, `chore:`, `docs:`.
- Run `pytest -q` before/after risky changes when feasible.

If you later want a clean history, squash before merge; do not avoid checkpoint
commits during exploration.

## Fast Recovery After Context Loss

If you come back after a break (or a model context reset), do this first:

1. Read `.planning/NOTES.md` (Casual Mode) or `.planning/STATE.md` (GSD Mode).
2. Run `git status -sb`.
3. Run `git log -10 --oneline --decorate`.
4. Run `pytest -q` (or the relevant subset).

If you need to hand work to someone else (or to a fresh assistant session), the
minimum useful bundle is:

- The contents of `.planning/NOTES.md`
- `git status -sb`
- `git diff --stat`
- The last 3-5 commit messages

## Avoiding Context Bloat (Human + Assistant)

Large, unfocused threads accumulate unrelated state. Prefer:

- One topic per session when possible.
- A written snapshot (in `.planning/NOTES.md`) when you change topics.
- A new session/chat after the snapshot if the current one is noisy.

This keeps the "authoritative state" in the repo, not in the chat transcript.
