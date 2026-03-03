# GSD (Get Shit Done) Setup

## Governance Metadata

- Doc role: workflow tooling guide
- Authority tier: support (process/tooling)
- Status: active
- Owner: repo maintainers
- Last reviewed: 2026-03-03
- Canonical for: repo-local GSD command usage

This repo vendors the OpenCode GSD workflow under `.opencode/` so phase planning and execution work on any machine without relying on `~/.opencode`.

## Install / Update

1) Ensure OpenCode is installed on your machine.

2) Install the repo-local OpenCode dependencies:

```bash
cd .opencode
bun install
```

3) Run GSD commands from the repo root.

## Recommended Workflow

### 1) Map Before Big Changes

Use `/gsd-map-codebase` when you need a structured inventory of:
- legacy codepaths that should be deleted
- residual filesystem state (`./data/**`) assumptions
- config/env toggles that are no longer real
- integration points (Postgres/S3/GROBID/etc)

This writes a durable map under `.planning/codebase/` which you can use as a pruning checklist.

### 2) Debug Only When Something Is Failing

Use `/gsd-debug` (or `/gsd-verify-work` → diagnose flow) when you have a concrete symptom:
- endpoint returns `500`
- state doesn’t persist across restart
- a specific E2E step fails

`/gsd-debug` is hypothesis-driven and is most useful when there’s a crisp reproduction.

### 3) Prune Backwards From Verified E2E

When migrating systems (e.g. “everything on the spine”), prune in this order:
1) Add/confirm a single scripted E2E that covers the full workflow.
2) Move one subsystem to the new durable store.
3) Re-run E2E.
4) Only then delete the legacy codepath/config.

This avoids “delete-first” refactors where you lose the last known-good behavior.

## Common Commands

- `/gsd-progress` — show current phase/milestone state
- `/gsd-map-codebase` — generate/update `.planning/codebase/*`
- `/gsd-plan-phase <N>` — produce an executable plan for phase N
- `/gsd-execute-phase <N>` — execute that plan with atomic commits
- `/gsd-verify-work <phase>` — manual acceptance testing with a persistent log
- `/gsd-debug` — investigate failures with a persistent debug log
- `/gsd-update` — update the vendored GSD workflow under `.opencode/`

## Notes

- Secrets must remain local. Do not commit `.env`.
- `.opencode/node_modules` is intentionally gitignored.
