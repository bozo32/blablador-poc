# OS-ERIN Use Case: GSD Phase Execution

Use this pack when running structured planning/execution workflows (not ad-hoc bug fixing).

## Primary Goals

- Execute phase-scoped work with explicit boundaries.
- Keep decisions, verification, and state continuity coherent.
- Minimize context loss across sessions.

## Read Order

1. `docs/GSD.md`
2. `.planning/PROJECT.md`
3. `.planning/ROADMAP.md`
4. `.planning/STATE.md`
5. `.planning/README.md`
6. Current phase `*-INDEX.md` and latest `*-VERIFICATION.md`

## Typical Command Flow

- `/gsd-progress`
- `/gsd-discuss-phase <phase>`
- `/gsd-plan-phase <phase>`
- `/gsd-execute-phase <phase>`
- `/gsd-verify-work <phase>`

## Guardrails

- Phase boundary from roadmap is fixed during execution.
- Capture deferred ideas instead of scope creep.
- Prefer additive changes while bug-hunt uncertainty is high.

## Handoff Discipline

- Update `.planning/STATE.md` and `.planning/README.md` when stopping.
- Keep verification artifact current for the active phase.
