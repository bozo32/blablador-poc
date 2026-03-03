# OS-ERIN Use Case: Bug Hunt (Chasing)

Use this pack when debugging the Chasing flow while upstream data/graph quality is still evolving.

## Primary Goals

- Validate Chasing behavior against current contracts.
- Fix user-facing blockers with minimal blast radius.
- Avoid graph-tail refactors that depend on unfinished upstream bug hunts.

## Read Order

1. `.planning/README.md`
2. `.planning/STATE.md`
3. `docs/IDENTITY_SCOPE_VISIBILITY_CONTRACT.md`
4. `docs/WORKFLOW_PROTOCOL.md`
5. `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-09-VERIFICATION.md`
6. `.planning/debug/*.md` (latest relevant notes)

## Guardrails

- Treat Reading/Chasing data correctness as upstream authority for graph quality.
- Keep strict scope/membership fail-closed behavior intact.
- Prefer narrow, test-backed fixes over architecture shifts.
- Log prune candidates; defer major pruning until bug-hunt stabilization.

## Verify First

- `./.venv/bin/pytest tests/test_chasing_panel_scope_keys.py -q`
- `./.venv/bin/pytest tests/test_frontend_scope_selector_lock.py -q`
- `./.venv/bin/pytest tests/test_projection_scope_consistency_api.py -q`

## Handoff Targets

- Add findings to `.planning/debug/`.
- Track deferred cleanup in `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-PRUNE-CANDIDATES.md`.
