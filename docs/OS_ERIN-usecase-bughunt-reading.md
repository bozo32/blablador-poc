# OS-ERIN Use Case: Bug Hunt (Reading/Data Quality)

Use this pack when the goal is improving reading-path extraction/resolution correctness that feeds later stages.

## Primary Goals

- Improve citation/body/reference quality at source.
- Validate fail-closed scope behavior in reading APIs.
- Reduce downstream noise in Chasing/Graph by fixing upstream data.

## Read Order

1. `docs/REPO_SPEC.md`
2. `docs/WORKFLOW_PROTOCOL.md`
3. `docs/IDENTITY_SCOPE_VISIBILITY_CONTRACT.md`
4. `.planning/README.md`
5. `.planning/STATE.md`
6. `.planning/debug/*.md` (reading/extraction/resolution issues)

## Guardrails

- Read endpoints should not trigger expensive compute implicitly.
- Keep extraction/resolve behavior explicit and observable.
- Avoid UI-local state as authority for cross-project behavior.

## Verify First

- `bash scripts/dev/verify_10_04_5_01_intake.sh`
- `./.venv/bin/pytest tests/test_project_scope_api.py -q`
- `./.venv/bin/pytest tests/test_scope_strict_remaining_routes_api.py -q`

## Handoff Targets

- Log bugs in `.planning/debug/`.
- Record contract-impacting changes in current phase verification docs.
