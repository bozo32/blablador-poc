# OS-ERIN Use Case: Contract Validation

Use this pack when auditing whether implementation still honors the active contracts.

## Primary Goals

- Validate identity/scope/visibility contract behavior.
- Validate Stage A-F workflow protocol behavior.
- Detect drift between docs, tests, and verifiers.

## Read Order

1. `docs/IDENTITY_SCOPE_VISIBILITY_CONTRACT.md`
2. `docs/WORKFLOW_PROTOCOL.md`
3. `docs/REPO_SPEC.md`
4. `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-09-VERIFICATION.md`

## Validation Commands

- `./.venv/bin/pytest -q tests/test_scope_observability.py tests/test_project_scope_api.py tests/test_scope_guard_primitive.py tests/test_scope_write_pilot_api.py tests/test_scope_strict_remaining_routes_api.py tests/test_opinion_visibility_acl_api.py tests/test_project_membership_spine.py tests/test_project_membership_api.py tests/test_project_api_scope_headers.py tests/test_frontend_scope_client_guards.py tests/test_frontend_scope_selector_lock.py tests/test_projection_scope_consistency_api.py tests/test_reference_retrieval.py`
- `bash scripts/dev/verify_10_04_5_spine_workflow.sh`

## Guardrails

- Contract docs outrank convenience behavior.
- If verifier scripts fail but tests pass, classify script drift vs product regression explicitly.
- Record any mismatch with repro steps and proposed source-of-truth update.

## Output

- Add findings to relevant verification markdown in `.planning/phases/10-contracts-core-workflow-simplification/`.
