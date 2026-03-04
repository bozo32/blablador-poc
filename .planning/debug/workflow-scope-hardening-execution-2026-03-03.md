# Workflow Scope Hardening Execution + Verification (2026-03-03)

## Trigger

- Execute the workflow scope hardening plan and produce execution artifacts + verification summary.

## Execution Sweep

- Verified scope hardening is applied in workflow persistence helpers:
  - `backend/spine/pipeline_runs.py`
  - `backend/spine/pipeline_run_scopes.py`
  - `backend/spine/pipeline_run_status.py`
- Verified route-level scope contracts and canonical run project resolution in `backend/main.py` for:
  - start/finalize/latest (explicit request project)
  - status/events (run-id canonical project, optional header consistency)
  - resume/cancel (run-id canonical project + required `X-Project-Id`)
- Verified orchestrator/project propagation in:
  - `backend/workflow_happy_path/orchestrator.py`
  - `backend/workflow_happy_path/builders.py`
- Verified contracts/frontend scope threading in:
  - `backend/pipeline_contracts/service.py`
  - `frontend/workflow_api.py`

## New Execution Artifact

- Added regression test: `tests/test_workflow_happy_path_orchestrator_scope.py`
  - proves `_run_one` uses canonical run `project_id` for `build_extract_data(...)`
  - asserts run status/event/list calls stay on canonical project scope

## Verification Results

- Command:
  - `./.venv/bin/pytest tests/test_workflow_happy_path_orchestrator_scope.py tests/test_workflow_happy_path_status_api.py tests/test_workflow_happy_path_stage_writes.py tests/test_pipeline_contract_store.py`
- Result:
  - 6 passed, 0 failed
- Coverage highlights:
  - latest-run project isolation
  - public-trace run-id reads + mismatch safeguards (`409`)
  - write endpoints requiring project header (`400` when missing)
  - canonical project propagation through orchestrator runtime path

## Remaining Gaps

- P2 migration and observability items remain open in `.planning/debug/workflow-scope-hardening-fix-checklist-2026-03-03.md`:
  - legacy default-project backfill script
  - scope-source/mismatch metrics + logs
  - operator runbook for legacy trace repair

## Artifact Updates

- Updated `.planning/debug/workflow-scope-hardening-audit-2026-03-03.md` status to `verified` and refreshed resolution/verification notes.
- Updated `.planning/debug/workflow-scope-hardening-fix-checklist-2026-03-03.md` to mark orchestrator regression coverage complete.
