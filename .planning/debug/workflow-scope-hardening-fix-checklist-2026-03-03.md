# Workflow Scope Hardening Fix Checklist (2026-03-03)

## P0 - Stop default-project persistence for workflow traces

- [x] Refactor `backend/spine/pipeline_runs.py` to require explicit `project_id` and `created_by_user_id` in `create_run(...)`; remove internal `_project_id()`/`_user_id()` fallbacks for run writes.
- [x] Refactor `backend/spine/pipeline_run_scopes.py` (`insert_scope`, `latest_run_id`, `list_runs`) to accept explicit `project_id` argument; remove `_project_id()` fallback.
- [x] Refactor `backend/spine/pipeline_run_status.py` read/write APIs to require explicit `project_id` (or resolve from canonical run context); remove `_project_id()` usage from all run/target/event operations.

## P0 - Unify workflow route scope contract

- [x] Add `X-Project-Id` handling to all `/workflow/*` endpoints that mutate or discover runs (`assessment/finalize`, `resume`, `latest`, `status`, `cancel`, `events`).
- [x] Decide read policy per endpoint:
- [x] If strict-scope: require header and fail closed (`400` missing, `404`/`409` mismatch).
- [x] If public-trace by `run_id`: resolve canonical project from run first, then query all run tables with that project; optional header becomes a consistency assertion.
- [x] Eliminate silent fallback paths by using `_resolve_scope_observability(..., require_project=True, allow_dev_project_default=False)` for scoped endpoints.

## P0 - Preserve background-run project consistency

- [x] Extend orchestrator context so run project is persisted in-memory/per-call and passed through all stage builders.
- [x] Update `backend/workflow_happy_path/builders.py::build_extract_data` to require non-empty `project_id` for production paths (keep fallback only in explicitly dev-gated code if needed).
- [x] In `backend/workflow_happy_path/orchestrator.py::_run_one`, resolve project from run context and pass it to `build_extract_data(...)`.

## P1 - Thread scope through contracts and frontend clients

- [x] Update `backend/pipeline_contracts/service.py::create_run(...)` signature to accept `project_id` and `created_by_user_id`, and pass into `pipeline_runs.create_run(...)`.
- [x] Update `frontend/workflow_api.py` to pass `project_id` for `finalize_assessment`, `resume_run`, `get_run_status`, `get_latest_run`, and `cancel_target` requests.
- [x] Add query/header support for project on SSE endpoint callers when polling events.

## P1 - Regression coverage for multi-project + public trace

- [x] Add backend tests that create two runs with same `claim_id`/`reviewer_uid` in different projects and verify `latest` isolation.
- [x] Add tests for `status/events/cancel/resume` with non-default project ensuring no `DEFAULT_PROJECT_ID` dependency.
- [x] Add compatibility tests for chosen public-trace model:
- [x] Run-id-only read works across projects without fallback drift, or
- [ ] Strict-header read denies mismatched project with explicit error (not selected for current public-trace read model).
- [x] Add orchestrator test proving `_run_one` reads extraction from run project, not default.

## Goal-Backward Verification (2026-03-03)

- [x] Goal 1 — no `DEFAULT_PROJECT_ID` fallback in workflow spine paths: run/scope/status/event helpers now require explicit `project_id`; workflow route/orchestrator paths provide concrete scope instead of spine defaults.
- [x] Goal 2 — `run_id`-resolved canonical project for `status/events/targets` endpoints: run-id endpoints resolve canonical `project_id` via `pipeline_runs.get_run(run_id)` and use it for downstream status/target/event queries.
- [x] Goal 3 — preserve cross-project public-trace reads without write leakage: `status/events` allow run-id-only reads (optional header consistency check); write endpoints (`resume`, `cancel`) require `X-Project-Id` and reject mismatches.

## P2 - Data migration and observability

- [ ] Add one-time SQL/backfill script to identify legacy workflow rows where `project_id='default'` but associated citing docs/scopes belong to non-default projects.
- [ ] Add metrics/log fields for workflow scope source and run-project mismatch detection.
- [ ] Create operator runbook entry for legacy trace repair and validation after deploy.
