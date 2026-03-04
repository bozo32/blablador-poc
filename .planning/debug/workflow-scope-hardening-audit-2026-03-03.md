---
status: verified
trigger: "/gsd-debug Multi-project workflow scope hardening audit: identify all places where workflow run/scope/status APIs or storage default to DEFAULT_PROJECT_ID instead of request scope; verify impact on cross-project public-trace requirement; produce debug artifacts and concrete fix checklist."
created: 2026-03-03T00:00:00Z
updated: 2026-03-03T23:59:00Z
---

## Current Focus

hypothesis: Workflow run/scope/status persistence is still keyed by module-level DEFAULT_PROJECT_ID fallbacks instead of per-request scope, causing cross-project drift and making later scope-hardening risky unless read/write contracts are redesigned together.
test: Static code-path audit across backend API routes, workflow orchestrator, spine stores, and frontend workflow client calls.
expecting: Complete map of fallback points + concrete fix sequence that preserves run trace readability without cross-project bleed.
next_action: Complete P2 migration/observability tasks in `.planning/debug/workflow-scope-hardening-fix-checklist-2026-03-03.md`.

## Evidence

- timestamp: 2026-03-03T00:00:00Z
  checked: workflow route scope entrypoints in `backend/main.py`
  found: only `POST /workflow/claimspans/{claim_id}/runs` accepts `X-Project-Id`; all other workflow run/scope/status routes omit project header and call stores that resolve project via defaults (`/workflow/.../assessment/finalize`, `/workflow/.../runs/latest`, `/workflow/runs/{run_id}/status`, `/workflow/runs/{run_id}/resume`, `/workflow/runs/{run_id}/targets/{target_id}/cancel`, `/workflow/runs/{run_id}/events`)
  evidence_lines: `backend/main.py:3034`, `backend/main.py:3068`, `backend/main.py:3176`, `backend/main.py:3186`, `backend/main.py:3199`, `backend/main.py:3224`, `backend/main.py:3234`
  implication: request scope is not consistently present at API boundary for workflow trace lifecycle.

- timestamp: 2026-03-03T00:00:00Z
  checked: run record creation path
  found: `pipeline_contracts.service.create_run()` does not accept project/user scope and calls `pipeline_runs.create_run()` without scoped identifiers
  evidence_lines: `backend/pipeline_contracts/service.py:56`, `backend/pipeline_contracts/service.py:65`
  implication: every run created through contracts layer inherits storage helper defaults unless helper API is changed.

- timestamp: 2026-03-03T00:00:00Z
  checked: pipeline run storage helper
  found: `pipeline_runs._project_id()` and `_user_id()` read app defaults; `create_run()` uses them for persisted run metadata
  evidence_lines: `backend/spine/pipeline_runs.py:27`, `backend/spine/pipeline_runs.py:31`, `backend/spine/pipeline_runs.py:50`, `backend/spine/pipeline_runs.py:51`
  implication: run ownership project is default-derived, not request-derived.

- timestamp: 2026-03-03T00:00:00Z
  checked: run scope index helper
  found: `pipeline_run_scopes.insert_scope()`, `latest_run_id()`, and `list_runs()` all resolve `project_id` via `_project_id()` default helper
  evidence_lines: `backend/spine/pipeline_run_scopes.py:16`, `backend/spine/pipeline_run_scopes.py:54`, `backend/spine/pipeline_run_scopes.py:90`, `backend/spine/pipeline_run_scopes.py:132`
  implication: claim/reviewer latest-run lookup can collapse across projects into the default bucket.

- timestamp: 2026-03-03T00:00:00Z
  checked: mutable run status and event helpers
  found: all write/read paths in `pipeline_run_status.py` resolve project from `_project_id()` defaults (`upsert_run_status`, `upsert_target_status`, `get_run_status`, `get_target_status`, `list_target_status`, `append_event`, `list_events`)
  evidence_lines: `backend/spine/pipeline_run_status.py:46`, `backend/spine/pipeline_run_status.py:251`, `backend/spine/pipeline_run_status.py:352`, `backend/spine/pipeline_run_status.py:457`, `backend/spine/pipeline_run_status.py:507`, `backend/spine/pipeline_run_status.py:553`, `backend/spine/pipeline_run_status.py:609`, `backend/spine/pipeline_run_status.py:650`
  implication: status/event reads and writes are scoped by process default, not run scope.

- timestamp: 2026-03-03T00:00:00Z
  checked: orchestrator stage builder scope propagation
  found: `start_run_for_claimspan()` passes `project_id` only to initial `build_extract_data(...)`, but background worker `_run_one()` later calls `build_extract_data(citing_doc_id=...)` without project_id; builder then falls back to default project
  evidence_lines: `backend/workflow_happy_path/orchestrator.py:394`, `backend/workflow_happy_path/orchestrator.py:211`, `backend/workflow_happy_path/builders.py:17`, `backend/workflow_happy_path/builders.py:22`
  implication: same run can read citing document from request project at initialization but from default project during async progression.

- timestamp: 2026-03-03T00:00:00Z
  checked: frontend workflow API scope propagation
  found: only `start_claimspan_run()` optionally sends `X-Project-Id`; `finalize_assessment`, `resume_run`, `get_run_status`, `get_latest_run`, and `cancel_target` omit project header
  evidence_lines: `frontend/workflow_api.py:29`, `frontend/workflow_api.py:59`, `frontend/workflow_api.py:91`, `frontend/workflow_api.py:101`, `frontend/workflow_api.py:111`, `frontend/workflow_api.py:127`
  implication: even if backend routes are hardened, current frontend callers cannot satisfy strict scope contracts for most workflow APIs.

## Scope-Drift Inventory

| Area | File | Drift mode | Severity |
| --- | --- | --- | --- |
| Run creation storage | `backend/spine/pipeline_runs.py` | Writes `project_id`/`created_by_user_id` from defaults | P0 |
| Scope index storage/read | `backend/spine/pipeline_run_scopes.py` | Insert + latest/list keyed by defaults | P0 |
| Status/target/event storage/read | `backend/spine/pipeline_run_status.py` | All mutations/queries keyed by defaults | P0 |
| Orchestrator background reads | `backend/workflow_happy_path/orchestrator.py`, `backend/workflow_happy_path/builders.py` | Async extract stage falls back to default project | P0 |
| API contract surface | `backend/main.py` workflow endpoints | Missing/optional project header on run lifecycle endpoints | P0 |
| Frontend API client | `frontend/workflow_api.py` | Missing project propagation for non-start operations | P1 |
| Contracts service signature | `backend/pipeline_contracts/service.py` | No project/user scope threading to run creation | P1 |

## Cross-Project Public-Trace Impact

assumed_requirement: Run trace endpoints (`status`, `events`, and latest-run discovery) must remain debuggable/shareable across projects without silently rebinding to `DEFAULT_PROJECT_ID`.

verified_impact:

1. Current behavior does not satisfy strict multi-project isolation: most workflow metadata is written into default scope regardless request scope, so project A/B runs can collide in same logical bucket.
2. Naive hardening (write scoped project only) will break current read paths because read helpers still filter by module default project and many routes do not pass scope; non-default project run traces will start returning `404`/empty lists.
3. To satisfy both isolation and "public trace" usability, run trace reads should derive authoritative project from run identity (or explicit request scope), then use that project for downstream scope-scoped queries.

recommended_trace_model:

- `run_id` remains externally shareable trace handle.
- First lookup resolves canonical `project_id` from run row (or a dedicated resolver).
- All status/target/event/scope queries execute against resolved `project_id`, never `DEFAULT_PROJECT_ID`.
- Optional `X-Project-Id` can be validated as a consistency check (`409` on mismatch) rather than being required for every trace read.

## Resolution

root_cause: Workflow run/scope/status system had split scope authority: request/project scope was accepted in only one entrypoint, while persistence/read helpers and most route contracts derived project from `DEFAULT_PROJECT_ID`, causing cross-project drift and inconsistent background progression.
fix: Scope hardening changes are now applied across route contracts, workflow spine helpers, orchestrator scope propagation, contracts service signatures, and frontend API propagation. Run-id endpoints resolve canonical project via `pipeline_runs.get_run(run_id)` and validate optional header mismatches (`409`), while write endpoints require `X-Project-Id`.
verification: Static code-path sweep plus targeted regression tests, including project-isolated latest-run coverage, public-trace read/write scope guard coverage, and orchestrator `_run_one` canonical project propagation.
files_changed:
  - `.planning/debug/workflow-scope-hardening-audit-2026-03-03.md`
  - `.planning/debug/workflow-scope-hardening-fix-checklist-2026-03-03.md`

## Verification Checklist

- [x] Enumerated workflow API endpoints missing strict/explicit project scope.
- [x] Enumerated storage helpers defaulting to `DEFAULT_PROJECT_ID`.
- [x] Traced orchestrator/background path for scope loss after run creation.
- [x] Traced frontend caller paths that do not propagate project scope.
- [x] Assessed impact against cross-project public-trace behavior.
- [x] Implement and test staged hardening checklist (P2 migration/observability tasks still open).
