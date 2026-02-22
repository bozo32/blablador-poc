---
phase: 10-contracts-core-workflow-simplification
plan: 02
subsystem: api
tags: [fastapi, streamlit, postgres, docker-compose, pydantic, sse]

# Dependency graph
requires:
  - phase: 10-contracts-core-workflow-simplification (10-01)
    provides: Immutable pipeline stage artifacts + store/fetch API surface
provides:
  - Mutable workflow run + per-target status tables with append-only events
  - /workflow API for start/resume/status/events/cancel and assessment finalization
  - Happy-path orchestrator that writes immutable terminal stage artifacts via pipeline contracts
  - Streamlit polling-first requested-works queue and claimspan status drilldown
  - Docker-compose-backed happy-path verifier
affects: [10-03, 10-04, workflow, contracts, ui]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Immutable stage artifacts + mutable run status (Postgres) with pollable stage_state_json
    - Polling-first UI (SSE optional) for run status and queue refresh

key-files:
  created:
    - backend/spine/pipeline_run_scopes.py
    - backend/spine/pipeline_run_status.py
    - backend/workflow_happy_path/orchestrator.py
    - backend/workflow_happy_path/builders.py
    - frontend/workflow_api.py
    - tests/test_workflow_happy_path_status_api.py
    - tests/test_workflow_happy_path_stage_writes.py
    - scripts/dev/verify_10_02_happy_path.sh
  modified:
    - backend/db/migrate.py
    - backend/main.py
    - backend/contracts/pipeline_v1.py
    - backend/pipeline_contracts/service.py
    - frontend/components/chase_queue.py
    - frontend/components/chasing_panel.py
    - frontend/ui.py

key-decisions:
  - "Track workflow progress/retries/cancel only in mutable Postgres tables; stage artifacts remain immutable and written once per (run_id, stage)."
  - "UI correctness must not depend on SSE; Streamlit polling is the primary refresh mechanism."
  - "Mirror final reviewer judgments into an immutable assessment stage artifact for audit/replay via store_stage(stage='assessment')."

patterns-established:
  - "stage_state_json schema: explicit per-stage state/progress + ISO timestamps with updated_at changes on mutation"

# Metrics
duration: 3 min
completed: 2026-02-22
---

# Phase 10 Plan 02: Core Workflow Simplification Summary

**Happy-path workflow runs with mutable Postgres status + immutable stage artifacts, pollable /workflow API, Streamlit polling queue UX, and assessment-stage mirroring for audit/replay**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-22T10:59:05Z
- **Completed:** 2026-02-22T11:02:44Z
- **Tasks:** 3
- **Files modified:** 15

## Accomplishments
- Added Postgres-backed workflow run persistence (run scopes, run status, per-target status, append-only events) with a stable `stage_state_json` contract.
- Implemented happy-path orchestrator + `/workflow/*` endpoints (start/resume/status/events/cancel) and an additive `assessment` stage with explicit `store_stage(..., stage='assessment')` finalize write path.
- Shipped Streamlit polling-first requested-works queue + claimspan run status drilldown, plus docker-compose-backed end-to-end verifier + API tests.

## Task Commits

Each task was committed atomically:

1. **Task 1: Mutable run-status persistence (tables + spine helpers + stage_state_json schema)** - `e5fed63`, `71728e0`, `e5ae9cf` (feat/fix)
2. **Task 2: Happy-path orchestrator + /workflow API + assessment stage write path** - `ca09e0a` (feat)
3. **Task 3: Streamlit polling UX + tests + verifier script** - `4009077`, `eb07a86` (feat/fix)

**Plan metadata:** `0d4c42d` (docs: complete plan)

## Files Created/Modified
- `backend/db/migrate.py` - Adds Phase 10-02 run-status DDL (idempotent migrations).
- `backend/spine/pipeline_run_scopes.py` - Append-only run scope index + latest/list helpers.
- `backend/spine/pipeline_run_status.py` - Mutable run + per-target status persistence, stage_state_json, and append-only events.
- `backend/workflow_happy_path/orchestrator.py` - Background orchestrator that advances stages, updates status, and writes terminal artifacts via `store_stage()`.
- `backend/workflow_happy_path/builders.py` - Stage payload builders for extract/citespans and candidate stages.
- `backend/contracts/pipeline_v1.py` - Adds `assessment` stage contract models + stage mapping.
- `backend/pipeline_contracts/service.py` - Allows `assessment` stage writes (without candidate-id enforcement).
- `backend/main.py` - `/workflow` endpoints, assessment finalize route, and dev wipe support.
- `frontend/workflow_api.py` - Small HTTP wrapper used by Streamlit components.
- `frontend/components/chase_queue.py` - Citation-ordered requested-works queue with polling refresh.
- `frontend/components/chasing_panel.py` - Run minting on claimspan save + status icons + drilldown.
- `frontend/ui.py` - Judgment-finalization mirrors assessment via `/workflow/.../assessment/finalize`.
- `tests/test_workflow_happy_path_status_api.py` - Status API contract coverage (ordering, schema, cancel state).
- `tests/test_workflow_happy_path_stage_writes.py` - Stage immutability + assessment artifact write/fetch coverage.
- `scripts/dev/verify_10_02_happy_path.sh` - Compose-backed end-to-end verifier for happy path.

## Decisions Made
- Track progress/retries/cancel only in mutable Postgres run-status tables; immutable stage artifacts are written once at terminal outcome snapshots.
- Make Streamlit polling-first the functional baseline; keep SSE available for debugging/future UX but non-required.
- Mirror final judgment payload into an immutable `assessment` stage artifact via `pipeline_contracts_service.store_stage(..., stage='assessment')` for audit/replay.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Scoped run status by project_id**
- **Found during:** Task 1 (run-status persistence)
- **Issue:** Run status queries/updates were insufficiently scoped, risking cross-project leakage.
- **Fix:** Enforced consistent `project_id` scoping in spine helpers.
- **Files modified:** `backend/spine/pipeline_run_status.py`
- **Verification:** `bash scripts/dev/pytest_docker.sh`
- **Committed in:** `e5ae9cf`

**2. [Rule 1 - Bug] Made happy-path verifier idempotent**
- **Found during:** Task 3 (verifier automation)
- **Issue:** Existing compose containers could keep port 8000 busy and cause verifier reruns to fail.
- **Fix:** Ensured verifier tears down prior app-api run containers before starting.
- **Files modified:** `scripts/dev/verify_10_02_happy_path.sh`
- **Verification:** `bash scripts/dev/verify_10_02_happy_path.sh`
- **Committed in:** `eb07a86`

---

**Total deviations:** 2 auto-fixed (2 bug)
**Impact on plan:** Both fixes were required for correctness and repeatable verification. No scope creep.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Workflow run status + events + immutable artifacts are in place for 10-03 graph navigation to consume.
- Ready to extend eventing/decision durability in 10-04 using `pipeline_run_events` and the run-status tables.

---
*Phase: 10-contracts-core-workflow-simplification*
*Completed: 2026-02-22*
