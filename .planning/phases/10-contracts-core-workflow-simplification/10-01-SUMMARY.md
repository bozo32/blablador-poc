---
phase: 10-contracts-core-workflow-simplification
plan: 01
subsystem: api
tags: [fastapi, pydantic, postgres, s3, minio, contracts]

# Dependency graph
requires:
  - phase: 09.3-spine-everywhere-legacy-removal
    provides: Postgres spine + works-bucket object store + /dev/wipe
provides:
  - Versioned per-stage contract envelope models (v1)
  - Write-once stage artifact persistence (Postgres pointer + S3 JSON)
  - Minimal /pipeline API to store/fetch stage artifacts
affects: [10-02-core-workflow-simplification, workflow-orchestration, rerank-swaps]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "S3 JSON artifact + Postgres pointer row (server returns JSON bodies)"
    - "Write-once per (run_id, stage) enforced by DB reserve -> S3 put -> DB finalize"

key-files:
  created:
    - backend/contracts/pipeline_v1.py
    - backend/contracts/upgrade.py
    - backend/spine/pipeline_runs.py
    - backend/spine/pipeline_artifacts.py
    - backend/pipeline_contracts/service.py
    - tests/test_pipeline_contract_store.py
    - tests/test_pipeline_contract_api.py
    - scripts/dev/verify_10_01_contracts.sh
  modified:
    - backend/contracts/__init__.py
    - backend/db/migrate.py
    - backend/main.py
    - tests/conftest.py

key-decisions:
  - "span_id_for() matches SpanGraphStore's pipe-joined sha256 algorithm for compatibility"
  - "Stage store conflict maps to HTTP 409 and never overwrites pipeline/{run_id}/{stage}.json"
  - "Verifier uses docker compose --build to guarantee code is packaged"

patterns-established:
  - "contracts/*: validate+upgrade entrypoints own payload shape enforcement"
  - "spine/pipeline_artifacts.py: reserve row first (ON CONFLICT DO NOTHING) before any S3 write"

# Metrics
duration: 9h 48m
completed: 2026-02-21
---

# Phase 10 Plan 01: Contracts Arc Summary

**Versioned per-stage artifact envelopes with write-once S3 persistence and a minimal /pipeline API surface for store/fetch.**

## Performance

- **Duration:** 9h 48m
- **Started:** 2026-02-20T22:48:07Z
- **Completed:** 2026-02-21T08:36:34Z
- **Tasks:** 3
- **Files modified:** 12

## Accomplishments

- Added v1 Pydantic contract envelopes and stage-specific data blocks with deterministic helpers.
- Implemented spine-backed persistence: Postgres pointer rows + works-bucket JSON objects at `pipeline/{run_id}/{stage}.json`, write-once enforced via `(run_id, stage)` uniqueness.
- Exposed minimal endpoints to create runs and store/fetch stage artifacts, with `/dev/wipe` clearing both DB rows and works-bucket objects.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add v1 pipeline contract models + upgrade shim** - `79db837` (feat)
2. **Task 2: Add pipeline run + stage artifact persistence (Postgres pointers + S3 JSON)** - `2b8e98b` (feat)
3. **Task 3: Add minimal API endpoints to create runs and store/fetch stage artifacts (+ wipe + verifier)** - `9d3737a` (feat)

## Files Created/Modified

- `backend/contracts/pipeline_v1.py` - Common envelope + stage v1 models + deterministic id/bytes helpers.
- `backend/contracts/upgrade.py` - Upgrade/validate entrypoints (additive-only evolution).
- `backend/spine/pipeline_runs.py` - Create/get pipeline run rows.
- `backend/spine/pipeline_artifacts.py` - Write-once stage artifact store (DB reserve -> S3 put -> DB finalize).
- `backend/pipeline_contracts/service.py` - Run + stage payload builders used by HTTP handlers.
- `backend/db/migrate.py` - Adds `pipeline_runs` and `pipeline_stage_artifacts` DDL + indexes.
- `backend/main.py` - Adds `/pipeline/*` endpoints + includes pipeline tables in `/dev/wipe`.
- `tests/test_pipeline_contract_store.py` - Immutability + round-trip store/load coverage.
- `tests/test_pipeline_contract_api.py` - API store/fetch + 409 conflict + wipe behavior.
- `scripts/dev/verify_10_01_contracts.sh` - Compose-backed end-to-end verifier.

## Decisions Made

- Used SpanGraphStore's existing deterministic span id algorithm (pipe-joined sha256) as the contract helper to guarantee stable cross-module span ids.
- Returned 409 conflicts as a top-level JSON error body `{error:{code,message}}` to keep client handling explicit (no S3 pointer exposure).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Enforced artifact_type consistency via model-level validation**

- **Found during:** Task 2 (persistence layer relies on artifact_type correctness)
- **Issue:** Field-level validation could not reliably check `artifact_type` against `stage` due to validation order.
- **Fix:** Added a v1 envelope `model_validator` enforcing `artifact_type == contracts/{stage}@v{schema_version}`.
- **Files modified:** backend/contracts/pipeline_v1.py
- **Verification:** `bash scripts/dev/pytest_docker.sh` (full suite)
- **Committed in:** `2b8e98b`

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Correctness-only; no scope creep.

## Issues Encountered

- Pre-commit hooks reformatted new files (black) and required docstring/lint fixes during Task 1/2 commits.
- Initial verifier draft needed `docker compose up -d --build` to ensure the API container packaged new code.

## User Setup Required

None - uses existing Postgres + MinIO compose services.

## Next Phase Readiness

Ready for `10-02-PLAN.md`: stage contracts + persistence + minimal API are in place for orchestrating a happy-path core workflow.

---
*Phase: 10-contracts-core-workflow-simplification*
*Completed: 2026-02-21*
