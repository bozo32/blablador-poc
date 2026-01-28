---
phase: 05-evidence-matching-ranking
plan: 02
subsystem: api
tags: [fastapi, evidence-pipeline, pytest]

# Dependency graph
requires:
  - phase: 05-evidence-matching-ranking
    provides: EvidencePipeline + deterministic seeds (05-01 foundation)
provides:
  - Disk-backed EvidenceRunStore with claim snapshot metadata and delta badges
  - EvidenceMatchingService with rerun queueing, auto triggers, and list/history helpers
  - FastAPI `/claims/{claim_id}/evidence*` endpoints plus schemas + regression tests
affects: [frontend-evidence-ui, rerun-orchestration, attachment-pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Evidence runs record attachment fingerprints so reruns are skipped unless inputs change
    - Background rerun queue serialized per-claim with automatic attachment hooks

key-files:
  created:
    - backend/evidence_matching/store.py
    - backend/evidence_matching/service.py
  modified:
    - backend/settings.py
    - backend/attachment_pipeline.py
    - backend/main.py
    - backend/schemas.py
    - tests/test_evidence_matching_api.py

key-decisions:
  - Persist claim_text and attachment snapshots with each run to detect staleness cheaply.
  - Enforce a serialized rerun queue per claim, defaulting to one worker, so UI locking semantics remain predictable.
  - Stub FAISS/retriever modules inside tests to decouple FastAPI imports from optional native dependencies.

patterns-established:
  - Evidence API responses always surface `lock_state` and latest run metadata to drive UI state and timers.
  - Attachment pipeline publishes rerun intents immediately after `mark_matched`, ensuring candidates refresh without manual intervention.

# Metrics
duration: 22 min
completed: 2026-01-28
---

# Phase 05 Plan 02: Evidence service + API Summary

**Evidence runs persist to disk with history + delta metadata, a rerun orchestration service auto-refreshes claims, and FastAPI routes now expose list/rerun/history endpoints with regression tests.**

## Performance

- **Duration:** 22 min
- **Started:** 2026-01-28T07:35:40Z
- **Completed:** 2026-01-28T07:58:07Z
- **Tasks:** 3
- **Files modified:** 7
- **Verification:** `pytest tests/test_evidence_matching_api.py`

## Accomplishments

- Implemented `EvidenceRunStore` with atomic JSON writes, history trimming, run summaries, and delta metadata, wired through new settings knobs for store path/history depth/rerun tuning.
- Built `EvidenceMatchingService` that wraps loaders, matcher, and pipeline, records runs via the store, enforces a rerun queue, exposes list/history helpers, and auto-triggers reruns from `attachment_pipeline` once attachments reach `matched`.
- Added Pydantic schemas plus FastAPI routes for listing candidates, requesting reruns, and fetching history; expanded pytest coverage (including FAISS/retriever stubs) to exercise store/service/API flows end-to-end.

## Task Commits

1. **Task 1: Persist ranking runs and history** - `d1081e2` (feat)
2. **Task 2: Implement evidence matching service + auto reruns** - `cfc2791` (feat)
3. **Task 3: Add FastAPI evidence endpoints** - `1c25130` (feat)

## Files Created/Modified

- `backend/evidence_matching/store.py` – Implements `EvidenceRunStore` with atomic writes, history pruning, label summaries, and delta annotations.
- `backend/evidence_matching/service.py` – Coordinates reruns (ensure/list/history, queue management, auto triggers) atop pipeline + store.
- `backend/settings.py` – Adds store directory/history depth and rerun worker/timeouts.
- `backend/attachment_pipeline.py` – Fires `evidence_service.trigger_auto_rerun` after successful matches.
- `backend/schemas.py` – Introduces evidence payload/list/history/rerun models for FastAPI responses/requests.
- `backend/main.py` – Registers `/claims/{claim_id}/evidence`, `/evidence/rerun`, and `/evidence/history` routes backed by the service.
- `tests/test_evidence_matching_api.py` – Adds store/service/API tests plus FAISS/retriever stubs so FastAPI imports run in CI.

## Decisions Made

- Store each run with the originating claim text and attachment fingerprint, enabling cheap freshness checks for `ensure_current_run`.
- Limit reruns to a serialized queue per claim (default worker count = 1) so UI locks remain deterministic while still supporting queued manual reruns.
- Stub heavy optional dependencies (FAISS + retriever) inside the pytest module to keep API tests lightweight without gatekeeping the runtime installation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Stubbed FAISS and retriever modules for API tests**
- **Found during:** Task 3 (FastAPI evidence endpoints)
- **Issue:** Importing `backend.main` for TestClient spun up optional FAISS/retriever dependencies that are not installed in CI, causing import errors.
- **Fix:** Added lightweight module stubs in `tests/test_evidence_matching_api.py` to satisfy import-time expectations without requiring native FAISS bindings.
- **Files modified:** tests/test_evidence_matching_api.py
- **Verification:** `pytest tests/test_evidence_matching_api.py -k api`

**Total deviations:** 1 auto-fixed (blocking dependency stub). No scope change beyond ensuring tests run.

## Issues Encountered

- Needed to normalize FastAPI `BackgroundTasks` annotations (no optional typing) so the router could register new evidence endpoints under Pydantic v2.

## User Setup Required

None – no external services introduced.

## Next Phase Readiness

- Backend now persists evidence runs, exposes rerun orchestration, and serves paginated evidence data, so frontend evidence cards/rationale sidebar can bind directly to `/claims/{claim_id}/evidence*` endpoints.
- Auto reruns are triggered from attachment lifecycle; future work can focus on UI wiring and reviewer workflows without needing more backend plumbing.

---
*Phase: 05-evidence-matching-ranking*
*Completed: 2026-01-28*
