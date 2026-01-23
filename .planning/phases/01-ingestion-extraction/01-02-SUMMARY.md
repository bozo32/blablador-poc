---
phase: 01-ingestion-extraction
plan: 02
subsystem: api
tags: [grobid, tei, lxml, fastapi, extraction]

# Dependency graph
requires:
  - phase: 01-ingestion-extraction
    provides: Local ingestion storage and upload API
provides:
  - GROBID client for TEI/XML extraction
  - TEI parsing for metadata, citations, and bibliography
  - Extraction endpoints storing TEI and parsed payloads
affects: [reference-resolution, ingestion-ui]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Extraction artifacts persisted alongside ingestion metadata

key-files:
  created:
    - backend/grobid_client.py
    - backend/extraction.py
    - .planning/phases/01-ingestion-extraction/01-USER-SETUP.md
  modified:
    - backend/settings.py
    - backend/ingestion_store.py
    - backend/schemas.py
    - backend/main.py

key-decisions:
  - "None - followed plan as specified"

patterns-established:
  - "Extraction payloads stored under ingestion metadata with status and data"

# Metrics
duration: 10 min
completed: 2026-01-23
---

# Phase 1 Plan 2: GROBID Extraction Summary

**GROBID extraction now produces stored TEI/XML plus structured metadata, citations, and references via new ingestion endpoints.**

## Performance

- **Duration:** 10 min
- **Started:** 2026-01-23T16:36:16Z
- **Completed:** 2026-01-23T16:46:19.281971Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments
- Added a configurable GROBID client for TEI extraction with URL/timeout settings.
- Implemented TEI parsing for metadata, citation callouts, and bibliography entries.
- Wired extraction endpoints to persist TEI/XML and parsed payloads per ingestion record.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement GROBID client + settings** - `8cbb27a` (feat)
2. **Task 2: Parse TEI for metadata and citations** - `41e3260` (feat)
3. **Task 3: Wire extraction into ingestion records** - `894b96e` (feat)

**Plan metadata:** Pending

_Note: TDD tasks may have multiple commits (test → feat → refactor)_

## Files Created/Modified
- `backend/grobid_client.py` - HTTP client for GROBID TEI extraction.
- `backend/extraction.py` - TEI parsing helpers for metadata, citations, and references.
- `backend/settings.py` - GROBID URL/timeout settings defaults.
- `backend/ingestion_store.py` - Extraction storage helpers and TEI persistence.
- `backend/schemas.py` - Extraction response schemas.
- `backend/main.py` - Extraction endpoints for ingestion records.
- `.planning/phases/01-ingestion-extraction/01-USER-SETUP.md` - GROBID service setup checklist.

## Decisions Made
None - followed plan as specified.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed missing API server dependencies for verification**

- **Found during:** Task 3 (Extraction endpoint verification)
- **Issue:** `uvicorn`, `fastapi`, `faiss-cpu`, `sentence-transformers`, and `python-multipart` were missing, blocking local server startup.
- **Fix:** Installed required packages locally to run the extraction verification flow.
- **Files modified:** None (environment-only changes)
- **Verification:** `curl -X POST /ingest/{doc_id}/extract` succeeded against a stub GROBID service.
- **Committed in:** 894b96e (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Verification dependencies were required to complete local endpoint testing. No scope creep.

## Issues Encountered
None.

## User Setup Required

**External services require manual configuration.** See `./.planning/phases/01-ingestion-extraction/01-USER-SETUP.md` for:
- Environment variables to add
- Verification commands

## Next Phase Readiness
Ready for 01-03-PLAN.md (reference resolution via Crossref).

---
*Phase: 01-ingestion-extraction*
*Completed: 2026-01-23*
