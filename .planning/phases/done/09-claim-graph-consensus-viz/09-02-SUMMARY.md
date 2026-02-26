---
phase: 09-claim-graph-consensus-viz
plan: 02
subsystem: api
tags: [fastapi, pydantic, judgments, export, reviewer_uid]

# Dependency graph
requires:
  - phase: 09-claim-graph-consensus-viz
    provides: Reviewer identities persisted in project metadata (active/compare user selection)
provides:
  - Reviewer-scoped per-claim judgment persistence on disk
  - Reviewer-aware judgment endpoints and per-claim reviewer listing
  - Judgment exports that always include reviewer_uid (legacy-safe)
affects: [09-03-claim-graph-store, 09-04-streamlit-judgment-ui, 09-06-compare-mode]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Per-(claim_id, reviewer_uid) JSON storage with legacy default fallback
    - reviewer_uid is always present in API payloads and exports

key-files:
  created: []
  modified:
    - backend/schemas.py
    - backend/judgment_store.py
    - backend/main.py
    - tests/test_judgment_store.py

key-decisions:
  - "Use reviewer_uid query param to scope GET/PUT /claims/{claim_id}/judgment, defaulting to 'default'"
  - "Treat legacy per-claim judgment files as reviewer_uid='default' when no reviewer-specific file exists"

patterns-established:
  - "Reviewer-scoped persistence uses collision-safe filenames: claim safe+hash + reviewer safe+hash"

# Metrics
duration: 1h 36m
completed: 2026-02-06
---

# Phase 9 Plan 02: Reviewer-Scoped Judgments Summary

**Reviewer-scoped per-claim judgments with backward-compatible disk format and reviewer_uid-inclusive exports.**

## Performance

- **Duration:** 1h 36m
- **Started:** 2026-02-06T21:25:55Z
- **Completed:** 2026-02-06T23:02:09Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Added reviewer_uid to judgment schemas so API payloads are always attributable.
- Updated judgment persistence to store one file per (claim_id, reviewer_uid) while keeping legacy single-judgment files readable as reviewer_uid="default".
- Exposed reviewer-aware judgment endpoints, including an endpoint to fetch all reviewers' judgments for a claim.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add reviewer_uid to judgment request/response schemas** - `4837320` (feat)
2. **Task 2: Persist judgments per (claim_id, reviewer_uid) with backward compatibility** - `a948d4d` (feat)
3. **Task 3: Expose reviewer-aware judgment endpoints** - `8548754` (feat)

## Files Created/Modified

- `backend/schemas.py` - Adds reviewer_uid to judgment request/response models and a per-claim reviewer listing response.
- `backend/judgment_store.py` - Stores judgments per reviewer, falls back to legacy default file, and includes reviewer_uid in exports.
- `backend/main.py` - Adds reviewer_uid query scoping for judgment routes and a per-claim judgments listing endpoint.
- `tests/test_judgment_store.py` - Adds regression tests for multi-reviewer writes, legacy reads, and reviewer_uid exports.

## Decisions Made

- Scoped per-claim judgment reads/writes via `reviewer_uid` query param (defaulting to "default") to align with the "Current user" UI model.
- Kept legacy per-claim JSON files readable by treating them as reviewer_uid="default" only when no reviewer-specific default file exists.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Reviewer-scoped judgment plumbing is in place; ready for `09-03-PLAN.md` to build reviewer-attributed claim graph votes and consensus aggregation.

---
*Phase: 09-claim-graph-consensus-viz*
*Completed: 2026-02-06*
