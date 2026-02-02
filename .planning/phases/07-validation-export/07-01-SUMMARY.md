---
phase: 07-validation-export
plan: 01
subsystem: api
tags: [fastapi, pydantic, pytest, json, csv]

# Dependency graph
requires:
  - phase: 06-evidence-review-selection
    provides: On-disk per-claim store pattern + FastAPI CRUD precedent
provides:
  - On-disk per-claim judgment persistence (draft/final) keyed by claim_id
  - Judgment CRUD/list endpoints and raw JSON/CSV export (claim/callout shapes)
  - Deterministic export ordering and stable callout grouping rules
affects: [07-validation-export, frontend, exports]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Store-backed per-claim JSON persistence under data/ with collision-safe filenames
    - Export builders return raw bytes for FastAPI Response downloads (no JSON envelope)

key-files:
  created:
    - backend/judgment_store.py
    - tests/test_judgment_store.py
    - .planning/phases/07-validation-export/07-01-SUMMARY.md
  modified:
    - backend/schemas.py
    - backend/main.py
    - .planning/STATE.md

key-decisions:
  - "Per-callout CSV export is flattened row-per-(callout, claim) to avoid nested cells"

patterns-established:
  - "Callout export groups by (doc_id, citation_index, target_id) and sorts deterministically for stable tests"

# Metrics
duration: 10 min
completed: 2026-02-02
---

# Phase 07 Plan 01: Judgment Persistence + Export Summary

**Draft/final per-claim judgments persisted on disk, with FastAPI CRUD/list endpoints and deterministic JSON/CSV exports (claim + callout shapes).**

## Performance

- **Duration:** 10 min
- **Started:** 2026-02-02T08:34:09Z
- **Completed:** 2026-02-02T08:44:36Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Added a `JudgmentStore` that persists one record per `claim_id` under `data/judgments/` with collision-safe filenames.
- Implemented draft/final semantics in Pydantic models (final requires verdict; draft allows missing verdict) with optional structured notes and provenance fields.
- Added judgment CRUD/list endpoints plus raw downloadable exports as JSON (default) and CSV for both per-claim and per-callout shapes.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement on-disk JudgmentStore + Pydantic models (draft/final + notes + provenance)** - `62f5a1a` (feat)
2. **Task 1 follow-up: restore missing schema models** - `4fc6de4` (fix)
3. **Task 2: Wire FastAPI endpoints for judgment CRUD/list and export (JSON/CSV; claim/callout shapes)** - `edf5d4f` (feat)

## Files Created/Modified
- `backend/judgment_store.py` - JSON store keyed by claim_id + deterministic export builders.
- `backend/schemas.py` - Judgment models + validation rules + list response model.
- `backend/main.py` - Judgment CRUD/list endpoints + raw export endpoint.
- `tests/test_judgment_store.py` - Unit coverage for draft/final rules, collision-guarded filenames, and export shaping.

## Decisions Made
- Per-callout CSV export is flattened to one row per (callout, claim) to keep CSV usable and avoid nested JSON-in-cells.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- A staging/pre-commit interaction caused the initial Task 1 commit to miss `backend/schemas.py`; fixed immediately in a follow-up commit (`4fc6de4`).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Ready for `07-02-PLAN.md` (frontend judgment store + API helpers) to consume:
  - `GET/PUT /claims/{claim_id}/judgment`
  - `GET /judgments`
  - `GET /judgments/export?shape=claim|callout&format=json|csv&include_drafts=...&mode=core|verbose`

---
*Phase: 07-validation-export*
*Completed: 2026-02-02*
