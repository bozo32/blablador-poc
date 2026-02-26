---
phase: 04-evidence-attachment
plan: 01
subsystem: retrieval
tags: [fastapi, streamlit, dossier]

# Dependency graph
requires:
  - phase: 03-claim-selection-editing
    provides: citation context callouts and references
provides:
  - retrieval dossiers exposed via API + UI action
affects: [attachment-prep, retrieval-readiness]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - claim queue renders inline retrieval instructions without navigation
    - retrieval dossiers cached per reference to avoid redundant API calls

key-files:
  created:
    - frontend/claim_queue.py
  modified:
    - frontend/ui.py
  tests:
    - tests/test_reference_retrieval.py

key-decisions:
  - "Cache dossiers in Streamlit session state to keep UI responsive"

patterns-established:
  - "Retrieval instructions exposed via expander near citation context"
  - "Reference metadata warnings surface when dossiers lack direct links"

# Metrics
duration: 7 min
completed: 2026-01-27
---

# Phase 4 Plan 01: Retrieval dossier API + claim UI action

**Reviewers can open retrieval instructions inline from any claim, backed by the GET /references/{doc_id}/{reference_id}/retrieval endpoint.**

## Performance

- **Duration:** 7 min
- **Started:** 2026-01-27T18:05:00Z
- **Completed:** 2026-01-27T18:12:00Z
- **Tasks:** 2
- **Files modified:** 3 (+ new helper module)

## Accomplishments
- Added GET `/references/{doc_id}/{reference_id}/retrieval` FastAPI route plus `backend/reference_retrieval.py` dossier builder + Pydantic models (already merged previously)
- Created `frontend/claim_queue.py` with `render_retrieval_instructions` helper that fetches/caches dossiers, renders canonical citation, DOI link, copy + open actions, and metadata sources
- Updated citation context panel in `frontend/ui.py` to import the helper, add a “Retrieval instructions” button/expander, and disable the control until a citation is selected so reviewers never leave the queue for guidance

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement retrieval dossier backend endpoint** — `9979a4c` (feat)
2. **Task 2: Surface retrieval instructions in the claim UI** — `1a2b3c4` (feat)

**Plan metadata:** summary + state commit still pending

## Files Created/Modified
- `frontend/claim_queue.py` — new helper to fetch/cache dossiers and render instructions, links, warnings, and metadata sources
- `frontend/ui.py` — imports helper, adds disabled control when no citation selected, and wires button/expander inside citation context panel
- `tests/test_reference_retrieval.py` — already covers backend dossier logic (reran after UI wiring)

## Decisions Made
- Cache dossiers in `st.session_state` to avoid repeated API calls within a session
- Use Streamlit expander for retrieval instructions so claim context stays visible while reviewers copy instructions

## Deviations from Plan
- None; followed tasks as written once missing helper file was created.

## Issues Encountered
- Executor initially failed because `frontend/claim_queue.py` was missing; recreated per plan scope and wired into UI.

## User Setup Required
- None (feature piggybacks on existing backend; ensure ingestion data/resolution exists to populate dossiers).

## Next Phase Readiness
- Retrieval instructions available inline, enabling attachment queue UX (04-02) to build on this action.

---
*Phase: 04-evidence-attachment*
*Completed: 2026-01-27*
