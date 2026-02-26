---
phase: 02-citation-context-navigation
plan: 05
subsystem: api
tags: [grobid, crossref, openalex, fastapi, streamlit]

# Dependency graph
requires:
  - phase: 01-ingestion-extraction
    provides: grobid extraction output and ingestion storage
provides:
  - consolidated grobid reference metadata in extraction outputs
  - crossref/openalex comparison with mismatch status and selection endpoint
  - mismatch resolution selector in citation context UI
affects: [citation-judgment, evidence-review]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - resolution payloads include grobid/crossref/openalex candidates with status
    - user-selected resolution source persisted per reference

key-files:
  created: []
  modified:
    - backend/grobid_client.py
    - backend/extraction.py
    - backend/reference_resolver.py
    - backend/schemas.py
    - backend/main.py
    - frontend/ingestion_api.py
    - frontend/ui.py

key-decisions:
  - "None - followed plan as specified"

patterns-established:
  - "Crossref/OpenAlex candidates compared against GROBID consolidated metadata"
  - "Resolution selections update derived DOI/title/year fields"

# Metrics
duration: 6 min
completed: 2026-01-24
---

# Phase 2 Plan 05: Citation Context Navigation Summary

**GROBID-consolidated bibliography metadata plus Crossref/OpenAlex mismatch detection with selectable resolution candidates.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-01-24T17:51:01+01:00
- **Completed:** 2026-01-24T17:57:25+01:00
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments
- Added consolidated GROBID metadata for each bibliography entry to guide resolution comparison
- Resolved Crossref/OpenAlex candidates in parallel with mismatch status and selection persistence
- Added a Streamlit mismatch selector to apply user-chosen reference sources

## Task Commits

Each task was committed atomically:

1. **Task 1: Capture GROBID consolidated reference metadata** - `6190d92` (feat)
2. **Task 2: Compare Crossref/OpenAlex results and persist user selection** - `2ede370` (feat)
3. **Task 3: Surface mismatches in the UI and allow selection** - `159bc45` (feat)

**Plan metadata:** (docs commit for summary/state)

## Files Created/Modified
- `backend/grobid_client.py` - Request consolidated GROBID bibliography data
- `backend/extraction.py` - Parse consolidated GROBID reference metadata into bibliography entries
- `backend/reference_resolver.py` - Add Crossref/OpenAlex comparison, candidates, and selection helpers
- `backend/schemas.py` - Extend resolution payload schema with candidates and status
- `backend/main.py` - Persist user-selected resolution sources via new endpoint
- `frontend/ingestion_api.py` - Add helper to submit resolution selections
- `frontend/ui.py` - Render mismatch selector and apply selections in citation context UI

## Decisions Made
None - followed plan as specified.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Streamlit mismatch selector verification requires manual review (`streamlit run frontend/ui.py`).

## User Setup Required

**External services require manual configuration.** See `.planning/phases/02-citation-context-navigation/02-USER-SETUP.md` for:
- Environment variables to add
- Account setup steps
- Verification command

## Next Phase Readiness
- Phase 2 citation navigation and mismatch resolution are complete
- OpenAlex API key still required for live OpenAlex lookups
- Manual UI verification needed for mismatch selector behavior

---
*Phase: 02-citation-context-navigation*
*Completed: 2026-01-24*
