---
phase: 02-citation-context-navigation
plan: 02
subsystem: ui
tags: [streamlit, graphviz, citation-context, openalex]

# Dependency graph
requires:
  - phase: 01-ingestion-extraction
    provides: ingestion, extraction, and reference resolution APIs
provides:
  - Callout-driven citation context navigation pane
  - Citation graph visualization with local fallback data
  - Frontend helpers for citation context and graph APIs
affects: [citation-judgment, evidence-review]

# Tech tracking
tech-stack:
  added: [graphviz, python-multipart]
  patterns: [streamlit session state for callout navigation, local citation graph fallback]

key-files:
  created: []
  modified:
    - environment.yml
    - frontend/ingestion_api.py
    - frontend/ui.py
    - backend/citation_graph.py
    - backend/main.py
    - backend/utils.py

key-decisions:
  - "Use local citation graph data when DOI resolution is unavailable to avoid OpenAlex 404s"

patterns-established:
  - "Callout chips replace in-sentence tokens for quick selection"
  - "Citation graph renders from session-state request cache"

# Metrics
duration: 2m 27s
completed: 2026-01-24
---

# Phase 02 Plan 02: Citation Context Navigation Summary

**Streamlit citation context pane with callout selection, follow-up metadata, and local citation graph rendering.**

## Performance

- **Duration:** 2m 27s
- **Started:** 2026-01-24T11:34:34Z
- **Completed:** 2026-01-24T11:37:01Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments
- Callout-driven context pane updates with inline metadata follow-through
- Citation graph controls render references with local fallback data
- Upload flow can auto-run extraction/resolution with debug toggles

## Task Commits

Each task was committed atomically:

1. **Task 1: Add citation context API helpers + Graphviz dependency** - `e1d3ae3` (feat)
2. **Task 2: Build citation context navigation UI** - `6921772` (feat)
3. **Task 3: Refine citation graph and callout UX** - `02ca7a7` (fix)

**Plan metadata:** (docs commit for summary/state)

## Files Created/Modified
- `environment.yml` - Graphviz and dependency pins for UI rendering
- `frontend/ingestion_api.py` - Citation context/graph helper calls
- `frontend/ui.py` - Callout navigation, context rendering, and graph UI
- `backend/citation_graph.py` - Local citation graph builder for fallback data
- `backend/main.py` - Citation graph endpoint wiring to local builder
- `backend/utils.py` - Python 3.9-compatible typing fixes

## Decisions Made
- Use local citation graph data when DOI resolution is unavailable to avoid OpenAlex 404s.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Avoided citation graph failures when DOI is missing**
- **Found during:** Post-checkpoint polish
- **Issue:** Graph endpoint returned 404 when DOI resolution was missing
- **Fix:** Added local citation graph builder and endpoint wiring to rely on extraction/resolution data
- **Files modified:** `backend/citation_graph.py`, `backend/main.py`, `frontend/ui.py`
- **Verification:** UI checkpoint approval with updated graph behavior
- **Committed in:** `02ca7a7`

### Other Deviations
- Added auto extraction/resolution toggles on upload to streamline demo flow
- Pinned core dependencies (numpy/pandas/scipy/sklearn) and added python-multipart for uploads

---

**Total deviations:** 1 auto-fixed (Rule 1) + 2 scope adjustments
**Impact on plan:** Reliability improvements and workflow polish; scope stayed within citation navigation.

## Issues Encountered
None.

## User Setup Required

**External services require manual configuration.** See `./02-USER-SETUP.md` for:
- Environment variables to add
- Dashboard configuration steps
- Verification commands

## Next Phase Readiness
- Phase 2 citation navigation complete and ready for next-phase evaluation flows
- OpenAlex API key still required to unlock external citation graph data

---
*Phase: 02-citation-context-navigation*
*Completed: 2026-01-24*
