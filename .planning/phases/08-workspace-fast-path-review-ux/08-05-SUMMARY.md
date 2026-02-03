---
phase: 08-workspace-fast-path-review-ux
plan: 05
subsystem: ui
tags: [streamlit, python, citations, ux]

# Dependency graph
requires:
  - phase: 08-workspace-fast-path-review-ux
    provides: 3-pane workspace shell with Document/Review split (08-04)
provides:
  - Inline citation chips are hyperlinks that drive chase selection via query params
  - Paragraph-level citation button rows removed in favor of in-text interactions
  - Right-rail chase queue stays stable in document order on activation
affects: [08-06, workspace, chasing, review-ux]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Query-param-driven navigation (?doc, ?cite, ?target) as the interaction bus between panes

key-files:
  created: []
  modified: [frontend/ui.py, frontend/components/chase_queue.py]

key-decisions:
  - "Inline citation chips are clickable <a> links using existing ?doc=&cite=&target= routing (no Streamlit button rows)."
  - "Collector UI ordering is stable and driven by citation_index + target_id, not activation state."

patterns-established:
  - "Inline citations are the primary interaction surface; right pane is the citing-span collector."

# Metrics
duration: 2h 26m
completed: 2026-02-03
---

# Phase 8 Plan 05: Inline Citation Interactions Summary

**Inline citation chips are hyperlink-style callouts that drive the right-rail collector via existing query-param routing, with per-paragraph citation button rows removed.**

## Performance

- **Duration:** 2h 26m
- **Started:** 2026-02-02T23:47:56Z
- **Completed:** 2026-02-03T02:14:31Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Document tab citations are clickable inline (anchor links) and update selection/collector through `?doc=&cite=&target=`.
- Removed end-of-block citation button rows so the document text is the primary interaction surface.
- Chase queue rendering is stable and ordered; opening an item does not reshuffle the list.

## Task Commits

Each task was committed atomically:

1. **Task 1: Render inline citation chips as hyperlinks (no button rows)** - `6dc10d3` (feat)
2. **Task 2: Ensure right-pane collector activates without breaking document order** - `bc73fad` (feat)

**Plan metadata:** (docs commit created after task commits)

## Files Created/Modified
- `frontend/ui.py` - Render inline citation chips as hyperlinks and remove per-paragraph citation button rows.
- `frontend/components/chase_queue.py` - Enforce stable ordering (citation_index, target_id) regardless of activation.

## Decisions Made
None - followed plan as specified.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Preserve sentence_id provenance when switching to hyperlink navigation**

- **Found during:** Task 1 (Render inline citation chips as hyperlinks)
- **Issue:** Removing the button-row click handler eliminated the only path that populated `selected_callout_tuple.sentence_id`, reducing judgment provenance and downstream context.
- **Fix:** Added best-effort lookup of the first matching `sentence_id` from the loaded document body when selecting a citation.
- **Files modified:** `frontend/ui.py`
- **Verification:** `python -m py_compile frontend/ui.py`
- **Committed in:** `6dc10d3`

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Auto-fix preserved existing provenance behavior while meeting UX-01; no scope creep.

## Issues Encountered
- Pre-commit `black` reformatted `frontend/ui.py` during Task 1 commit; restaged and committed successfully.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Ready for `08-06-PLAN.md` (Source Bin UI + background processing wiring).

---
*Phase: 08-workspace-fast-path-review-ux*
*Completed: 2026-02-03*
