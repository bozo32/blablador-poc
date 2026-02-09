---
phase: 09-claim-graph-consensus-viz
plan: 05
subsystem: ui
tags: [streamlit, streamlit-agraph, claim-graph, consensus, votes]

requires:
  - phase: 09-claim-graph-consensus-viz/09-03
    provides: Backend claim-graph endpoints (subgraph, votes, claim-link CRUD)
  - phase: 09-claim-graph-consensus-viz/09-04
    provides: Project meta reviewer identity + persistence for active reviewer
provides:
  - Interactive claim graph tab with pan/zoom, selection, inspection, and voting
  - Frontend API client helpers for claim graph endpoints
  - Streamlit renderer wrapper that maps backend payloads to agraph nodes/edges
affects: [09-claim-graph-consensus-viz/09-06, compare-mode]

tech-stack:
  added: [streamlit-agraph]
  patterns:
    - "frontend/*_api.py request helper module with typed-ish dict payloads + GraphApiError"
    - "Streamlit component wrapper returns selection dict {type,id}"

key-files:
  created:
    - frontend/graph_api.py
    - frontend/components/claim_graph_panel.py
  modified:
    - environment.yml
    - frontend/ui.py
    - frontend/assets/workspace.css

key-decisions:
  - "Use streamlit-agraph to avoid a custom JS build while still supporting pan/zoom + selection"
  - "Persist Graph tab filter settings under project_meta.graph_settings"

patterns-established:
  - "Graph tab renders graph + inspector in a 3:2 column split"
  - "Edge badge uses support/contradict counts (n_support/n_contradict) for at-a-glance consensus"

duration: 27min
completed: 2026-02-07
---

# Phase 9 Plan 05: Claim Graph Tab Summary

**Streamlit Graph tab now renders an interactive claim consensus graph (pan/zoom) with node/edge inspection, per-reviewer vote attribution, and manual edge materialization.**

## Performance

- **Duration:** 27 min
- **Started:** 2026-02-06T23:38:52Z
- **Completed:** 2026-02-07T00:06:00Z
- **Tasks:** 3/3
- **Files modified:** 5

## Accomplishments

- Added `streamlit-agraph` dependency for an embedded interactive graph renderer.
- Implemented `frontend/graph_api.py` + `frontend/components/claim_graph_panel.py` to fetch/render claim subgraphs and return selection events.
- Replaced the legacy citation Graphviz tab with a claim graph + right-side inspection panel supporting voting, vote attribution, and candidate edge materialization.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add interactive Streamlit graph component dependency** - `35f9859` (chore)
 
2. **Task 2: Create frontend graph_api client and claim_graph_panel renderer** - `d9093e4` (feat)
3. **Task 3: Replace Graph tab with claim graph + inspection panel (MVP)** - `91293d9` (feat)

**Plan metadata:** _added after execution_ (docs: complete plan)

## Files Created/Modified

- `environment.yml` - Adds `streamlit-agraph` to the pip dependencies.
- `frontend/graph_api.py` - Request helpers for `/graph/*` claim-graph endpoints.
- `frontend/components/claim_graph_panel.py` - Maps backend nodes/edges to agraph primitives; returns selection `{type,id}`.
- `frontend/ui.py` - Graph tab now renders claim graph controls, graph canvas, and inspection/voting UI.
- `frontend/assets/workspace.css` - Adds inspector + vote column styling for the claim graph panel.

## Decisions Made

- Used `streamlit-agraph` for interactive rendering (pan/zoom + selection) without introducing a custom JS build step.
- Stored Graph tab filters (hops, edge_cap, min_votes, provenance toggles) in `project_meta.graph_settings` for per-project persistence.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Pre-commit hooks reformatted `frontend/ui.py` (black) and initially failed flake8 line-length checks; resolved by wrapping long strings/HTML rendering into small helpers.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Graph tab MVP wiring is in place; ready for follow-up work like compare mode and richer graph navigation/search.

---

*Phase: 09-claim-graph-consensus-viz*
*Completed: 2026-02-07*
