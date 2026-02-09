---
phase: 09-claim-graph-consensus-viz
plan: 06
subsystem: ui
tags: [streamlit, claim-graph, compare, reviewers]

requires:
  - phase: 09-claim-graph-consensus-viz/09-05
    provides: Interactive claim graph tab (inspection + voting)
provides:
  - Compare mode for claim judgments and edge votes across two reviewers
  - Disagreements list with navigation/focus back into the graph selection model
affects: [09-claim-graph-consensus-viz/09-07]

key-files:
  modified:
    - frontend/ui.py

verification:
  automated:
    - python -m py_compile frontend/ui.py
  manual:
    - Pending: 09-06-PLAN.md checkpoint walkthrough

commits:
  - 3617b31 feat(09-06): add compare mode for graph disagreements
completed: 2026-02-07
---

# Phase 9 Plan 06: Compare Mode Summary

Compare mode is implemented in the Graph tab inspector so two reviewers can be selected and their claim judgments + edge votes compared side-by-side, with a disagreements list that can focus/select the relevant node/edge.

## What Changed

- Added compare pair persistence in project meta (`compare_reviewer_a`, `compare_reviewer_b`).
- Compute visible-subgraph disagreements (claims + edges), including missing-vs-present as a difference.
- Clicking a disagreement focuses the graph by setting a pending selection/center and rerunning.

## Verification Status

- Automated: `py_compile` passes for `frontend/ui.py`.
- Manual: the end-to-end checkpoint in `.planning/phases/09-claim-graph-consensus-viz/09-06-PLAN.md` has not been re-run/recorded in planning docs yet.

---

*Phase: 09-claim-graph-consensus-viz*
*Completed: 2026-02-07*
