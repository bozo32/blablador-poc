---
phase: 06-evidence-review-selection
plan: 02
subsystem: frontend
tags: [streamlit, evidence-review, selection, css]

# Dependency graph
requires:
  - phase: 06-evidence-review-selection
    provides: Excerpt/jump + selection backend APIs
provides:
  - Streamlit Source-view evidence review with paragraph-bounded excerpt previews
  - Session-backed selection controls persisted via `/claims/{claim_id}/evidence/selection`
affects: [06-evidence-review-selection, 07-validation-export]

# Tech tracking
key-files:
  modified:
    - frontend/evidence_api.py
    - frontend/evidence_store.py
    - frontend/ui.py
    - frontend/assets/evidence_review.css

key-decisions:
  - "UI normalizes backend labels `entails/contradicts/neutral` into `entail/contradict/neutral` for consistent rendering."
  - "Source view groups candidates by TEI section path with a Body fallback, and shows the top 5 hits first."
  - "Evidence selections persist per claim_id with support/contradict/uncertain/none verdicts, optional primary, optional secondaries (each with rationale), and an uncertain-note requirement."
---

# Phase 06 Plan 02: Evidence Review UI + Selection Summary

Implemented the Phase 06 reviewer UX in Streamlit: a `Source` evidence view that renders TEI paragraph-bounded excerpt windows (with highlighted sentences) and a right-rail selection panel that saves/restores reviewer verdicts and chosen spans.

## What Landed

- `frontend/evidence_api.py`
  - Added helpers for excerpt/jump endpoints and evidence selection GET/PUT.
- `frontend/evidence_store.py`
  - Added session-cached selection state, selection save wiring, and excerpt preview caching.
  - Normalized incoming candidate labels (`entails/contradicts/...`) so the UI can treat them consistently.
- `frontend/ui.py`
  - Added `Citing` vs `Source` mode toggle in the evidence panel.
  - `Source` mode renders Top hits, then groups remaining candidates by section path, with expandable excerpt previews.
  - Added selection controls (verdict + primary/secondary + required notes/rationales) with persistence.
- `frontend/assets/evidence_review.css`
  - Added lightweight styling for excerpt highlights, muted candidates, and selection summary rail.

## Verification

- Automated: `pytest -q`
- Manual checkpoint still required (see `.planning/phases/06-evidence-review-selection/06-02-PLAN.md` human verify steps).

## Task Commits

- `96e34ee` feat(06-02): add excerpt + selection helpers
- `96a0fe9` feat(06-02): add source view evidence review + selection rail

---

*Phase: 06-evidence-review-selection*
