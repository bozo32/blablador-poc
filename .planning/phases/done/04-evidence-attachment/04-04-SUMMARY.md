---
phase: 04-evidence-attachment
plan: 04
subsystem: claim-queue
tags: [streamlit, ui, clipboard]

# Dependency graph
requires:
  - phase: 04-01
    provides: Retrieval dossier endpoint + claim UI entry point
provides:
  - Shared clipboard helper component for Streamlit apps
  - Retrieval instructions panel with real copy workflow + manual fallback
affects: [08-workspace-organization]

# Tech tracking
tech-stack:
  added:
    - streamlit.components.v1 html clipboard helper
  patterns:
    - "All copy actions expose manual fallback text areas when clipboard APIs fail"

key-files:
  created:
    - frontend/clipboard.py
  modified:
    - frontend/claim_queue.py

key-decisions:
  - "Use a lightweight HTML component to trigger navigator.clipboard while reporting success/fallback via Streamlit events"
  - "Always show the retrieval instructions text block so reviewers can verify what was copied"

patterns-established:
  - "Success toasts include a snippet of the copied payload for instant confirmation"
  - "Fallback textarea appears automatically when clipboard permissions are denied"

# Metrics
duration: 15m
completed: 2026-01-27
---

# Phase 04 Plan 04: Clipboard-backed retrieval instructions copy action

**Reusable clipboard helper plus Streamlit retrieval panel wiring so reviewers can trust the copy control and see what was captured.**

## Performance

- **Duration:** ~15m
- **Started:** 2026-01-27T19:20:00Z
- **Completed:** 2026-01-27T19:35:00Z
- **Tasks:** 2
- **Files touched:** 2

## Accomplishments

- Built `frontend/clipboard.py`, a tiny component that uses `navigator.clipboard` when available, reports success/failure back to Streamlit, shows manual fallback text areas, and emits standardized snippets + timestamps for downstream UI.
- Reworked `render_retrieval_instructions` to call the helper, show success toasts with a truncated payload, surface the full instructions within `st.code`, and note when the text was last copied.

## Task Commits

1. **Task 1: Build clipboard helper with fallback messaging** — `be67f32` (`feat`)
2. **Task 2: Integrate helper into retrieval instructions panel** — `0411965` (`feat`)

## Files Created/Modified

- `frontend/clipboard.py` — HTML-based copy control with session tracking, toasts, and fallback textarea.
- `frontend/claim_queue.py` — Retrieval instructions UI now imports the helper, surfaces success banners, captures copy timestamps, and keeps instructions visible via `st.code`.

## Verification Notes

- Clipboard behavior relies on browser APIs; manual Streamlit run recommended (`python -m streamlit run frontend/ui.py`). Automated tests not run in this repo context.

## Issues / Follow-ups

- None. Helper is generic and can be reused later (e.g., Phase 08 workspace copy affordances).

---
*Phase: 04-evidence-attachment*
*Plan: 04*
