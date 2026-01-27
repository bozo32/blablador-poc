---
phase: 04-evidence-attachment
plan: 06
subsystem: attachment-queue
tags: [streamlit, heuristics, testing]

# Dependency graph
requires:
  - phase: 04-05
    provides: Queue lifecycle statuses + backend plumbing
provides:
  - Auto-match orchestration tied to claim heuristics + timeline logging
  - Queue UI for ambiguous attachments with manual reassignment controls
affects: [05-evidence-matching, 06-evidence-review]

# Tech tracking
tech-stack:
  added:
    - streamlit selectbox-driven reassignment controls
  patterns:
    - "Ambiguous attachments are highlighted and tracked until user resolves them"
    - "Auto-matched items are uploaded via the same backend pathway as manual drops"

key-files:
  created:
    - tests/test_claim_queue.py
  modified:
    - frontend/attachment_queue.py
    - frontend/claim_queue.py
    - frontend/ui.py
    - frontend/assets/attachment_panel.css

key-decisions:
  - "Treat auto- and manual assignments the same way by routing through `_upload_to_backend` so backend lifecycle status remains the source of truth"
  - "Surface claim dropdown labels as `callout • truncated claim` so reviewers can quickly choose targets"

patterns-established:
  - "Queue summary banner warns when ambiguous attachments require attention"
  - "Accepting a suggestion or manual assignment logs timeline events (`assigned`, `auto-suggestion`) for audit trails"

# Metrics
duration: 30m
completed: 2026-01-27
---

# Phase 04 Plan 06: Auto-match heuristics + manual reassignment controls

**Automated attachment matching with fallback UI so reviewers can resolve ambiguous drops directly from the queue panel.**

## Performance

- **Duration:** ~30m
- **Started:** 2026-01-27T20:00:00Z
- **Completed:** 2026-01-27T20:30:00Z
- **Tasks:** 2
- **Files touched:** 5
- **Tests:** `pytest tests/test_claim_queue.py`

## Accomplishments

- Implemented `auto_match_queue_items` to run claim heuristics whenever new queue items appear or backend state syncs, logging ambiguous cases, persisting candidate metadata, and routing confident matches through the normal backend upload flow.
- Added `claim_queue.get_claim_options` plus UI controls: ambiguous attachments now display warnings, offer “accept suggestion” buttons, and expose a claim dropdown + assign button; queue summary also highlights how many files need attention, and new tests cover matching heuristics + option labels.

## Task Commits

1. **Task 1: Implement queue auto-match + ambiguity tracking** — `77ef4af` (`feat`)
2. **Task 2: Add manual reassignment controls + alerts in the queue panel** — `01261a5` (`feat`)

## Files Created/Modified

- `frontend/attachment_queue.py` — normalized status helper, auto-match logic, ambiguous tracking, and refactored assignment helper to re-use backend uploads.
- `frontend/claim_queue.py` — claim option label helper for dropdowns.
- `frontend/ui.py` — summary banner, ambiguous warnings, auto-suggestion button, claim selectbox, and queue-level attention indicator.
- `frontend/assets/attachment_panel.css` — queue card styling per status (from earlier wave) reused for the new controls.
- `tests/test_claim_queue.py` — regression coverage for heuristics + label truncation without requiring a Streamlit runtime.

## Verification Notes

- Automated: `pytest tests/test_claim_queue.py`
- Manual: `python -m streamlit run frontend/ui.py` → drop PDFs, observe ambiguous banner and reassignment controls.

## Issues / Follow-ups

- Backend still retains legacy attachments when reassigned; a future enhancement should add an API endpoint to move/delete obsolete attachments after manual reassignment.

---
*Phase: 04-evidence-attachment*
*Plan: 06*
