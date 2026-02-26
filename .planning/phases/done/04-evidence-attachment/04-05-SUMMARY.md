---
phase: 04-evidence-attachment
plan: 05
subsystem: attachments
tags: [fastapi, streamlit, pytest]

# Dependency graph
requires:
  - phase: 04-03
    provides: Attachment persistence API + polling
provides:
  - Backend lifecycle helpers for converting/parsing/matched statuses
  - Streamlit queue UI and styles that surface lifecycle counts + pills
affects: [05-evidence-matching]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Attachment lifecycle events always emit converting → parsing → matched, logged in timelines"
    - "Queue summary chip mirrors backend status counts so reviewers can see progress"

key-files:
  modified:
    - backend/attachment_store.py
    - backend/attachment_pipeline.py
    - scripts/attachment_smoke.py
    - tests/test_attachment_store.py
    - tests/test_attachment_pipeline.py
    - frontend/attachment_queue.py
    - frontend/ui.py
    - frontend/assets/attachment_panel.css

key-decisions:
  - "Treat legacy 'ready' statuses as 'matched' to keep existing attachments compatible"
  - "Queue panel shows a status digest banner plus color-coded cards per lifecycle state"

patterns-established:
  - "CLI smoke exits when attachments reach `matched`, matching UI terminology"
  - "All status chips + pills share the same label map to avoid drift between backend and UI"

# Metrics
duration: 25m
completed: 2026-01-27
---

# Phase 04 Plan 05: Attachment lifecycle statuses (pending → converting → parsing → matched)

**Backend + frontend lifecycle polish so reviewers see converting/matched progress with matching timelines, tests, and CLI coverage.**

## Performance

- **Duration:** ~25m
- **Started:** 2026-01-27T19:35:00Z
- **Completed:** 2026-01-27T20:00:00Z
- **Tasks:** 2
- **Files touched:** 8
- **Tests:** `pytest tests/test_attachment_store.py tests/test_attachment_pipeline.py`

## Accomplishments

- Added `STATUS_CONVERTING`/`STATUS_MATCHED`, lifecycle helpers (`mark_converting`, `mark_matched`), legacy status normalization, and resumable filters so backend timelines now record pending → converting → parsing → matched; updated pipeline, tests, and the CLI smoke runner to follow the new sequence.
- Refreshed Streamlit queue UI to consume the new statuses: summary chip now lists pending/converting/matched counts, the queue panel shows a digest banner plus contextual info, and per-item cards gain status-specific styling with color-coded borders.

## Task Commits

1. **Task 1: Extend backend lifecycle to convert → parse → match** — `cefaae5` (`feat`)
2. **Task 2: Surface converting/matched states in the Streamlit queue** — `bf1a6bc` (`feat`)

## Files Created/Modified

- `backend/attachment_store.py` — new lifecycle helpers, legacy status normalization, resumable filter updates.
- `backend/attachment_pipeline.py` — emits converting + matched events around parsing.
- `scripts/attachment_smoke.py` — waits for `matched` or `error` before exiting.
- `tests/test_attachment_store.py`, `tests/test_attachment_pipeline.py` — cover new statuses and timeline events.
- `frontend/attachment_queue.py` — normalizes backend statuses, recomputes summaries, exposes converting counts.
- `frontend/ui.py` — queue panel summary banner + new helper copy, hooking to updated status labels.
- `frontend/assets/attachment_panel.css` — queue card accents for pending/converting/parsing/matched/error.

## Verification Notes

- Automated: `pytest tests/test_attachment_store.py tests/test_attachment_pipeline.py`
- Manual UI run recommended to see updated chips (`python -m streamlit run frontend/ui.py`).

## Issues / Follow-ups

- None. Legacy `ready` statuses are normalized to `matched`, so old attachments remain visible without migration.

---
*Phase: 04-evidence-attachment*
*Plan: 05*
