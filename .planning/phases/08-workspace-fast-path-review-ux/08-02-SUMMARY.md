---
phase: 08-workspace-fast-path-review-ux
plan: 02
subsystem: api
tags: [fastapi, background-jobs, pause, json-state, attachments, evidence]

requires:
  - phase: 04-evidence-attachment
    provides: persisted attachment pipeline + background processing hooks
  - phase: 05-evidence-matching-ranking
    provides: EvidenceMatchingService rerun queueing + run persistence
provides:
  - Persistent global pause flag for background work under data/
  - FastAPI endpoints for UI polling/toggling background state
  - Pause-aware enqueue gates for attachments and auto evidence reruns
affects: [phase-08-ui, source-bin, review-ux, background-queues]

tech-stack:
  added: []
  patterns:
    - file-backed JSON state with atomic replace
    - pause gates at enqueue/dequeue boundaries (no mid-flight cancellation)

key-files:
  created:
    - backend/background_state.py
  modified:
    - backend/main.py
    - backend/attachment_pipeline.py
    - backend/evidence_matching/service.py
    - backend/tei_body.py

key-decisions:
  - "Pause is enforced only when starting new background work; in-flight work runs to completion."
  - "Manual evidence reruns ignore pause, but queued jobs only start when unpaused."

patterns-established:
  - "background_state.get_state()/set_paused() as the single pause source of truth"

duration: 6 min
completed: 2026-02-02
---

# Phase 8 Plan 02: Global Background Pause Summary

**Persistent global pause toggle with FastAPI polling endpoints, plus enqueue-time gates for attachment processing and auto evidence reruns.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-02T22:16:30Z
- **Completed:** 2026-02-02T22:22:45Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added stdlib-only persistent pause state stored under `data/background_state.json` with atomic writes
- Exposed `/background/state` and `/background/pause` so the UI can poll/toggle paused mode
- Prevented new attachment-processing threads and suppressed auto evidence reruns while paused (manual reruns still execute)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement persistent background pause state** - `3cbea87` (feat)
2. **Task 2: Wire pause-aware enqueuing + FastAPI endpoints** - `4632908` (feat)

## Files Created/Modified

- `backend/background_state.py` - file-backed pause flag with atomic temp-write + replace helpers
- `backend/main.py` - `GET /background/state` + `POST /background/pause` routes and paused-startup gating
- `backend/attachment_pipeline.py` - checks pause before starting attachment worker threads and records `timeline_event="paused"`
- `backend/evidence_matching/service.py` - gates auto reruns when paused and only dequeues queued work when unpaused
- `backend/tei_body.py` - fallback segmentation for suspicious single-<s> TEI paragraphs (test-suite unblock)

## Decisions Made

- Pause is checked at boundaries where new work starts (enqueue/dequeue), rather than trying to cancel in-flight threads.
- Auto evidence reruns enqueue while paused and start on the next request after unpausing; manual reruns still start immediately.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed failing TEI body segmentation test by adding suspicious single-<s> fallback splitting**

- **Found during:** Task 2 (pytest verification)
- **Issue:** `tests/test_tei_body_segmentation.py` expected long single `<s>` blocks to split into multiple sentences; implementation returned 1 sentence and failed the suite
- **Fix:** Added a heuristic fallback in `backend/tei_body.py` to split long single-`<s>` segments by sentence-ending punctuation while keeping citation segments attached to the correct sentence
- **Files modified:** `backend/tei_body.py`
- **Verification:** `pytest -q`
- **Committed in:** `4632908` (part of Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Deviation was required to keep the test suite green; no scope creep beyond correctness.

## Issues Encountered

- Pre-commit hooks (black/flake8) required minor formatting/docstring adjustments before Task 2 could commit cleanly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Backend now exposes a single durable pause toggle that the upcoming Source Bin + workspace UI can surface as a global banner/toggle.
- Ready for `.planning/phases/08-workspace-fast-path-review-ux/08-03-PLAN.md`.

---

*Phase: 08-workspace-fast-path-review-ux*
*Completed: 2026-02-02*
