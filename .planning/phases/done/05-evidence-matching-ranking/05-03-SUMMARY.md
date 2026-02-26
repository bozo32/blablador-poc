---
phase: 05-evidence-matching-ranking
plan: 03
subsystem: ui
tags: [streamlit, evidence, state, api]

# Dependency graph
requires:
  - phase: 05-evidence-matching-ranking
    provides: 05-02 evidence service endpoints
provides:
  - Streamlit evidence API client with concurrency guard
  - Session-backed EvidenceStore coordinating claims, filters, and reruns
  - UI hooks that auto-sync claim focus and attachment lifecycle with evidence refreshes
affects: [05-evidence-matching-ranking, 05-04-evidence-matching-ranking]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Session-state EvidenceStore ensures deterministic claim state across reruns and load-more actions
    - Timeline-driven stale markers trigger automatic evidence refresh when attachments finish matching

key-files:
  created:
    - frontend/evidence_api.py
    - frontend/evidence_store.py
    - tests/test_evidence_store.py
  modified:
    - frontend/ui.py
    - frontend/claim_queue.py
    - frontend/attachment_queue.py

key-decisions:
  - Auto-refresh evidence when attachment timeline emits matched, so reviewers don't have to rerun manually when parsing completes.
  - Persist claim selection via EvidenceStore active-claim tracking and drive UI syncs from that canonical state.

patterns-established:
  - "Evidence fetch guard": keep per-claim counters in session state and annotate responses with remaining slots to govern load-more UX.
  - "Timeline-driven refresh": reuse claim_queue timeline events to mark EvidenceStore entries stale whenever attachments change.

# Metrics
duration: 15 min
completed: 2026-01-28
---

# Phase 05 Plan 03: Evidence store + claim sync wiring Summary

**Streamlit evidence panel now auto-syncs claim selections and attachment completions with backend rerun state**

## Performance

- **Duration:** 15 min
- **Started:** 2026-01-28T08:13:20Z
- **Completed:** 2026-01-28T08:28:36Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- Added `frontend/evidence_api.py` client helpers with Streamlit-aware error handling and a hard limit of two concurrent fetches per claim.
- Built the session-backed `EvidenceStore` plus pytest coverage for filter toggles, load-more persistence, rerun serialization, and stale-mark handling.
- Wired claim queue timeline events and a new evidence panel in `frontend/ui.py` so claim selection, attachment matches, rerun requests, and load-more controls stay in sync without manual refresh.

## Task Commits

Each task was committed atomically:

1. **Task 1: Create evidence API client helpers** - `87f6380` (feat)
2. **Task 2: Implement evidence session store** - `bfee423` (feat)
3. **Task 3: Hook store into claim + attachment flows** - `d9b3f18` (feat)

**Plan metadata:** _pending commit_ (docs update will capture SUMMARY + STATE)

## Files Created/Modified

- `frontend/evidence_api.py` - Streamlit-aware HTTP client for evidence list/rerun/history/export endpoints with concurrency guard metadata.
- `frontend/evidence_store.py` - Session-backed state machine handling candidates, filters, load-more counts, rerun queue, and stale markers.
- `frontend/ui.py` - Adds the Ranked Evidence panel with claim selector, rerun/load-more actions, and placeholder rendering while data loads.
- `frontend/claim_queue.py` - Notifies the EvidenceStore whenever claim timelines log attachment events and exposes an active-claim setter.
- `frontend/attachment_queue.py` - Emits a `matched` timeline event when backend status transitions to matched so the store can refresh.
- `tests/test_evidence_store.py` - Pytest coverage for API guard logic, store transitions, rerun serialization, and stale refresh scenarios.

## Decisions Made

- Auto-refresh evidence via timeline events: when attachment statuses advance (attach/detach/matched), mark the claim’s EvidenceStore entry stale and immediately resync if it is the active claim.
- Persist claim focus in session state and drive all UI evidence interactions (rerun, load-more, filters) through the canonical EvidenceStore instance for determinism and concurrency safety.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Emit matched timeline events to trigger evidence refresh**

- **Found during:** Task 3 (Hook store into claim + attachment flows)
- **Issue:** Attachment lifecycle never recorded a `matched` event, so the evidence store could not detect when parsing completed to auto-sync candidates.
- **Fix:** Extended `_hydrate_from_backend` in `frontend/attachment_queue.py` to fire a `matched` timeline entry when status transitions, allowing `claim_queue` to mark the claim stale and refresh evidence automatically.
- **Verification:** Manual UI-driven tests plus `pytest tests/test_evidence_store.py` confirm EvidenceStore stale markers trigger a new sync.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Evidence store, API helpers, and UI hooks are in place so Plan 05-04 can focus on visual evidence cards, filters, and rationale sidebar rendering.
- Need to design the final evidence card layout plus accept/reject actions now that claim sync and load-more/rerun plumbing are stable.

---
*Phase: 05-evidence-matching-ranking*
*Completed: 2026-01-28*
