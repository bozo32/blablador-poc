---
phase: 07-validation-export
plan: 02
subsystem: ui
tags: [streamlit, python, requests, pytest]

# Dependency graph
requires:
  - phase: 06-evidence-review-selection
    provides: EvidenceStore/evidence_api patterns for session caching and Streamlit-safe API helpers
provides:
  - Frontend judgment_api wrappers for judgment CRUD/list and export downloads
  - Session-backed JudgmentStore for per-claim caching and callout status aggregation
  - Unit tests covering caching semantics and callout outcome computation
affects: [07-validation-export, frontend/ui.py, exports]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Mirror EvidenceStore patterns for new session-backed UI state (store + api modules)"
    - "Return Streamlit download_button-ready export payloads without JSON decoding"

key-files:
  created:
    - frontend/judgment_api.py
    - frontend/judgment_store.py
    - tests/test_judgment_frontend_store.py
  modified: []

key-decisions:
  - "Callout validated=true only when any matching claim has status=final and a verdict set"
  - "When multiple final verdicts disagree for a callout, outcome collapses to 'uncertain'"
  - "If stored judgments omit target_id, callout_status falls back to (doc_id, citation_index, None)"

patterns-established:
  - "Per-doc callout index keyed by (doc_id, citation_index, target_id) stored in session_state"

# Metrics
duration: 6 min
completed: 2026-02-02
---

# Phase 7 Plan 2: Frontend Judgment Store Summary

**Frontend can fetch/cache per-claim judgments, compute callout validated/outcome from stored state, and download export payloads via Streamlit-safe helpers.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-02T08:35:18Z
- **Completed:** 2026-02-02T08:41:48Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Added `frontend/judgment_api.py` wrappers for `/claims/{claim_id}/judgment`, `/judgments`, and `/judgments/export` with Streamlit toast-style error handling.
- Added `frontend/judgment_store.py` session-backed caching layer (per-claim + per-doc indexes) and callout status aggregation with legacy `target_id` fallback.
- Added `tests/test_judgment_frontend_store.py` unit tests for caching behavior and validated/outcome semantics.

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement judgment_api helpers (CRUD/list/export) following evidence_api conventions** - `38c5fa6` (feat)
2. **Task 2: Implement session-backed JudgmentStore + unit tests for caching and callout aggregation** - `4032c4e` (feat)

**Plan metadata:** [pending]

## Files Created/Modified
- `frontend/judgment_api.py` - Judgment API wrappers + export downloader returning `st.download_button`-ready payloads.
- `frontend/judgment_store.py` - Session-backed JudgmentStore with doc-level callout indexes and aggregation helpers.
- `tests/test_judgment_frontend_store.py` - Store unit tests using Streamlit + API stubs.

## Decisions Made
None - followed plan as specified.

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
- Pre-commit initially failed due to unrelated staged changes; unstaged them to keep task commits atomic.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Ready for `07-03-PLAN.md` to integrate JudgmentStore into `frontend/ui.py` (inline editing, callout indicators, and download buttons).

---
*Phase: 07-validation-export*
*Completed: 2026-02-02*
