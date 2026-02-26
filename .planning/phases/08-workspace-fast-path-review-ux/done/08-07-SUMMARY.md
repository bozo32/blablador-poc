---
phase: 08-workspace-fast-path-review-ux
plan: 07
subsystem: [api, ui]
tags: [fastapi, streamlit, evidence-matching, colbert, reranking]

# Dependency graph
requires:
  - phase: 05-evidence-matching-ranking
    provides: EvidenceMatchingService + EvidencePipeline orchestration and rerun/history endpoints
  - phase: 08-workspace-fast-path-review-ux
    provides: Workspace settings drawer + Source Bin-driven auto reruns
provides:
  - Named execution profile registry (Fast/Local, Best/Local) with deterministic overrides
  - Profile-aware evidence reruns that apply per-run settings (including ColBERT path)
  - Compact UI selector that sends `advanced_settings.profile` to the backend on reruns
affects: [ml-02, evidence, workspace-ux]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-run settings overrides via settings model copy (no global mutation)"
    - "Named execution profiles carried through `advanced_settings.profile`"

key-files:
  created: []
  modified:
    - backend/settings.py
    - backend/schemas.py
    - backend/evidence_matching/service.py
    - frontend/ui.py
    - frontend/attachment_queue.py

key-decisions:
  - "Use a small hardcoded profile registry in backend/settings.py and apply it per rerun via model_copy."
  - "Carry the chosen profile via EvidenceRerunRequest.advanced_settings.profile (no new top-level field)."

patterns-established:
  - "Backend-owned routing: frontend selects profile by name; backend decides pipeline/reranker configuration"

# Metrics
duration: 1h 3m
completed: 2026-02-04
---

# Phase 8 Plan 7: Workspace + Fast-Path Review UX Summary

**Named execution profiles now route evidence reruns through backend-owned settings, including a ColBERT-enabled hybrid option.**

## Performance

- **Duration:** 1h 3m
- **Started:** 2026-02-03T23:09:18Z
- **Completed:** 2026-02-04T00:12:42Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Added a small server-side execution profile registry (`Fast/Local`, `Best/Local`) with deterministic pipeline/reranker overrides.
- Applied the selected profile inside backend evidence rerun execution without mutating global settings.
- Added a compact profile selector to the workspace settings drawer and threaded it through manual + auto rerun requests.

## Task Commits

Each task was committed atomically:

1. **Task 1: Define execution profiles and expose them in settings + API models** - `12acc9c` (feat)
2. **Task 2: Apply profile selection in backend evidence reruns + add UI selector** - `c7853e1` (feat)

**Plan metadata:** (added in final docs commit)

## Files Created/Modified
- `backend/settings.py` - Define named execution profiles and helpers to apply them per run.
- `backend/schemas.py` - Validate `advanced_settings.profile` typing for rerun requests.
- `backend/evidence_matching/service.py` - Apply selected profile overrides during rerun execution and record the profile in run metadata.
- `frontend/ui.py` - Add execution profile selector and attach profile to rerun requests.
- `frontend/attachment_queue.py` - Ensure auto reruns include the selected execution profile.

## Decisions Made
- Use a hardcoded profile registry (small named set) instead of exposing raw settings in the UI.
- Apply overrides via a copied settings object per rerun to keep routing backend-owned and deterministic.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Auto reruns could not include profile from ui.py alone**
- **Found during:** Task 2 (Apply profile selection in backend evidence reruns + add UI selector)
- **Issue:** Auto reruns are triggered from `frontend/attachment_queue.py` via a direct `/evidence/rerun` POST and did not include `advanced_settings.profile`.
- **Fix:** Added `advanced_settings={"profile": ...}` to the auto rerun request payload.
- **Files modified:** `frontend/attachment_queue.py`
- **Verification:** `pytest -q`
- **Committed in:** `c7853e1` (part of task commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Required to meet the plan's must-have that auto reruns also carry the profile; no scope creep.

## Issues Encountered
- Pre-commit `black` reformatted `backend/evidence_matching/service.py` during the Task 2 commit; re-staged and committed normally.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Ready for `.planning/phases/08-workspace-fast-path-review-ux/08-08-PLAN.md` (HF inference mode) with profiles available as the routing primitive.

---
*Phase: 08-workspace-fast-path-review-ux*
*Completed: 2026-02-04*
