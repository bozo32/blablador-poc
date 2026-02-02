---
phase: 07-validation-export
plan: 03
subsystem: ui
tags: [streamlit, judgments, export, json, csv]

# Dependency graph
requires:
  - phase: 07-validation-export/07-01
    provides: FastAPI judgment CRUD + export endpoints (claim/callout; JSON/CSV; draft/final semantics)
  - phase: 07-validation-export/07-02
    provides: Session-backed JudgmentStore + judgment_api export download helpers
provides:
  - Inline per-claim judgment editing (draft/final, verdict, structured notes) on the Streamlit claim card
  - Outcome-colored callout chips with validated/unvalidated indicators and jump-to-judgment navigation
  - Streamlit download buttons for judgment exports (claim/callout x JSON/CSV; core/verbose; final-only default)
affects: [frontend/ui.py, reviewer-workflow, exports]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "JudgmentStore drives UI hydration + deterministic provenance for saves and callout status"
    - "Export downloads use judgment_api payloads suitable for st.download_button (no JSON decoding)"

key-files:
  created: []
  modified:
    - frontend/ui.py
    - frontend/assets/judgment.css
    - frontend/judgment_store.py
    - frontend/components/chase_queue.py
    - frontend/components/chasing_panel.py
    - frontend/state_keys.py

key-decisions:
  - "Verdict/status UI uses human labels but persists lowercase literals to match backend Literal models"

patterns-established:
  - "Citation chips receive validated/unvalidated + outcome CSS classes derived from callout_status(doc_id, citation_index, target_id)"

# Metrics
duration: 5h 56m
completed: 2026-02-02
---

# Phase 7 Plan 3: Streamlit Judgment UI + Exports Summary

**Streamlit claim cards now capture draft/final judgments with structured notes, callout chips show validated/outcome status with jump-to-judgment navigation, and exports download as JSON/CSV (claim/callout; core/verbose; final-only default).**

## Performance

- **Duration:** 5h 56m
- **Started:** 2026-02-02T13:37:45+01:00
- **Completed:** 2026-02-02T19:34:36+01:00
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Added inline judgment controls on the claim card: Draft/Final, verdict selection, and optional structured notes with collapse-after-save behavior.
- Styled citation callout chips to reflect validated/unvalidated state with outcome-specific icon+color styling and a click target that routes to the relevant claim/judgment view.
- Added export downloads from the UI for claim and callout shapes in JSON (default) and CSV, supporting core/verbose modes and final-only-by-default filtering.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add inline claim-card judgment controls (draft/final + verdict + structured notes)** - `69c7d4d` (feat)
2. **Task 2: Add callout validated indicators + export downloads (JSON/CSV; claim/callout; core/verbose; final-only default)** - `6b4be3e` (feat)

Follow-up fixes after integration testing:

- `740b8b1` (fix): avoid empty Streamlit radio label
- `19f566c` (fix): stabilize chase context + judgment save feedback

**Plan metadata:** [pending]

## Files Created/Modified
- `frontend/ui.py` - Claim-card judgment controls, callout chip status badges + navigation, and export download section.
- `frontend/assets/judgment.css` - Outcome-specific chip styling + compact judgment form layout.
- `frontend/judgment_store.py` - Integration support for deterministic provenance and callout status consumption.
- `frontend/components/chase_queue.py` - Chase-mode integration touchpoints for judgment navigation.
- `frontend/components/chasing_panel.py` - Chase panel selection/state flow updates supporting the new routing behavior.
- `frontend/state_keys.py` - Centralized session state keys used by jump-to-claim/judgment navigation.

## Decisions Made
None - followed plan as specified.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Avoided empty-label Streamlit radio that caused rendering issues**
- **Found during:** Task 1 (UI integration)
- **Issue:** Streamlit radio control was created with an empty label in one code path.
- **Fix:** Provided a non-empty label (while keeping the UI compact) to prevent Streamlit rendering warnings/errors.
- **Files modified:** frontend/ui.py
- **Committed in:** 740b8b1

**2. [Rule 1 - Bug] Stabilized chase-context selection + save feedback wiring for judgment actions**
- **Found during:** Task 2 (callout navigation + exports integration)
- **Issue:** Chase-mode selection context and/or session keys caused inconsistent claim focus after clicking callout indicators; save feedback could be confusing.
- **Fix:** Consolidated/standardized session state key usage and chase selection wiring so callout navigation reliably focuses the intended claim and save feedback is consistent.
- **Files modified:** frontend/components/chase_queue.py, frontend/components/chasing_panel.py, frontend/state_keys.py, frontend/ui.py
- **Committed in:** 19f566c

---

**Total deviations:** 2 auto-fixed (2 bug)
**Impact on plan:** Both fixes were required for Streamlit correctness and stable navigation; no scope creep.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 7 UI flows (judgment capture + indicators + exports) are complete and verified by human checkpoint.

---
*Phase: 07-validation-export*
*Completed: 2026-02-02*
