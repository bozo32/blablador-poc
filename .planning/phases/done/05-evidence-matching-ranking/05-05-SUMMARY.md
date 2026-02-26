---
phase: 05-evidence-matching-ranking
plan: 05
subsystem: api
tags: [python, fastapi, attachments, testing, reruns]

# Dependency graph
requires:
  - phase: 04-evidence-attachment
    provides: Attachment persistence, timelines, and metadata storage that this plan enriches with claim text.
provides:
  - "Attachment APIs now carry claim_text from uploads through storage so the UI can display it and reruns can reuse it."
  - "Evidence reruns resolve claim_text from attachments before erroring, keeping the service deterministic for newcomers."
  - 05-06-PLAN.md (frontend claim_text wiring)
  - 06-evidence-review-selection

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Claim_text metadata lives with attachments and seeds reruns so the evidence service never depends on ad-hoc manual data.
    - EvidenceMatchingService resolves claim_text from attachments before raising 409 so APIs only fail when no text exists at all.

key-files:
  created: []
  modified:
    - backend/schemas.py
    - backend/main.py
    - backend/attachment_store.py
    - backend/attachment_pipeline.py
    - backend/evidence_matching/service.py
    - tests/test_attachment_pipeline.py
    - tests/test_evidence_matching_api.py

key-decisions:
  - "Persist attachment claim_text so reruns inherit the UI-provided text without additional inputs."
  - "EvidenceMatchingService resolves claim_text from attachment metadata before failing to keep initial requests stable."

patterns-established:
  - "Claim_text metadata now travels with attachments and rerun jobs, making deterministic windows possible without manual seeding."
  - "API endpoints surface informative errors only when no claim_text exists across attachments or run history."

# Metrics
completed: 2026-01-28
---

# Phase 05: Evidence Matching + Ranking Summary

**Attachment claim_text now flows from uploads through reruns so the evidence service can build deterministic candidates immediately.**

## Performance

- **Duration:** 7 min
- **Started:** 2026-01-28T22:06:06Z
- **Completed:** 2026-01-28T22:13:58Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Persisted claim_text on attachments and surfaced it through the attachment API so the UI always knows what text was requested.
- Auto reruns now carry the stored claim_text into the evidence service, closing the 409 gap for new claims.
- EvidenceMatchingService reuses attachment/run metadata when resolving claim_text and regression tests guard both manual and attachment-triggered paths.

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend attachment schema + storage with claim text** - `37ad0fd` (feat)
2. **Task 2: Pipe claim text into rerun orchestration + harden ensure_current_run** - `36c92b7` (fix)

**Plan metadata:** docs(05-05): complete backend claim_text plan

_Note: TDD tasks may have multiple commits (test → feat → refactor)_

## Files Created/Modified

- `backend/schemas.py` - Added claim_text to the attachment create/request schemas so HTTP payloads can carry the text.
- `backend/main.py` - Forwarded claim_text from the attachment route into the attachment store.
- `backend/attachment_store.py` - Persisted claim_text in metadata so it survives both private/public views.
- `backend/attachment_pipeline.py` - Reloaded the record after parsing so the auto rerun picks up the stored claim_text.
- `backend/evidence_matching/service.py` - Resolved claim_text from attachments and previous runs, and improved the auto rerun plumbing.
- `tests/test_attachment_pipeline.py` - Asserted claim_text survives attachment parsing and public status responses.
- `tests/test_evidence_matching_api.py` - Added regression coverage for claim_text-aware evidence APIs and auto rerun plumbing.

## Decisions Made

- Persisted attachment claim_text so reruns inherit the UI-provided text without extra requests.
- EvidenceMatchingService resolves claim_text from attachment metadata before raising 409 so APIs remain stable.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external services or user configuration required.

## Next Phase Readiness

- Claim_text persistence closes the 409 gap so the UI’s next plan (05-06) can surface a rerun button that never fails.
- With reruns consuming stored claim_text, Phase 06’s evidence review workflows can rely on deterministic candidate lists.
