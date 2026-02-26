---
phase: 08-workspace-fast-path-review-ux
plan: 01
subsystem: api
tags: [tei, grobid, segmentation, citations, sentence-id, pytest]

requires:
  - phase: 06-evidence-review-selection
    provides: TEI sentence_id used as stable excerpt/jump anchor
provides:
  - Deterministic fallback sentence segmentation for suspicious TEI <s> boundaries in build_document_body
  - Pytest coverage for suspicious-boundary detection and citation preservation
affects: [phase-08-ui, citation-ux, inline-citations]

tech-stack:
  added: []
  patterns:
    - citation-safe placeholder token stream for sentence splitting
    - stable sentence_id derived from paragraph_id + index + sha1(text)

key-files:
  created:
    - tests/test_tei_body_segmentation.py
  modified:
    - backend/tei_body.py

key-decisions:
  - "Suspicious TEI sentence detection uses explicit length thresholds (480 per <s>, 600 for single-<s> paragraph)."
  - "Fallback sentence_id format: {paragraph_id}-fb-{index}-{sha1(text)[:10]}."

patterns-established:
  - "Rehydrate citation segments after splitting by placeholders to preserve citation_index/target_id."

duration: 8 min
completed: 2026-02-02
---

# Phase 8 Plan 01: Robust TEI Sentence Fallback Segmentation Summary

**Deterministic fallback sentence segmentation for suspicious TEI/GROBID boundaries while preserving inline citation metadata and stable sentence ids.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-02T22:16:03Z
- **Completed:** 2026-02-02T22:24:58Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Added suspicious-boundary detection that triggers a deterministic fallback segmenter for paragraphs whose TEI <s> nodes are likely malformed
- Implemented citation-safe sentence splitting via placeholder tokenization so citation segments keep their `citation_index` and `target_id`
- Generated stable fallback `sentence_id` values and ensured citation segments inherit the containing fallback sentence id

## Task Commits

Each task was committed atomically:

1. **Task 1: RED: Add failing tests for suspicious TEI sentence boundaries** - `91baf95` (test)
2. **Task 2: GREEN: Implement fallback sentence segmentation in tei_body** - `af43ead` (feat)

## Files Created/Modified

- `tests/test_tei_body_segmentation.py` - regression tests for suspicious-boundary fallback + citation preservation
- `backend/tei_body.py` - fallback sentence segmentation path with stable generated sentence ids

## Decisions Made

- Used explicit length-based heuristics (480 chars per <s>, 600 chars for single-<s> paragraph) to keep fallback triggers simple and predictable.
- Generated fallback sentence ids deterministically from paragraph id + sentence index + short hash of sentence text.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Ready for `.planning/phases/08-workspace-fast-path-review-ux/08-03-PLAN.md`.

---

*Phase: 08-workspace-fast-path-review-ux*
*Completed: 2026-02-02*
