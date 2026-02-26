---
phase: 05-evidence-matching-ranking
plan: 06
subsystem: ui
tags: [streamlit, evidence-store, attachments, claim-text, pytest]

# Dependency graph
requires:
  - phase: 05-evidence-matching-ranking (Plan 05)
    provides: Backend claim_text plumbing + metadata caching
provides:
  - Claim_text metadata is seeded when claim records register so EvidenceStore knows the text for every fetch/rerun.
  - The evidence panel now passes the active claim text into sync/load-more/rerun controls and surfaces a remediation banner/button when the backend requests claim_text.
  - Attachment uploads and API helpers now send claim_text and extract backend error detail so auto reruns run immediately.
affects:
  - Phase 06 (Evidence Review + Selection)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Claim_text metadata now flows from the claim registry through the session store into every fetch/rerun/attachment path.
    - UI remediation banners rely on backend error detail to prompt reviewers to resubmit claim text before reruns.

key-files:
  created: []
  modified:
    - frontend/claim_queue.py
    - frontend/evidence_store.py
    - frontend/ui.py
    - frontend/attachment_queue.py
    - frontend/evidence_api.py
    - tests/test_evidence_store.py

key-decisions:
  - "Cache claim_text metadata in EvidenceStore via claim_queue registry updates so background refreshes and reruns always have the text."
  - "Surface an explicit remediation path whenever the backend rejects evidence fetches for missing claim_text and use the cached/edited text to recover."
  - "Send claim_text from attachment uploads so backend auto reruns triggered after parsing have the data they need."

patterns-established:
  - "Claim_text flows end-to-end: claim registry → EvidenceStore metadata → API calls/attachments, eliminating silent 409s."
  - "UI remediation banner + Send claim text button use backend error detail to guide reviewers to resend text when needed."

# Metrics
completed: 2026-01-28
---
# Phase 05 Plan 06: Claim text plumbing and remediation summary

**Claim text metadata now travels through the claim registry, evidence store, UI, and attachment uploads so reruns succeed and reviewers receive remediation guidance.**

## Performance
- **Duration:** 10 min
- **Started:** 2026-01-28T22:18:45Z
- **Completed:** 2026-01-28T22:28:55Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- EvidenceStore caches claim_text metadata when claims register, falling back to that metadata for sync/load-more/rerun calls and refreshing when the text changes.
- The evidence panel now injects the active claim text into every fetch/rerun/load-more path and surfaces a warning + Send claim text button when the backend complains.
- Attachment uploads and API helpers send claim_text plus backend detail so auto reruns run immediately and reviewers know how to recover.

## Task Commits
1. **Task 1: Persist claim_text in store metadata and reuse it for sync/rerun calls** - `8ad7000` (feat)
2. **Task 2: Send claim_text from UI + attachment uploads and surface remediation prompt** - `ded665a` (feat)

**Plan metadata:** docs(05-06): complete claim text plan

## Files Created/Modified
- `frontend/claim_queue.py` - push claim_text/callout metadata into EvidenceStore state and forward claim_text when background events refresh the active claim.
- `frontend/evidence_store.py` - cache claim_text metadata, honor overrides when it changes, let load-more/apply-filter accept text, and keep reruns sending it.
- `frontend/ui.py` - pass the active claim text into each evidence API call, enrich rerun/load-more controls, and surface a missing-claim_text banner with a Send claim text action.
- `frontend/attachment_queue.py` - include claim_text in uploads so backend auto reruns are seeded immediately.
- `frontend/evidence_api.py` - surface backend error detail to help the UI detect missing claim_text errors.
- `tests/test_evidence_store.py` - cover metadata seeding, fallback fetches, and rerun payloads with claim_text.

## Decisions Made
- Cache claim_text metadata in the EvidenceStore whenever claims register so refreshes and reruns can reuse the text without manual seeding.
- Surface an explicit remediation flow that resends cached or edited text when the backend reports a missing claim_text (409) and unblock the evidence fetch.
- Send claim_text as part of every attachment upload so backend auto reruns triggered after parsing have the data they need.

## Deviations from Plan
None - plan executed exactly as written.

**Total deviations:** 0 auto-fixed
**Impact on plan:** None – all work directly supported the planned scope.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Evidence fetching now always ships claim_text, and attachments seed backend reruns, so the evidence review phase can focus on presenting candidates instead of troubleshooting missing text.
