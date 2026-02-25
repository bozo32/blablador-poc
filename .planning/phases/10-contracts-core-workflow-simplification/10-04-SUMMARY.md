---
phase: 10-contracts-core-workflow-simplification
plan: 04
subsystem: api
tags: [postgres, fastapi, streamlit, occ, idempotency, ndjson, event-sourcing]

# Dependency graph
requires:
  - phase: 10-contracts-core-workflow-simplification (10-02)
    provides: "Stable evidence happy-path + project export/import baseline"
provides:
  - "Append-only evidence decision events + fast projection reads (pins/triage)"
  - "OCC + idempotency-protected decision writes with 409 current_version recovery"
  - "Pinned placeholders that persist even when not in current evidence run"
  - "Project export/import round-trip for decision events"
affects: [10-05-demo-trace-replay, multi-user-reviewer-streams, project-export-import]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Event-sourced reviewer decisions: append-only events + per-target projection"
    - "OCC via per-stream version row + 409 with current_version"
    - "Idempotency via (project_id, claim_id, reviewer_uid, idempotency_key) uniqueness + request fingerprint"

key-files:
  created:
    - backend/spine/evidence_decisions.py
    - scripts/dev/verify_10_04_decisions.sh
    - tests/test_evidence_decisions_api.py
  modified:
    - backend/db/migrate.py
    - backend/main.py
    - backend/schemas.py
    - frontend/evidence_api.py
    - frontend/evidence_store.py
    - frontend/ui.py
    - frontend/components/evidence_card.py
    - tests/conftest.py

key-decisions:
  - "Stable decision target is (attachment_id, span_id) with target_key=attachment_id:span_id (not candidate_id)"
  - "Conflicts are scoped to (project_id, claim_id, reviewer_uid) streams; no cross-reviewer OCC"
  - "Decision state is server-authoritative; Streamlit caches version and retries once on 409"

patterns-established:
  - "Pinned section always returned; missing-in-run pins represented as placeholders with metadata.not_in_current_run"
  - "Export/import carries evidence_decision_events.ndjson and rebuilds projections by replay"

# Metrics
duration: 3h 43m
completed: 2026-02-25
---

# Phase 10 Plan 04: Durable decisions/events Summary

**Audit-grade, per-reviewer evidence decision events (pins + triage) persisted in Postgres with OCC + idempotency, replayable via export/import, and wired into the Streamlit evidence panel as the source of truth.**

## Performance

- **Duration:** 3h 43m
- **Started:** 2026-02-25T00:58:40+01:00
- **Completed:** 2026-02-25T04:42:04+01:00
- **Tasks:** 3/3 (2 auto + 1 human-verify)
- **Files modified:** 11

## Accomplishments

- Added durable decision persistence as append-only events + fast per-target projection scoped by (project_id, claim_id, reviewer_uid).
- Exposed decision endpoints and integrated decision overlays into evidence listing, including pinned placeholders that persist across reruns.
- Wired Streamlit evidence UI to server-backed decisions with 409 auto-refresh + single retry, visible banner gating, and recent-event timeline.
- Included decision events in project export/import so a new project instance can replay reviewer decision state.
- Added automated verification (pytest + compose-backed verifier script) for OCC/idempotency/pinned placeholders/export-import.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add append-only evidence decision event tables + transactional spine helper (OCC + idempotency + projection)** - `8e3a149` (feat)
2. **Task 2: Expose decision endpoints, integrate into evidence list + export/import, and add automated verification (pytest + compose verifier)** - `0e7dad9` (feat)
3. **Task 3: Manual checkpoint (Streamlit UX verification)** - approved (no code commit)

**Plan metadata:** (this commit)

## Files Created/Modified

- `backend/db/migrate.py` - Adds streams/events/targets tables for durable evidence decisions.
- `backend/spine/evidence_decisions.py` - Transactional append/replay for decision events (OCC + idempotency) + projection reads.
- `backend/main.py` - Decision API endpoints; evidence list overlays pinned/triage and pinned placeholders; export/import wiring.
- `backend/schemas.py` - Pydantic models for decision read/append requests and evidence list additive fields.
- `frontend/evidence_api.py` - Client helpers for decision reads/appends with 409 surfaced for store recovery.
- `frontend/evidence_store.py` - Server-backed decision cache + action methods + 409 refresh/retry + banner gating.
- `frontend/ui.py` - Evidence panel pinned section + decision timeline + clear decisions action.
- `frontend/components/evidence_card.py` - Renders decision badges (pinned/accepted/rejected) in the card header.
- `tests/conftest.py` - Adds decision tables to truncate/wipe lists for test isolation.
- `tests/test_evidence_decisions_api.py` - Coverage for idempotency, OCC conflicts, pinned placeholders, overlay behavior.
- `scripts/dev/verify_10_04_decisions.sh` - Compose-backed end-to-end verifier for durable decisions + export/import.

## Decisions Made

- Stable decision targets are anchored on `(attachment_id, span_id)` so decisions survive evidence reranks and pagination.
- Per-reviewer OCC applies only within a reviewer stream; write conflicts return 409 with `current_version` and the UI retries once.
- Idempotency is enforced with a replayable `idempotency_key` guarded by a request fingerprint to prevent accidental double-apply.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Repo contained unrelated debugging/seed work changes during execution; the plan-completion docs commit stages only the plan metadata artifacts.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Automated verification passes: `bash scripts/dev/pytest_docker.sh` and `bash scripts/dev/verify_10_04_decisions.sh`.
- Manual checkpoint approved; ready for `10-05` (demo + trace replay) to build on durable decision events.

---
*Phase: 10-contracts-core-workflow-simplification*
*Completed: 2026-02-25*
