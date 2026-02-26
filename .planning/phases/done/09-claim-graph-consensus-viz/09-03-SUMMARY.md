---
phase: 09-claim-graph-consensus-viz
plan: 03
subsystem: [api, database, testing]
tags: [fastapi, sqlite, claim-graph, consensus, votes]

# Dependency graph
requires:
  - phase: 09-claim-graph-consensus-viz/09-02
    provides: reviewer_uid-scoped persistence conventions
provides:
  - Claim-link edges (CLAIM_LINK) with provenance metadata and creator-scoped deletion
  - Per-edge per-reviewer votes with consensus aggregates
  - FastAPI endpoints for claim subgraph queries, candidates, edge CRUD, and voting
affects: [09-04, 09-05, 09-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - SQLite edge_votes table keyed by (edge_id, reviewer_uid)
    - REST endpoints under /graph/* using reviewer_uid query param
    - Subgraph returns only materialized CLAIM_LINK edges; candidates are computed on demand

key-files:
  created:
    - backend/graph_store.py
    - tests/test_claim_graph.py
  modified:
    - backend/main.py
    - backend/schemas.py
    - tests/test_judgment_frontend_store.py

key-decisions:
  - "Store claim-to-claim edges in existing edges table as kind=CLAIM_LINK with provenance in properties_json"
  - "Store votes in edge_votes with one row per (edge_id, reviewer_uid) and compute consensus via counts"
  - "Generate candidates deterministically via token overlap + SequenceMatcher (no persistence)"

patterns-established:
  - "Edge responses always include aggregates {n_support,n_contradict,n_neutral,n_uncertain,n_total}"
  - "Manual edge deletion is creator-scoped and implemented as edges.enabled=0"

# Metrics
duration: 28min
completed: 2026-02-06
---

# Phase 09 Plan 03: Claim Graph Store + Endpoints Summary

**Claim-to-claim edges with per-user voting and consensus aggregates, exposed via FastAPI subgraph/candidates/link/vote endpoints for the Streamlit Graph tab.**

## Performance

- **Duration:** 28 min
- **Started:** 2026-02-06T23:04:44Z
- **Completed:** 2026-02-06T23:32:34Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

- Extended the SQLite graph store with materialized CLAIM_LINK edges and per-edge per-reviewer votes
- Added API schemas + endpoints for claim neighborhoods, candidate generation, edge vote inspection, and vote upserts
- Added FastAPI tests covering subgraph payloads, vote aggregates, and creator-scoped deletes

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend GraphStore schema for edge votes + claim links** - `4bf7866` (feat)
2. **Task 2: Add claim-graph schemas + FastAPI endpoints** - `3b4c9bd` (feat)
3. **Task 3: Add tests for claim subgraph + votes** - `2dce1d1` (test)

## Files Created/Modified

- `backend/graph_store.py` - Adds CLAIM_LINK helpers plus edge_votes storage and aggregate helpers
- `backend/main.py` - Implements /graph endpoints (subgraph, candidates, link CRUD, votes)
- `backend/schemas.py` - Adds claim graph node/edge/vote request+response models
- `tests/test_claim_graph.py` - Covers manual link creation, votes/aggregates, and delete enforcement

## Decisions Made

- Kept edges in the existing edges table (kind=CLAIM_LINK) and stored provenance in properties_json to avoid a new schema split.
- Consensus aggregates are computed from raw vote counts (no confidence weighting).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed reviewer_uid kwarg mismatch in judgment store test stub**
- **Found during:** Task 2 (pytest verification)
- **Issue:** `frontend/judgment_store.py` calls API methods with `reviewer_uid=...`, but `StubJudgmentApi` in tests did not accept the kwarg
- **Fix:** Updated stub method signatures to accept and ignore `reviewer_uid`
- **Files modified:** `tests/test_judgment_frontend_store.py`
- **Verification:** `pytest -q`
- **Committed in:** `38b5c71`

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary to restore green tests; no scope creep.

## Issues Encountered

- Pre-commit hooks required a restage after auto-formatting changes (handled during commits).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- API surface is ready for Streamlit Graph tab wiring (09-05), including subgraph queries and edge voting.

---
*Phase: 09-claim-graph-consensus-viz*
*Completed: 2026-02-06*
