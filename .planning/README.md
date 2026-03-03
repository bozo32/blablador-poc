# Planning Handoff (Next Model Pass)

Last updated: 2026-03-03

This file is the fast handoff for the next model pass. Read this first, then jump to the linked docs.

## 1) What changed most recently

- Phase plans added/executed for walkthrough regressions:
  - `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-07-PLAN.md`
  - `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-08-PLAN.md`
- Retrieval-scope regression plan set added and partially executed:
  - `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-09-INDEX.md`
  - `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-09-VERIFICATION.md`
- Runtime/version visibility added in UI/API so stale images are easier to detect.
- Scope/workflow hardening landed around:
  - user discovery robustness in scope selector
  - explicit-only project creation intent (`+ New project...`)
  - intake unknown-note dedupe
  - ledger cache invalidation after auto-place
  - bootstrap failure visibility + stale workspace state clearing

## 2) Read order (minimum)

1. `.planning/phases/10-contracts-core-workflow-simplification/10-INDEX.md`
2. `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-09-INDEX.md`
3. `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-09-VERIFICATION.md`
4. `.planning/phases/10-contracts-core-workflow-simplification/deferred-items.md`
5. `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-PRUNE-CANDIDATES.md`
6. `docs/WORKFLOW_PROTOCOL.md`

## 3) Current debugging stance

- Do not prune aggressively during live walkthrough bug-finding.
- Continue stepwise walkthrough, classify each issue as:
  - real bug
  - stale runtime/image mismatch
  - expected strict scope behavior
- Keep adding prune candidates as you discover dead/legacy coupling, but defer removals until walkthrough is stable.

## 4) Known operational gotcha (important)

Compose does not mount frontend/backend source code into running app containers. Local code changes require rebuild/recreate.

- Use:
  - `docker compose up -d --build app-api app-ui`
- If scope/citation fixes seem missing, assume stale image first and rebuild before deeper contract changes.

## 5) Focus areas for next pass

- Validate end-to-end scope behavior under refresh/navigation:
  - existing user visibility in selector
  - project selection/apply semantics
  - no silent bootstrap failures
- Re-check cited/citing counters for stale-cache symptoms after auto-place and scope switches.
- Re-check citation chip -> rail -> focus-in-center path with strict headers.
- Confirm no duplicate intent guidance copy in intake rows.

## 6) Tests to run first

- `./.venv/bin/pytest tests/test_frontend_scope_selector_lock.py -q`
- `./.venv/bin/pytest tests/test_intake_auto_route_phase1.py -q`
- `./.venv/bin/pytest tests/test_project_api_scope_headers.py -q`
- `./.venv/bin/pytest tests/test_project_membership_api.py -q`
- `./.venv/bin/pytest tests/test_project_membership_spine.py -q`
- `./.venv/bin/pytest tests/test_projection_scope_consistency_api.py -q`

## 7) Where to log findings

- Bug investigation notes: `.planning/debug/`
- Deferred but known issues: `.planning/phases/10-contracts-core-workflow-simplification/deferred-items.md`
- Dead/stale code candidates only: `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-PRUNE-CANDIDATES.md`

## 8) Guardrails

- Preserve strict scope/membership contracts; do not reintroduce silent fallback behavior.
- UI should remain projection/display; spine/session remains authority.
- Prefer minimal, test-backed changes per bug cluster.
