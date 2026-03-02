---
status: diagnosed
trigger: "/gsd-debug Enforce durable user+project scoping for frontend/backend"
created: 2026-03-01T00:00:00Z
updated: 2026-03-01T00:28:00Z
---

## Current Focus

hypothesis: confirmed — drift is caused by mixed strict vs fallback scoping on backend plus frontend clients that either omit X-Project-Id or default to "default" and reviewer "default"
test: complete endpoint-by-endpoint enforcement matrix and derive backend-first hardening plan
expecting: explicit map of strict routes, fallback routes, and unscoped routes with associated risks
next_action: return structured diagnosis with risks and minimal backend-first fix strategy

## Symptoms

expected: user must have uid and project_id before any activity, and all reads/writes stay locked to that pair across reload/new browser
actual: on reload or new browser, project scope drifts to default; UI appears to show older documents
errors: no specific runtime error reported; behavior is silent scope fallback/drift
reproduction: open app, work in non-default project, reload or open new browser context, observe effective scope reverting and stale/default docs shown
started: current issue (exact start time not provided)

## Eliminated

## Evidence

- timestamp: 2026-03-01T00:05:00Z
  checked: user-reported context and requirements
  found: strict uid+project_id lock is required, but current behavior drifts on reload/new browser
  implication: need endpoint-by-endpoint and initialization-path audit for fallback behavior

- timestamp: 2026-03-01T00:12:00Z
  checked: codebase-wide symbol search for X-Project-Id, project_id, and visibility markers
  found: backend/main.py contains many X-Project-Id headers but also many `or DEFAULT_PROJECT_ID` patterns; frontend/ui.py references project_id with default fallback usage
  implication: mixed strict/fallback endpoint behavior likely exists and must be enumerated route-by-route

- timestamp: 2026-03-01T00:16:00Z
  checked: frontend API modules and session initialization
  found: ui init sets session `project_id` default from settings; get_project_id() falls back to default; project_api.py sends no X-Project-Id at all; many API helpers only attach header when project_id provided (else omit) and some force "default"
  implication: frontend can easily issue unscoped requests that backend resolves to default scope

- timestamp: 2026-03-01T00:24:00Z
  checked: backend/main.py route handlers using AST + source inspection
  found: backend mixes strict `_require_project_id_for_upload` enforcement with many handlers that use `str(x_project_id or "").strip() or DEFAULT_PROJECT_ID`; many non-ingest routes have no project header support at all
  implication: missing/empty project header silently shifts requests into default project on affected endpoints

- timestamp: 2026-03-01T00:25:00Z
  checked: user identity handling across routes
  found: many handlers default reviewer identity to `"default"` and actor user to `DEFAULT_USER_ID` (`"local"`) when explicit user identity is absent
  implication: identity isolation is optional in current design, enabling cross-user mixing under default identities

- timestamp: 2026-03-01T00:26:00Z
  checked: opinion visibility persistence/enforcement path (`backend/spine/opinion_events.py`, opinion endpoints)
  found: `visibility` is stored and exported; read APIs primarily filter by `project_id` + `owner_uid`; no general runtime ACL policy enforces public/selectable/private semantics across readers
  implication: visibility acts mostly as metadata (plus export filter), not comprehensive authorization control

- timestamp: 2026-03-01T00:27:00Z
  checked: defaults in settings and frontend components
  found: defaults are `DEFAULT_PROJECT_ID="default"`, `DEFAULT_USER_ID="local"`; frontend components and helpers explicitly fallback to these defaults
  implication: legacy/default data remains highly likely to surface whenever scope is missing during reload/new session

## Resolution

root_cause: Scope drift is caused by a systemic mismatch: frontend frequently emits unscoped requests (missing X-Project-Id and default reviewer/user values) while backend inconsistently enforces scope. Strict endpoints reject missing project IDs, but many others silently fallback to DEFAULT_PROJECT_ID/DEFAULT_USER_ID. On reload/new browser, session state resets to defaults, so API traffic is routed to default/local scope and appears as old/default documents.
fix: Not applied (research-only mode). Recommended backend-first hardening is to require explicit project and user identity at API boundary for all stateful routes and remove fallback-to-default behavior behind a temporary compatibility flag.
verification: Not executed (no code changes). Verification checklist prepared in diagnosis output.
files_changed: []
