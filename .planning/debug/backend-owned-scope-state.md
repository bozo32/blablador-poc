---
status: diagnosed
trigger: "/gsd-debug Diagnose and root-cause the current scope persistence architecture end-to-end. Goal: migrate to backend-owned scope state so clients (including Streamlit) only read/write scope via API, with no client-memory authority. Include: current data flow, failure modes (navigation/reload/multi-tab), required backend API contract, migration strategy, and acceptance tests. Output should include concrete implementation recommendations and risks."
created: 2026-03-03T06:33:01+00:00
updated: 2026-03-03T06:36:39+00:00
---

## Current Focus

hypothesis: Confirmed split-authority scope architecture causes persistence drift and fragility.
test: Synthesize findings into backend-owned scope contract and migration plan.
expecting: A single backend scope session resource removes navigation/reload/multi-tab drift.
next_action: return diagnosis with implementation recommendations and risks

## Symptoms

expected: Scope selection should persist across navigation, reloads, and tabs with backend as sole source of truth.
actual: Current scope persistence architecture is suspected to rely on client memory/local state, creating drift and loss scenarios.
errors: No single stack trace provided; failure class requested includes navigation/reload/multi-tab inconsistency.
reproduction: Select or change scope, then navigate/reload/open second tab and observe whether active scope remains consistent.
started: Existing architecture concern; not tied to a single recent regression in prompt.

## Eliminated

<!-- none yet -->

## Evidence

- timestamp: 2026-03-03T06:33:46+00:00
  checked: repository-wide scope references and scope-related test files
  found: Scope logic is concentrated in frontend/scope_lock.py with multiple tests for scope API behavior (pilot/strict routes)
  implication: There is an existing mixed model with client scope lock and partial backend scope APIs to evaluate

- timestamp: 2026-03-03T06:35:30+00:00
  checked: frontend/scope_lock.py and scope selector implementation in frontend/ui.py
  found: apply_draft_scope writes applied uid/project into Streamlit session_state and mirrors legacy keys; selector auto-applies from uid/project query params only when applied scope is empty
  implication: Applied scope authority is client-memory-first, with URL params as fragile recovery path

- timestamp: 2026-03-03T06:35:30+00:00
  checked: backend scope guards and project membership endpoints in backend/main.py + backend/spine/project_membership.py
  found: backend enforces scoped headers for many writes and stores active project per user (`/projects/select`, `/projects/active`) but has no canonical full scope session resource including reviewer identity
  implication: Backend has partial persistence (active project), not complete authoritative scope state for clients

- timestamp: 2026-03-03T06:35:30+00:00
  checked: frontend modules (evidence_api.py, attachment_queue.py, components/chase_queue.py, live_surfing_panel.py)
  found: several modules still read `st.session_state["project_id"]` / `active_reviewer_uid` directly and construct headers locally, including mixed fallback behavior
  implication: Scope propagation is inconsistent and susceptible to drift across pages, tabs, and reruns

- timestamp: 2026-03-03T06:36:22+00:00
  checked: frontend usage of project membership APIs and query-param handling in ui.py
  found: frontend writes backend active project via `/projects/select` but does not consume `/projects/active`; draw_ingestion_panel clears query params wholesale after navigation actions
  implication: Reload or hyperlink flows can lose scope recovery hints while applied scope remains local-only, causing re-apply loops

- timestamp: 2026-03-03T06:36:22+00:00
  checked: reviewer identity handling in ui.py project panel vs request identity helpers
  found: project panel persists `project_meta.active_reviewer_uid`, but runtime request identity helper `_active_reviewer_uid()` uses scope_lock applied uid only
  implication: multiple reviewer identity channels exist, enabling drift between persisted project metadata and effective API scope headers

## Resolution

root_cause: "Scope authority is split across client memory (Streamlit session applied scope + legacy keys), URL query parameters, and partial backend project membership state. Backend lacks a canonical full scope session resource (user + project + reviewer identity), so clients reconstruct scope locally and inconsistently; this causes scope loss on reload/navigation and drift across tabs/modules."
fix: "Not applied (diagnosis-only). Recommend introducing backend-owned scope session endpoints and migrating clients to consume scope exclusively via API-backed context with no local authority."
verification: "Static code-path verification: traced frontend scope apply/read/write paths and backend scope guards/membership APIs; identified absence of authoritative full-scope read endpoint and inconsistent frontend scope sources."
files_changed: []
