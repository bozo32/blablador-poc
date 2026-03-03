---
status: diagnosed
trigger: "/gsd-debug Scope selection is lost after UI hyperlink navigation (clicking citation chips and previously placed documents). Symptoms: user gets dumped back to unapplied scope, document body endpoint errors with missing X-Project-Id, and loops re-selecting uid/project. Investigate root causes across frontend/session state/query-param behavior and backend scope expectations. Propose robust architecture where scope stickiness is backend-grounded (not fragile Streamlit-only state), plus minimal immediate patch recommendations. Return concise findings, root cause confidence, and prioritized fixes."
created: 2026-03-02T21:23:49+00:00
updated: 2026-03-02T21:30:18+00:00
---

## Current Focus

hypothesis: Scope stickiness fails due to frontend-only ephemeral state + query-param clearing + inconsistent scope propagation in live navigation paths
test: Correlate query-param lifecycle, scope hydration guard conditions, and backend strict header requirements with concrete callsites
expecting: Explain all reported symptoms (unapplied loop + missing X-Project-Id body errors) through specific code paths
next_action: finalize root-cause synthesis and prioritized fix recommendations

## Symptoms

expected: Scope selection (uid/project) remains applied when clicking citation chips or previously placed document links, and downstream document body fetches succeed.
actual: After hyperlink navigation, UI returns to unapplied scope state and user is forced to re-select uid/project repeatedly.
errors: Document body endpoint fails due to missing X-Project-Id header.
reproduction: 1) Apply uid/project scope in UI. 2) Click citation chip or previously placed document hyperlink. 3) Observe scope reset and document body request error with missing X-Project-Id.
started: Not specified by reporter.

## Eliminated

## Evidence

- timestamp: 2026-03-02T21:24:16+00:00
  checked: Reporter symptom statement
  found: Scope is lost specifically on hyperlink-based navigation paths and backend errors mention missing X-Project-Id
  implication: Investigation should focus on state propagation across navigation events and request header construction

- timestamp: 2026-03-02T21:25:03+00:00
  checked: Repo-wide search for X-Project-Id and citation/navigation code
  found: frontend/ui.py builds citation links with uid/project query params via _citation_href, and many requests set X-Project-Id from get_project_id() (scope_lock-backed)
  implication: Root cause likely in scope_lock state hydration/lifecycle rather than missing link parameters alone

- timestamp: 2026-03-02T21:27:12+00:00
  checked: frontend/ui.py scope selector and query-param handling (render_scope_selector_block + draw_ingestion_panel)
  found: scope auto-apply from query params only runs when applied scope is empty (lines 2360-2369) and suppresses all exceptions; draw_ingestion_panel then clears query params wholesale (lines 6293-6300), including uid/project used for recovery
  implication: Any navigation/session restart path that loses Streamlit session state after params are cleared cannot recover scope automatically, producing "Scope: unapplied" loops

- timestamp: 2026-03-02T21:28:26+00:00
  checked: backend /ingest/{doc_id}/body contract and frontend callers
  found: backend strictly requires X-Project-Id for /ingest/{doc_id}/body (backend/main.py lines 4457-4467), while live_surfing_panel calls get_document_body(api_url, wid) without project_id (frontend/components/live_surfing_panel.py line 1596)
  implication: body endpoint errors with missing X-Project-Id can occur on navigation/surfing paths even when other UI paths are scoped correctly

- timestamp: 2026-03-02T21:29:09+00:00
  checked: live_surfing_panel scope propagation consistency
  found: component mixes scope_lock access with legacy st.session_state["project_id"] and calls graph_api.resolve_references without required user_id in some paths
  implication: scope propagation is non-uniform across frontend modules, increasing fragility and making hyperlink-driven transitions prone to partial scope loss/failures

## Resolution

root_cause: Scope state is not backend-authoritative and is inconsistently propagated in frontend navigation paths. Hyperlink/query-param navigation depends on Streamlit session + transient uid/project query params; these params are cleared aggressively, and auto-apply failures are swallowed, causing re-entry into unapplied state. In parallel, at least one active navigation path (live_surfing_panel) calls strict scoped backend endpoints without X-Project-Id, directly triggering document body errors.
fix: Not applied (diagnosis-only). Prioritize immediate scope-propagation patch plus backend-grounded scope session design.
verification:
files_changed: []
