---
status: diagnosed
trigger: "/gsd-debug Perform a full causal-path debug for the placed-document click failure: RuntimeError {\"detail\":\"Document <doc_id> not found\"} raised from retrieval path when opening chase panel/retrieval instructions. Treat this as potentially systemic scope-path regression, not single endpoint bug. Requirements: (1) Read planning context first: .planning/PRIORITIES.md, .planning/README.md, .planning/CLEANUP_PROTOCOL.md, .planning/phases/10-contracts-core-workflow-simplification/10-04.5-07-PLAN.md, .planning/phases/10-contracts-core-workflow-simplification/10-04.5-08-PLAN.md, .planning/debug/*.md. (2) Execute full /gsd-debug process with explicit step artifacts at each stage (state, hypotheses, evidence, eliminated, resolution, verification checklist). (3) Create or update debug artifact docs under .planning/debug/ for this incident, including timeline and causal graph across frontend/backend endpoints. (4) Identify all sibling endpoints/callers on same causal path (scope headers, project resolution, session cache keys) and classify risk/priorities. (5) Propose minimal safe patch plan preserving strict scope contracts. Return: concise diagnosis summary, exact artifact file paths created/updated, prioritized fix list, and recommended immediate next implementation slice."
created: 2026-03-03T18:08:37+00:00
updated: 2026-03-03T18:12:03+00:00
---

## Current Focus

hypothesis: Confirmed systemic contract mismatch: retrieval dossier endpoint/callers are unscoped + default-project-bound while neighboring endpoints are strict-scoped; frontend retrieval cache is also scope-agnostic.
test: Completed code-path and sibling-endpoint audit; causal graph and timeline captured in companion artifact.
expecting: Root cause accepted as scope-path regression requiring strict contract alignment.
next_action: return structured diagnosis + prioritized minimal safe patch plan

## Symptoms

expected: Opening chase panel/retrieval instructions for a placed document should resolve and load that document.
actual: RuntimeError is raised in retrieval path with detail "Document <doc_id> not found".
errors: RuntimeError {"detail":"Document <doc_id> not found"}
reproduction: Click a placed document, then open chase panel/retrieval instructions.
started: Unknown from user report (needs evidence from history/related debug docs).

## Eliminated

- hypothesis: Document is genuinely absent from storage (isolated data-loss) and retrieval endpoint is correct
  evidence: Retrieval loader forces DEFAULT_PROJECT_ID regardless active scope, while placed-document selection path is scope-aware; this can produce "Document <id> not found" for valid docs in non-default projects
  timestamp: 2026-03-03T18:10:55+00:00

- hypothesis: Stale runtime/image mismatch is primary cause for this incident
  evidence: Code-level contract mismatch is deterministic (no retrieval scope headers + backend default project binding) and independently sufficient to reproduce 404 path
  timestamp: 2026-03-03T18:10:55+00:00

## Evidence

- timestamp: 2026-03-03T18:09:39+00:00
  checked: required planning context + prior debug files
  found: prior diagnoses repeatedly identified split scope authority and navigation-path header inconsistency; strict-scope guardrails were explicitly marked non-negotiable in current phase plans
  implication: current incident should be treated as systemic scope contract regression candidate, not isolated UI bug

- timestamp: 2026-03-03T18:09:39+00:00
  checked: backend retrieval dossier implementation (backend/reference_retrieval.py)
  found: _load_document uses settings.DEFAULT_PROJECT_ID unconditionally, then build_ingested_document_from_spine(..., project_id=DEFAULT_PROJECT_ID); missing doc in default project raises FileNotFoundError("Document <id> not found")
  implication: retrieval dossier cannot resolve docs that exist only in non-default project scope

- timestamp: 2026-03-03T18:09:39+00:00
  checked: retrieval endpoint contract (backend/main.py /references/{doc_id}/{reference_id}/retrieval)
  found: endpoint accepts no X-Project-Id/X-User-Id and directly calls build_retrieval_dossier; FileNotFoundError is surfaced as HTTP 404 detail
  implication: request path is structurally unscoped and cannot honor active scope even when frontend has it

- timestamp: 2026-03-03T18:09:39+00:00
  checked: frontend retrieval callers (frontend/nav_api.py, frontend/ingestion_api.py, live_surfing_panel.py, claim_queue.py)
  found: both get_reference_retrieval callers perform plain GET with no scope headers; chase/surfing UI invokes them in retrieval instructions path
  implication: frontend and backend contracts are aligned on unscoped retrieval, creating deterministic project-mismatch failures in scoped sessions

- timestamp: 2026-03-03T18:09:39+00:00
  checked: strict sibling document endpoints (/ingest/{doc_id}/body, /ingest/{doc_id}/citation-context)
  found: these routes require X-Project-Id (and citation-context also X-User-Id + membership), unlike retrieval endpoint
  implication: mixed strict/unscoped behavior exists on same user flow; a single panel can call both strict and unscoped reads against different project resolutions

- timestamp: 2026-03-03T18:09:39+00:00
  checked: retrieval UI cache keying in frontend/claim_queue.py
  found: retrieval_cache key is retrieval__{doc_id}__{reference_id}, not scoped by project/user
  implication: even after contract fix, cache can leak stale dossiers across scope switches unless key invalidation/keying includes scope

- timestamp: 2026-03-03T18:10:55+00:00
  checked: chase-panel retrieval callsite and graph/body sibling calls (frontend/components/chasing_panel.py, live_surfing_panel.py)
  found: retrieval instructions call render_retrieval_instructions(doc_id, reference_id) -> ingestion_api.get_reference_retrieval without project/user headers, while same panels call resolve_references and body APIs with scoped headers
  implication: within one UI surface, retrieval dossier call can diverge from scoped graph/body state

- timestamp: 2026-03-03T18:10:55+00:00
  checked: scope apply invalidation path (frontend/ui.py::_invalidate_scope_cached_state)
  found: cache invalidates ledger/docs/graph context but never clears claim_queue retrieval_cache
  implication: retrieval dossier cache can survive scope switch and present stale/wrong data even if endpoint contract is corrected

- timestamp: 2026-03-03T18:10:55+00:00
  checked: frontend ingestion API list/get document calls vs retrieval calls
  found: list_documents/get_document/get_document_body accept project scope headers; get_reference_retrieval omits all scope headers
  implication: reporter can click a legitimately scoped placed document then fail only on retrieval path due to missing scope propagation

- timestamp: 2026-03-03T18:10:55+00:00
  checked: tests/test_reference_retrieval.py execution and assertions
  found: tests pass (3/3) but monkeypatch get_ingested_document and contain no scoped-project coverage for retrieval endpoint contract
  implication: regression escaped because test suite validates dossier formatting, not scope contract behavior

## Resolution

root_cause: Retrieval instructions call an unscoped endpoint (/references/{doc_id}/{reference_id}/retrieval) whose loader hard-binds to DEFAULT_PROJECT_ID. In scoped sessions, placed docs can exist in active project but not default project, yielding deterministic "Document <doc_id> not found". Secondary systemic issue: retrieval_cache keys ignore scope and are not cleared on scope switch.
fix: Not applied in this debug pass (diagnosis + patch plan only). Proposed minimal safe patch: enforce strict scope headers on retrieval endpoint/callers, pass resolved project into retrieval builder, and scope-key + invalidate retrieval cache.
verification: Root cause verified by static causal-path trace across frontend callsites, backend endpoint contract, and spine project-membership checks; supportive unit tests pass but currently lack scope-contract coverage.
files_changed: []

## Verification Checklist

- [x] Located exact throw site for reported message (`Document <doc_id> not found`).
- [x] Traced full causal path from placed-document click/chase panel to backend loader.
- [x] Compared retrieval path contract against sibling strict-scoped endpoints.
- [x] Identified scope/session cache-key risk on same path.
- [ ] Runtime repro in two real projects (pending implementation patch).
- [ ] Post-fix regression tests for retrieval scope headers + cache invalidation.
