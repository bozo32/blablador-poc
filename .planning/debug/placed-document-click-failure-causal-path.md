# Placed-Document Click Failure — Causal Path Analysis

Last updated: 2026-03-03T18:10:55+00:00
Primary debug session: `.planning/debug/placed-document-click-failure.md`

## Incident

- Symptom: opening Chase panel / Retrieval instructions from a placed document throws
  `RuntimeError {"detail":"Document <doc_id> not found"}`.
- Scope: treat as systemic scope-path regression candidate.

## Timeline (investigation)

1. **Planning context read first** (priorities, phase plans, existing scope debug notes) confirmed strict scope contracts are intentional and active.
2. Located error source in `backend/reference_retrieval.py` (`FileNotFoundError("Document {did} not found")`).
3. Traced endpoint `GET /references/{doc_id}/{reference_id}/retrieval` in `backend/main.py`; observed no scope headers accepted.
4. Traced retrieval loader `_load_document()`; observed unconditional `project_id=settings.DEFAULT_PROJECT_ID`.
5. Traced frontend retrieval callers (`frontend/ingestion_api.py`, `frontend/nav_api.py`); observed no `X-Project-Id` / `X-User-Id` headers sent.
6. Traced sibling calls in same UI surfaces (`/graph/resolve-references`, `/ingest/{doc_id}/body`, `/ingest/{doc_id}/citation-context`); observed strict scoped headers required.
7. Audited session cache keying; found retrieval dossier cache keyed only by `doc_id + reference_id`, not scope.
8. Ran `tests/test_reference_retrieval.py`; pass confirms formatting coverage but no scope-contract coverage.

## Causal Graph (frontend → backend)

```mermaid
flowchart TD
  A[Placed document selected in scoped session] --> B[Open Chase / Retrieval instructions]
  B --> C[frontend.claim_queue _get_cached_dossier]
  C --> D[frontend.ingestion_api.get_reference_retrieval]
  D --> E[GET /references/{doc_id}/{reference_id}/retrieval]
  E --> F[backend.reference_retrieval._load_document]
  F --> G[build_ingested_document_from_spine(project_id=DEFAULT_PROJECT_ID)]
  G -->|doc not in default project| H[FileNotFoundError Document <id> not found]
  H --> I[HTTP 404 detail propagated]
  I --> J[RuntimeError shown in UI path]

  B --> K[Sibling calls in same flow]
  K --> L[/graph/resolve-references (requires X-Project-Id + X-User-Id)]
  K --> M[/ingest/{doc_id}/body (requires X-Project-Id)]
  K --> N[/ingest/{doc_id}/citation-context (requires X-Project-Id + X-User-Id)]
```

## Sibling Endpoint/Caller Inventory (same causal path)

| Layer | Item | Scope contract today | Risk | Priority |
|---|---|---|---|---|
| Frontend caller | `ingestion_api.get_reference_retrieval()` | **Unscoped** GET, no headers | Direct source of mismatch | P0 |
| Frontend caller | `nav_api.get_reference_retrieval()` | **Unscoped** GET, no headers | Same mismatch in graph nav retrieval card | P0 |
| Backend endpoint | `GET /references/{doc_id}/{reference_id}/retrieval` | **Unscoped** (no header args) | Cannot honor active project scope | P0 |
| Backend retrieval loader | `_load_document()` in `reference_retrieval.py` | **Hard-wired DEFAULT_PROJECT_ID** | Deterministic false 404 for non-default project docs | P0 |
| Frontend cache | `st.session_state["retrieval_cache"]` key=`doc_id+reference_id` | **Scope-agnostic** | Cross-project stale dossier bleed post switch | P1 |
| Sibling strict endpoint | `POST /graph/resolve-references` | Requires `X-Project-Id` + `X-User-Id` | Diverges from retrieval dossier behavior | P1 (consistency) |
| Sibling strict endpoint | `GET /ingest/{doc_id}/body` | Requires `X-Project-Id` | Works scoped while retrieval does not | P1 (consistency) |
| Sibling strict endpoint | `GET /ingest/{doc_id}/citation-context` | Requires `X-Project-Id` + `X-User-Id` + membership | Works scoped while retrieval does not | P1 (consistency) |

## Minimal Safe Patch Plan (strict scope preserved)

1. **Scope the retrieval endpoint contract (P0)**
   - Add headers to `/references/{doc_id}/{reference_id}/retrieval`:
     - `x_project_id: str = Header(..., alias="X-Project-Id")`
     - `x_user_id: str = Header(..., alias="X-User-Id")` (or at least parity with neighboring strict read routes)
   - Resolve project via `_resolve_scope_observability(... require_project=True, require_user=True, allow_dev_project_default=False)`.
   - Pass resolved project into retrieval builder.

2. **Remove default-project binding in retrieval loader (P0)**
   - Refactor `build_retrieval_dossier(doc_id, reference_id, *, project_id)`.
   - Refactor `_load_document(doc_id, *, project_id)` to use supplied project_id.
   - Keep `FileNotFoundError` semantics unchanged, but now scoped correctly.

3. **Propagate scope headers from frontend retrieval callers (P0)**
   - Update `ingestion_api.get_reference_retrieval(..., project_id, user_id)` and send required headers.
   - Update `nav_api.get_reference_retrieval(..., project_id, user_id)` similarly.
   - Update callsites in `claim_queue.py` + `live_surfing_panel.py` using applied scope (`get_project_id()` / active uid helper).

4. **Scope-safe retrieval cache keying + invalidation (P1)**
   - Expand cache key to include scope tuple: `(project_id, user_id, doc_id, reference_id)`.
   - Invalidate `retrieval_cache` in `_invalidate_scope_cached_state()`.

5. **Regression tests (P0/P1)**
   - Add API tests: retrieval endpoint rejects missing scope headers and resolves document only in supplied project.
   - Add frontend API tests: retrieval callers send strict scope headers.
   - Add UI/session test: scope switch clears/isolates retrieval cache.

## Why this is systemic (not isolated)

- Scope/session architecture has already moved toward backend-owned scope sessions.
- Most adjacent endpoints now enforce strict scope headers.
- Retrieval path remains legacy-unscoped + default-project-bound, creating a contract discontinuity in the middle of an otherwise strict path.
