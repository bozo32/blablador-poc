# Architecture

**Analysis Date:** 2026-02-20

## Pattern Overview

**Overall:** Multi-service, UI-driven workflow with a spine-backed system-of-record (Postgres + S3/MinIO) and explicit background pipelines for expensive compute.

**Key Characteristics:**
- Streamlit UI orchestrates the workflow and calls a single FastAPI backend.
- Durable state is centralized in the “spine” (Postgres metadata + S3 object keys); large artifacts live in object storage.
- Expensive work (PDF extraction, attachment processing, evidence reruns, OCR fallback) runs only via explicit triggers/background workers.

## Layers

**Presentation (Streamlit UI):**
- Purpose: Interactive reviewer workspace (upload PDFs, navigate citations, confirm claims, attach sources, review evidence, graph/surfing views).
- Location: `frontend/ui.py`
- Contains: Streamlit views, session state init, CSS injection, panel composition.
- Depends on: HTTP client helpers (`frontend/ingestion_api.py`, `frontend/evidence_api.py`, `frontend/graph_api.py`, `frontend/ledger_api.py`, `frontend/project_api.py`, `frontend/judgment_api.py`), and UI components (`frontend/components/*`).
- Used by: Docker service `app-ui` (`docker/app-ui/Dockerfile`), local runner (`application.py`).

**API Surface (FastAPI):**
- Purpose: Single backend API exposing ingestion, resolution, attachments, evidence, judgments, graph, and maintenance endpoints.
- Location: `backend/main.py`
- Contains: Route handlers, startup checks (migrations + S3 bucket), CORS middleware, endpoint orchestration.
- Depends on: Spine primitives (`backend/spine/*`), stores (`backend/attachment_store.py`, `backend/graph_store.py`, `backend/span_graph_store.py`, `backend/evidence_matching/*`), and pipelines (`backend/spine/extraction_pool.py`, `backend/attachment_pipeline.py`).
- Used by: Streamlit UI (`frontend/*_api.py`), Docker service `app-api` (`docker/app-api/Dockerfile`).

**Domain/Pipeline Modules:**
- Purpose: PDF extraction, TEI parsing, body segmentation, reference resolution, evidence matching, and NLI scoring.
- Location: `backend/extraction.py`, `backend/tei_body.py`, `backend/grobid_client.py`, `backend/reference_resolver.py`, `backend/evidence_matching/pipeline.py`, `backend/nli.py`, `backend/attachment_pipeline.py`
- Contains: Deterministic parsing/serialization logic and pipeline orchestration.
- Depends on: External services (GROBID; optional fallback worker), model/runtime settings (`backend/settings.py`).
- Used by: `backend/main.py`, `backend/spine/extraction_pool.py`.

**Persistence (“Spine” + Stores):**
- Purpose: Durable system-of-record for works/documents, attempts/jobs/artifacts, attachments, evidence runs/selections, judgments, and graph/span-graph state.
- Location: `backend/db/migrate.py`, `backend/db/pg.py`, `backend/spine/*`, `backend/object_store/s3.py`, `backend/attachment_store.py`, `backend/evidence_matching/store.py`, `backend/graph_store.py`, `backend/span_graph_store.py`
- Contains: Idempotent DDL, minimal DB connect wrapper, Postgres helpers, and S3/MinIO wrapper.
- Depends on: Settings (`backend/settings.py`).
- Used by: API layer and background pipelines.

## Data Flow

**Ingest + Extract + Body Model:**

1. UI uploads citing PDF via `frontend/ingestion_api.py:upload_pdf()` → `POST /ingest`.
2. API stores the PDF in S3/MinIO (`backend/object_store/s3.py`) and upserts spine records (`backend/spine/works.py`, `backend/spine/documents.py`).
3. API schedules extraction via `backend/spine/extraction_pool.py:SpineExtractionPool.enqueue()` (created in `backend/main.py`).
4. Worker pulls the PDF (`backend/spine/pdf_source.py`, `backend/object_store/s3.py`) → calls GROBID (`backend/grobid_client.py`) → parses TEI (`backend/extraction.py`) → persists artifacts (`backend/spine/artifacts.py`).
5. UI reads status/data via `GET /ingest` and `GET /ingest/{doc_id}` (`backend/spine/ingest_view.py`) and renders the body model via `GET /ingest/{doc_id}/body` (built by `backend/tei_body.py`).

**Reference Resolution:**

1. UI triggers resolution via `frontend/ingestion_api.py:trigger_resolution()` → `POST /ingest/{doc_id}/resolve`.
2. API resolves bibliography entries using `backend/reference_resolver.py:resolve_references()` (Crossref/OpenAlex) and persists `resolution.json` as an artifact (`backend/spine/artifacts.py`).
3. Graph index is refreshed using `backend/graph_store.py:index_resolution()` to attach resolved identifiers to reference nodes.

**Claim Confirmation → Graph + Span Graph:**

1. UI persists edited/confirmed segments via `frontend/ingestion_api.py:confirm_claims()` → `POST /claims/confirm`.
2. API writes confirmed claim rows (`backend/db/migrate.py` table `confirmed_claims`) and indexes:
   - Document/claim graph nodes/edges in `backend/graph_store.py:index_confirmed_claims()`.
   - Stable citation-window spans + claim spans in `backend/span_graph_store.py:index_claim_confirmation()`.

**Attachments (Cited PDFs) → Background Processing:**

1. UI enqueues uploads and placement via `frontend/attachment_queue.py` → `POST /attachments` and `PATCH /attachments/{attachment_id}`.
2. API persists attachment metadata + timeline in Postgres (`backend/attachment_store.py`; tables `attachments`, `attachment_events`, `attachment_artifacts` created by `backend/db/migrate.py`) and stores the PDF bytes in S3/MinIO (`backend/object_store/s3.py`).
3. Background attachment threads (`backend/attachment_pipeline.py:enqueue_processing()`) run:
   - download PDF from object store,
   - TEI extraction via GROBID (`backend/grobid_client.py`) + parse (`backend/extraction.py`),
   - sentence extraction + embeddings (`backend/attachment_pipeline.py`, `backend/utils.py`),
   - artifact persistence (`backend/attachment_store.py:save_artifacts()`),
   - optional promotion to an ingested Work (`backend/attachment_pipeline.py:_maybe_ingest_matched_attachment()`).
4. Attachment completion triggers best-effort evidence rerun queueing via `backend/evidence_matching/service.py:EvidenceMatchingService.trigger_auto_rerun()`.

**Evidence Matching (Reruns + Paging):**

1. UI lists candidates via `frontend/evidence_api.py:list_evidence()` → `GET /claims/{claim_id}/evidence`.
2. Backend serves cached/latest run without forcing compute using `backend/evidence_matching/service.py:EvidenceMatchingService.list_candidates()`.
3. UI triggers compute via `frontend/evidence_api.py:request_rerun()` → `POST /claims/{claim_id}/evidence/rerun`.
4. Evidence jobs queue and run in background threads (`backend/evidence_matching/service.py`), executing:
   - window loading (`backend/evidence_matching/loaders.py`),
   - deterministic seeding (`backend/evidence_matching/deterministic_matcher.py`),
   - pipeline run (`backend/evidence_matching/pipeline.py`) including retrieval/rerank/NLI (`backend/nli.py`).
5. Runs and selections are persisted spine-first (`backend/evidence_matching/store.py`, `backend/evidence_selection_store.py`), with large payloads optionally stored in S3.

**Graph/Surfing View:**

1. UI “Surfing” composes work-level graph from the ledger (`frontend/components/live_surfing_panel.py`) using `GET /ledger` (`backend/graph_store.py:ledger_rows()`).
2. UI expands CiteSpans by calling `GET /ingest/{doc_id}/body` and resolving reference targets via `POST /graph/resolve-references` (`backend/graph_store.py:resolve_reference_to_ingest_id()`).
3. Stable span bundles and statuses are read from `backend/span_graph_store.py` via endpoints in `backend/main.py` (e.g. `/spans/{span_id}/bundle`, `/spans/{span_id}/status`).

## Key Abstractions

**Spine (Durable System-of-Record):**
- Purpose: Keep durable metadata in Postgres and store large artifacts in S3/MinIO.
- Examples: `backend/db/migrate.py`, `backend/spine/attempts.py`, `backend/spine/artifacts.py`, `backend/spine/ingest_view.py`, `backend/object_store/s3.py`
- Pattern: “pointers in Postgres, blobs in object store”; idempotent operations keyed by content hashes (`sha256`) and attempt settings hashes.

**GraphStore (Work/Claim Graph):**
- Purpose: Postgres-backed nodes/edges/votes for document-level and claim-level navigation.
- Examples: `backend/graph_store.py`, endpoints in `backend/main.py` under `/graph/*` and `/ledger/*`.
- Pattern: Upserts with alias resolution (`graph_aliases`) and deterministic doc keys (`sha256:` / `doi:` / `bib:`).

**SpanGraphStore (Stable Anchoring):**
- Purpose: Span-first addressing of citation windows, claim spans, claim atoms, assertions, and neighborhood search.
- Examples: `backend/span_graph_store.py`, span endpoints in `backend/main.py` under `/spans/*`.
- Pattern: Deterministic ids computed from selectors + fingerprints (`span:{sha256(...)}`), persisted to `span_graph_*` tables.

**EvidenceMatchingService:**
- Purpose: Coordinate evidence reruns with concurrency limits and attachment-state snapshots.
- Examples: `backend/evidence_matching/service.py`, pipeline `backend/evidence_matching/pipeline.py`.
- Pattern: Background queue with per-claim locking; “read endpoints do not trigger reruns”.

## Entry Points

**FastAPI backend:**
- Location: `backend/main.py`
- Triggers: `uvicorn backend.main:app` (Compose `docker/app-api/Dockerfile`, dev `application.py`).
- Responsibilities: API routes + startup checks (`apply_migrations()` and `object_store_s3.ensure_bucket()`), orchestration of pipelines/stores.

**Streamlit UI:**
- Location: `frontend/ui.py`
- Triggers: `streamlit run frontend/ui.py` (Compose `docker/app-ui/Dockerfile`, dev `application.py`).
- Responsibilities: Session-state workflow UI; calls API via `frontend/*_api.py`.

**Local dev launcher (multi-process):**
- Location: `application.py`
- Triggers: `python application.py ...`
- Responsibilities: Starts `uvicorn backend.main:app` and `streamlit run frontend/ui.py` (and optionally ColBERT server).

**Docker Compose stack:**
- Location: `docker-compose.yml`
- Triggers: `bash scripts/dev/up.sh`
- Responsibilities: Runs API + UI + GROBID + Postgres + MinIO + optional fallback worker.

**Fallback OCR worker (optional service):**
- Location: `backend/ocr_worker_app.py`
- Triggers: `uvicorn backend.ocr_worker_app:app` (Compose `docker/fallback-worker/Dockerfile`).
- Responsibilities: OCR-capable fallback extraction, authenticated by `INTERNAL_SERVICE_TOKEN`.

**Optional ColBERT reranker service:**
- Location: `colbert_server/colbert.py`
- Triggers: `uvicorn colbert_server.colbert:app` (started by `application.py` when configured).
- Responsibilities: External ColBERT reranking used by hybrid pipeline modes.

## Error Handling

**Strategy:** Explicit HTTP errors at boundaries; background pipelines log exceptions and persist state transitions.

**Patterns:**
- FastAPI endpoints raise `fastapi.HTTPException` for validation/availability (`backend/main.py`).
- Long-running extraction/processing uses durable states (attempt/job tables created by `backend/db/migrate.py`) and best-effort retries (e.g. GROBID 503 backoff in `backend/grobid_client.py`).
- Background work can be globally paused via `backend/background_state.py` and `/background/*` endpoints in `backend/main.py`.

## Cross-Cutting Concerns

**Logging:** Python logging configured centrally in `backend/main.py` (module-level `logging.basicConfig`).
**Validation:** Pydantic models in `backend/schemas.py` for request/response contracts; Streamlit-side validation is minimal.
**Authentication:** No end-user auth; internal worker calls can require `INTERNAL_SERVICE_TOKEN` (`backend/ocr_worker_app.py`).

---

*Architecture analysis: 2026-02-20*
