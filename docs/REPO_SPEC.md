# OS-ERIN (`blablador-poc`) — Functional Description + Technical Spec

## High-level overview (1 paragraph)

OS-ERIN is a local-first citation validation web app: a reviewer uploads an academic PDF (“citing” paper), navigates in-text citation callouts, follows a chosen citation to a target reference, segments the citing sentence into reviewable claim text, attaches the cited-source PDF, and then runs an evidence pipeline that retrieves and ranks candidate spans from the cited source and labels them (support/contradict/neutral) using NLI—after which the reviewer records a judgment and exports those judgments for downstream analysis.

Identity/scope/visibility contract for current POC behavior is documented in: `docs/IDENTITY_SCOPE_VISIBILITY_CONTRACT.md`.

Directional narrative for EU-library citation-walking target use is documented in: `docs/EU_LIBRARY_CITATION_WALKING_PATH.md`.

Design intent and architecture tradeoffs are documented in: `docs/SYSTEM_DESIGN_RATIONALE.md`.

---

## End-to-end flow (≈1 page)

### Actors
- **Reviewer (human)**: chooses callouts, edits claims, selects evidence, finalizes judgments.
- **UI (Streamlit)**: workspace, panels, state, calls backend APIs.
- **API (FastAPI)**: orchestrates ingestion, reference resolution, graph, attachments, evidence reruns, judgments.
- **GROBID**: primary PDF→TEI conversion and reference extraction.
- **Spine (Postgres + S3/MinIO)**: durable system-of-record for PDFs, attempts/jobs, artifacts, attachments, evidence runs/selections, and judgments.
- **Models**: embeddings + reranker (optionally ColBERT server) + NLI classifier; optional remote LLM (Blablador API) for some paths.

### Sequence (happy path)
1. **Upload citing PDF**
   - UI `POST /ingest` uploads PDF.
   - API stores PDF in object store (`pdf/...`), creates/updates spine records (`works`, `documents`, `document_versions`), and schedules extraction.
2. **Extract structure and references**
   - API triggers extraction (`/ingest/{doc_id}/extract`) using GROBID; persists artifacts (TEI, JSON) and attempt/job state.
   - API builds a document body model (paragraphs/sentences with stable-ish `sentence_id`) and exposes it via `GET /ingest/{doc_id}/body`.
3. **Resolve bibliography entries**
   - API resolves references (`/ingest/{doc_id}/resolve`), storing “best-effort” identifiers (DOI/title/year) with confidence and mismatch reasons; selection is editable.
4. **Navigate citations**
   - UI highlights in-text callouts; reviewer clicks a callout.
   - UI fetches local context via `GET /citation/context` and optional `GET /citation/graph` for graph expansion (OpenAlex when available).
5. **Create claims (reviewable units)**
   - Reviewer confirms/edits claim text for the chosen callout span; UI persists the “confirmed claim(s)” via `POST /claims/confirm`.
   - (In some modes) the claim ID encodes reviewer/citation context for span-first graph mirroring.
6. **Attach the cited-source PDF**
   - Reviewer uploads/drops a cited PDF (either claim-scoped or global “source bin” mode).
   - UI calls `POST /claims/{claim_id}/attachments` or `POST /attachments`; API stores the PDF, records an attachment row + timeline events, and starts background processing.
7. **Process attachments**
   - Background attachment pipeline parses the cited PDF (GROBID primary; fallback worker optional), extracts sentences/paragraphs, and tries to match/associate the PDF to the intended reference target.
   - Best-effort: attachments can be “promoted” into an ingested Work for work-level graphs and reuse.
8. **Run evidence matching (explicit rerun)**
   - UI requests candidates for a claim via `GET /claims/{claim_id}/evidence` and/or enqueues a rerun via `POST /claims/{claim_id}/evidence/rerun`.
   - Evidence pipeline seeds candidate windows (deterministic), retrieves/reranks (FAISS/BM25 + optional ColBERT), scores with NLI, and persists an immutable “run” with candidates + metadata.
9. **Review evidence and record outcomes**
   - UI renders evidence cards (entail/contradict/neutral), allows pinning/selection, and writes the chosen evidence via `PUT /claims/{claim_id}/evidence/selection`.
   - Reviewer records an overall judgment (draft/final + verdict + notes) via `PUT /claims/{claim_id}/judgment`.
10. **Export**
   - UI downloads exports from `GET /judgments/export` (claim-shaped or callout-shaped; JSON/CSV).

### Key design rules (repo-wide)
- **Read endpoints must not trigger expensive compute** (evidence reruns, parsing, OCR happen only via explicit triggers/background workers).
- **Artifacts are durable**: large blobs in S3/MinIO; pointers + metadata in Postgres; UI treats backend as source of truth.
- **Local-first by default**: external calls are optional/replaceable (Blablador, OpenAlex/Crossref, HF inference).

---

## Full technical spec (for a dev team)

### 1) Product goals
- Provide an end-to-end workflow to **validate whether a cited source supports a specific claim** made in a citing PDF.
- Make validation results **reviewable, attributable, exportable**, and reusable.
- Keep the system **runnable on a laptop** (Docker Compose path), while supporting “move heavy parts remote” later.

### 2) Non-goals (current)
- Multi-tenant auth/permissions (POC; internal-service token only).
- Large-corpus batch validation across hundreds of works.
- Fully automated adjudication without human review.

### 3) Primary user stories
- Upload a citing PDF and click an in-text citation to see context.
- Follow a citation target, attach the cited PDF, and see evidence candidates.
- Select evidence (or none), record a verdict + notes, and export judgments.
- (Power user) Compare reviewer outcomes and inspect disagreement in graph views.

### 4) Runtime architecture

**Services (Compose “full stack”)**
- `app-api`: FastAPI (`backend/main.py`).
- `app-ui`: Streamlit (`frontend/ui.py`).
- `grobid`: TEI extraction service.
- `postgres`: spine system-of-record (schema created idempotently on API startup).
- `minio`: S3-compatible object store for PDFs and large artifacts.
- `fallback-worker`: optional OCR-capable extraction worker (Phase 9.2).

**Optional/legacy service**
- `colbert_server`: FastAPI microservice (`colbert_server/colbert.py`) used when the hybrid pipeline enables ColBERT reranking.

### 5) Key modules (code map)
- **Backend**
  - `backend/main.py`: API surface + orchestration.
  - `backend/settings.py`: centralized config (`AppSettings`) via env vars + `.env`.
  - Ingestion (spine-first):
    - `backend/spine/*`: work/document/attempt/job/artifact primitives + views.
    - `backend/db/migrate.py`: idempotent table creation for spine tables.
    - `backend/object_store/s3.py`: object store wrappers (MinIO/S3).
  - Extraction:
    - `backend/grobid_client.py`, `backend/extraction.py`, `backend/tei_body.py`: TEI parsing + body model.
    - `backend/fallback_body.py`, `backend/fallback_text.py`, `backend/fallback_worker_app.py`: deterministic/fallback extraction paths.
    - `backend/spine/extraction_pool.py`: background extraction pool for spine attempts.
  - Attachments:
    - `backend/attachment_store.py`: attachment persistence + timeline events.
    - `backend/attachment_pipeline.py`: background processing (parse → match → artifacts).
  - Evidence:
    - `backend/evidence_matching/service.py`: rerun orchestration + queueing, “ensure current run”, history/candidates.
    - `backend/evidence_matching/pipeline.py`: retrieval/rerank/NLI scoring pipeline.
    - `backend/nli.py`: NLI scoring implementation + batching/fallback.
  - Graph:
    - `backend/graph_store.py`: graph nodes/edges/votes for “graph as architecture”.
    - `backend/span_graph_store.py`: span-first graph and mirroring from evidence selection.

- **Frontend**
  - `frontend/ui.py`: app shell + tabs + backend API calls.
  - `frontend/components/*`: workspace panels (chasing, evidence cards, surfing/graph).
  - `frontend/*_api.py`: typed HTTP clients + error handling.

### 6) Data model (conceptual)

**Ingestion spine (durable)**
- **Work**: canonical PDF unit (deduped by sha256), scoped by `project_id`.
- **Document / DocumentVersion**: identity split that supports future multiple versions per document.
- **Attempt**: an extraction/resolution run for a work, with state machine:
  `queued → running → succeeded | partial | failed | cancelled`
  plus `settings_hash`, `quality_json`, `failure_reason`.
- **Job**: worker slot tracking progress + heartbeat.
- **Artifact**: pointer to an object-store key (TEI, extraction JSON, resolution JSON, fallback bundles).

**Attachments + evidence + judgments**
- **Attachment**: a reviewer-attached cited-source PDF; has placement fields (claim/doc/citation/target) and a **timeline** (events) plus **artifacts** (TEI, extracted JSON).
- **EvidenceRun**: immutable results of a pipeline run for a claim (candidates + metadata).
- **EvidenceSelection**: reviewer’s chosen evidence and derived verdict (or “none”).
- **Judgment**: reviewer’s overall verdict/status and notes; exports derive from these.

**Graph / Surfing**
- **GraphNode/Edge**: application-level graph primitives for works/anchors/claims/assertions.
- **Edge votes**: reviewer-attributed votes/comments on relationships.
- **Span graph**: stable anchoring of citation spans and claim atoms; evidence selections can be mirrored as span-first assertions.

### 7) Persistence details

**Postgres (created by `backend/db/migrate.py`)**
- Works/ingestion: `works`, `attempts`, `jobs`, `artifacts`
- Documents/projects: `documents`, `document_versions`, `project_documents`, `project_meta`
- Settings/workflows/locators: `settings_bundles`, `settings_versions`, `settings_field_policies`, `workflow_definitions`, `workflow_versions`, `locators`
- Runtime state: `background_state`
- Workflow data: `attachments`, `attachment_events`, `attachment_artifacts`, `evidence_runs`, `evidence_selections`, `judgments`, `confirmed_claims`
- Graph: `graph_nodes`, `graph_aliases`, `graph_edges`, `graph_edge_votes`
- Span graph: `span_graph_*` tables (works, cites, spans, atoms, assertions, neighborhood runs/candidates, etc.)

**Object store layout (representative)**
- PDFs: `pdf/{work_id}/{sha256}.pdf`
- Primary extraction: `extract/{work_id}/attempts/{attempt_id}/primary/tei.xml` and `.../extraction.json`
- Resolution: `extract/{work_id}/attempts/{attempt_id}/resolution.json`
- Attachment artifacts: keyed under attachment artifact pointers (stored in Postgres).

**Local filesystem (legacy/POC)**
- Some legacy stores still exist under `data/` (bind-mounted in Compose), but the target direction is “spine everywhere” (Phase 9.3).

### 8) API surface (high-level)

**Ingestion**
- `POST /ingest`: upload PDF.
- `GET /ingest`: list ingests (spine-backed).
- `GET /ingest/{doc_id}`: ingest metadata.
- `POST /ingest/{doc_id}/extract`: run extraction (async spine attempt).
- `GET /ingest/{doc_id}/extraction`: extraction status/artifacts view.
- `POST /ingest/{doc_id}/resolve`: resolve references; `GET /ingest/{doc_id}/resolution`.
- `GET /ingest/{doc_id}/body`: document body model for UI navigation.

**Citation navigation**
- `GET /citation/context`: sentence-level context for a callout.
- `GET /citation/graph`: small citation tree expansion (OpenAlex when resolved).

**Claims**
- `POST /claims/confirm`: persist confirmed claim text for a citation context.
- `GET /claims/{claim_id}/status`, `GET /claims/{claim_id}/span-context`: UI support endpoints.

**Attachments**
- `POST /claims/{claim_id}/attachments`: claim-scoped attachment.
- `POST /attachments`: global attachment (“source bin”) mode.
- `GET /attachments`, `PATCH /attachments/{attachment_id}`, `GET /attachments/{attachment_id}`: list + placement/archive updates.

**Evidence**
- `GET /claims/{claim_id}/evidence`: list candidates (paged); does not force rerun.
- `POST /claims/{claim_id}/evidence/rerun`: enqueue/trigger rerun (background workers).
- `GET /claims/{claim_id}/evidence/history`: recent runs.
- `GET/PUT /claims/{claim_id}/evidence/selection`: read/write chosen evidence.

**Judgments**
- `GET/PUT /claims/{claim_id}/judgment`: per-reviewer judgment.
- `GET /claims/{claim_id}/judgments`: list all reviewers’ judgments.
- `GET /judgments`, `GET /judgments/export`: listing + export.

**Project + operational**
- `GET/PUT /project`, `GET /project/export`, `POST /project/import`
- `GET/POST /spine/settings`, `GET/POST /spine/workflows`, locator endpoints
- `GET/POST /background/*` pause/resume for background work gating

### 9) Evidence pipeline (implementation expectations)
- Deterministic **window seeding** over cited-source sentences/paragraphs (stable output for the same inputs).
- Retrieval/merging can include:
  - lexical seeding (BM25-like),
  - embedding similarity (FAISS),
  - optional ColBERT reranking (via `colbert_server`),
  - cross-encoder reranker (HF model).
- NLI scoring labels each candidate span as `entail`, `contradict`, or `neutral` with confidence.
- Runs are persisted as immutable records; UI pages through candidates and fetches history.
- Profiles (e.g. “Fast/Local”, “Best/Local”) map to a settings overlay applied per rerun.

### 10) UI spec (Streamlit)
- One primary workspace with:
  - **Document** view: read the citing doc body and click callouts.
  - **Chasing/Review**: claim queue, claim editing, citation follow panel.
  - **Evidence**: evidence cards + rerun controls + selection + rationale sidebar.
  - **Graph/Surfing**: claim/work graph inspection (“graph as architecture” POC).
- State rules:
  - one canonical session-state key per data item;
  - duplicated widgets use unique widget keys and synchronize back to canonical state.

### 11) Configuration
- Centralized via `backend/settings.py` (`AppSettings`) with env vars for:
  - runtime endpoints (`BACKEND_URL`, `GROBID_URL`, `FALLBACK_WORKER_URL`)
  - spine (`POSTGRES_DSN`, `S3_*`, bucket)
  - model settings (embedding/reranker/NLI thresholds, profiles).
- Docker Compose wires the defaults for local operation (`docker-compose.yml`).

### 12) Security and privacy (POC posture)
- No end-user auth; assume trusted local environment.
- Internal services can use `INTERNAL_SERVICE_TOKEN` for bearer auth on worker calls.
- External requests (Crossref/OpenAlex/Blablador/HF) are optional and should be treated as:
  - configurable,
  - minimize data sent (privacy constraint),
  - replaceable for future hosted deployments.

### 13) Observability and operations
- API logs are structured-ish via Python logging; major state transitions are recorded in spine tables.
- Attempt/job state and artifact pointers are the primary audit trail for ingestion/extraction.
- Background work can be globally paused to avoid runaway CPU usage during demos/dev.

### 14) Testing and verification
- Unit/integration tests via `pytest` (`pytest.ini`, `tests/`).
- Smoke workflows via `scripts/dev/*` and `smoke.py` (upload → extract → follow → attach → evidence → export).

---

## Plans (execution + roadmap)

### A) Current roadmap position (repo planning docs)
- Roadmap phases 1–7 are marked complete; Phase 9.2 is documented as complete in `.planning/STATE.md`.
- The next major delivery target is **Phase 9.3: Spine Everywhere + Legacy Removal** (move remaining legacy disk stores to Postgres/S3 and delete compat tails).

### B) Next milestones (recommended for a dev team)
1. **Phase 9.3-01..06**: move attachments/judgments/evidence/project meta fully onto spine; delete legacy stores.
2. **Phase 8/8.1**: polish reviewer UX (inline citations, background processing ergonomics) and integrate graph navigation into the primary workflow.
3. **Hardening**: add correlation IDs across attempts/jobs + worker calls; define explicit readiness/liveness endpoints; add safe “wipe all” admin gating for dev only.

### C) Delivery plan template (2–4 week increments)
- Week 1: spine-only persistence for judgments + evidence selections + exports (no `data/` dependency).
- Week 2: spine-only attachments (records + artifacts) + backfill/migration strategy for existing local data.
- Week 3: remove legacy codepaths and update smoke tests + docs; stabilize Compose “one command runs all”.
- Week 4: UX/ops pass: Works Manager-style status surfacing and better failure remediation loops.
