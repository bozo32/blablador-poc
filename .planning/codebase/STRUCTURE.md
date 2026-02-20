# Codebase Structure

**Analysis Date:** 2026-02-20

## Directory Layout

```
[project-root]/
├── backend/                 # FastAPI backend, spine persistence, pipelines
├── frontend/                # Streamlit UI, components, API clients
├── tests/                   # Pytest test suite
├── docker/                  # Dockerfiles for app-api/app-ui/fallback-worker
├── scripts/                 # Dev/smoke/e2e helper scripts
├── docs/                    # Technical specs + workflow docs
├── requirements/            # Pinned runtime deps (app.in + app.lock.txt)
├── fixtures/                # Small PDFs and fixture inputs for smoke/tests
├── data/                    # Local disposable artifacts + legacy stores
├── corpus/                  # Local corpus folders (focus/local/ok)
├── colbert_server/          # Optional ColBERT microservice (uvicorn app)
├── ColBERT/                 # Vendored upstream ColBERT repo
├── .planning/               # Project planning + codebase maps
├── .opencode/               # Vendored OpenCode/GSD workflow tooling
├── docker-compose.yml       # Multi-service local stack
├── application.py           # Local dev launcher (backend + UI + optional ColBERT)
└── README.md                # Quickstart + service overview
```

## Directory Purposes

**backend/:**
- Purpose: Backend API surface + all durable persistence + pipelines.
- Contains: `backend/main.py` routes; spine modules in `backend/spine/`; graph stores (`backend/graph_store.py`, `backend/span_graph_store.py`); evidence pipeline (`backend/evidence_matching/*`); attachment processing (`backend/attachment_pipeline.py`, `backend/attachment_store.py`).
- Key files: `backend/main.py`, `backend/settings.py`, `backend/db/migrate.py`, `backend/object_store/s3.py`.

**backend/spine/:**
- Purpose: Postgres-first primitives for ingestion and durable workflow state.
- Contains: helpers for works/documents/versions, attempts/jobs/artifacts, project meta, settings/workflows/locators.
- Key files: `backend/spine/ingest_view.py`, `backend/spine/extraction_pool.py`, `backend/spine/attempts.py`.

**backend/evidence_matching/:**
- Purpose: Evidence rerun orchestration + pipeline code.
- Contains: deterministic seeding, pipeline execution, serialization, persistence.
- Key files: `backend/evidence_matching/service.py`, `backend/evidence_matching/pipeline.py`, `backend/evidence_matching/store.py`.

**frontend/:**
- Purpose: Streamlit UI shell and client-side state helpers.
- Contains: UI app (`frontend/ui.py`), API clients (`frontend/*_api.py`), session-state stores (`frontend/*_store.py`), and panel components.
- Key files: `frontend/ui.py`, `frontend/ingestion_api.py`, `frontend/evidence_api.py`, `frontend/components/live_surfing_panel.py`.

**frontend/components/:**
- Purpose: Reusable Streamlit UI panels and renderers.
- Contains: “workspace” panels (chasing, evidence cards, graph/surfing) and Cytoscape component wrappers.
- Key files: `frontend/components/chasing_panel.py`, `frontend/components/evidence_card.py`, `frontend/components/live_surfing_panel.py`.

**tests/:**
- Purpose: Automated verification via pytest.
- Contains: `test_*.py` files and shared fixtures in `tests/conftest.py`.
- Key files: `tests/conftest.py`, `tests/test_segment_endpoint.py`, `tests/test_evidence_matching_pipeline.py`.

**docker/:**
- Purpose: Container builds for the Compose stack.
- Contains: `docker/app-api/Dockerfile`, `docker/app-ui/Dockerfile`, `docker/fallback-worker/Dockerfile`.

**scripts/:**
- Purpose: Dev/smoke/e2e utilities for running the stack and workflows.
- Contains: `scripts/dev/up.sh`, `scripts/dev/smoke_ingest.sh`, `scripts/dev/pytest_docker.sh`.

**docs/:**
- Purpose: Repo specification and operational docs.
- Contains: `docs/REPO_SPEC.md` (technical spec), `docs/GSD.md` (workflow), `docs/PUBLISH_PUBLIC.md`.

**requirements/:**
- Purpose: Dependency inputs and pinned lock.
- Contains: `requirements/app.in`, `requirements/app.lock.txt`.

**fixtures/:**
- Purpose: Small deterministic fixture PDFs and corpus lists.
- Contains: `fixtures/sample.pdf`, `fixtures/scanned-1.pdf`, `fixtures/text-1.pdf`.

**data/:**
- Purpose: Local/dev artifacts and legacy durable stores.
- Contains: `data/graph.db`, `data/claims.db`, `data/poc_graph.json`, and temporary caches.
- Generated: Yes (runtime output).
- Committed: Mixed (directory exists in repo; contents are not a durable contract).

**colbert_server/:**
- Purpose: Optional ColBERT reranking service.
- Contains: `colbert_server/colbert.py` uvicorn app.

**ColBERT/:**
- Purpose: Vendored upstream ColBERT codebase.
- Contains: its own `ColBERT/server.py`, package sources, and docs.
- Generated: No (vendored source).
- Committed: Yes.

## Key File Locations

**Entry Points:**
- `backend/main.py`: FastAPI application and routes.
- `frontend/ui.py`: Streamlit application.
- `application.py`: Local launcher for backend + UI (+ optional ColBERT server).
- `backend/ocr_worker_app.py`: Optional OCR fallback worker FastAPI app.
- `docker-compose.yml`: Compose orchestration for API/UI/GROBID/Postgres/MinIO.

**Configuration:**
- `backend/settings.py`: Centralized `AppSettings` (env + `.env`).
- `docker-compose.yml`: Service wiring + env vars for local stack.
- `requirements/app.in`: Unpinned dependency list.
- `requirements/app.lock.txt`: Pinned install list used in Dockerfiles.

**Core Logic:**
- Ingestion spine: `backend/spine/*`, schema DDL in `backend/db/migrate.py`.
- Object store: `backend/object_store/s3.py`.
- Extraction: `backend/grobid_client.py`, `backend/extraction.py`, `backend/tei_body.py`, worker pool `backend/spine/extraction_pool.py`.
- Attachments: `backend/attachment_store.py`, `backend/attachment_pipeline.py`.
- Evidence: `backend/evidence_matching/*`, NLI `backend/nli.py`.
- Graph: `backend/graph_store.py`, `backend/span_graph_store.py`.

**Testing:**
- `tests/`: pytest tests.
- `pytest.ini`: pytest configuration.

## Naming Conventions

**Files:**
- Backend stores/persistence helpers: `backend/*_store.py` (e.g. `backend/attachment_store.py`).
- Frontend HTTP clients: `frontend/*_api.py` (e.g. `frontend/ingestion_api.py`).
- Streamlit components/panels: `frontend/components/*_panel.py` (e.g. `frontend/components/chasing_panel.py`).
- Tests: `tests/test_*.py`.

**Directories:**
- Top-level app modules: `backend/`, `frontend/`.
- Feature clusters: `backend/spine/`, `backend/evidence_matching/`, `frontend/components/`.

## Where to Add New Code

**New Feature (end-to-end workflow step):**
- Primary code: add API endpoints and orchestration in `backend/main.py` (or introduce a new module and import from `backend/main.py` to keep route handlers thin).
- Persistence: prefer Postgres + S3 via `backend/spine/*` and `backend/object_store/s3.py` (avoid adding new durable state under `data/`).
- UI: add a new panel/component in `frontend/components/` and wire it into `frontend/ui.py`.
- Tests: add pytest coverage in `tests/` (follow existing `test_*.py` naming).

**New Component/Module:**
- Backend domain module: place under `backend/` with a clear suffix (`*_store.py`, `*_client.py`, `*_pipeline.py`).
- Spine primitive/helper: place under `backend/spine/` when it reads/writes durable tables or artifacts.
- Frontend API wrapper: add a new `frontend/<area>_api.py` when introducing new endpoints.

**Utilities:**
- Shared backend helpers: `backend/utils.py`.
- Shared frontend state keys: `frontend/state_keys.py`.

## Special Directories

**.planning/:**
- Purpose: Project roadmap/state and generated codebase maps.
- Generated: Mixed.
- Committed: Yes.

**.opencode/:**
- Purpose: Vendored OpenCode/GSD tooling (Bun-based).
- Generated: `.opencode/node_modules/` is generated.
- Committed: Source is committed; dependencies are not.

---

*Structure analysis: 2026-02-20*
