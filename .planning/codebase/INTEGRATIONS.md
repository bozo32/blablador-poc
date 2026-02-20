# External Integrations

**Analysis Date:** 2026-02-20

## APIs & External Services

**LLM / NLP APIs:**
- Blablador API (Helmholtz Blablador)
  - Used for: completions and embeddings in some paths (e.g. best-passage rationale)
  - Implementation: `backend/bl_client.py`, `backend/utils.py`, `frontend/ui.py`
  - SDK/Client: `requests` (custom client)
  - Auth: `API_KEY`
  - Base URL: `API_BASE`
  - Default model alias: `DEFAULT_LLM_MODEL`

**Scholarly metadata / citation graph:**
- OpenAlex Works API
  - Used for: citation graph expansion and reference resolution
  - Implementation: `backend/citation_graph.py`, `backend/reference_resolver.py`, `backend/main.py`
  - SDK/Client: `requests`
  - Auth: optional `OPENALEX_API_KEY`
  - Base URL: `OPENALEX_API_URL`

- Crossref REST API
  - Used for: reference resolution by DOI / bibliographic query
  - Implementation: `backend/reference_resolver.py`
  - SDK/Client: `requests`
  - Auth: none (requires contact email)
  - Contact param: `CROSSREF_MAILTO`
  - Base URL: `CROSSREF_API_URL`

**Model hosting:**
- Hugging Face Inference API
  - Used for: optional remote NLI scoring when enabled
  - Implementation: `backend/utils.py` (`hf_inference_post`), `backend/nli.py`
  - Auth: `HF_API_TOKEN`
  - Toggle: `HF_REMOTE_INFERENCE`

**Document structure extraction:**
- GROBID (containerized service)
  - Used for: PDF -> TEI conversion and reference extraction
  - Client: `backend/grobid_client.py`
  - Service URL: `GROBID_URL`
  - Compose image: `lfoppiano/grobid:0.8.0` (`docker-compose.yml`)

**Optional internal services:**
- Fallback OCR Worker (FastAPI)
  - Used for: optional OCR-capable fallback extraction delegated from spine extraction
  - API: `POST /v1/fallback` (`backend/ocr_worker_app.py`)
  - Service URL: `FALLBACK_WORKER_URL` (`backend/settings.py`, `docker-compose.yml`)
  - Auth: `INTERNAL_SERVICE_TOKEN` (Bearer)

- ColBERT API Server (FastAPI microservice)
  - Used for: optional reranking when hybrid pipeline enables ColBERT
  - Implementation: `colbert_server/colbert.py`
  - Service URL: `COLBERT_API_URL` (default `http://localhost:7001`, `backend/settings.py`)

## Data Storage

**Databases:**
- Postgres (spine system-of-record)
  - Connection: `POSTGRES_DSN` (`backend/settings.py`, `docker-compose.yml`)
  - Client: `psycopg` wrapper (`backend/db/pg.py`)
  - Schema bootstrap/migrations: `backend/db/migrate.py`
  - Compose image: `postgres:16-alpine` (`docker-compose.yml`)

**File Storage:**
- S3-compatible object store (MinIO in dev)
  - Used for: PDFs and large extraction artifacts
  - Client: `backend/object_store/s3.py` (boto3 client, path-style addressing)
  - Connection/auth:
    - `S3_ENDPOINT_URL`
    - `S3_ACCESS_KEY`
    - `S3_SECRET_KEY`
    - `S3_BUCKET_WORKS`
    - `S3_REGION`
    - `S3_USE_SSL`
  - Compose image: `minio/minio:latest` + init job `minio/mc:latest` (`docker-compose.yml`)

**Caching:**
- In-memory process caches only (e.g. OpenAlex response caches in `backend/citation_graph.py`)

## Authentication & Identity

**Auth Provider:**
- None for end-user authentication (single-user/POC mode)
  - Internal service auth: shared bearer token `INTERNAL_SERVICE_TOKEN` used by worker endpoints (`backend/ocr_worker_app.py`, `backend/spine/extraction_pool.py`)
  - External API keys:
    - `API_KEY` for Blablador (`backend/bl_client.py`)
    - `OPENALEX_API_KEY` optional (`backend/citation_graph.py`, `backend/reference_resolver.py`)
    - `HF_API_TOKEN` optional (`backend/nli.py`)

## Monitoring & Observability

**Error Tracking:**
- Not detected

**Logs:**
- Standard Python logging / stdout (e.g. `backend/nli.py`, Docker container logs)

## CI/CD & Deployment

**Hosting:**
- Docker Compose local stack (API + UI + Grobid + Postgres + MinIO + optional worker) (`docker-compose.yml`, `scripts/dev/up.sh`)

**CI Pipeline:**
- Not detected (no `.github/workflows/*`)

## Environment Configuration

**Required env vars:**
- Core (local-first): `BACKEND_URL` (UI -> API), `GROBID_URL`
- Spine persistence: `POSTGRES_DSN`, `S3_ENDPOINT_URL`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET_WORKS`, `S3_REGION`, `S3_USE_SSL`
- Internal auth: `INTERNAL_SERVICE_TOKEN`

**Optional env vars (feature toggles / external calls):**
- Blablador: `API_KEY`, `API_BASE`, `DEFAULT_LLM_MODEL` (`backend/settings.py`, `.env.example`)
- OpenAlex: `OPENALEX_API_KEY`, `OPENALEX_API_URL`
- Crossref: `CROSSREF_MAILTO`, `CROSSREF_API_URL`
- HF inference: `HF_REMOTE_INFERENCE`, `HF_API_TOKEN`
- Fallback worker: `FALLBACK_WORKER_URL`, `FALLBACK_OCR_LANGUAGES`, `FALLBACK_TEXT_MIN_CHARS`, `FALLBACK_OCR_DPI_STEPS`
- ColBERT: `COLBERT_MODE`, `COLBERT_API_URL` (and service-side `COLBERT_INDEX`, `COLBERT_COLLECTION`, `COLBERT_CHECKPOINT` in `colbert_server/colbert.py`)

**Secrets location:**
- Local dev: repo-root `.env` loaded by `backend/settings.py` and `python-dotenv` (`application.py`)
- Compose dev: service env injected directly in `docker-compose.yml`

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- None detected (HTTP calls are direct API requests, not webhook delivery)

---

*Integration audit: 2026-02-20*
