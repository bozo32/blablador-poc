# External Integrations

**Analysis Date:** 2026-01-23

## APIs & External Services

**LLM/Embedding API:**
- Blablador API (OpenAI-compatible) - completions and embeddings
  - SDK/Client: `requests` in `backend/bl_client.py`, `frontend/ui.py`
  - Auth: `API_KEY` via `backend/settings.py`, `application.py`

**Model Hub:**
- Hugging Face model downloads for embeddings/NLI
  - SDK/Client: `transformers` and `sentence-transformers` in `backend/nli.py`, `backend/utils.py`
  - Auth: Not detected

**Reranker Service:**
- ColBERT API server (local FastAPI service)
  - SDK/Client: `requests` in `backend/utils.py`, `backend/hybrid.py`
  - Auth: Not detected

**Bibliographic Metadata:**
- Crossref REST API for reference resolution
  - SDK/Client: `requests` in `backend/reference_resolver.py`
  - Auth: `CROSSREF_MAILTO` contact email in `backend/settings.py`

## Data Storage

**Databases:**
- Not detected

**File Storage:**
- Local filesystem only (FAISS index files) - `backend/retriever.py`
- ColBERT index/collection storage - `colbert_server/colbert.py`

**Caching:**
- Local in-memory and on-disk model caches (Hugging Face) - `backend/nli.py`, `backend/utils.py`

## Authentication & Identity

**Auth Provider:**
- API key header for Blablador API
  - Implementation: Bearer token via `requests` in `backend/bl_client.py`, `frontend/ui.py`

## Monitoring & Observability

**Error Tracking:**
- None

**Logs:**
- Python logging/stdout in `backend/main.py`, `backend/nli.py`, `application.py`

## CI/CD & Deployment

**Hosting:**
- Local process execution (uvicorn + streamlit) - `application.py`

**CI Pipeline:**
- None

## Environment Configuration

**Required env vars:**
- `API_KEY` - Blablador API auth in `backend/settings.py`, `application.py`
- `API_BASE` - Blablador API base URL in `backend/settings.py`, `application.py`
- `BACKEND_URL` - UI to backend routing in `backend/settings.py`, `frontend/ui.py`
- `PIPELINE_MODE` - classic/hybrid selection in `backend/settings.py`
- `CROSSREF_MAILTO` - Crossref REST API contact email in `backend/settings.py`
- `CROSSREF_API_URL` - Crossref REST API base URL in `backend/settings.py`
- `COLBERT_API_URL` - ColBERT service URL in `backend/settings.py`, `backend/hybrid.py`
- `COLBERT_INDEX`, `COLBERT_COLLECTION`, `COLBERT_CHECKPOINT` - ColBERT paths in `colbert_server/colbert.py`
- `COLBERT_CONDA_ENV` - ColBERT server env name in `application.py`

**Secrets location:**
- `.env` via `backend/settings.py`
- CLI flags `--api_key`/`--api_base` in `application.py`

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

---

*Integration audit: 2026-01-23*
