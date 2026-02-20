# Technology Stack

**Analysis Date:** 2026-02-20

## Languages

**Primary:**
- Python 3.10 - Backend API (`backend/*.py`), Streamlit UI (`frontend/*.py`), workers (`backend/ocr_worker_app.py`)

**Secondary:**
- Shell (POSIX sh) - Dev scripts (`scripts/dev/up.sh`, `scripts/dev/down.sh`, `scripts/dev/smoke_ingest.sh`)
- HTML/JavaScript - Streamlit component vendor assets (`frontend/components/cytoscape_component/index.html`, `frontend/components/cytoscape_component/vendor/*.js`)
- YAML/TOML - Environment + app config (`environment.yml`, `.streamlit/config.toml`, `docker-compose.yml`)

## Runtime

**Environment:**
- Python 3.10 (Conda dev env: `environment.yml`; containers: `docker/app-api/Dockerfile`, `docker/app-ui/Dockerfile`, `docker/fallback-worker/Dockerfile`)

**Package Manager:**
- pip (runtime deps installed from a compiled lock)
- Lockfile: present (`requirements/app.lock.txt`, generated from `requirements/app.in`)

## Frameworks

**Core:**
- FastAPI 0.129.0 - HTTP API service (`backend/main.py`, `docker/app-api/Dockerfile`)
- Streamlit 1.54.0 - Web UI (`frontend/ui.py`, `docker/app-ui/Dockerfile`)
- Pydantic 2.12.5 + pydantic-settings 2.12.0 - Config and request/response schemas (`backend/settings.py`, `backend/schemas.py`)

**Testing:**
- pytest - Test runner (config: `pytest.ini`, tests in `tests/`)
- FastAPI TestClient - HTTP API tests (`tests/test_*.py`)

**Build/Dev:**
- Docker + Docker Compose - Local multi-service stack (`docker-compose.yml`, `scripts/dev/up.sh`)
- pre-commit (Black + Flake8) - Formatting/lint hooks (`.pre-commit-config.yaml`, `.flake8`)

## Key Dependencies

**Critical:**
- requests 2.32.5 - HTTP client for external/internal services (`backend/bl_client.py`, `backend/grobid_client.py`, `backend/reference_resolver.py`)
- lxml 6.0.2 - TEI/XML parsing (`backend/extraction.py`, `backend/grobid_client.py`)

**Infrastructure:**
- psycopg[binary] 3.3.2 - Postgres spine persistence (`backend/db/pg.py`, `backend/db/migrate.py`)
- boto3 1.42.49 - S3/MinIO object store (`backend/object_store/s3.py`)

**ML/NLP (local-first):**
- torch 2.10.0 + transformers 5.1.0 - NLI model inference (`backend/nli.py`)
- sentence-transformers 5.2.2 - Embeddings + cross-encoder reranking (`backend/utils.py`)
- faiss-cpu 1.13.2 - Vector search (`backend/retriever.py`, `backend/utils.py`)
- rank-bm25 0.2.2 - Lexical retrieval option (`backend/hybrid.py`, `backend/utils.py`)

**PDF/Fallback extraction:**
- pypdf 6.7.0 - Non-TEI text extraction fallback (`backend/fallback_text.py`)
- PyMuPDF 1.27.1 + pytesseract 0.3.13 + Pillow 12.1.1 - OCR-capable fallback worker (`backend/ocr_worker_app.py`, `docker/fallback-worker/Dockerfile`)

## Configuration

**Environment:**
- Centralized settings via Pydantic settings (`backend/settings.py`)
  - Loads repo-root `.env` by default (see `backend/settings.py` `env_file=.../.env`)
  - `.env.example` documents a minimal local setup (`.env.example`)

**Build:**
- Docker images install pinned Python deps from `requirements/app.lock.txt` (`docker/app-api/Dockerfile`, `docker/app-ui/Dockerfile`, `docker/fallback-worker/Dockerfile`)
- Compose wires services + default env (`docker-compose.yml`)
- Streamlit watcher config (`.streamlit/config.toml`)

## Platform Requirements

**Development:**
- Docker + Docker Compose for the full stack (`docker-compose.yml`, `scripts/dev/up.sh`)
- Python 3.10 for host-run workflows (`environment.yml`, `application.py`)
- Optional: separate Conda env for ColBERT service (`colbert_server/environment.yml`, `application.py`)

**Production:**
- Containerized deployment (Compose-compatible) with externalized Postgres + S3-compatible object store recommended (`docker-compose.yml`, `backend/settings.py`)

---

*Stack analysis: 2026-02-20*
