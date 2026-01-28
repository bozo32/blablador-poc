# Codebase Structure

**Analysis Date:** 2026-01-23

## Directory Layout

```
[project-root]/
├── application.py             # CLI launcher for backend/UI/ColBERT services
├── backend/                   # FastAPI app + ML pipeline modules
├── frontend/                  # Streamlit UI
├── colbert_server/            # ColBERT FastAPI reranker service
├── ColBERT/                   # Vendored ColBERT repository
├── data/                      # ColBERT collections and indexes
├── experiments/               # Experiment artifacts (indexes, metadata)
├── tests/                     # Pytest tests and fixtures
├── environment.yml            # Primary environment definition
├── smoke.py                   # Dependency smoke check
└── README.md                  # Project overview and setup
```

## Directory Purposes

**backend/**
- Purpose: FastAPI API layer plus retrieval, parsing, and NLI pipeline logic.
- Contains: Core pipeline modules, settings, schemas, utilities.
- Key files: `backend/main.py`, `backend/retriever.py`, `backend/hybrid.py`, `backend/parser.py`, `backend/settings.py`.

**frontend/**
- Purpose: Streamlit user interface for upload, segmentation, and results.
- Contains: Single Streamlit app entrypoint.
- Key files: `frontend/ui.py`.

**colbert_server/**
- Purpose: Separate FastAPI service that hosts ColBERT build/search APIs.
- Contains: ColBERT API app and its environment definition.
- Key files: `colbert_server/colbert.py`, `colbert_server/environment.yml`.

**ColBERT/**
- Purpose: Local ColBERT source checkout used by the reranker service.
- Contains: ColBERT library and utilities.
- Key files: `ColBERT/colbert`, `ColBERT/server.py`.

**data/**
- Purpose: Local ColBERT data artifacts (collections, indexes).
- Contains: Default collection/index data.
- Key files: `data/colbert/collections/default.tsv`.

**experiments/**
- Purpose: Index artifacts produced by ColBERT runs.
- Contains: Default experiment outputs.
- Key files: `experiments/default/indexes/default/metadata.json`.

**tests/**
- Purpose: Automated tests for parsing, retrieval, coreference, and endpoints.
- Contains: Pytest files and dummy data.
- Key files: `tests/test_parser.py`, `tests/test_retriever.py`, `tests/test_segment_endpoint.py`.

## Key File Locations

**Entry Points:**
- `application.py`: CLI launcher for all services.
- `backend/main.py`: FastAPI backend application.
- `frontend/ui.py`: Streamlit UI entrypoint.
- `colbert_server/colbert.py`: ColBERT FastAPI service.

**Configuration:**
- `backend/settings.py`: Runtime settings and env bindings.
- `.streamlit/config.toml`: Streamlit configuration.
- `environment.yml`: Base environment dependencies.
- `colbert_server/environment.yml`: ColBERT environment dependencies.

**Core Logic:**
- `backend/retriever.py`: FAISS index builder and query logic.
- `backend/hybrid.py`: Hybrid pipeline implementation.
- `backend/parser.py`: TEI/CSV parsing and windowing.
- `backend/nli.py`: NLI inference pipeline.
- `backend/utils.py`: Embedding, reranking, and helper utilities.

**Testing:**
- `tests/test_parser.py`: TEI/CSV parsing tests.
- `tests/test_retriever.py`: FAISS retrieval tests.
- `tests/test_segment_endpoint.py`: FastAPI endpoint tests.

## Naming Conventions

**Files:**
- Snake_case module files for Python code (`backend/retriever.py`, `frontend/ui.py`).

**Directories:**
- Lowercase for project code (`backend`, `frontend`, `tests`), capitalized for vendored repo (`ColBERT`).

## Where to Add New Code

**New Feature:**
- Primary code: `backend/main.py` (endpoint wiring) and `backend/schemas.py` (payload models).
- Tests: `tests/` with `test_*.py` naming (`tests/test_segment_endpoint.py`).

**New Component/Module:**
- Implementation: `backend/` as a new snake_case module imported by `backend/main.py`.

**Utilities:**
- Shared helpers: `backend/utils.py`.

## Special Directories

**data/**
- Purpose: ColBERT collections and index files.
- Generated: Yes.
- Committed: Yes.

**experiments/**
- Purpose: ColBERT experiment outputs.
- Generated: Yes.
- Committed: Yes.

**__pycache__/**
- Purpose: Python bytecode caches.
- Generated: Yes.
- Committed: No.

---

*Structure analysis: 2026-01-23*
