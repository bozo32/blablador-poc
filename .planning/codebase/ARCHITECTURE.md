# Architecture

**Analysis Date:** 2026-01-23

## Pattern Overview

**Overall:** Pipeline-centric ML service with a FastAPI backend, Streamlit UI, and a separate ColBERT reranker service orchestrated by a CLI launcher (`backend/main.py`, `frontend/ui.py`, `colbert_server/colbert.py`, `application.py`).

**Key Characteristics:**
- Mode-based pipeline selection (classic vs hybrid) via a registry and settings object (`backend/pipeline_registry.py`, `backend/settings.py`).
- In-memory retriever cache keyed by row/segment with per-request rebuilding when needed (`backend/main.py`, `backend/utils.py`).
- Heavy ML inference through local HF/SentenceTransformer pipelines with caching and optional external LLM calls (`backend/nli.py`, `backend/utils.py`, `backend/bl_client.py`).

## Layers

**CLI Orchestration:**
- Purpose: Launch FastAPI, Streamlit, and ColBERT services together.
- Location: `application.py`
- Contains: subprocess startup, environment setup, health checks.
- Depends on: OS env, uvicorn/streamlit commands.
- Used by: CLI user flows.

**Frontend UI:**
- Purpose: Upload data, configure pipeline settings, call backend endpoints, display results.
- Location: `frontend/ui.py`
- Contains: Streamlit UI, session state, LLM segmentation helpers.
- Depends on: HTTP calls to FastAPI, settings, model cache.
- Used by: Human user interacting with the app.

**API Layer:**
- Purpose: Expose segmentation and prebuild endpoints for retrieval/NLI flow.
- Location: `backend/main.py`
- Contains: FastAPI app, `/segment` and `/prebuild` endpoints, in-memory caches.
- Depends on: `backend/schemas.py`, `backend/pipeline_registry.py`, `backend/utils.py`, `backend/nli.py`.
- Used by: Streamlit UI and potential external callers.

**Pipeline/Domain Logic:**
- Purpose: Build FAISS indexes, retrieve candidates, rerank, and assess NLI.
- Location: `backend/retriever.py`, `backend/hybrid.py`, `backend/parser.py`, `backend/nli.py`, `backend/utils.py`
- Contains: Retriever class, HybridPipeline flow, TEI/CSV parsing, NLI scoring, reranking utilities.
- Depends on: SentenceTransformers, FAISS, HF models, ColBERT API client.
- Used by: FastAPI endpoints.

**External Reranker Service:**
- Purpose: Provide ColBERT indexing and search via HTTP for hybrid reranking.
- Location: `colbert_server/colbert.py`
- Contains: FastAPI app with `/build` and `/search` endpoints.
- Depends on: ColBERT Python package and local index storage.
- Used by: `backend/utils.py` via `colbert_api_rerank`.

## Data Flow

**Segment Evaluation Flow:**
1. User uploads CSV/TEI data and configures settings in Streamlit (`frontend/ui.py`).
2. UI generates segment candidates via the Blablador LLM API and posts `/segment` requests (`frontend/ui.py`, `backend/bl_client.py`).
3. Backend selects pipeline implementation using registry + settings (`backend/main.py`, `backend/pipeline_registry.py`, `backend/settings.py`).
4. Backend builds/reuses FAISS retrievers and embeds claims (`backend/main.py`, `backend/retriever.py`, `backend/utils.py`).
5. Backend optionally reranks with cross-encoder or ColBERT API (`backend/main.py`, `backend/utils.py`, `colbert_server/colbert.py`).
6. Backend runs NLI over candidate passages and returns evidence list (`backend/nli.py`, `backend/main.py`).
7. UI renders evidence, collects user assessments, and exposes results (`frontend/ui.py`).

**Prebuild Flow (Classic):**
1. UI triggers `/prebuild` to build indexes once (`frontend/ui.py`).
2. Backend constructs retrievers for the selected folder (`backend/main.py`, `backend/retriever.py`).

**State Management:**
- Backend uses process-local dictionaries for retrievers and results (`backend/main.py`).
- Streamlit uses `st.session_state` for UI state and cached segments (`frontend/ui.py`).
- Model caching uses LRU caches and in-memory maps (`backend/nli.py`, `backend/utils.py`).

## Key Abstractions

**Retriever:**
- Purpose: Build and query FAISS indexes over TEI/CSV-derived chunks.
- Examples: `backend/retriever.py`
- Pattern: Stateful class with build/load/query methods and internal docstore.

**HybridPipeline:**
- Purpose: Multi-step retrieval pipeline with windowing, filtering, reranking, and NLI.
- Examples: `backend/hybrid.py`
- Pattern: Static `build_all` that returns retriever-like outputs plus optional audit payloads.

**Settings Object:**
- Purpose: Centralized runtime configuration with env overrides.
- Examples: `backend/settings.py`
- Pattern: Pydantic BaseSettings singleton (`settings`).

**Request/Response Schemas:**
- Purpose: Validate API payloads for `/segment` and `/prebuild`.
- Examples: `backend/schemas.py`
- Pattern: Pydantic models with validators for numeric inputs.

**Blablador Client:**
- Purpose: External LLM completions and embeddings API wrapper.
- Examples: `backend/bl_client.py`
- Pattern: Simple HTTP client with convenience wrappers.

## Entry Points

**CLI Launcher:**
- Location: `application.py`
- Triggers: `python application.py ...`
- Responsibilities: Configure env vars, spawn FastAPI/Streamlit/ColBERT services.

**FastAPI Backend:**
- Location: `backend/main.py`
- Triggers: `uvicorn backend.main:app`
- Responsibilities: Request routing, retriever lifecycle, NLI processing.

**Streamlit UI:**
- Location: `frontend/ui.py`
- Triggers: `streamlit run frontend/ui.py`
- Responsibilities: Uploads, segmentation UI, result display.

**ColBERT Service:**
- Location: `colbert_server/colbert.py`
- Triggers: `uvicorn colbert_server.colbert:app`
- Responsibilities: Build/search endpoints for reranking.

## Error Handling

**Strategy:** HTTP exceptions for API failures and defensive checks around model/IO operations (`backend/main.py`, `colbert_server/colbert.py`, `backend/nli.py`).

**Patterns:**
- `HTTPException` with status codes for validation/runtime errors (`backend/main.py`, `colbert_server/colbert.py`).
- Early guards for missing data or empty indexes (`backend/retriever.py`, `backend/main.py`).
- Logging + fallback behavior for LLM parsing (`backend/utils.py`).

## Cross-Cutting Concerns

**Logging:** Structured logging setup for backend + NLI debug output (`backend/main.py`, `backend/nli.py`).
**Validation:** Pydantic models for request payloads and settings with constraints (`backend/schemas.py`, `backend/settings.py`).
**Authentication:** API key and base URL configuration for Blablador requests (`backend/settings.py`, `frontend/ui.py`, `backend/bl_client.py`).

---

*Architecture analysis: 2026-01-23*
