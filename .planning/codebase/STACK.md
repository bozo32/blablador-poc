# Technology Stack

**Analysis Date:** 2026-01-23

## Languages

**Primary:**
- Python 3.10 - `backend/`, `frontend/ui.py`, `application.py`, `colbert_server/colbert.py`

**Secondary:**
- Shell (bash) - `clean.sh`, `cleanup.sh`

## Runtime

**Environment:**
- Conda environment - `environment.yml`
- Conda environment (ColBERT server) - `colbert_server/environment.yml`

**Package Manager:**
- conda (with pip) - `environment.yml`, `colbert_server/environment.yml`
- Lockfile: missing

## Frameworks

**Core:**
- FastAPI - `backend/main.py`, `colbert_server/colbert.py`
- Streamlit - `frontend/ui.py`

**Testing:**
- pytest (implicit) - `tests/test_retriever.py`

**Build/Dev:**
- uvicorn - `application.py`, `backend/main.py`, `colbert_server/colbert.py`
- Streamlit dev server - `application.py`, `.streamlit/config.toml`

## Key Dependencies

**Critical:**
- faiss-cpu - `backend/retriever.py`, `backend/hybrid.py`
- transformers - `backend/nli.py`, `application.py`
- sentence-transformers - `backend/utils.py`
- torch - `environment.yml`, `colbert_server/environment.yml`
- pydantic-settings - `backend/settings.py`

**Infrastructure:**
- pandas - `backend/utils.py`, `frontend/ui.py`
- requests - `backend/bl_client.py`, `backend/utils.py`, `frontend/ui.py`
- fastcoref - `backend/hybrid.py`
- rank-bm25 - `backend/utils.py`, `backend/hybrid.py`
- spacy - `backend/utils.py`, `backend/hybrid.py`
- colbert-ai - `colbert_server/colbert.py`
- lxml, pymupdf, pypdf, pytesseract - `environment.yml`

## Configuration

**Environment:**
- Pydantic settings load `.env` via `backend/settings.py`
- Runtime env vars used throughout `backend/settings.py`, `application.py`, `frontend/ui.py`

**Build:**
- Conda env specs: `environment.yml`, `colbert_server/environment.yml`
- Streamlit config: `.streamlit/config.toml`

## Platform Requirements

**Development:**
- Python 3.10 conda env with FAISS, torch, transformers - `environment.yml`
- Optional ColBERT server env with PyTorch 2.1+ - `colbert_server/environment.yml`

**Production:**
- Local process execution via `application.py` (no deployment config detected)

---

*Stack analysis: 2026-01-23*
