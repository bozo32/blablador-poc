# OS-ERIN

An integrated system for segmenting citing sentences into discrete claims and testing those claims against the content of cited sources.

+**Status – alpha / work-in-progress. Expect breaking changes and bugs.**

## Table of Contents
- [Upstream](#upstream)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Backend (FastAPI)](#backend-fastapi)
  - [Frontend (Streamlit)](#frontend-streamlit)
  - [Command-line Interface](#command-line-interface)
- [Project Structure](#project-structure)
- [Modules](#modules)
- [Contributing](#contributing)
- [Test data](#test-data)
- [License](#license)


## Features
- Parses TEI-XML of cited articles into sentences and windows, retaining metadata.
- Segments citing sentences into independent claims.
- Flexible retrieval: FAISS-based embedding search (classic) or sliding-window with filtering, reranking, and audit (hybrid).
- NLI-based filtering for entailment/contradiction using Hugging Face models.
- ColBERT integration: sub-passage/token-level reranking and attribution (hybrid pipeline).
- Web UI (Streamlit) for interactive segmentation and evidence checking.
- FastAPI backend exposing endpoints for segmentation, index building, and search.
- Supports OpenAI-compatible API for LLM-based rationale selection.
- All local, open-source models unless using external LLM for rationale.

## Requirements
- Python 3.8 or higher
- FAISS (CPU or GPU)
- pip
- (For hybrid pipeline or reranking:)  
  [ColBERT](https://github.com/stanford-futuredata/ColBERT) (install locally in editable mode)
- (For coreference/audit:)  
  `biu-nlp/f-coref` or similar HF model

## Installation
```bash
git clone https://github.com/your-org/blablador-nli-backend.git
cd blablador-nli-backend
pip install -r requirements.txt
# For hybrid pipeline/ColBERT reranking:
git clone https://github.com/stanford-futuredata/ColBERT.git ~/ColBERT
cd ~/ColBERT
pip install -e .
# Return to your project root before running.
```

## Configuration
- All runtime settings are controlled by environment variables or `.env`, and/or overridable via UI/CLI.
- ColBERT index, collection, and checkpoint can be set via `COLBERT_INDEX`, `COLBERT_COLLECTION`, and `COLBERT_CHECKPOINT` environment variables.

## Usage

### Docker Compose Quickstart

The easiest way to run the full stack (API + UI + GROBID + Postgres + MinIO) is via Docker Compose:

```bash
bash scripts/dev/up.sh
```

Key URLs:

- UI: `http://localhost:8501`
- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- GROBID: `http://localhost:8070`
- MinIO console: `http://localhost:9001`

Note: Compose already starts GROBID on port 8070. If you also run a local GROBID service, you'll see an "Address already in use" error.

Smoke test (upload + extract):

```bash
bash scripts/dev/smoke_ingest.sh fixtures/sample.pdf
```

To remove the stack (including volumes / persisted data):

```bash
bash scripts/dev/down.sh
```

1. Drag and drop your CSV and TEI XML files into the sidebar.
2. Wait for the backend to process and report available models.
3. Choose segmentation and pipeline options as needed:
    - **Classic**: Simple sentence embedding search.
    - **Hybrid**: Sliding window, FAISS, reranking (ColBERT or SPLADE), audit, NLI.
4. Edit segments for clarity.
5. Submit for evidence checking.
6. Download JSON results.

### Backend (FastAPI)
- Start with:  
  ```bash
  python application.py --api_key <key> --api_base <url>
  ```
  This will launch FastAPI backend, Streamlit frontend, and ColBERT API server (if hybrid/reranker is enabled).

### Frontend (Streamlit)
- Access via the URL printed in terminal (`localhost:8501` or similar).

### Command-line Interface
- For batch/test runs:
  ```bash
  python application.py --folder <data-folder> --api_key <key> --api_base <url>
  ```

## Project Structure
```
.
├── backend
│   ├── main.py        # FastAPI backend
│   ├── bl_client.py   # API client for completions & embeddings
│   ├── parser.py      # TEI & CSV parsing
│   ├── retriever.py   # FAISS index/search
│   ├── nli.py         # NLI classification (HF)
│   ├── schemas.py     # Pydantic models
│   ├── hybrid.py      # Hybrid pipeline (sliding windows, rerank, audit)
│   ├── utils.py       # Utilities (embedding, cleaning, CSV)
│   └── settings.py    # Config & runtime settings
├── colbert_server/colbert.py  # FastAPI ColBERT server (reranker, token attributions)
├── frontend/ui.py     # Streamlit UI
├── application.py     # CLI entrypoint (spawns backend, frontend, ColBERT server)
└── requirements.txt   # Python dependencies
```

## Modules

### `bl_client.py`
Client for external completions/embedding APIs, with batching, error logging.

### `parser.py`
TEI XML and CSV parsing, metadata extraction, sliding window generation.

### `retriever.py`
Classic FAISS embedding index/search; per-paper or global indexes.

### `nli.py`
Hugging Face NLI inference, confidence filtering for entailment/contradiction.

### `hybrid.py`
Hybrid pipeline:  
- TEI sliding windows (configurable size/stride)  
- FAISS candidate retrieval  
- Reranking (ColBERT/SPLADE)  
- NLI audit and passage-level evidence scoring  
- (Optional) Token salience/color audit and discourse/coref masking

### `colbert_server/colbert.py`
FastAPI microservice exposing `/build` and `/search` endpoints for ColBERT reranking and (if supported) per-token salience attribution.

### `ui.py`
Streamlit UI for segmentation, configuration, result review and download.

### `application.py`
CLI for unified launch, health-checks, and multiprocess startup.

## Pipelines

### Classic Pipeline
- TEI/CSV parsing → sentence embedding (HF) → FAISS index/search → NLI classification.

### Hybrid Pipeline
- TEI/CSV parsing → windowing (configurable) → FAISS index/search (τ-thresholding) → parent window snapping → reranking (ColBERT/SPLADE) → NLI → audit/visualization.

## ColBERT Integration

- Reranker (hybrid pipeline only):  
  - `/colbert_server/colbert.py` provides build/search API, using your local [ColBERT](https://github.com/stanford-futuredata/ColBERT) install.
  - Passes settings (e.g., `nbits`) via `ColBERTConfig`.
  - **Token-level salience/attribution**: If available, `include_token_scores=True` is set; falls back to `return_token_scores=True` for older ColBERT.

- **Installation required:**  
  - Clone and install ColBERT from source in editable mode.
  - Set `COLBERT_INDEX`, `COLBERT_COLLECTION`, `COLBERT_CHECKPOINT` if using custom paths/models.

## Available Settings

| Setting                        | Default/Env      | Description / Effect                                              |
|--------------------------------|------------------|-------------------------------------------------------------------|
| `PIPELINE_MODE`                | classic/hybrid   | Select pipeline ("classic": fast, FAISS only; "hybrid": advanced) |
| `EMBED_MODEL`                  | HF model path    | Embedding model for index/query                                   |
| `MAX_SENTENCES`                | 1000             | Max candidates from FAISS per segment                             |
| `FAISS_MIN_SCORE`              | 0.2              | Min similarity for candidate                                      |
| `NLI_MODEL`                    | HF model path    | Local NLI model for entailment/contradiction                      |
| `NLI_THRESHOLD`                | 0.5              | NLI entailment confidence threshold                               |
| `RERANKER_MODEL`               | ColBERT/SPLADE   | Reranking model for hybrid pipeline                               |
| `RERANKER_TOP_K`               | 10               | Top reranked windows to keep for NLI                              |
| `HYBRID_WINDOW_SIZE`           | 3                | Sentences per window                                              |
| `HYBRID_STRIDE`                | 1                | Window stride                                                     |
| `RETRIEVAL_K`                  | 1500             | Top-K FAISS to consider pre-filter                                |
| `RETRIEVAL_TAU`                | 0.8              | Fractional threshold for similarity cut-off                       |
| `RETRIEVAL_MAX`                | 200              | Max windows before rerank                                         |
| `HYBRID_AUDIT_MODE`            | True             | Enable debug outputs, color attribution, etc.                     |
| `COLBERT_INDEX`                | ./data/colbert/... | Path override for ColBERT index                                   |
| `COLBERT_COLLECTION`           | ./data/colbert/... | Path override for ColBERT collection                              |
| `COLBERT_CHECKPOINT`           | colbert-ir/colbertv2.0 | Checkpoint to use for ColBERT                             |
| `nbits`                        | 2 (small data)   | Number of bits for quantization in ColBERT index                  |
| (other ColBERTConfig kwargs)   | See ColBERT docs | See [ColBERT](https://github.com/stanford-futuredata/ColBERT)     |

**Parameter notes:**  
- For small data, set `nbits` and (if available) `n_cells` to low values to avoid overfitting/noise in centroids.
- Most settings can be controlled via UI or `.env`; audit/debugging features are always available in hybrid mode.

## Contributing
1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/XYZ`)  
3. Commit your changes (`git commit -m "Add XYZ"`)  
4. Push to your fork (`git push origin feature/XYZ`)  
5. Open a Pull Request

## Test data
There are files in `tests/dummy data` to support testing of the scripts.

## License
MIT. See `LICENSE`.
