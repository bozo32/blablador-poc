# colbert_server/colbert.py
# start with uvicorn colbert_server.colbert:app --host 0.0.0.0 --port 7001 --reload

import os
import sys

os.environ["FAISS_DISABLE_OPENMP"] = "1"
sys.path.insert(0, os.path.expanduser("~/ColBERT"))

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import traceback
import aiofiles
from colbert.infra.config import ColBERTConfig

searcher = None  # Global variable to hold the Searcher instance


# Set ColBERT index, collection, checkpoint paths from env or defaults
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_COLBERT_ROOT = PROJECT_ROOT / "data" / "colbert"
DEFAULT_INDEX_PATH = DEFAULT_COLBERT_ROOT / "indexes" / "default"
DEFAULT_COLLECTION_PATH = DEFAULT_COLBERT_ROOT / "collections" / "default.tsv"
DEFAULT_CHECKPOINT = "colbert-ir/colbertv2.0"

INDEX_PATH = Path(os.environ.get("COLBERT_INDEX", DEFAULT_INDEX_PATH))
COLLECTION_PATH = Path(os.environ.get("COLBERT_COLLECTION", DEFAULT_COLLECTION_PATH))
CHECKPOINT = os.environ.get("COLBERT_CHECKPOINT", DEFAULT_CHECKPOINT)

app = FastAPI(title="ColBERT API Server")


class QueryInput(BaseModel):
    query: str
    k: int = 5


class SearchResult(BaseModel):
    text: str
    score: float
    token_scores: Optional[List[float]] = None


@app.post("/search", response_model=List[SearchResult])
def search(input: QueryInput):
    global searcher
    if searcher is None:
        raise HTTPException(
            status_code=500, detail="Index not loaded. POST to /build first."
        )
    # Ask ColBERT to return per‑token salience if the build supports it.
    # Newer versions use `include_token_scores`; older ones fall back to
    # `return_token_scores`.
    try:
        results = searcher.search(input.query, k=input.k, include_token_scores=True)
    except TypeError:
        # Older ColBERT signature
        results = searcher.search(input.query, k=input.k, return_token_scores=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Searcher failed: {e}")
    response = []
    for res in results:
        if isinstance(res, dict):
            text = res.get("text", "")
            score = res.get("score", 0.0)
            token_scores = res.get("token_scores", None)
        else:
            text = getattr(res, "text", "")
            score = getattr(res, "score", 0.0)
            token_scores = getattr(res, "token_scores", None)
        response.append({"text": text, "score": score, "token_scores": token_scores})
    return response


@app.get("/")
def root():
    return {"message": "ColBERT API is running."}


@app.post("/build")
async def build_index(tsv: UploadFile = File(...)):
    # ... other imports ...
    from colbert import Indexer, Searcher

    # Your preferred root
    colbert_data_root = DEFAULT_COLBERT_ROOT  # e.g., "data/colbert" in project

    os.makedirs(colbert_data_root / "collections", exist_ok=True)
    os.makedirs(colbert_data_root / "indexes", exist_ok=True)

    # Save uploaded TSV as usual (to colbert_data_root/collections/default.tsv)
    async with aiofiles.open(COLLECTION_PATH, "wb") as f:
        while chunk := await tsv.read(8192):
            await f.write(chunk)

    # ---- This is the key line: set working dir ----
    os.chdir(str(colbert_data_root))
    print("ColBERT working dir:", os.getcwd())
    # Now everything below is relative to colbert_data_root

    config = ColBERTConfig(nbits=2, n_cells=16)  # appropriate for small data

    try:
        indexer = Indexer(checkpoint=CHECKPOINT, config=config)

        indexer.index(
            name="default",
            collection="collections/default.tsv",
            overwrite=True,
        )
    except Exception as e:
        print("EXCEPTION DURING INDEX BUILD:\n", traceback.format_exc())
        raise HTTPException(500, f"Index build failed: {e}")

    # Instantiate the Searcher—use the correct relative paths
    global searcher
    try:
        searcher = Searcher(
            index="default",
            collection="collections/default.tsv",
            checkpoint=CHECKPOINT,
            config=config,
        )
    except Exception as e:
        print("EXCEPTION DURING SEARCHER INIT:\n", traceback.format_exc())
        raise HTTPException(500, f"Searcher init failed after build: {e}")

    return {"ok": True}
