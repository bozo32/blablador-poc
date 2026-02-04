# backend/utils.py

import requests
import json
import logging
import re
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union  # new import

import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

from backend.bl_client import BlabladorClient

from functools import lru_cache
import io
import hashlib


HF_INFERENCE_API_BASE = "https://api-inference.huggingface.co/models"


def hf_inference_post(
    model_id: str,
    *,
    token: str,
    payload: dict[str, Any],
    timeout: int = 30,
) -> Any:
    """POST a JSON payload to the Hugging Face Inference API.

    Raises RuntimeError on any non-200 response or malformed JSON.
    """
    model = (model_id or "").strip()
    if not model:
        raise RuntimeError("HF inference model_id is empty")
    url = f"{HF_INFERENCE_API_BASE}/{model}"
    headers = {}
    token_value = (token or "").strip()
    if token_value:
        headers["Authorization"] = f"Bearer {token_value}"
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - callers treat as fallback
        raise RuntimeError(f"HF inference request failed: {exc}") from exc
    if resp.status_code != 200:
        detail = (resp.text or "").strip()
        if len(detail) > 400:
            detail = detail[:400] + "…"
        raise RuntimeError(f"HF inference HTTP {resp.status_code}: {detail}")
    try:
        return resp.json()
    except Exception as exc:  # noqa: BLE001 - callers treat as fallback
        raise RuntimeError(f"HF inference returned invalid JSON: {exc}") from exc


# testing for parallelism support
def set_sane_threads():
    """Set a sane number of threads for heavy compute libraries.

    Use number of physical cores if possible.
    """
    import os

    try:
        import psutil

        num_threads = psutil.cpu_count(logical=False) or os.cpu_count()
    except ImportError:
        num_threads = os.cpu_count()

    # Set threading env vars before any imports of numpy/torch/transformers
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    try:
        import torch

        torch.set_num_threads(num_threads)
    except ImportError:
        pass
    return num_threads


def clean_text(text: str) -> str:
    # remove TEI tags remnants and citation markers
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\*\*\*#.*?\*\*\*", "", text)
    text = re.sub(r"\[\d+.*?\]", "", text)
    return text.strip()


def read_csv(path: Union[Path, str]) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame.

    Ensure correct path resolution and handle nested quoting.
    """
    import csv

    path = Path(path)
    # First, try pandas’ parser in a slightly more forgiving mode:
    try:
        df = pd.read_csv(
            path,
            dtype=str,
            quotechar='"',
            skipinitialspace=True,
            on_bad_lines="warn",
            engine="python",
        )
        # if it came back as a single column, likely the file is globally quoted
        if len(df.columns) > 1:
            return df
    except Exception:
        # fall back to the csv module below
        pass

    # Fallback: use the stdlib csv.reader to correctly handle nested quoting
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return pd.DataFrame()
    header, *data = rows
    return pd.DataFrame(data, columns=header)


# ----------------------------------------
# Local‐HF sentence‐transformers cache & loading
# ----------------------------------------

# Use a dedicated cache folder for local HF models
MODEL_CACHE_DIR = Path.home() / ".cache/hf_sentence_models"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# List locally cached models (those with a config.json file)
def list_local_models() -> list[str]:
    return sorted(
        [
            d.name
            for d in MODEL_CACHE_DIR.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        ]
    )


# Cache loaded models in memory for performance
_loaded_models: dict[str, SentenceTransformer] = {}


def get_model(model_name: str) -> SentenceTransformer:
    """Load a SentenceTransformer from local cache if present.

    Download to MODEL_CACHE_DIR when missing.
    """
    if model_name in _loaded_models:
        return _loaded_models[model_name]
    try:
        local_path = MODEL_CACHE_DIR / model_name
        if local_path.exists():
            model = SentenceTransformer(str(local_path))
        else:
            model = SentenceTransformer(model_name, cache_folder=str(MODEL_CACHE_DIR))
        _loaded_models[model_name] = model
        return model
    except Exception as e:
        raise RuntimeError(f"Could not load embedding model '{model_name}': {e}")


def embed(
    texts: list[str], model_name: str = "all-MiniLM-L6-v2", mode: str = None
) -> list[list[float]]:
    """Embed texts using a local Hugging Face model (as chosen by the user)."""
    # Models that require prefix
    models_with_prefix = ["e5", "infloat", "bge"]  # add/adjust as needed
    lower_model = model_name.lower()
    needs_prefix = any(prefix in lower_model for prefix in models_with_prefix)
    if mode and needs_prefix:
        prefix = f"{mode}: "
        texts = [f"{prefix}{t}" for t in texts]
    model = get_model(model_name)
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()


def pick_best_passage(
    claim: str,
    passages: List[str],
    mode: Literal["support", "contradict"],
    model_name: str,
    api_key: str,
    base_url: str,
) -> Tuple[Optional[int], Optional[str]]:
    """Pick the single best passage by index and produce a one-line rationale.

    Returns (best_index, rationale) or (None, None) on failure.
    """
    # 1) Build a minimal, unambiguous prompt
    verb = "SUPPORTS" if mode == "support" else "CONTRADICTS"
    instruction = (
        f"Claim: {claim}\n"
        f"Passages:\n" + "\n".join(f"{i}: {p}" for i, p in enumerate(passages)) + "\n\n"
        f"Which single passage index {verb} the claim?  \n"
        "Respond with exactly this JSON (no extra text):\n"
        '{"best_id": <index>, "rationale": "<very brief explanation>"}\n'
        "Now your answer:\n"
    )

    client = BlabladorClient(api_key=api_key, base_url=base_url)
    resp_text = client.completion(
        instruction,
        model=model_name,
        temperature=0.0,
        max_tokens=150,
    )

    # 2) Log raw output for debugging—even if it’s empty
    logging.debug(f"pick_best_passage raw response: {resp_text!r}")

    # 3) Try to extract the first {...} block
    if not resp_text or not resp_text.strip():
        logging.error("Empty response from LLM in pick_best_passage")
        return None, None

    # Regex to find {...} (non-greedy)
    match = re.search(r"\{.*?\}", resp_text, flags=re.DOTALL)
    if match:
        candidate = match.group()
        try:
            out = json.loads(candidate)
            bid = out.get("best_id")
            rat = out.get("rationale")
            if isinstance(bid, int) and isinstance(rat, str):
                return bid, rat
            else:
                logging.error(f"Parsed JSON missing expected types: {candidate}")
        except Exception as e:
            logging.error(
                "Failed to json.loads() in pick_best_passage: %s\nCandidate: %s",
                e,
                candidate,
            )

    else:
        logging.error(f"No JSON object found in pick_best_passage output:\n{resp_text}")

    # 4) Fallback: pick the highest-score passage automatically
    logging.warning(
        "Falling back to highest-FAISS-score passage (index 0) without rationale"
    )
    return 0, "no rationale"


def make_retriever_key(row_id: str, segment_id: Optional[str] = None) -> str:
    """Build a stable retriever key.

    Return row_id::segment_id when segment_id is provided; otherwise row_id.
    """
    return f"{row_id}::{segment_id}" if segment_id else row_id


@lru_cache(maxsize=4)  # adjust as needed
def get_cross_encoder(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank(
    query: str, candidates: list[dict], model_name: str, top_k: int
) -> list[dict]:
    """Rerank candidates with a cross-encoder.

    Candidates are dicts with keys 'text' plus metadata. Returns entries with
    'rerank_score', sorted and sliced to top_k.
    """
    model = get_cross_encoder(model_name)
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs).tolist()
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_k]


def filter_and_snap(
    ids: list[str],
    scores: list[float],
    windows: list[dict],
    tau: float,
    cap: int,
) -> list[dict]:
    """Filter and snap FAISS windows.

    Keep hits within tau of the best score and snap to parent3_id, keeping the
    highest score. Returns windows with added faiss_score.
    """
    # 1. pair & sort
    hits = sorted(
        [(i, s) for i, s in zip(ids, scores) if i != -1],
        key=lambda x: x[1],
        reverse=True,
    )
    if not hits:
        return []
    best = hits[0][1]
    kept = [(i, s) for i, s in hits if s >= best * tau][:cap]

    # 2. snap → dedupe
    snap: dict[str, float] = {}
    for idx_i, score in kept:
        w = windows[idx_i]
        p3 = w["meta"]["parent3"]
        snap[p3] = max(snap.get(p3, -1), score)

    # 3. materialize windows
    out = []
    for pid, sc in sorted(snap.items(), key=lambda x: x[1], reverse=True):
        w = next(w for w in windows if w["window_id"] == pid)
        w2 = w.copy()
        w2["faiss_score"] = sc
        out.append(w2)
    return out


def sbert_rerank(windows: list[dict], claim: str, model_name: str) -> list[dict]:
    """Attach '_sbert_score' to each window and return them sorted descending."""
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer(model_name)
    q_emb = model.encode([claim], convert_to_tensor=True)
    p_emb = model.encode([w["text"] for w in windows], convert_to_tensor=True)
    sims = util.pytorch_cos_sim(q_emb, p_emb)[0].cpu().tolist()
    for w, s in zip(windows, sims):
        w["_sbert_score"] = s
    return sorted(windows, key=lambda w: w["_sbert_score"], reverse=True)


def bm25_rerank(windows: list[dict], claim: str) -> list[dict]:
    """Attach '_bm25_score' to each window and return them sorted descending."""
    from rank_bm25 import BM25Okapi
    import spacy

    nlp = spacy.blank("en")
    tok_corpus = [[tok.text for tok in nlp(w["text"])] for w in windows]
    bm = BM25Okapi(tok_corpus)
    tok_query = [tok.text for tok in nlp(claim)]
    scores = bm.get_scores(tok_query)
    for w, s in zip(windows, scores):
        w["_bm25_score"] = s
    return sorted(windows, key=lambda w: w["_bm25_score"], reverse=True)


# ---- ColBERT wrappers ----------------------------------------------


# --------------------------------------------------------------------
#  ColBERT collection → /build mini‑cache
#  We hash the TSV payload; each unique (hash, api_url) pair is sent
#  to /build only once per interpreter session.
# --------------------------------------------------------------------
@lru_cache(maxsize=32)
def _ensure_index(tsv_sha1: str, api_url: str, tsv_bytes: bytes) -> None:
    r = requests.post(
        f"{api_url}/build",
        files={"tsv": ("collection.tsv", io.BytesIO(tsv_bytes), "text/tsv")},
        timeout=600,
    )
    r.raise_for_status()


def colbert_api_rerank(
    query: str, windows: list[dict], api_url: str, k: int = 20
) -> list[dict]:
    # 1) Build collection.tsv in-memory
    tsv_lines = []
    for i, w in enumerate(windows):
        clean_text = w["text"].replace("\n", " ")
        tsv_lines.append(f"{i}\t{clean_text}")

    # 2) POST /build  (cached on SHA‑1 of TSV)
    tsv_bytes = "\n".join(tsv_lines).encode("utf-8")
    sha1 = hashlib.sha1(tsv_bytes).hexdigest()
    _ensure_index(sha1, api_url, tsv_bytes)

    # 3) /search
    r = requests.post(f"{api_url}/search", json={"query": query, "k": k})
    r.raise_for_status()
    hits = r.json()  # [{text, score, token_scores}, …]

    out = []
    for h in hits:
        idx = next(i for i, w in enumerate(windows) if w["text"] == h["text"])
        w = windows[idx]
        w["colbert_score"] = h["score"]
        w["token_scores"] = h.get("token_scores")
        out.append(w)
    return out


__all__ = [
    "clean_text",
    "read_csv",
    "embed",
    "list_local_models",
    "get_model",
    "pick_best_passage",
    "rerank",
    "filter_and_snap",
    "sbert_rerank",
    "bm25_rerank",
    # "_index_id",  # removed if not needed
    # "get_searcher",  # removed if not needed
]
