from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os, json
from typing import Optional, Dict, Any
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from . import parser, retriever, schemas, utils
# Import assess directly for clarity and to update signature
from .nli import assess
from .parser import tei_and_csv_to_documents
from .retriever import build_all
from .utils import pick_best_passage

# Add requests and logging imports
import requests
import logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s ▶ %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


# Import embed_documents from bl_client
from .bl_client import embed_documents


# Import time for retry back-off
import time

# --------- Pydantic model for /segment request ----------
class SegmentRequest(BaseModel):
    folder: str
    row: int
    segments: List[str]
    settings: Optional[Dict[str, Any]] = None

app = FastAPI(title="Blablador NLI backend")

CSV_PATH  = Path(os.environ.get("CSV_PATH", "source.csv")).resolve()
SOURCE_DIR = Path(os.environ.get("SOURCE_DIR", CSV_PATH.parent / "source")).resolve()

@app.on_event("startup")
async def startup_event():
    # automatic prebuild disabled; will be triggered manually from the UI
    pass

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

results, retrievers = {}, {}

def _paper_key(author: str, year: int) -> str:
    return f"{author}-{year}"

# ---------- index builder ----------
def prebuild_indexes(
    csv_path: Path,
    embed_model: str = "alias-embeddings",
    max_chunks: int = 256,
    faiss_min_score: float = 0.2,
):
    df = utils.read_csv(csv_path)  # ✔ single call
    print("Prebuilding from:", csv_path, "rows:", len(df))
    required = {"Cited Author", "Cited Year", "tei_sentence", "TEI File"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing {missing}; found {list(df.columns)}")

    for (author, year), _ in df.groupby(["Cited Author", "Cited Year"]):
        key = _paper_key(author, year)
        tei_path = SOURCE_DIR / f"{author}.pdf.tei.xml"
        # Parse TEI chunks
        docs = parser.parse_chunks(tei_path)

        idx_base = Path("backend/models") / key
        idx_base.parent.mkdir(parents=True, exist_ok=True)
        retr = retriever.Retriever(
            index_path=idx_base,
            max_sentences=max_chunks,
            min_score=faiss_min_score,
            embed_model=embed_model,
            api_key=os.environ.get("API_KEY"),
            base_url=os.environ.get("BASE_URL"),
            # Pass the TEI chunk list as a lookup map for later retrieval
            docstore={chunk["id"]: chunk for chunk in docs},
        )
        retr.build(docs)
        retrievers[key] = retr
    
#
# Replace embeddings with embed_documents and fix session state references
from backend.schemas import SentencePayload, Settings

@app.post("/segment")
async def segment(req: SentencePayload):
    """
    Process a set of user-generated segments for one CSV row.
    """
    # if nobody has ever called /prebuild, do it now so we at least have a default index
    if not retrievers:
        try:
            retrievers.update(
                build_all(
                    folder=Path(req.folder),
                    embed_model=req.settings.embed_model,
                    api_key=req.settings.api_key,
                    base_url=req.settings.base_url,
                    max_sentences=req.settings.max_sentences,
                    min_score=req.settings.faiss_min_score,
                )
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Could not build FAISS indices on the fly: {e}"
            )

    # now pick the per‐paper retriever, or fall back to the global default
    key = f"{req.citing_title}-{req.settings.data_dir or 'default'}"
    retr = retrievers.get(key) or retrievers.get("default")
    if retr is None:
        raise HTTPException(
            status_code=500,
            detail=f"No FAISS index found for key={key} and no default index available"
        )

    # Embed all segments in one shot
    seg_texts = [seg.claim for seg in req.segments]
    logger.debug(f"[EMBED] embedding segments: {seg_texts}")
    seg_vecs = embed_documents(
        [{"text": t} for t in seg_texts],
        model=req.settings.embed_model,
        api_key=req.settings.api_key,
        base_url=req.settings.base_url,
    )
    logger.debug(f"[EMBED] returned {len(seg_vecs)} vectors; example vec[0]={seg_vecs[0] if seg_vecs else None}")

    response_segments = []
    for seg, vec in zip(req.segments, seg_vecs):
        logger.debug(f"[SEGMENT]={seg.segment_id} claim={seg.claim!r}")

        # 2) FAISS SEARCH
        k_cap = req.settings.max_sentences or len(retr.chunks)
        ids, scores = retr.search([vec], k=k_cap)
        logger.debug(f"[FAISS] raw ids={ids[0]} scores={scores[0]}")

        # 3) FAISS THRESHOLD + CAP
        filtered = [
            (doc_id, score)
            for doc_id, score in zip(ids[0], scores[0])
            if score >= req.settings.faiss_min_score
        ]
        # keep only top‐25 by score
        filtered.sort(key=lambda x: x[1], reverse=True)
        filtered = filtered[:25]
        logger.debug(f"[FAISS→filtered] {filtered}")

        # 4) PREPARE NLI INPUTS
        texts, metadatas = [], []
        for doc_id, faiss_score in filtered:
            chunk = retr.docstore.get(doc_id)
            if not chunk:
                logger.warning(f"[FAISS→filtered] missing chunk id={doc_id}")
                continue
            texts.append(chunk["text"])
            m = dict(chunk.get("meta", {}))
            m["faiss_score"] = faiss_score
            metadatas.append(m)
        logger.debug(f"[NLI] inputs texts={len(texts)} passages")

        # 5) NLI (we keep only entailment/contradiction above threshold)
        raw_nli = assess(
            seg.claim,
            texts,
            metadatas,
            nli_model=req.settings.nli_model
        )
        logger.debug(f"[NLI→raw] {raw_nli}")
        if not raw_nli:
            logger.warning(f"[NLI] no raw evidence returned for segment {seg.segment_id!r}")
        evidences = []
        for ev in raw_nli:
            label = ev.get("label")
            score = ev.get("score", 0.0)
            if label not in ("entailment", "contradiction"):
                continue
            if score < req.settings.nli_threshold:
                continue
            evidences.append({
                "text":  ev.get("text", ""),
                "score": score,
                "label": label,
                **{k: v for k, v in ev.items() if k not in ("text", "score", "label")},
            })
        logger.debug(f"[NLI→filtered] {evidences}")

        # 6) Dual LLM picks: support + contradiction, both over ALL evidences
        texts = [e["text"] for e in evidences]
        if texts:
            sup_id, sup_rat = pick_best_passage(
                seg.claim, texts, "support",
                model_name=req.settings.llm_model,
                api_key=req.settings.api_key,
                base_url=req.settings.base_url,
            )
            con_id, con_rat = pick_best_passage(
                seg.claim, texts, "contradict",
                model_name=req.settings.llm_model,
                api_key=req.settings.api_key,
                base_url=req.settings.base_url,
            )
        else:
            sup_id = con_id = None
            sup_rat = con_rat = None

        response_segments.append({
            "segment_id":              seg.segment_id,
            "claim":                   seg.claim,
            "evidence":                evidences,
            "best_support_id":         sup_id,
            "support_rationale":       sup_rat,
            "best_contradiction_id":   con_id,
            "contradiction_rationale": con_rat,
        })

    return {"row_id": req.row_id, "status": "done", "segments": response_segments}

@app.get("/progress/{row_id}")
async def progress(row_id: int):
    # Check if any segment for this row_id is present in results
    str_row_id = str(row_id)
    for paper in results.values():
        for cp in paper.get("citing_papers", {}).values():
            for sent in cp.get("sentences", []):
                for seg in sent.get("segments", []):
                    if str(seg.get("segment_id", "")).startswith(str_row_id):
                        return {"row_id": row_id, "status": "done"}
    return {"row_id": row_id, "status": "pending"}

# ---------- manual prebuild endpoint ----------
class PrebuildRequest(BaseModel):
    folder: str
    embed_model: str = "alias-embeddings"
    max_chunks: int = 256
    faiss_min_score: float = 0.2
    api_key: str
    base_url: str

@app.post("/prebuild")
async def prebuild(req: PrebuildRequest):
    try:
        global retrievers
        retrievers.clear()
        retrievers.update(retriever.build_all(
            folder=Path(req.folder),
            embed_model=req.embed_model,
            api_key=req.api_key,
            base_url=req.base_url,
            max_sentences=req.max_chunks,
            min_score=req.faiss_min_score
        ))
        # Log the successful build
        logging.info(f"FAISS index built for folder: {req.folder!r}")
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Prebuild error: {e}")
        raise HTTPException(status_code=500, detail=str(e))