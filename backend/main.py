# backend/main.py

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import schemas, utils
from .nli import assess
from .retriever import build_all
from .utils import pick_best_passage

# Configure logging (so that logger.debug/info/etc. actually prints)
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s ▶ %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Blablador NLI backend")

CSV_PATH = Path(os.environ.get("CSV_PATH", "source.csv")).resolve()
SOURCE_DIR = Path(os.environ.get("SOURCE_DIR", CSV_PATH.parent / "source")).resolve()


@app.on_event("startup")
async def startup_event():
    # automatic prebuild disabled; will be triggered manually from the UI
    pass


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store in-memory indices here
results, retrievers = {}, {}


# ---------- /segment endpoint ----------
@app.post("/segment")
async def segment(req: schemas.SentencePayload):
    """
    Process a set of user-generated segments for one CSV row.
    """
    # 1) If no FAISS index exists yet, build it on the fly
    if not retrievers:
        try:
            retrievers.update(
                build_all(
                    folder=Path(req.folder),
                    embed_model=req.settings.embed_model,
                    max_sentences=req.settings.max_sentences,
                    min_score=req.settings.faiss_min_score,
                )
            )
            logging.info("Built FAISS indices on the fly (default).")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Could not build FAISS indices on the fly: {e}"
            )

    # 2) Pick the per-paper retriever (or fall back to "default")
    key = f"{req.citing_title}-{req.settings.data_dir or 'default'}"
    retr = retrievers.get(key) or retrievers.get("default")
    if retr is None:
        raise HTTPException(
            status_code=500,
            detail=f"No FAISS index found for key={key} and no default index available",
        )

    # 3) Embed all submitted “segments” locally (via sentence-transformers)
    seg_texts = [seg.claim for seg in req.segments]
    logger.debug(f"[EMBED] embedding segments: {seg_texts}")
    try:
        seg_vecs = utils.embed(seg_texts, model_name=req.settings.embed_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local embedding failed: {e}")
    logger.debug(
        f"[EMBED] returned {len(seg_vecs)} vectors; example vec[0]={seg_vecs[0] if seg_vecs else None}"
    )

    response_segments = []
    for seg, vec in zip(req.segments, seg_vecs):
        logger.debug(f"[SEGMENT]={seg.segment_id} claim={seg.claim!r}")

        # 4) FAISS search (top-k candidates)
        k_cap = req.settings.max_sentences or len(retr.chunks)
        ids, scores = retr.search([vec], k=k_cap)
        logger.debug(f"[FAISS] raw ids={ids[0]} scores={scores[0]}")

        # 5) Threshold + cap to top 25
        filtered = [
            (doc_id, score)
            for doc_id, score in zip(ids[0], scores[0])
            if score >= req.settings.faiss_min_score
        ]
        filtered.sort(key=lambda x: x[1], reverse=True)
        filtered = filtered[:25]
        logger.debug(f"[FAISS→filtered] {filtered}")

        # ——— 5.5) Rerank the FAISS candidates if requested ———
        if req.settings.reranker_model:
            # build candidate dicts
            candidates = [
                {
                    "id": doc_id,
                    "text": retr.docstore[doc_id]["text"],
                    "meta": retr.docstore[doc_id]["meta"],
                    "faiss_score": score,
                }
                for doc_id, score in filtered
            ]
            reranked = utils.rerank(
                query=seg.claim,
                candidates=candidates,
                model_name=req.settings.reranker_model,
                top_k=req.settings.reranker_top_k,
            )
            # replace filtered list (preserving original FAISS score)
            filtered = [(c["id"], c["faiss_score"]) for c in reranked]
        # ————————————————————————————————

        # 6) Assemble texts and metadata for NLI
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

        # 7) Run NLI (entailment/contradiction) locally via HF pipeline
        raw_nli = assess(seg.claim, texts, metadatas, nli_model=req.settings.nli_model)
        logger.debug(f"[NLI→raw] {raw_nli}")
        if not raw_nli:
            logger.warning(
                f"[NLI] no raw evidence returned for segment {seg.segment_id!r}"
            )

        # 8) Filter NLI results by label & threshold
        evidences = []
        for ev in raw_nli:
            label = ev.get("label")
            score = ev.get("score", 0.0)
            if label not in ("entailment", "contradiction"):
                continue
            if score < req.settings.nli_threshold:
                continue
            evidences.append(
                {
                    "text": ev.get("text", ""),
                    "score": score,
                    "label": label,
                    "section_path": ev.get("meta", {}).get("section_path"),
                    **{
                        k: v
                        for k, v in ev.items()
                        if k not in ("text", "score", "label")
                    },
                }
            )
        logger.debug(f"[NLI→filtered] {evidences}")

        # 9) If there is at least one piece of evidence, run pick_best_passage
        texts_for_llm = [e["text"] for e in evidences]
        if texts_for_llm:
            sup_id, sup_rat = pick_best_passage(
                seg.claim,
                texts_for_llm,
                "support",
                model_name=req.settings.llm_model,
                api_key=req.settings.api_key,
                base_url=req.settings.base_url,
            )
            con_id, con_rat = pick_best_passage(
                seg.claim,
                texts_for_llm,
                "contradict",
                model_name=req.settings.llm_model,
                api_key=req.settings.api_key,
                base_url=req.settings.base_url,
            )
        else:
            sup_id = con_id = None
            sup_rat = con_rat = None

        response_segments.append(
            {
                "segment_id": seg.segment_id,
                "claim": seg.claim,
                "evidence": evidences,
                "best_support_id": sup_id,
                "support_rationale": sup_rat,
                "best_contradiction_id": con_id,
                "contradiction_rationale": con_rat,
            }
        )

    return {"row_id": req.row_id, "status": "done", "segments": response_segments}


# ---------- /prebuild endpoint ----------
@app.post("/prebuild")
def prebuild(req: schemas.PrebuildRequest):
    import math

    try:
        # 1) Validate numeric fields
        if req.max_chunks is not None and (
            not isinstance(req.max_chunks, int) or req.max_chunks <= 0
        ):
            raise HTTPException(
                status_code=400, detail="max_chunks must be a positive integer"
            )
        if req.faiss_min_score is not None and (
            not isinstance(req.faiss_min_score, float)
            or math.isnan(req.faiss_min_score)
            or math.isinf(req.faiss_min_score)
        ):
            raise HTTPException(
                status_code=400,
                detail="faiss_min_score must be a real float between 0 and 1",
            )

        # 2) Rebuild global retrievers using local embedding
        global retrievers
        retrievers.clear()
        retrievers.update(
            build_all(
                folder=Path(req.folder),
                embed_model=req.embed_model,
                max_sentences=req.max_chunks,
                min_score=req.faiss_min_score,
            )
        )
        logging.info(f"FAISS index built for folder: {req.folder!r}")
        return {"status": "ok"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prebuild error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
