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

# Add requests and logging imports
import requests
import logging
logger = logging.getLogger(__name__)

# Import embed_documents from bl_client
from .bl_client import embed_documents

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
    df = utils.read_csv(csv_path)                 # ✔ single call
    print("Prebuilding from:", csv_path, "rows:", len(df))
    required = {"Cited Author", "Cited Year", "tei_sentence", "TEI File"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing {missing}; found {list(df.columns)}")

    for (author, year), _ in df.groupby(["Cited Author", "Cited Year"]):
        key      = _paper_key(author, year)
        tei_path = SOURCE_DIR / f"{author}.pdf.tei.xml"
        # 1) TEI chunks
        chunks   = parser.tei_to_chunks(tei_path)

        idx_base = Path("backend/models") / key
        idx_base.parent.mkdir(parents=True, exist_ok=True)
        r = retriever.Retriever(
            idx_base,
            embed_model=embed_model,
            max_chunks=max_chunks,
            min_score=faiss_min_score,
        )
        # Load existing index if the .faiss file exists, otherwise build it
        if r.index_path.exists():
            r.load()
            # Re-attach chunks list so queries can map back to text
            r.chunks = chunks
        else:
            r.build(chunks)
            r.chunks = chunks
        retrievers[key] = r
    
# ---------- per-sentence worker ----------
def _process_sentence(row_id: int, segments: list[str], settings: Optional[Dict[str, Any]] = None):
    # establish models (defaults if not overridden)
    nli_model = settings.get("nli_model", None) if settings else None
    llm_model = settings.get("llm_model", None) if settings else None
    # if UI supplied a data_dir, use that to locate the CSV; otherwise fall back to CSV_PATH
    if settings and settings.get("data_dir"):
        csv_dir = Path(settings["data_dir"])
        csv_files = list(csv_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in {csv_dir}")
        csv_path = csv_files[0]
    else:
        csv_path = CSV_PATH
    df = utils.read_csv(csv_path)
    row = df.iloc[row_id]
    key = _paper_key(row["Cited Author"], row["Cited Year"])
    r = retrievers.get(key) or retrievers.get("default")
    if r is None:
        raise KeyError(f"Retriever for {key} not built")

    sent_result = {"text": row["tei_sentence"], "segments": []}
    for i, claim in enumerate(segments):
        # retrieve more candidates and bias toward prose sentences
        candidates = r.query(claim, k=12)
        # take up to 6 sentence chunks, backfill with others if needed
        sentence_chunks = [c for c in candidates if c["meta"].get("type") == "sentence"]
        if len(sentence_chunks) >= 6:
            chosen = sentence_chunks[:6]
        else:
            chosen = sentence_chunks + candidates[: 6 - len(sentence_chunks)]
        passages = [c["text"] for c in chosen]
        metadatas = [c["meta"] for c in chosen]
        print(f"\n\n🔎 Checking evidence for claim: {claim}")
        for j, p in enumerate(passages):
            print(f" Passage {j} ({len(p)} chars): {p[:200]!r} …")
        ass = assess(
            claim, passages, metadatas,
            nli_model=nli_model,
            llm_model=llm_model
        )
        # normalize NLI output into a list of evidence dicts
        if isinstance(ass, dict):
            evs = ass.get("evidence") or [ass]
        elif isinstance(ass, list):
            evs = ass
        else:
            evs = []
        evidence = []
        for idx, ev in enumerate(evs):
            # ensure ev is a dict before annotating
            if not isinstance(ev, dict):
                continue
            ev["source_id"] = chosen[idx]["meta"].get("id", "")
            ev["source_type"] = chosen[idx]["meta"].get("type", "")
            evidence.append(ev)
        sent_result["segments"].append({
            "segment_id": f"{row_id}{chr(97+i)}",
            "claim": claim,
            "quoted_evidence": evidence
        })

        # — now call LLM for final ranking/justification —
        # collect only the “yes”/“partial” snippets
        top_passages = [ev["quote"] for ev in evidence if ev["assessment"] in ("yes","partial")]
        if top_passages:
            llm_prompt = {
                "claim": claim,
                "evidence": top_passages
            }
            llm_model = settings.get("llm_model") if settings else None
            justification = utils.call_llm_justification(llm_prompt, model_name=llm_model)
            best_id = justification.get("best_id") if isinstance(justification, dict) else None
            rationale = justification.get("rationale") if isinstance(justification, dict) else justification
            seg = sent_result["segments"][-1]
            seg["best_evidence_id"] = best_id
            seg["llm_rationale"] = rationale

    # --- nest into results structure ---
    citing_id    = row["TEI File"]
    citing_title = citing_id.split("-DOI")[0]
    results.setdefault(key, {
        "title": "",
        "id": key,
        "doi": row["Cited DOI"],
        "citing_papers": {}
    })
    cpapers = results[key]["citing_papers"]
    cpapers.setdefault(citing_title, {
        "title": citing_title,
        "id": citing_id,
        "sentences": []
    })
    cpapers[citing_title]["sentences"].append(sent_result)

    with open("output.json", "w") as fh:
        json.dump(results, fh, indent=2)

# Replace embeddings with embed_documents and fix session state references
@app.post("/segment")
async def segment(req: SegmentRequest):
    # Look up the right FAISS retriever
    key = f"{req.citing_title}-{req.settings.data_dir or 'default'}"
    retr = retrievers.get(key) or retrievers["default"]

    # Embed all segments in one shot
    seg_texts = [seg.claim for seg in req.segments]
    seg_vecs  = embed_documents(
        [{"text": text} for text in seg_texts],
        model=req.settings.embed_model,
        api_key=req.settings.api_key,
        base_url=req.settings.base_url
    )

    response_segments = []
    for seg, vec in zip(req.segments, seg_vecs):
        # retrieve top-k 
        ids, scores = retr.search([vec], k=5)
        evidences = []
        for doc_id, score in zip(ids[0], scores[0]):
            # get the original TEI sentence text & meta
            matched = retr.docstore.get_document(doc_id)
            evidences.append({
                "text": matched.text,
                "score": score,
                **matched.meta
            })
        response_segments.append({
            "segment_id": seg.segment_id,
            "claim":      seg.claim,
            "evidence":   evidences
        })

    return {
        "row": req.row, 
        "status": "done",
        "segments": response_segments
    }

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
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Prebuild error: {e}")
        raise HTTPException(status_code=500, detail=str(e))