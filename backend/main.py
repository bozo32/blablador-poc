# backend/main.py

import os
import shutil

# Transformers v5 removes TRANSFORMERS_CACHE; use HF_HOME.
os.environ.setdefault("HF_HOME", str(os.path.expanduser("~/.cache/huggingface")))

# Set threading env vars *before* numpy/torch
from backend import utils

utils.set_sane_threads()
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from backend import (
    background_state,
    attachment_pipeline,
    attachment_store,
    attachment_spans,
    citation_context,
    citation_graph,
    extraction,
    evidence_selection_store,
    grobid_client,
    judgment_store,
    schemas,
    tei_body,
    utils,
)
from backend.nli import assess
from backend.evidence_matching.service import evidence_service
from pydantic import BaseModel, ValidationError

try:
    from transformers import AdamW  # noqa: F401
except ImportError:
    import torch
    import transformers

    transformers.AdamW = torch.optim.AdamW

# New imports after other backend imports
from backend.pipeline_registry import get_pipeline
from backend.reference_resolver import apply_resolution_selection, resolve_references
from backend.settings import settings as app_settings  # global default settings
from backend.ingestion_store import (
    create_ingested_document,
    get_document_source_path,
    get_ingested_document,
    get_tei_xml,
    list_ingested_documents,
    store_extraction,
    store_resolution,
    update_ingested_document,
)
from backend.claim_store import claim_store
from backend.reference_retrieval import build_retrieval_dossier
from backend.graph_store import GraphStore
from backend import project_io

# Configure logging.
# Default to INFO to avoid extremely noisy dependency logs (urllib3/HF).
_log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper().strip()
_log_level = getattr(logging, _log_level_name, logging.INFO)
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s ▶ %(message)s",
    level=_log_level,
)
for _logger_name in (
    "urllib3",
    "huggingface_hub",
    "transformers",
    "sentence_transformers",
):
    logging.getLogger(_logger_name).setLevel(max(logging.WARNING, _log_level))
logger = logging.getLogger(__name__)


graph_store = GraphStore(app_settings.GRAPH_DB_PATH)

# Setup pipeline via registry and settings
build_all = get_pipeline(app_settings)
print(f"Pipeline mode: {app_settings.PIPELINE_MODE}")  # Optional: log on startup

app = FastAPI(title="Blablador NLI backend")


CSV_PATH = Path(os.environ.get("CSV_PATH", "source.csv")).resolve()
SOURCE_DIR = Path(os.environ.get("SOURCE_DIR", CSV_PATH.parent / "source")).resolve()


@app.on_event("startup")
async def startup_event():
    if background_state.get_state().get("paused"):
        logger.info("Background work is paused; skipping resumable attachment startup")
        return
    pending = attachment_store.list_resumable()
    if pending:
        logger.info("Resuming %s attachment(s) from previous session", len(pending))
    for record in pending:
        attachment_pipeline.enqueue_processing(record["id"])


class BackgroundPauseRequest(BaseModel):
    paused: bool
    reason: Optional[str] = None


@app.get("/background/state")
def get_background_state():
    return background_state.get_state()


@app.post("/background/pause")
def set_background_pause(payload: BackgroundPauseRequest):
    return background_state.set_paused(payload.paused, reason=payload.reason)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store in-memory indices here
results, retrievers = {}, {}


def _is_pdf_upload(file: UploadFile) -> bool:
    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()
    return filename.endswith(".pdf") and content_type == "application/pdf"


def _serialize_attachment(record: Optional[dict]) -> schemas.AttachmentStatus:
    if record is None:
        raise HTTPException(status_code=404, detail="Attachment not found")
    return schemas.AttachmentStatus(**record)


@app.post("/ingest", response_model=schemas.IngestUploadResponse)
async def ingest_document(file: UploadFile = File(...)):
    if not _is_pdf_upload(file):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

    file_bytes = await file.read()
    metadata = create_ingested_document(file_bytes, file.filename or "document.pdf")
    try:
        graph_store.index_ingest_upload(metadata)
    except Exception:
        logger.exception("Graph index failed for ingest upload")
    return {"document": metadata}


@app.get("/ingest", response_model=schemas.IngestListResponse)
def list_ingest_documents():
    documents = list_ingested_documents()
    return {"documents": documents}


@app.get("/ingest/{doc_id}", response_model=schemas.IngestedDocument)
def get_ingest_document(doc_id: str):
    document = get_ingested_document(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@app.get("/ledger", response_model=schemas.LedgerResponse)
def get_document_ledger():
    rows = graph_store.ledger_rows()
    options = graph_store.ledger_options()
    return {"rows": rows, "options": options}


@app.get("/project", response_model=schemas.ProjectMeta)
def get_project_meta():
    return project_io.read_project_meta()


class ProjectMetaUpdate(BaseModel):
    name: Optional[str] = None
    reviewers: Optional[list[str]] = None
    active_reviewer_uid: Optional[str] = None
    compare_reviewer_a: Optional[str] = None
    compare_reviewer_b: Optional[str] = None
    graph_settings: Optional[dict[str, Any]] = None


@app.put("/project", response_model=schemas.ProjectMeta)
def put_project_meta(payload: ProjectMetaUpdate):
    patch = payload.model_dump(exclude_unset=True)
    try:
        return project_io.write_project_meta_update(patch)
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/project/export")
def export_project():
    payload = project_io.export_project_zip()
    meta = project_io.read_project_meta()
    name = (meta.get("name") or "project").strip() or "project"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)
    filename = f"{safe}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=payload, media_type="application/zip", headers=headers)


@app.post("/project/import", response_model=schemas.ProjectImportResponse)
async def import_project(file: UploadFile = File(...), overwrite: bool = False):
    blob = await file.read()
    try:
        result = project_io.import_project_zip(blob, overwrite=bool(overwrite))
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result


@app.post(
    "/claims/{claim_id}/auto-place",
    response_model=schemas.AutoPlaceResponse,
)
def auto_place_claim_source(
    claim_id: str,
    payload: schemas.AutoPlaceRequest,
    background_tasks: BackgroundTasks,
):
    doc_id = str(payload.doc_id or "").strip()
    target_id = (payload.target_id or "").strip() or None
    citation_index = payload.citation_index
    if not doc_id or not target_id:
        raise HTTPException(status_code=422, detail="doc_id and target_id are required")

    # If we already have an attachment for this claim+target, reuse it.
    for rec in attachment_store.list_attachments(
        claim_id=claim_id, archived=None, public=False
    ):
        if str(rec.get("doc_id") or "") != doc_id:
            continue
        if str(rec.get("target_id") or "") != str(target_id):
            continue
        public_record = attachment_store.public_status(rec["id"]) or {}
        return {"attachment": _serialize_attachment(public_record), "reused": True}

    cited_ingest_id = graph_store.resolve_reference_to_ingest_id(
        citing_doc_id=doc_id,
        reference_id=str(target_id),
    )
    if not cited_ingest_id:
        raise HTTPException(
            status_code=404,
            detail=(
                "No uploaded cited PDF found for this reference yet. "
                "Upload the cited PDF and run extraction/resolution "
                "to merge it into the tree."
            ),
        )

    try:
        source_path = get_document_source_path(cited_ingest_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    cited_meta = get_ingested_document(cited_ingest_id) or {}
    filename = cited_meta.get("filename") or source_path.name

    # Attempt to reuse parsed artifacts from any prior attachment for this cited doc.
    donor = None
    for rec in attachment_store.list_attachments(archived=None, public=False):
        if str(rec.get("source_ingest_id") or "") != str(cited_ingest_id):
            continue
        if attachment_store.is_ready(rec):
            donor = rec
            break

    try:
        record = attachment_store.create_attachment(
            claim_id=claim_id,
            doc_id=doc_id,
            local_path=source_path,
            filename=filename,
            reference_hint={"reference_id": target_id},
            citation_index=citation_index,
            target_id=target_id,
            source_ingest_id=cited_ingest_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # If a donor exists, clone artifacts and mark ready (skip reprocessing).
    if donor and donor.get("artifacts"):
        donor_artifacts = donor.get("artifacts") or {}
        dest_dir = Path(app_settings.ATTACHMENT_DIR) / str(record["id"])
        dest_dir.mkdir(parents=True, exist_ok=True)
        copied = {}
        for key in ("tei_xml", "tei_json", "sentences"):
            src = donor_artifacts.get(key)
            if not src:
                continue
            src_path = Path(str(src))
            if not src_path.exists():
                continue
            dst_path = dest_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            copied[key] = str(dst_path)
        if copied:
            attachment_store.mark_matched(str(record["id"]), artifacts=copied)
    else:
        if not background_state.get_state().get("paused"):
            if background_tasks is not None:
                background_tasks.add_task(
                    attachment_pipeline.process_attachment, record["id"]
                )
            else:
                attachment_pipeline.enqueue_processing(record["id"])

    public_record = attachment_store.public_status(record["id"]) or {}
    return {"attachment": _serialize_attachment(public_record), "reused": False}


@app.patch("/ledger/{doc_num}/outgoing", response_model=schemas.LedgerResponse)
def update_ledger_outgoing(doc_num: int, payload: schemas.LedgerLinksUpdateRequest):
    try:
        graph_store.set_outgoing(source_num=int(doc_num), target_nums=payload.targets)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return get_document_ledger()


@app.patch("/ledger/{doc_num}/incoming", response_model=schemas.LedgerResponse)
def update_ledger_incoming(doc_num: int, payload: schemas.LedgerLinksUpdateRequest):
    try:
        graph_store.set_incoming(target_num=int(doc_num), source_nums=payload.targets)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return get_document_ledger()


@app.patch("/ledger/{doc_num}/assign", response_model=schemas.LedgerResponse)
def update_ledger_assigned(doc_num: int, payload: schemas.LedgerAssignRequest):
    try:
        graph_store.set_assigned(doc_num=int(doc_num), assigned=bool(payload.assigned))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return get_document_ledger()


@app.post("/ingest/{doc_id}/extract", response_model=schemas.ExtractionResponse)
def extract_ingested_document(doc_id: str):
    document = get_ingested_document(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        pdf_path = get_document_source_path(doc_id)
        tei_xml = grobid_client.extract_tei(pdf_path)
        extraction_payload = extraction.parse_tei(tei_xml)
        stored = store_extraction(doc_id, tei_xml, extraction_payload)
        try:
            graph_store.index_extraction(
                ingest_meta=document, extraction_data=extraction_payload
            )
        except Exception:
            logger.exception("Graph index failed for extraction")
    except Exception as exc:
        logger.exception("Extraction failed for document %s", doc_id)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")

    return {"document_id": doc_id, "extraction": stored.get("extraction")}


@app.get("/ingest/{doc_id}/extraction", response_model=schemas.ExtractionResult)
def get_ingested_extraction(doc_id: str):
    document = get_ingested_document(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document.get("extraction")


@app.get("/ingest/{doc_id}/body", response_model=schemas.DocumentBodyResponse)
def get_ingested_body(doc_id: str):
    document = get_ingested_document(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        tei_xml = get_tei_xml(doc_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    payload = tei_body.build_document_body(tei_xml)
    return {"document_id": doc_id, **payload}


@app.post("/ingest/{doc_id}/resolve", response_model=schemas.ResolutionResponse)
def resolve_ingested_references(doc_id: str):
    document = get_ingested_document(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    extraction = document.get("extraction") or {}
    extraction_data = extraction.get("data") or {}
    references = extraction_data.get("references")
    if not references:
        raise HTTPException(
            status_code=404, detail="Extraction data not found for document"
        )

    try:
        resolved = resolve_references(references)
        stored = store_resolution(doc_id, resolved)
        try:
            graph_store.index_resolution(ingest_meta=document, resolution_data=resolved)
        except Exception:
            logger.exception("Graph index failed for resolution")
    except Exception as exc:
        logger.exception("Reference resolution failed for document %s", doc_id)
        raise HTTPException(
            status_code=500, detail=f"Reference resolution failed: {exc}"
        )

    return {"document_id": doc_id, "resolution": stored.get("resolution")}


@app.get("/ingest/{doc_id}/resolution", response_model=schemas.ResolutionResult)
def get_ingested_resolution(doc_id: str):
    document = get_ingested_document(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    resolution = document.get("resolution")
    if not resolution or not resolution.get("data"):
        raise HTTPException(status_code=404, detail="Resolution data not found")
    return resolution


@app.post(
    "/ingest/{doc_id}/resolution/{reference_id}/select",
    response_model=schemas.ResolutionResponse,
)
def select_resolution_source(
    doc_id: str, reference_id: str, payload: schemas.ResolutionSelectionRequest
):
    document = get_ingested_document(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    resolution = document.get("resolution") or {}
    resolution_data = resolution.get("data") or []
    if not resolution_data:
        raise HTTPException(status_code=404, detail="Resolution data not found")

    updated_entries = []
    updated = False
    for entry in resolution_data:
        if entry.get("reference_id") == reference_id:
            try:
                updated_entry = apply_resolution_selection(
                    entry, payload.selected_source, override_status=True
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            updated_entries.append(updated_entry)
            updated = True
        else:
            updated_entries.append(entry)

    if not updated:
        raise HTTPException(status_code=404, detail="Resolution entry not found")

    resolved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    resolution_payload = {
        **resolution,
        "status": "complete",
        "resolved_at": resolved_at,
        "data": updated_entries,
    }
    stored = update_ingested_document(doc_id, {"resolution": resolution_payload})
    return {"document_id": doc_id, "resolution": stored.get("resolution")}


@app.post(
    "/claims/{claim_id}/attachments",
    response_model=schemas.AttachmentResponse,
)
def create_claim_attachment(
    claim_id: str,
    payload: schemas.AttachmentCreateRequest,
    background_tasks: BackgroundTasks,
):
    try:
        record = attachment_store.create_attachment(
            claim_id=claim_id,
            doc_id=payload.doc_id,
            local_path=payload.local_path,
            filename=payload.filename,
            size_bytes=payload.size_bytes,
            reference_hint=payload.reference_hint,
            claim_text=payload.claim_text,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    public_record = attachment_store.public_status(record["id"])
    if background_tasks is not None:
        background_tasks.add_task(attachment_pipeline.process_attachment, record["id"])
    else:
        attachment_pipeline.enqueue_processing(record["id"])
    return {"attachment": _serialize_attachment(public_record)}


@app.post("/attachments", response_model=schemas.AttachmentResponse)
def create_global_attachment(
    payload: schemas.AttachmentGlobalCreateRequest,
    background_tasks: BackgroundTasks,
):
    try:
        record = attachment_store.create_attachment(
            claim_id=payload.claim_id,
            doc_id=payload.doc_id,
            local_path=payload.local_path,
            filename=payload.filename,
            size_bytes=payload.size_bytes,
            reference_hint=payload.reference_hint,
            claim_text=payload.claim_text,
            citation_index=payload.citation_index,
            target_id=payload.target_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    public_record = attachment_store.public_status(record["id"])
    if not background_state.get_state().get("paused"):
        if background_tasks is not None:
            background_tasks.add_task(
                attachment_pipeline.process_attachment, record["id"]
            )
        else:
            attachment_pipeline.enqueue_processing(record["id"])
    return {"attachment": _serialize_attachment(public_record)}


@app.get("/attachments", response_model=schemas.AttachmentListResponse)
def list_global_attachments(archived: bool = False):
    # Default: return only non-archived. If archived=true: include archived.
    store_archived: Optional[bool] = None if archived else False
    records = attachment_store.list_attachments(archived=store_archived, public=True)
    return {
        "attachments": [schemas.AttachmentStatus(**rec) for rec in records],
    }


@app.patch("/attachments/{attachment_id}", response_model=schemas.AttachmentResponse)
def patch_attachment_status(
    attachment_id: str, payload: schemas.AttachmentUpdateRequest
):
    record = attachment_store.get_attachment(attachment_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Attachment not found")

    fields = set(getattr(payload, "model_fields_set", set()))

    if "archived" in fields:
        if payload.archived is None:
            raise HTTPException(status_code=422, detail="archived must be a boolean")
        attachment_store.set_archived(attachment_id, archived=payload.archived)
        record = attachment_store.get_attachment(attachment_id) or record

    placement_fields = {"claim_id", "doc_id", "citation_index", "target_id"}
    if fields & placement_fields:
        next_claim_id = (
            payload.claim_id if "claim_id" in fields else record.get("claim_id")
        )
        next_doc_id = payload.doc_id if "doc_id" in fields else record.get("doc_id")
        next_citation_index = (
            payload.citation_index
            if "citation_index" in fields
            else record.get("citation_index")
        )
        next_target_id = (
            payload.target_id if "target_id" in fields else record.get("target_id")
        )

        if (
            next_claim_id != record.get("claim_id")
            or next_doc_id != record.get("doc_id")
            or next_citation_index != record.get("citation_index")
            or next_target_id != record.get("target_id")
        ):
            attachment_store.set_placement(
                attachment_id,
                claim_id=next_claim_id,
                doc_id=next_doc_id,
                citation_index=next_citation_index,
                target_id=next_target_id,
            )
            try:
                graph_store.mark_doc_assigned_for_attachment(
                    doc_id=next_doc_id,
                    claim_id=next_claim_id,
                )
            except Exception:
                logger.exception("Graph index failed for attachment placement")

    public_record = attachment_store.public_status(attachment_id)
    return {"attachment": _serialize_attachment(public_record)}


@app.get(
    "/claims/{claim_id}/attachments/status",
    response_model=schemas.AttachmentListResponse,
)
def claim_attachment_status(claim_id: str):
    records = attachment_store.public_claim_status(claim_id)
    return {
        "attachments": [schemas.AttachmentStatus(**rec) for rec in records],
    }


@app.get("/attachments/{attachment_id}", response_model=schemas.AttachmentResponse)
def get_attachment_status(attachment_id: str):
    public_record = attachment_store.public_status(attachment_id)
    return {"attachment": _serialize_attachment(public_record)}


@app.get(
    "/attachments/{attachment_id}/spans/{span_id}/jump",
    response_model=schemas.AttachmentSpanJumpResponse,
)
def get_attachment_span_jump(attachment_id: str, span_id: str):
    try:
        index = attachment_spans.AttachmentSpanIndex.for_attachment(attachment_id)
    except attachment_store.AttachmentNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        jump = index.jump(span_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Span not found") from exc
    return schemas.AttachmentSpanJumpResponse(
        attachment_id=attachment_id, span_id=span_id, **jump
    )


@app.get(
    "/attachments/{attachment_id}/spans/{span_id}/excerpt",
    response_model=schemas.AttachmentSpanExcerptResponse,
)
def get_attachment_span_excerpt(
    attachment_id: str,
    span_id: str,
    before: int = 2,
    after: int = 1,
):
    if before < 0 or after < 0:
        raise HTTPException(status_code=422, detail="before/after must be >= 0")

    try:
        index = attachment_spans.AttachmentSpanIndex.for_attachment(attachment_id)
    except attachment_store.AttachmentNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        sentences = index.excerpt(span_id, before=before, after=after)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Span not found") from exc
    return schemas.AttachmentSpanExcerptResponse(
        attachment_id=attachment_id,
        span_id=span_id,
        before=before,
        after=after,
        sentences=sentences,
    )


@app.post(
    "/attachments/{attachment_id}/retry", response_model=schemas.AttachmentResponse
)
def retry_attachment(
    attachment_id: str,
    background_tasks: BackgroundTasks,
):
    try:
        attachment_store.reset_for_retry(attachment_id)
    except attachment_store.AttachmentNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    public_record = attachment_store.public_status(attachment_id)
    if background_tasks is not None:
        background_tasks.add_task(attachment_pipeline.process_attachment, attachment_id)
    else:
        attachment_pipeline.enqueue_processing(attachment_id)
    return {"attachment": _serialize_attachment(public_record)}


@app.get(
    "/references/{doc_id}/{reference_id}/retrieval",
    response_model=schemas.ReferenceRetrievalResponse,
)
def get_reference_retrieval(doc_id: str, reference_id: str):
    try:
        return build_retrieval_dossier(doc_id, reference_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get(
    "/ingest/{doc_id}/citation-context",
    response_model=schemas.CitationContextResponse,
)
def get_citation_context(
    doc_id: str,
    citation_index: int = 0,
    target_id: Optional[str] = None,
):
    document = get_ingested_document(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        tei_xml = get_tei_xml(doc_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    context = citation_context.get_citation_context(tei_xml, citation_index, target_id)
    if context is None:
        return {"document_id": doc_id, "context": None}

    lookup_id = target_id or context.get("target_id")
    reference_entry = None
    resolution_entry = None
    if lookup_id:
        extraction_data = (document.get("extraction") or {}).get("data") or {}
        references = extraction_data.get("references") or []
        reference_entry = next(
            (ref for ref in references if ref.get("id") == lookup_id),
            None,
        )
        resolution_data = (document.get("resolution") or {}).get("data") or []
        resolution_entry = next(
            (ref for ref in resolution_data if ref.get("reference_id") == lookup_id),
            None,
        )

    return {
        "document_id": doc_id,
        "context": {
            **context,
            "citation_index": citation_index,
            "reference": reference_entry,
            "resolution": resolution_entry,
        },
    }


@app.get(
    "/ingest/{doc_id}/citation-graph",
    response_model=schemas.CitationGraphResponse,
)
def get_citation_graph(
    doc_id: str,
    target_id: Optional[str] = None,
    doi: Optional[str] = None,
    depth: int = 1,
    max_nodes: int = 10,
):
    document = get_ingested_document(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        resolved_identifier = doi
        if not resolved_identifier and target_id:
            resolution_data = (document.get("resolution") or {}).get("data") or []
            resolution_entry = next(
                (
                    ref
                    for ref in resolution_data
                    if ref.get("reference_id") == target_id
                ),
                None,
            )
            if resolution_entry:
                resolved_identifier = (
                    resolution_entry.get("openalex_id")
                    or resolution_entry.get("openalex_work_id")
                    or resolution_entry.get("doi")
                )

        if resolved_identifier:
            graph = citation_graph.build_citation_graph(
                resolved_identifier,
                depth=depth,
                max_nodes=max_nodes,
            )
        else:
            graph = citation_graph.build_local_citation_graph(
                document,
                target_id,
                depth=depth,
                max_nodes=max_nodes,
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return {"document_id": doc_id, **graph}


@app.post("/claims/confirm", response_model=schemas.ClaimConfirmationResponse)
def confirm_claims(payload: schemas.ClaimConfirmationRequest):
    inserted = claim_store.persist_confirmed_claims(payload)
    try:
        graph_store.index_confirmed_claims(payload.model_dump())
    except Exception:
        logger.exception("Graph index failed for confirmed claims")
    return {"inserted": inserted}


@app.get(
    "/claims/{claim_id}/evidence",
    response_model=schemas.EvidenceListResponse,
)
def list_claim_evidence(
    claim_id: str,
    label: Optional[str] = None,
    offset: int = 0,
    limit: int = 10,
    include_neutral: bool = True,
    pinned_only: bool = False,
    claim_text: Optional[str] = None,
):
    # Register claim_text without triggering a compute-heavy rerun.
    if claim_text:
        try:
            evidence_service.ensure_current_run(
                claim_id,
                claim_text=claim_text,
                execute=False,
            )
        except Exception:
            pass
    payload = evidence_service.list_candidates(
        claim_id,
        label=label,
        include_neutral=include_neutral,
        offset=offset,
        limit=limit,
    )
    return schemas.EvidenceListResponse(
        claim_id=claim_id,
        candidates=[
            schemas.EvidenceCandidatePayload(**cand) for cand in payload["candidates"]
        ],
        total=payload["total"],
        offset=payload["offset"],
        limit=payload["limit"],
        lock_state=payload["lock_state"],
        run=payload.get("run"),
    )


@app.post(
    "/claims/{claim_id}/evidence/rerun",
    response_model=schemas.EvidenceRerunResponse,
)
def request_evidence_rerun(
    claim_id: str, payload: schemas.EvidenceRerunRequest
) -> schemas.EvidenceRerunResponse:
    job = evidence_service.request_rerun(
        claim_id,
        claim_text=payload.claim_text,
        note=payload.note,
        advanced_settings=payload.advanced_settings,
    )
    return schemas.EvidenceRerunResponse(**job)


@app.get(
    "/claims/{claim_id}/evidence/history",
    response_model=schemas.EvidenceHistoryResponse,
)
def list_evidence_history(
    claim_id: str, limit: int = 5
) -> schemas.EvidenceHistoryResponse:
    runs = evidence_service.get_history(claim_id)
    if limit > 0:
        runs = runs[:limit]
    entries = [
        schemas.EvidenceHistoryEntry(
            run_id=run.get("run_id", "unknown"),
            created_at=run.get("created_at", ""),
            summary=run.get("summary", {}),
            metadata=run.get("metadata", {}),
        )
        for run in runs
    ]
    return schemas.EvidenceHistoryResponse(claim_id=claim_id, runs=entries)


@app.get(
    "/claims/{claim_id}/evidence/selection",
    response_model=schemas.EvidenceSelectionPayload,
)
def get_evidence_selection(claim_id: str):
    try:
        stored = evidence_selection_store.selection_store.read(claim_id)
    except ValidationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if stored is None:
        return schemas.EvidenceSelectionPayload(claim_id=claim_id, verdict="none")
    return stored


@app.put(
    "/claims/{claim_id}/evidence/selection",
    response_model=schemas.EvidenceSelectionPayload,
)
def put_evidence_selection(
    claim_id: str, payload: schemas.EvidenceSelectionUpsertRequest
):
    try:
        stored = evidence_selection_store.selection_store.upsert(claim_id, payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return stored


@app.get(
    "/claims/{claim_id}/judgment",
    response_model=schemas.JudgmentPayload,
)
def get_claim_judgment(claim_id: str, reviewer_uid: str = "default"):
    stored = judgment_store.judgment_store.read(claim_id, reviewer_uid=reviewer_uid)
    if stored is None:
        return schemas.JudgmentPayload(
            claim_id=claim_id,
            reviewer_uid=reviewer_uid,
            status="draft",
            verdict=None,
            notes=None,
        )
    return stored


@app.put(
    "/claims/{claim_id}/judgment",
    response_model=schemas.JudgmentPayload,
)
def put_claim_judgment(
    claim_id: str,
    payload: schemas.JudgmentUpsertRequest,
    reviewer_uid: str = "default",
):
    try:
        data = payload.model_dump(mode="json")
        data["reviewer_uid"] = reviewer_uid
        stored = judgment_store.judgment_store.upsert(claim_id, data)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return stored


@app.get(
    "/claims/{claim_id}/judgments",
    response_model=schemas.JudgmentByReviewerResponse,
)
def list_claim_judgments(claim_id: str):
    judgments = judgment_store.judgment_store.list_for_claim(claim_id)
    return {"judgments": judgments}


@app.get(
    "/judgments",
    response_model=schemas.JudgmentListResponse,
)
def list_judgments(
    doc_id: Optional[str] = None,
    include_drafts: bool = False,
    status: Optional[Literal["final", "draft", "all"]] = None,
):
    effective = status or ("all" if include_drafts else "final")
    judgments = judgment_store.judgment_store.list_filtered(
        status=effective, doc_id=doc_id
    )
    return {"judgments": judgments}


@app.get("/judgments/export")
def export_judgments(
    shape: Literal["claim", "callout"],
    format: Literal["json", "csv"] = "json",
    include_drafts: bool = False,
    mode: Literal["core", "verbose"] = "core",
):
    if shape == "claim":
        payload = judgment_store.judgment_store.export_claims(
            include_drafts=include_drafts,
            mode=mode,
            format=format,
        )
    else:
        payload = judgment_store.judgment_store.export_callouts(
            include_drafts=include_drafts,
            mode=mode,
            format=format,
        )

    media_type = "application/json" if format == "json" else "text/csv"
    scope = "all" if include_drafts else "final"
    filename = f"judgments_{shape}_{mode}_{scope}.{format}"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=payload, media_type=media_type, headers=headers)


# ---------- /segment endpoint ----------
@app.post("/segment")
async def segment(req: schemas.SentencePayload):
    """Process a set of user-generated segments for one CSV row."""
    # Read the UI’s choice (nested under req.settings), or fall back to ENV/app default
    pipeline_mode = (
        getattr(req.settings, "pipeline_mode", None)
        or os.environ.get("DEFAULT_PIPELINE_MODE")
        or getattr(app_settings, "DEFAULT_PIPELINE_MODE", "classic")
    )
    # Re-bind build_all to the pipeline implementation for this request
    cfg = app_settings.model_copy(deep=True)
    cfg.PIPELINE_MODE = pipeline_mode
    build_all = get_pipeline(cfg)

    # 1) Make sure a retriever exists for every (row_id, segment_id)
    # we are about to query
    try:
        for seg in req.segments:
            key = utils.make_retriever_key(req.row_id, seg.segment_id)
            if key in retrievers:
                # already cached – nothing to do
                continue

            if pipeline_mode == "hybrid":
                # --- one FAISS index *per segment* ---
                row_key = utils.make_retriever_key(req.row_id)
                if row_key not in retrievers:
                    raw_dict = build_all(
                        folder=Path(req.folder),
                        embed_model=req.settings.embed_model,
                        max_sentences=req.settings.max_sentences,
                        min_score=req.settings.faiss_min_score,
                        claim="; ".join(s.claim for s in req.segments),
                    )
                    retrievers[row_key] = next(iter(raw_dict.values()))
                retrievers[key] = retrievers[row_key]
                logging.info(f"[BUILD] hybrid index for {key}")
            else:
                # --- classic: one shared index for the whole row ---
                # Build it **once** (first segment) and reuse for the rest
                row_key = utils.make_retriever_key(req.row_id)  # no segment_id
                if row_key not in retrievers:
                    raw_dict = build_all(
                        folder=Path(req.folder),
                        embed_model=req.settings.embed_model,
                        max_sentences=req.settings.max_sentences,
                        min_score=req.settings.faiss_min_score,
                    )
                    retrievers[row_key] = raw_dict.get("default") or next(
                        iter(raw_dict.values())
                    )
                    logging.info(f"[BUILD] classic index for {row_key}")

                # Register the same retriever under this segment’s full key
                retrievers[key] = retrievers[row_key]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not build FAISS indices: {e}",
        )
    # 3) Embed all submitted “segments” locally (via sentence-transformers)
    seg_texts = [seg.claim for seg in req.segments]
    logger.debug(f"[EMBED] embedding segments: {seg_texts}")
    try:
        seg_vecs = utils.embed(
            seg_texts, model_name=req.settings.embed_model, mode="query"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local embedding failed: {e}")
    logger.debug(
        "[EMBED] returned %s vectors; example vec[0]=%s",
        len(seg_vecs),
        seg_vecs[0] if seg_vecs else None,
    )

    response_segments = []
    for seg, vec in zip(req.segments, seg_vecs):
        logger.debug(f"[SEGMENT]={seg.segment_id} claim={seg.claim!r}")

        # Retrieve the appropriate retriever for this segment
        key = f"{req.row_id}::{seg.segment_id}"
        retr = retrievers.get(key)
        if retr is None:
            raise HTTPException(
                status_code=500, detail=f"No FAISS index found for key={key}"
            )

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
        if req.settings.reranker_model and filtered:
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
            filtered = [(c["id"], c["faiss_score"]) for c in reranked]
        elif not filtered:
            logger.debug("[RERANK] Skipping rerank: no FAISS candidates to rerank")
            # >>> ADD THIS BLOCK <<<
            logger.debug(
                "[EVIDENCE] No candidates after retrieval+rerank; "
                "skipping NLI for segment %r",
                seg.segment_id,
            )
            response_segments.append(
                {
                    "segment_id": seg.segment_id,
                    "claim": seg.claim,
                    "evidence": [],
                    "best_support_id": None,
                    "support_rationale": None,
                    "best_contradiction_id": None,
                    "contradiction_rationale": None,
                }
            )
            continue  # move to next segment

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

        # ——— Dedupe within each label by overlapping sentence IDs ———
        deduped = []
        for lbl in ("entailment", "contradiction"):
            # get and sort evidences for this label by descending NLI score
            lbl_evs = sorted(
                (ev for ev in evidences if ev["label"] == lbl),
                key=lambda ev: ev["score"],
                reverse=True,
            )
            kept_sent_ids = set()
            for ev in lbl_evs:
                sids = ev.get("sent_ids", [])
                # if any sent_id already kept, skip this chunk
                if any(sid in kept_sent_ids for sid in sids):
                    continue
                # otherwise keep it and mark its sentences as seen
                deduped.append(ev)
                kept_sent_ids.update(sids)
        evidences = deduped
        logger.debug(f"[NLI→deduped by label] {evidences}")

        # 9) If there is at least one piece of evidence, run pick_best_passage
        texts_for_llm = [e["text"] for e in evidences]
        if texts_for_llm:
            sup_id, sup_rat = utils.pick_best_passage(
                seg.claim,
                texts_for_llm,
                "support",
                model_name=req.settings.llm_model,
                api_key=req.settings.api_key,
                base_url=req.settings.base_url,
            )
            con_id, con_rat = utils.pick_best_passage(
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

    return {
        "row_id": req.row_id,
        "status": "done",
        "original_sentence": req.original_sentence,
        "segments": response_segments,
    }


# ---------- /prebuild endpoint ----------
@app.post("/prebuild")
def prebuild(req: schemas.PrebuildRequest):
    import math

    pipeline_mode = getattr(req, "pipeline_mode", None)
    if not pipeline_mode:
        pipeline_mode = os.environ.get("DEFAULT_PIPELINE_MODE") or getattr(
            app_settings, "DEFAULT_PIPELINE_MODE", None
        )

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
