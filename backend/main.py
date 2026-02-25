# backend/main.py

import os
import re
import shutil
import asyncio
import json
import hashlib
import io
import tempfile
import zipfile
from uuid import uuid4

# Transformers v5 removes TRANSFORMERS_CACHE; use HF_HOME.
os.environ.setdefault("HF_HOME", str(os.path.expanduser("~/.cache/huggingface")))

# Set threading env vars *before* numpy/torch
from backend import utils

utils.set_sane_threads()
import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from backend import (
    background_state,
    attachment_pipeline,
    attachment_store,
    attachment_spans,
    citation_context,
    citation_graph,
    evidence_selection_store,
    judgment_store,
    schemas,
    tei_body,
    utils,
)
from backend.nli import assess
from backend.evidence_matching.service import evidence_service
from pydantic import BaseModel, Field, ValidationError

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
from backend.db import apply_migrations, connect
from backend.object_store import s3 as object_store_s3
from backend.spine.ids import pdf_object_key_for_work_pdf, work_id_from_doc_id
from backend.spine.ingest_view import (
    build_ingested_document_from_spine,
    list_ingests_from_spine,
    get_tei_xml_from_spine,
    get_resolution_from_spine,
)
from backend.spine.works import upsert_work_from_pdf
from backend.spine.attempts import (
    create_or_get_attempt,
    get_latest_attempt_for_work,
    mark_attempt_cancelled,
)
from backend.spine.artifacts import (
    create_artifact,
    get_artifact_for_attempt,
    list_artifacts_for_attempt,
)
from backend.spine.jobs import list_jobs_for_attempt
from backend.spine.documents import (
    ensure_project_document,
    get_document_version_by_sha256,
    get_latest_document_version,
    get_or_create_document,
    upsert_document_version,
)
from backend.spine.settings_store import (
    append_settings_version,
    get_latest_settings_version,
    get_or_create_settings_bundle,
)
from backend.spine.workflow_store import (
    append_workflow_version,
    create_workflow_definition,
    get_latest_workflow_version,
    list_workflows,
)
from backend.spine.locators import (
    create_locator,
    get_locator,
    list_locators_for_document_version,
)
from backend.spine.pdf_source import cleanup_temp_path, get_pdf_temp_path
from backend.claim_store import claim_store
from backend.reference_retrieval import build_retrieval_dossier
from backend.graph_store import GraphStore
from backend.span_graph_store import SpanGraphStore
from backend.spine.extraction_pool import SpineExtractionPool
from backend import fallback_body
from backend.pipeline_contracts import service as pipeline_contracts_service
from backend.spine.pipeline_artifacts import StageArtifactAlreadyExists
from backend.spine import pipeline_run_scopes as run_scopes_spine
from backend.spine import pipeline_run_status as run_status_spine
from backend.spine import evidence_decisions as evidence_decisions_spine
from backend.workflow_happy_path import orchestrator as happy_path_orchestrator
from backend.spine.project_meta import (
    export_project_meta_json,
    get_or_create_project_meta,
    import_project_meta_json,
    update_project_meta,
)

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


graph_store = GraphStore(settings=app_settings)
span_graph_store = SpanGraphStore(settings=app_settings)

spine_extract_pool = SpineExtractionPool(
    max_workers=max(1, int(getattr(app_settings, "INGEST_PIPELINE_WORKERS", 1) or 1)),
    max_queue=max(
        16, int(getattr(app_settings, "INGEST_PIPELINE_QUEUE_MAX", 64) or 64)
    ),
)

# Setup pipeline via registry and settings
build_all = get_pipeline(app_settings)
print(f"Pipeline mode: {app_settings.PIPELINE_MODE}")  # Optional: log on startup

app = FastAPI(title="Blablador NLI backend")


CSV_PATH = Path(os.environ.get("CSV_PATH", "source.csv")).resolve()
SOURCE_DIR = Path(os.environ.get("SOURCE_DIR", CSV_PATH.parent / "source")).resolve()


@app.on_event("startup")
async def startup_event():
    _require_spine_only_runtime()

    # Ensure V2 spine infra is usable before serving requests.
    for attempt in range(1, 31):
        try:
            apply_migrations()
            break
        except Exception:
            if attempt >= 30:
                logger.exception("Postgres migrations failed; giving up")
                raise
            logger.warning("Postgres not ready; retrying migrations (%s/30)", attempt)
            await asyncio.sleep(1)

    for attempt in range(1, 31):
        try:
            object_store_s3.ensure_bucket()
            break
        except Exception:
            if attempt >= 30:
                logger.exception("S3 bucket check failed; giving up")
                raise
            logger.warning("S3 not ready; retrying bucket check (%s/30)", attempt)
            await asyncio.sleep(1)

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


# ---------------------------------------------------------------------------
# Phase 10-03: Graph as navigation (/nav/*)
# ---------------------------------------------------------------------------


_NAV_STATE_VOCAB = {
    "missing",
    "requested",
    "blocked",
    "available",
    "processing",
    "done",
    "error",
    "cancelled",
    "mixed",
}


def _nav_state_for_work(
    *, work_id: str, requested_ids: set[str], resolved: bool
) -> str:
    wid = str(work_id or "").strip()
    if wid and wid in requested_ids:
        return "requested"
    if resolved:
        return "available"
    return "missing"


def _nav_state_for_claim_span_status(status: Optional[str]) -> str:
    s = str(status or "").strip().lower()
    if s in {"supported", "contradicted", "contested", "not_supported"}:
        return "done"
    return "available"


def _nav_state_for_span_rollup(span_status: Optional[dict]) -> str:
    raw = span_status or {}
    s = str(raw.get("status") or "").strip().lower()
    if s in {"supported"}:
        return "done"
    if s in {"contradicted", "contested", "not_supported"}:
        return "mixed"
    return "available"


def _nav_work_node_id(work_id: str) -> str:
    return f"work:{str(work_id or '').strip()}"


def _nav_citespan_node_id(span_id: str) -> str:
    return f"citespan:{str(span_id or '').strip()}"


def _nav_claimspan_node_id(claim_span_id: str) -> str:
    csid = str(claim_span_id or "").strip()
    if csid.startswith("claimspan:"):
        return csid
    return f"claimspan:{csid}"


def _nav_doc_label(
    *, document_id: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    """Return (label, hover) for an ingested document id when available."""
    did = str(document_id or "").strip()
    if not did:
        return None, None
    try:
        ver = get_latest_document_version(document_id=did)
    except Exception:
        ver = None
    if not isinstance(ver, dict):
        return None, None
    fname = str(ver.get("filename") or "").strip() or None
    if not fname:
        return None, None
    stem = Path(fname).name
    label = Path(stem).stem or stem
    hover = f"{stem}\n{did}".strip()
    return label, hover


def _nav_short_work_label(work_id: str) -> str:
    wid = str(work_id or "").strip()
    if not wid:
        return "work"
    # Keep UUID-ish ids readable in the graph.
    if len(wid) >= 8:
        return wid[:8]
    return wid


def _nav_edge_id(*, src: str, tgt: str, kind: str) -> str:
    return f"edge:{kind}:{src}->{tgt}"


def _nav_bool(value: Optional[bool]) -> bool:
    return bool(value) if value is not None else False


@app.get("/nav/works/{work_id}/contexts")
def nav_work_contexts(work_id: str, reviewer_uid: str = Query("default")):
    reviewer = str(reviewer_uid or "").strip() or "default"
    try:
        contexts = span_graph_store.list_citing_contexts_for_work(
            work_id=str(work_id), reviewer_uid=reviewer
        )
    except Exception:
        logger.exception("nav contexts failed for work_id=%s", work_id)
        contexts = []
    # Never 500 on best-effort snippet enrichment.
    cleaned = []
    for row in contexts or []:
        if not isinstance(row, dict):
            continue
        cleaned.append(
            {
                "citing_doc_id": row.get("citing_doc_id"),
                "citation_index": row.get("citation_index"),
                "reference_id": row.get("reference_id"),
                "sentence_id": row.get("sentence_id"),
                "claim_id": row.get("claim_id"),
                "snippet": row.get("snippet"),
            }
        )
    return {"work_id": str(work_id), "reviewer_uid": reviewer, "contexts": cleaned}


@app.get("/nav/graph")
def nav_graph(
    reviewer_uid: str = Query("default"),
    focus_type: Optional[str] = None,
    focus_id: Optional[str] = None,
    show_claimspans: bool = Query(True),
):
    reviewer = str(reviewer_uid or "").strip() or "default"
    f_type = str(focus_type or "").strip().lower() or None
    f_id = str(focus_id or "").strip() or None

    # Settings (persisted via /project read-modify-write; no /nav POST yet).
    try:
        meta = get_or_create_project_meta(
            project_id=str(app_settings.DEFAULT_PROJECT_ID)
        )
    except Exception:
        meta = {}
    graph_settings = (
        (meta or {}).get("graph_settings") if isinstance(meta, dict) else None
    )
    graph_settings = graph_settings if isinstance(graph_settings, dict) else {}

    expanded_raw = graph_settings.get("expanded_work_ids")
    requested_raw = graph_settings.get("requested_work_ids")
    expanded_ids = [str(x) for x in (expanded_raw or []) if str(x).strip()]
    requested_ids = [str(x) for x in (requested_raw or []) if str(x).strip()]
    expanded_ids = expanded_ids[:500]
    requested_set = set(requested_ids[:500])

    if not f_type or not f_id:
        return {
            "focus": None,
            "elements": [],
            "meta": {
                "reviewer_uid": reviewer,
                "show_claimspans": _nav_bool(show_claimspans),
                "expanded_work_ids": expanded_ids,
                "requested_work_ids": list(requested_set),
            },
        }

    # Resolve focus into a citation_window span bundle when possible.
    focus_span_id: Optional[str] = None
    if f_type == "citespan":
        focus_span_id = f_id
    elif f_type == "claimspan":
        try:
            cs = span_graph_store.get_claim_span_by_id(claim_span_id=f_id)
        except Exception:
            cs = None
        if isinstance(cs, dict):
            focus_span_id = str(cs.get("span_id") or "").strip() or None
    elif f_type == "work":
        focus_span_id = None
    else:
        focus_span_id = None

    elements: list[dict] = []
    meta_out: dict[str, Any] = {
        "reviewer_uid": reviewer,
        "show_claimspans": _nav_bool(show_claimspans),
        "expanded_work_ids": expanded_ids,
        "requested_work_ids": list(requested_set),
    }

    def add_node(data: dict) -> None:
        if not isinstance(data, dict):
            return
        node_id = str((data.get("id") or "")).strip()
        if not node_id:
            return
        elements.append({"data": data})

    def add_edge(data: dict) -> None:
        if not isinstance(data, dict):
            return
        edge_id = str((data.get("id") or "")).strip()
        src = str((data.get("source") or "")).strip()
        tgt = str((data.get("target") or "")).strip()
        if not edge_id or not src or not tgt:
            return
        elements.append({"data": data})

    def work_node(
        *,
        work_id: str,
        label: Optional[str],
        state: str,
        routing: Optional[dict] = None,
        resolved_ingest_id: Optional[str] = None,
    ) -> dict:
        st_val = state if state in _NAV_STATE_VOCAB else "available"
        data = {
            "id": _nav_work_node_id(work_id),
            "label": str(label or "work"),
            "selectable_type": "work",
            "work_id": str(work_id),
            "resolved_ingest_id": str(resolved_ingest_id)
            if resolved_ingest_id
            else None,
            "state": st_val,
        }
        if isinstance(routing, dict):
            data.update({k: v for k, v in routing.items() if v is not None})
        return {k: v for k, v in data.items() if v is not None}

    def citespan_node(*, span_id: str, label: str, state: str, routing: dict) -> dict:
        st_val = state if state in _NAV_STATE_VOCAB else "available"
        data = {
            "id": _nav_citespan_node_id(span_id),
            "label": str(label or "citespan"),
            "selectable_type": "citespan",
            "span_id": str(span_id),
            "state": st_val,
        }
        data.update({k: v for k, v in (routing or {}).items() if v is not None})
        return data

    def claimspan_node(
        *, claim_span_id: str, label: str, state: str, routing: dict
    ) -> dict:
        st_val = state if state in _NAV_STATE_VOCAB else "available"
        data = {
            "id": _nav_claimspan_node_id(claim_span_id),
            "label": str(label or "claim"),
            "selectable_type": "claimspan",
            "claim_span_id": str(claim_span_id),
            "state": st_val,
        }
        data.update({k: v for k, v in (routing or {}).items() if v is not None})
        return data

    def edge(
        *, src: str, tgt: str, kind: str, state: str, routing: Optional[dict] = None
    ) -> dict:
        st_val = state if state in _NAV_STATE_VOCAB else "available"
        data = {
            "id": _nav_edge_id(src=src, tgt=tgt, kind=kind),
            "source": src,
            "target": tgt,
            "selectable_type": "edge",
            "kind": str(kind),
            "state": st_val,
        }
        if isinstance(routing, dict):
            data.update({k: v for k, v in routing.items() if v is not None})
        return data

    # Focus: citation-window span
    if focus_span_id:
        try:
            bundle = span_graph_store.span_bundle(
                span_id=str(focus_span_id),
                reviewer_uid=reviewer,
                include_history=False,
            )
        except Exception:
            bundle = None
        if not isinstance(bundle, dict):
            return {
                "focus": {"type": f_type, "id": f_id},
                "elements": [],
                "meta": {**meta_out, "error": "focus_not_found"},
            }

        span = bundle.get("span") or {}
        span_status = bundle.get("span_status") or {}
        citing_doc_id = str(span.get("ingest_id") or "").strip() or None
        cites = bundle.get("cites") or []

        # Use first cite row as routing anchor (best-effort).
        primary = None
        for c in cites:
            if isinstance(c, dict) and (
                c.get("reference_id") is not None or c.get("citation_index") is not None
            ):
                primary = c
                break
        primary = primary if isinstance(primary, dict) else (cites[0] if cites else {})
        try:
            citation_index = (
                int(primary.get("citation_index"))
                if primary.get("citation_index") is not None
                else None
            )
        except Exception:
            citation_index = None
        reference_id = str(primary.get("reference_id") or "").strip() or None

        routing = {
            "citing_doc_id": citing_doc_id,
            "citation_index": citation_index,
            "reference_id": reference_id,
        }

        # Nodes
        if citing_doc_id:
            cite_label, cite_hover = _nav_doc_label(document_id=citing_doc_id)
            add_node(
                work_node(
                    work_id=str(citing_doc_id),
                    label=cite_label or "Citing",
                    state="available",
                    routing=routing,
                    resolved_ingest_id=str(citing_doc_id),
                )
            )

        cs_id = _nav_citespan_node_id(str(focus_span_id))
        add_node(
            citespan_node(
                span_id=str(focus_span_id),
                label="CiteSpan",
                state=_nav_state_for_span_rollup(
                    span_status if isinstance(span_status, dict) else {}
                ),
                routing=routing,
            )
        )

        # Edges from work->citespan
        if citing_doc_id:
            w_id = _nav_work_node_id(str(citing_doc_id))
            add_edge(
                edge(
                    src=w_id,
                    tgt=cs_id,
                    kind="HAS_CITESPAN",
                    state="available",
                    routing=routing,
                )
            )

        # ClaimSpan nodes
        if bool(show_claimspans):
            for cs in bundle.get("claim_spans") or []:
                if not isinstance(cs, dict):
                    continue
                csid = str(cs.get("claim_span_id") or "").strip()
                if not csid:
                    continue
                order_index = cs.get("order_index")
                label = (
                    f"Claim {int(order_index)}" if order_index is not None else "Claim"
                )
                state = _nav_state_for_claim_span_status(cs.get("status"))
                node_id = _nav_claimspan_node_id(csid)
                add_node(
                    claimspan_node(
                        claim_span_id=csid,
                        label=label,
                        state=state,
                        routing={
                            **routing,
                            "order_index": int(order_index)
                            if order_index is not None
                            else None,
                        },
                    )
                )
                add_edge(
                    edge(
                        src=cs_id,
                        tgt=node_id,
                        kind="HAS_CLAIMSPAN",
                        state=state,
                        routing=routing,
                    )
                )

        # Cited work nodes (1-hop)
        for c in cites or []:
            if not isinstance(c, dict):
                continue
            cited_work_id = str(c.get("cited_work_id") or "").strip()
            if not cited_work_id:
                continue
            resolved_ingest_id = None
            if citing_doc_id and reference_id:
                try:
                    resolved_ingest_id = graph_store.resolve_reference_to_ingest_id(
                        citing_doc_id=str(citing_doc_id),
                        reference_id=str(c.get("reference_id") or reference_id),
                    )
                except Exception:
                    resolved_ingest_id = None

            state = _nav_state_for_work(
                work_id=cited_work_id,
                requested_ids=requested_set,
                resolved=bool(resolved_ingest_id),
            )
            cited_label = None
            if resolved_ingest_id:
                cited_label, _cited_hover = _nav_doc_label(
                    document_id=resolved_ingest_id
                )
            if not cited_label:
                cited_label = _nav_short_work_label(cited_work_id)
            w_node_id = _nav_work_node_id(cited_work_id)
            add_node(
                work_node(
                    work_id=cited_work_id,
                    label=cited_label,
                    state=state,
                    routing=routing,
                    resolved_ingest_id=str(resolved_ingest_id)
                    if resolved_ingest_id
                    else None,
                )
            )
            add_edge(
                edge(
                    src=cs_id, tgt=w_node_id, kind="CITES", state=state, routing=routing
                )
            )

    # Focus: work (ingest id) -> list spans as neighbors (best-effort)
    if f_type == "work" and f_id:
        ingest_id = str(f_id)
        w_id = _nav_work_node_id(ingest_id)
        seed_label, _seed_hover = _nav_doc_label(document_id=ingest_id)
        add_node(
            work_node(
                work_id=ingest_id,
                label=seed_label or "Work",
                state="available" if ingest_id not in requested_set else "requested",
                routing=None,
                resolved_ingest_id=ingest_id,
            )
        )
        try:
            spans = span_graph_store.list_spans_for_ingest(
                ingest_id=ingest_id, kind="citation_window", limit=25
            )
        except Exception:
            spans = []
        for sp in spans or []:
            if not isinstance(sp, dict):
                continue
            sid = str(sp.get("span_id") or "").strip()
            if not sid:
                continue

            # Derive routing anchor for this citespan (best-effort).
            try:
                cites_for_span = span_graph_store.list_span_cites(span_id=sid)
            except Exception:
                cites_for_span = []
            primary = None
            for c in cites_for_span or []:
                if isinstance(c, dict) and (
                    c.get("reference_id") is not None
                    or c.get("citation_index") is not None
                ):
                    primary = c
                    break
            primary = primary if isinstance(primary, dict) else {}
            try:
                citation_index = (
                    int(primary.get("citation_index"))
                    if primary.get("citation_index") is not None
                    else None
                )
            except Exception:
                citation_index = None
            reference_id = str(primary.get("reference_id") or "").strip() or None
            routing = {
                "citing_doc_id": ingest_id,
                "citation_index": citation_index,
                "reference_id": reference_id,
            }

            # Prefer a rollup state when possible.
            try:
                span_status = span_graph_store.span_status(
                    span_id=str(sid), reviewer_uid=reviewer
                )
            except Exception:
                span_status = {}
            span_state = _nav_state_for_span_rollup(
                span_status if isinstance(span_status, dict) else {}
            )

            cs_node_id = _nav_citespan_node_id(sid)
            add_node(
                citespan_node(
                    span_id=sid,
                    label="CiteSpan",
                    state=span_state,
                    routing=routing,
                )
            )
            add_edge(
                edge(
                    src=w_id,
                    tgt=cs_node_id,
                    kind="HAS_CITESPAN",
                    state=span_state,
                    routing=routing,
                )
            )

            # ClaimSpan nodes (toggle-controlled).
            if bool(show_claimspans):
                try:
                    claim_spans = span_graph_store.list_claim_spans_for_span(
                        span_id=str(sid)
                    )
                except Exception:
                    claim_spans = []
                for cs in claim_spans or []:
                    if not isinstance(cs, dict):
                        continue
                    csid = str(cs.get("claim_span_id") or "").strip()
                    if not csid:
                        continue
                    order_index = cs.get("order_index")
                    label = (
                        f"Claim {int(order_index)}"
                        if order_index is not None
                        else "Claim"
                    )
                    try:
                        status = span_graph_store.claim_span_status(
                            claim_span_id=csid, reviewer_uid=reviewer
                        )
                    except Exception:
                        status = {}
                    state = _nav_state_for_claim_span_status(
                        (status or {}).get("status")
                    )
                    node_id = _nav_claimspan_node_id(csid)
                    add_node(
                        claimspan_node(
                            claim_span_id=csid,
                            label=label,
                            state=state,
                            routing={
                                **routing,
                                "order_index": int(order_index)
                                if order_index is not None
                                else None,
                            },
                        )
                    )
                    add_edge(
                        edge(
                            src=cs_node_id,
                            tgt=node_id,
                            kind="HAS_CLAIMSPAN",
                            state=state,
                            routing=routing,
                        )
                    )

            # Cited work nodes (1-hop) from this citespan.
            for c in cites_for_span or []:
                if not isinstance(c, dict):
                    continue
                cited_work_id = str(c.get("cited_work_id") or "").strip()
                if not cited_work_id:
                    continue
                ref2 = str(c.get("reference_id") or "").strip() or reference_id
                try:
                    cite_idx2 = (
                        int(c.get("citation_index"))
                        if c.get("citation_index") is not None
                        else citation_index
                    )
                except Exception:
                    cite_idx2 = citation_index
                routing2 = {
                    "citing_doc_id": ingest_id,
                    "citation_index": cite_idx2,
                    "reference_id": ref2,
                }

                resolved_ingest_id = None
                if ingest_id and ref2:
                    try:
                        resolved_ingest_id = graph_store.resolve_reference_to_ingest_id(
                            citing_doc_id=str(ingest_id), reference_id=str(ref2)
                        )
                    except Exception:
                        resolved_ingest_id = None
                state = _nav_state_for_work(
                    work_id=cited_work_id,
                    requested_ids=requested_set,
                    resolved=bool(resolved_ingest_id),
                )
                cited_label = None
                if resolved_ingest_id:
                    cited_label, _cited_hover = _nav_doc_label(
                        document_id=resolved_ingest_id
                    )
                if not cited_label:
                    cited_label = _nav_short_work_label(cited_work_id)
                w_node_id = _nav_work_node_id(cited_work_id)
                add_node(
                    work_node(
                        work_id=cited_work_id,
                        label=cited_label,
                        state=state,
                        routing=routing2,
                        resolved_ingest_id=str(resolved_ingest_id)
                        if resolved_ingest_id
                        else None,
                    )
                )
                add_edge(
                    edge(
                        src=cs_node_id,
                        tgt=w_node_id,
                        kind="CITES",
                        state=state,
                        routing=routing2,
                    )
                )

    # Expansion: show additional citing contexts for expanded works.
    #
    # Semantics: expanded work IDs represent "unlock one-hop incoming context".
    # For each expanded work node, append other citation-window spans that cite
    # it (across the current project), plus optional claimspans.
    def _expand_work_incoming(*, work_id: str) -> None:
        wid = str(work_id or "").strip()
        if not wid:
            return
        try:
            contexts = span_graph_store.list_citing_contexts_for_work(
                work_id=wid, reviewer_uid=reviewer, limit=120
            )
        except Exception:
            contexts = []
        if not contexts:
            return

        for ctx in contexts[:80]:
            if not isinstance(ctx, dict):
                continue
            citing_doc_id = str(ctx.get("citing_doc_id") or "").strip() or None
            reference_id = str(ctx.get("reference_id") or "").strip() or None
            citation_index = ctx.get("citation_index")
            try:
                cite_idx = int(citation_index) if citation_index is not None else None
            except Exception:
                cite_idx = None
            if not citing_doc_id or cite_idx is None or not reference_id:
                continue

            span_id = None
            try:
                span_id = span_graph_store.get_citation_span_id(
                    ingest_id=str(citing_doc_id),
                    citation_index=int(cite_idx),
                    target_id=str(reference_id),
                )
            except Exception:
                span_id = None
            if not span_id:
                try:
                    span = span_graph_store.find_citation_span(
                        ingest_id=str(citing_doc_id),
                        citation_index=int(cite_idx),
                        target_id=str(reference_id),
                    )
                except Exception:
                    span = None
                if isinstance(span, dict):
                    span_id = str(span.get("span_id") or "").strip() or None
            if not span_id:
                continue

            routing = {
                "citing_doc_id": citing_doc_id,
                "citation_index": int(cite_idx),
                "reference_id": reference_id,
                "sentence_id": ctx.get("sentence_id"),
            }

            # Citing work node + its citespan.
            cite_label, cite_hover = _nav_doc_label(document_id=citing_doc_id)
            add_node(
                work_node(
                    work_id=str(citing_doc_id),
                    label=cite_label or "Citing",
                    state="available",
                    routing=routing,
                    resolved_ingest_id=str(citing_doc_id),
                )
            )

            try:
                span_status = span_graph_store.span_status(
                    span_id=str(span_id), reviewer_uid=reviewer
                )
            except Exception:
                span_status = {}
            span_state = _nav_state_for_span_rollup(
                span_status if isinstance(span_status, dict) else {}
            )

            cs_node_id = _nav_citespan_node_id(str(span_id))
            add_node(
                citespan_node(
                    span_id=str(span_id),
                    label="CiteSpan",
                    state=span_state,
                    routing=routing,
                )
            )
            add_edge(
                edge(
                    src=_nav_work_node_id(str(citing_doc_id)),
                    tgt=cs_node_id,
                    kind="HAS_CITESPAN",
                    state=span_state,
                    routing=routing,
                )
            )

            resolved_ingest_id = None
            try:
                resolved_ingest_id = graph_store.resolve_reference_to_ingest_id(
                    citing_doc_id=str(citing_doc_id), reference_id=str(reference_id)
                )
            except Exception:
                resolved_ingest_id = None
            work_state = _nav_state_for_work(
                work_id=wid,
                requested_ids=requested_set,
                resolved=bool(resolved_ingest_id),
            )
            work_label = None
            if resolved_ingest_id:
                work_label, _work_hover = _nav_doc_label(document_id=resolved_ingest_id)
            if not work_label:
                work_label, _work_hover2 = _nav_doc_label(document_id=wid)
            if not work_label:
                work_label = _nav_short_work_label(wid)
            add_node(
                work_node(
                    work_id=str(wid),
                    label=work_label,
                    state=work_state,
                    routing=routing,
                    resolved_ingest_id=str(resolved_ingest_id)
                    if resolved_ingest_id
                    else None,
                )
            )
            add_edge(
                edge(
                    src=cs_node_id,
                    tgt=_nav_work_node_id(str(wid)),
                    kind="CITES",
                    state=work_state,
                    routing=routing,
                )
            )

            if bool(show_claimspans):
                try:
                    claim_spans = span_graph_store.list_claim_spans_for_span(
                        span_id=str(span_id)
                    )
                except Exception:
                    claim_spans = []
                for cs in claim_spans or []:
                    if not isinstance(cs, dict):
                        continue
                    csid = str(cs.get("claim_span_id") or "").strip()
                    if not csid:
                        continue
                    order_index = cs.get("order_index")
                    label = (
                        f"Claim {int(order_index)}"
                        if order_index is not None
                        else "Claim"
                    )
                    try:
                        status = span_graph_store.claim_span_status(
                            claim_span_id=csid, reviewer_uid=reviewer
                        )
                    except Exception:
                        status = {}
                    cs_state = _nav_state_for_claim_span_status(
                        (status or {}).get("status")
                    )
                    node_id = _nav_claimspan_node_id(csid)
                    add_node(
                        claimspan_node(
                            claim_span_id=csid,
                            label=label,
                            state=cs_state,
                            routing={
                                **routing,
                                "order_index": int(order_index)
                                if order_index is not None
                                else None,
                            },
                        )
                    )
                    add_edge(
                        edge(
                            src=cs_node_id,
                            tgt=node_id,
                            kind="HAS_CLAIMSPAN",
                            state=cs_state,
                            routing=routing,
                        )
                    )

    for wid in expanded_ids:
        _expand_work_incoming(work_id=str(wid))

    return {
        "focus": {"type": f_type, "id": f_id},
        "elements": elements,
        "meta": meta_out,
    }


def _is_pdf_upload(file: UploadFile) -> bool:
    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()
    return filename.endswith(".pdf") and content_type == "application/pdf"


def _require_spine_only_runtime() -> None:
    mode = str(os.environ.get("SPINE_MODE", "spine") or "spine").strip().lower()
    if mode and mode != "spine":
        raise RuntimeError(
            "Legacy/dual SPINE_MODE values were removed; set SPINE_MODE=spine"
        )


def _serialize_attachment(record: Optional[dict]) -> schemas.AttachmentStatus:
    if record is None:
        raise HTTPException(status_code=404, detail="Attachment not found")
    data = dict(record or {})

    # Backward-compatible field normalization.
    # Older payloads use uploaded_at; spine-backed attachments use created_at.
    if not data.get("uploaded_at"):
        data["uploaded_at"] = (
            data.get("created_at")
            or data.get("updated_at")
            or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
    if not data.get("updated_at"):
        data["updated_at"] = data.get("uploaded_at") or data.get("created_at")
    if data.get("size") is None and data.get("size_bytes") is not None:
        data["size"] = data.get("size_bytes")

    return schemas.AttachmentStatus(**data)


@app.post("/ingest", response_model=schemas.IngestUploadResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auto_process: bool = Query(True),
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    if not _is_pdf_upload(file):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

    file_bytes = await file.read()

    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)
    user_id = str(app_settings.DEFAULT_USER_ID)
    filename = str(file.filename or "").strip() or "document.pdf"

    sha256 = hashlib.sha256(file_bytes).hexdigest().lower()
    existing = get_document_version_by_sha256(sha256=sha256)

    if existing is not None:
        doc_id = str(existing.get("document_id") or "").strip()
        if not doc_id:
            raise HTTPException(
                status_code=500, detail="Spine dedupe missing document_id"
            )
        pdf_object_key = str(existing.get("pdf_object_key") or "").strip()
        size_bytes = int(existing.get("size_bytes") or len(file_bytes))
    else:
        doc_id = str(uuid4())
        pdf_object_key = pdf_object_key_for_work_pdf(doc_id, sha256)
        size_bytes = int(len(file_bytes))

    work_id = work_id_from_doc_id(doc_id)

    try:
        if not object_store_s3.exists(pdf_object_key):
            object_store_s3.put_bytes(
                pdf_object_key,
                file_bytes,
                content_type="application/pdf",
            )
    except Exception as exc:
        logger.exception("S3 upload failed for work_id=%s", work_id)
        raise HTTPException(status_code=503, detail="Object store unavailable") from exc

    try:
        upsert_work_from_pdf(
            work_id=work_id,
            project_id=project_id,
            created_by_user_id=user_id,
            filename=filename,
            sha256=sha256,
            size_bytes=size_bytes,
            pdf_object_key=pdf_object_key,
        )
        get_or_create_document(doc_id, created_by_user_id=user_id)
        upsert_document_version(
            doc_id,
            document_id=doc_id,
            sha256=sha256,
            size_bytes=size_bytes,
            pdf_object_key=pdf_object_key,
            filename=filename,
            created_by_user_id=user_id,
        )
        ensure_project_document(
            project_id,
            document_id=doc_id,
            added_by_user_id=user_id,
        )
    except Exception as exc:
        logger.exception("Spine upsert failed for doc_id=%s", doc_id)
        raise HTTPException(status_code=503, detail="Postgres unavailable") from exc

    doc = build_ingested_document_from_spine(
        work_id=doc_id,
        project_id=project_id,
        include_extraction_data=False,
    )
    if doc is None:
        raise HTTPException(status_code=500, detail="Spine ingest record missing")

    try:
        graph_store.index_ingest_upload(doc)
    except Exception:
        logger.exception("Graph index failed for ingest upload")

    if bool(auto_process) and not background_state.get_state().get("paused"):
        try:
            spine_extract_pool.enqueue(
                doc_id=work_id,
                project_id=project_id,
                user_id=user_id,
                force_fallback=False,
            )
        except Exception:
            logger.exception("Unable to enqueue extraction for doc_id=%s", doc_id)

    return {"document": doc}


@app.get("/ingest", response_model=schemas.IngestListResponse)
def list_ingest_documents(
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)

    try:
        out = list_ingests_from_spine(project_id=project_id, limit=200)
    except Exception as exc:
        logger.exception("Spine list failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    out = list(out or [])
    out.sort(key=lambda item: str(item.get("uploaded_at") or ""), reverse=True)
    return {"documents": out}


@app.get("/ingest/{doc_id}", response_model=schemas.IngestedDocument)
def get_ingest_document(
    doc_id: str,
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)
    try:
        doc = build_ingested_document_from_spine(
            work_id=str(doc_id),
            project_id=project_id,
            include_extraction_data=True,
        )
    except Exception as exc:
        logger.exception("Spine get failed for doc_id=%s", doc_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.get("/ledger", response_model=schemas.LedgerResponse)
def get_document_ledger():
    rows = graph_store.ledger_rows()
    options = graph_store.ledger_options()

    # Enrich ledger rows with ingestion stage status/errors so the UI can show
    # failures without requiring a click.
    by_ingest: dict[str, dict] = {}

    project_id = str(app_settings.DEFAULT_PROJECT_ID)
    try:
        for doc in list_ingests_from_spine(project_id=project_id, limit=2000):
            ingest_id = str(doc.get("id") or "").strip()
            if ingest_id:
                by_ingest[ingest_id] = doc
    except Exception:
        logger.exception("Spine list failed for ledger enrichment")

    for row in rows:
        if not isinstance(row, dict):
            continue
        ingest_id = str(row.get("ingest_id") or "").strip()
        if not ingest_id:
            continue
        doc = by_ingest.get(ingest_id) or {}
        extraction = doc.get("extraction") or {}
        body_extraction = doc.get("body_extraction") or {}
        resolution = doc.get("resolution") or {}
        if isinstance(extraction, dict):
            row["extraction_status"] = extraction.get("status")
            row["extraction_error"] = extraction.get("error")
        if isinstance(body_extraction, dict):
            row["body_extraction_status"] = body_extraction.get("status")
            row["body_extraction_error"] = body_extraction.get("error")
        if isinstance(resolution, dict):
            row["resolution_status"] = resolution.get("status")
            row["resolution_error"] = resolution.get("error")

    return {"rows": rows, "options": options}


@app.get("/project", response_model=schemas.ProjectMeta)
def get_project_meta(
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)
    user_id = str(app_settings.DEFAULT_USER_ID)
    return get_or_create_project_meta(project_id=project_id, user_id=user_id)


class ProjectMetaUpdate(BaseModel):
    name: Optional[str] = None
    reviewers: Optional[list[str]] = None
    active_reviewer_uid: Optional[str] = None
    compare_reviewer_a: Optional[str] = None
    compare_reviewer_b: Optional[str] = None
    graph_settings: Optional[dict[str, Any]] = None


@app.put("/project", response_model=schemas.ProjectMeta)
def put_project_meta(
    payload: ProjectMetaUpdate,
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    patch = payload.model_dump(exclude_unset=True)
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)
    user_id = str(app_settings.DEFAULT_USER_ID)
    try:
        return update_project_meta(project_id=project_id, user_id=user_id, patch=patch)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
#  Spine: versioned settings + workflows (additive endpoints)
# ---------------------------------------------------------------------------


class SpineSettingsUpsertRequest(BaseModel):
    config: dict = {}
    schema_version: int = 1
    name: str = "default"


@app.get("/spine/settings")
def spine_get_settings(
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)
    user_id = str(app_settings.DEFAULT_USER_ID)
    bundle = get_or_create_settings_bundle(
        project_id=project_id,
        name="default",
        created_by_user_id=user_id,
    )
    latest = get_latest_settings_version(
        bundle_id=str(bundle["bundle_id"]),
        project_id=project_id,
    )
    return {"bundle_id": bundle["bundle_id"], "latest": latest}


@app.post("/spine/settings")
def spine_post_settings(
    payload: SpineSettingsUpsertRequest,
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)
    user_id = str(app_settings.DEFAULT_USER_ID)
    name = str(payload.name or "default").strip() or "default"
    bundle = get_or_create_settings_bundle(
        project_id=project_id,
        name=name,
        created_by_user_id=user_id,
    )
    version = append_settings_version(
        bundle_id=str(bundle["bundle_id"]),
        project_id=project_id,
        created_by_user_id=user_id,
        config_json=dict(payload.config or {}),
        schema_version=int(payload.schema_version or 1),
    )
    return {"bundle_id": bundle["bundle_id"], "version": version}


class SpineWorkflowCreateRequest(BaseModel):
    name: str
    graph: dict = {}


class SpineWorkflowVersionRequest(BaseModel):
    graph: dict = {}


@app.get("/spine/workflows")
def spine_list_workflows(
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)
    return {"workflows": list_workflows(project_id=project_id)}


@app.post("/spine/workflows")
def spine_create_workflow(
    payload: SpineWorkflowCreateRequest,
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)
    user_id = str(app_settings.DEFAULT_USER_ID)
    wf = create_workflow_definition(
        project_id=project_id,
        name=str(payload.name or "").strip(),
        created_by_user_id=user_id,
    )
    ver = append_workflow_version(
        workflow_id=str(wf["workflow_id"]),
        project_id=project_id,
        created_by_user_id=user_id,
        graph_json=dict(payload.graph or {}),
    )
    return {"workflow_id": wf["workflow_id"], "version": ver}


@app.post("/spine/workflows/{workflow_id}/versions")
def spine_append_workflow_version(
    workflow_id: str,
    payload: SpineWorkflowVersionRequest,
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)
    user_id = str(app_settings.DEFAULT_USER_ID)
    ver = append_workflow_version(
        workflow_id=str(workflow_id),
        project_id=project_id,
        created_by_user_id=user_id,
        graph_json=dict(payload.graph or {}),
    )
    latest = get_latest_workflow_version(
        workflow_id=str(workflow_id), project_id=project_id
    )
    return {"workflow_id": workflow_id, "version": ver, "latest": latest}


class SpineLocatorCreateRequest(BaseModel):
    document_version_id: str
    type: str
    payload: dict = {}


@app.post("/spine/locators")
def spine_create_locator(
    payload: SpineLocatorCreateRequest,
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)
    user_id = str(app_settings.DEFAULT_USER_ID)
    loc = create_locator(
        project_id=project_id,
        created_by_user_id=user_id,
        document_version_id=str(payload.document_version_id),
        type=str(payload.type),
        payload_json=dict(payload.payload or {}),
    )
    return {"locator_id": loc["locator_id"], "locator": loc}


@app.get("/spine/locators/{locator_id}")
def spine_get_locator(
    locator_id: str,
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)
    loc = get_locator(locator_id=str(locator_id), project_id=project_id)
    if loc is None:
        raise HTTPException(status_code=404, detail="Locator not found")
    return {"locator": loc}


@app.get("/spine/document-versions/{document_version_id}/locators")
def spine_list_locators_for_document_version(
    document_version_id: str,
    limit: int = 200,
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)
    locs = list_locators_for_document_version(
        project_id=project_id,
        document_version_id=str(document_version_id),
        limit=int(limit),
    )
    return {"locators": locs}


@app.get("/project/export")
def export_project():
    project_id = str(app_settings.DEFAULT_PROJECT_ID)
    meta = get_or_create_project_meta(project_id=project_id)
    name = (meta.get("name") or "project").strip() or "project"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)
    filename = f"{safe}.zip"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.json", export_project_meta_json(project_id=project_id))
        try:
            zf.writestr(
                "evidence_decision_events.ndjson",
                evidence_decisions_spine.export_events_ndjson(project_id=project_id),
            )
        except Exception:
            logger.exception("Decision export failed")
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers=headers,
    )


@app.post("/project/import", response_model=schemas.ProjectImportResponse)
async def import_project(file: UploadFile = File(...), overwrite: bool = False):
    blob = await file.read()
    project_id = str(app_settings.DEFAULT_PROJECT_ID)
    user_id = str(app_settings.DEFAULT_USER_ID)
    try:
        with zipfile.ZipFile(io.BytesIO(blob), "r") as zf:
            meta_raw = json.loads(zf.read("project.json").decode("utf-8"))
            decision_blob = None
            try:
                decision_blob = zf.read("evidence_decision_events.ndjson")
            except Exception:
                decision_blob = None
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    import_project_meta_json(project_id=project_id, user_id=user_id, meta=meta_raw)
    if decision_blob:
        try:
            evidence_decisions_spine.import_events_ndjson(
                project_id=project_id,
                user_id=user_id,
                blob=decision_blob,
                overwrite=bool(overwrite),
            )
        except Exception:
            logger.exception("Decision import failed")
    project = get_or_create_project_meta(project_id=project_id, user_id=user_id)
    return {"ok": True, "backup_zip": "", "project": project}


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

    # Fallback: if another claim already has a ready attachment for the same
    # citing doc + reference target, clone it rather than requiring the graph
    # store to have merged the cited ingest id.
    donor_same_target = None
    for rec in attachment_store.list_attachments(archived=None, public=False):
        if str(rec.get("doc_id") or "") != doc_id:
            continue
        if str(rec.get("target_id") or "") != str(target_id):
            continue
        if attachment_store.is_ready(rec):
            donor_same_target = rec
            break

    if donor_same_target:
        donor_pdf_key = str(donor_same_target.get("pdf_object_key") or "").strip()
        if donor_pdf_key:
            tmp_path = None
            try:
                blob = object_store_s3.get_bytes(donor_pdf_key)
                tmp = tempfile.NamedTemporaryFile(
                    prefix="attachment-",
                    suffix=".pdf",
                    delete=False,
                )
                try:
                    tmp.write(blob)
                    tmp.flush()
                finally:
                    tmp.close()
                tmp_path = Path(tmp.name)

                filename = donor_same_target.get("filename") or "attachment.pdf"
                record = attachment_store.create_attachment(
                    claim_id=claim_id,
                    doc_id=doc_id,
                    local_path=tmp_path,
                    filename=filename,
                    reference_hint={"reference_id": target_id},
                    citation_index=citation_index,
                    target_id=target_id,
                    source_ingest_id=donor_same_target.get("source_ingest_id"),
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except Exception:
                logger.exception("Unable to clone donor attachment PDF")
                record = None
            finally:
                try:
                    if tmp_path and tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass

            if record is not None:
                donor_artifacts = donor_same_target.get("artifacts") or {}
                if isinstance(donor_artifacts, dict) and donor_artifacts:
                    try:
                        attachment_store.mark_matched(
                            str(record["id"]), artifacts=dict(donor_artifacts)
                        )
                    except Exception:
                        logger.exception("Unable to reuse donor attachment artifacts")
                else:
                    # No donor artifacts; process like a normal upload.
                    if not background_state.get_state().get("paused"):
                        if background_tasks is not None:
                            background_tasks.add_task(
                                attachment_pipeline.process_attachment, record["id"]
                            )
                        else:
                            attachment_pipeline.enqueue_processing(record["id"])

                public_record = attachment_store.public_status(record["id"]) or {}
                return {
                    "attachment": _serialize_attachment(public_record),
                    "reused": False,
                }

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

    tmp_pdf_path = None
    try:
        tmp_pdf_path = get_pdf_temp_path(
            doc_id=str(cited_ingest_id),
            work_id=str(cited_ingest_id),
            project_id=str(app_settings.DEFAULT_PROJECT_ID),
            document=None,
        )
        source_path = tmp_pdf_path
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    cited_meta = (
        build_ingested_document_from_spine(
            work_id=str(cited_ingest_id),
            project_id=str(app_settings.DEFAULT_PROJECT_ID),
            include_extraction_data=False,
        )
        or {}
    )
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
    finally:
        cleanup_temp_path(tmp_pdf_path)

    # If a donor exists, clone artifacts and mark ready (skip reprocessing).
    if donor and donor.get("artifacts"):
        donor_artifacts = donor.get("artifacts") or {}
        if isinstance(donor_artifacts, dict) and donor_artifacts:
            try:
                attachment_store.mark_matched(
                    str(record["id"]), artifacts=dict(donor_artifacts)
                )
            except Exception:
                logger.exception("Unable to reuse donor attachment artifacts")
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


# --- Claim graph (Phase 09) -------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _claim_text(node: dict) -> str:
    props = node.get("properties") or {}
    raw = props.get("parsed_text") or props.get("claim_text") or ""
    return str(raw or "").strip()


def _node_label(node: dict) -> str:
    label = node.get("label") or _claim_text(node) or node.get("node_id") or ""
    text = str(label or "").strip()
    if len(text) > 140:
        return text[:137] + "..."
    return text


def _node_payload(node: dict) -> schemas.ClaimGraphNode:
    return schemas.ClaimGraphNode(
        id=str(node.get("node_id") or ""),
        kind=str(node.get("kind") or ""),
        label=_node_label(node),
        properties=dict(node.get("properties") or {}),
    )


def _edge_payload(
    edge: dict, *, aggregates: Optional[dict] = None
) -> schemas.ClaimGraphEdge:
    aggs = aggregates or graph_store.edge_vote_aggregates(int(edge.get("edge_id") or 0))
    return schemas.ClaimGraphEdge(
        edge_id=int(edge.get("edge_id") or 0),
        source_id=str(edge.get("source_id") or ""),
        target_id=str(edge.get("target_id") or ""),
        kind=str(edge.get("kind") or ""),
        properties=dict(edge.get("properties") or {}),
        aggregates=schemas.ClaimGraphEdgeAggregates(**aggs),
    )


def _token_set(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text or "") if t.strip()}


def _candidate_score(a: str, b: str) -> float:
    a_norm = " ".join(_WORD_RE.findall(str(a or "").lower()))
    b_norm = " ".join(_WORD_RE.findall(str(b or "").lower()))
    if not a_norm or not b_norm:
        return 0.0
    a_tokens = _token_set(a_norm)
    b_tokens = _token_set(b_norm)
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens) or 1
    jaccard = inter / union
    seq = SequenceMatcher(None, a_norm, b_norm).ratio()
    return 0.6 * jaccard + 0.4 * seq


@app.get("/graph/claim/{claim_id}", response_model=schemas.ClaimNodeResponse)
def get_claim_graph_node(claim_id: str):
    node = graph_store.get_claim_node(claim_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Claim not found")
    return {"node": _node_payload(node)}


@app.get("/graph/claim-nodes", response_model=schemas.ClaimNodeListResponse)
def list_claim_graph_nodes(
    doc_id: Optional[str] = None,
    limit: int = Query(200, ge=1, le=5000),
):
    """List available claim nodes for UI bootstrap.

    This exists so the claim graph can load without requiring users to know a
    claim_id string.
    """
    want_doc = str(doc_id or "").strip() or None
    out: list[schemas.ClaimGraphNode] = []
    for node in graph_store.list_claim_nodes():
        payload = _node_payload(node)
        if want_doc:
            props = payload.properties or {}
            if str(props.get("document_id") or "").strip() != want_doc:
                continue
        out.append(payload)
        if len(out) >= int(limit):
            break
    return {"nodes": out, "count": int(len(out))}


@app.post("/graph/resolve-references", response_model=schemas.ReferenceResolveResponse)
def resolve_graph_references(payload: schemas.ReferenceResolveRequest):
    citing = str(payload.citing_doc_id or "").strip()
    mapping: dict[str, Optional[str]] = {}
    for ref_id in payload.reference_ids or []:
        rid = str(ref_id or "").strip()
        if not rid:
            continue
        try:
            ingest_id = graph_store.resolve_reference_to_ingest_id(
                citing_doc_id=citing,
                reference_id=rid,
            )
        except Exception:
            ingest_id = None
        mapping[rid] = str(ingest_id) if ingest_id else None
    return {"citing_doc_id": citing, "mapping": mapping}


@app.post("/graph/reindex-docs")
def reindex_graph_documents():
    """Best-effort reindex of document-level graph state from ingestion store.

    This is safe to run after code changes that affect document/alias inference
    (eg. DOI/bib aliasing). It does not touch span graph tables.
    """
    project_id = str(app_settings.DEFAULT_PROJECT_ID)
    docs: list[dict] = []
    try:
        for brief in list_ingests_from_spine(project_id=project_id, limit=2000):
            did = str(brief.get("id") or "").strip()
            if not did:
                continue
            full = build_ingested_document_from_spine(
                work_id=did,
                project_id=project_id,
                include_extraction_data=True,
            )
            if full is not None:
                docs.append(full)
    except Exception:
        logger.exception("Spine list/get failed for reindex")
    indexed = 0
    errors: list[str] = []
    for doc in docs:
        try:
            graph_store.index_ingest_upload(doc)
        except Exception as exc:
            errors.append(f"ingest_upload:{doc.get('id')}: {exc}")
            continue
        try:
            extraction = (doc.get("extraction") or {}).get("data")
            if isinstance(extraction, dict) and extraction:
                graph_store.index_extraction(
                    ingest_meta=doc, extraction_data=extraction
                )
        except Exception as exc:
            errors.append(f"extraction:{doc.get('id')}: {exc}")
        try:
            resolution = (doc.get("resolution") or {}).get("data")
            if isinstance(resolution, list) and resolution:
                graph_store.index_resolution(
                    ingest_meta=doc, resolution_data=resolution
                )
        except Exception as exc:
            errors.append(f"resolution:{doc.get('id')}: {exc}")
        indexed += 1

    return {"ok": True, "documents": int(indexed), "errors": errors}


@app.post("/ingest/reextract-all")
def reextract_all_ingested_documents(
    background_tasks: BackgroundTasks,
    limit: int = Query(0, ge=0, le=5000),
):
    """Re-run extraction+resolution for all ingested documents.

    Use this after changing extraction/graph indexing logic (eg. DOI/bib aliasing)
    so all stored docs get re-indexed consistently.
    """
    project_id = str(app_settings.DEFAULT_PROJECT_ID)
    docs: list[dict] = []

    try:
        docs = list_ingests_from_spine(project_id=project_id, limit=2000)
    except Exception:
        logger.exception("Spine list failed for reextract-all")
        docs = []

    if int(limit) > 0:
        docs = docs[: int(limit)]
    doc_ids = [
        str(d.get("id") or "").strip() for d in docs if str(d.get("id") or "").strip()
    ]

    scheduled = 0
    for doc_id in doc_ids:
        background_tasks.add_task(extract_ingested_document, doc_id, project_id)
        background_tasks.add_task(resolve_ingested_references, doc_id, project_id)
        scheduled += 1
    return {"ok": True, "scheduled": int(scheduled)}


class DevWipeRequest(BaseModel):
    confirm: str


@app.post("/dev/wipe")
def dev_wipe(payload: DevWipeRequest):
    confirm = str(payload.confirm or "").strip()
    if confirm != "WIPE":
        raise HTTPException(
            status_code=422,
            detail='To wipe everything, set JSON body {"confirm": "WIPE"}',
        )

    # 1) Postgres spine tables.
    #
    # Keep this list aligned with `tests/conftest.py` truncation so local dev
    # and tests have consistent "wipe everything" behavior.
    tables = [
        # 10-04: durable evidence decision events
        "evidence_decision_targets",
        "evidence_decision_events",
        "evidence_decision_streams",
        # 10-01: pipeline stage contracts
        # 10-02: mutable workflow run status
        "pipeline_run_events",
        "pipeline_target_status",
        "pipeline_run_status",
        "pipeline_run_scopes",
        "pipeline_stage_artifacts",
        "pipeline_runs",
        # V2 ingestion spine
        "locators",
        "workflow_versions",
        "workflow_definitions",
        "settings_field_policies",
        "settings_versions",
        "settings_bundles",
        "project_documents",
        "document_versions",
        "documents",
        "artifacts",
        "jobs",
        "attempts",
        "works",
        # 09.3: attachments/evidence/judgments/project meta
        "attachment_artifacts",
        "attachment_events",
        "attachments",
        "background_state",
        "evidence_runs",
        "evidence_selections",
        "judgments",
        "confirmed_claims",
        "project_meta",
        # 09.3-05: durable graph + span graph workboard state
        "graph_edge_votes",
        "graph_edges",
        "graph_aliases",
        "graph_nodes",
        "span_graph_neighborhood_candidates",
        "span_graph_neighborhood_runs",
        "span_graph_review_marks",
        "span_graph_assertions",
        "span_graph_claim_span_atoms",
        "span_graph_claim_atoms",
        "span_graph_claim_spans",
        "span_graph_span_cite_roles",
        "span_graph_citation_span_index",
        "span_graph_span_cites",
        "span_graph_spans",
        "span_graph_work_cites",
        "span_graph_works",
    ]
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE {', '.join(tables)} CASCADE")

    # 2) Object store (MinIO/S3).
    deleted_objects = object_store_s3.delete_all()

    # 3) Graph/claim workboard stores.
    #
    # Historically these were local SQLite stores; as of Phase 09.3 they are
    # spine-backed. Keep these calls as best-effort in case future variants
    # reintroduce disposable local caches.
    try:
        graph_store.wipe()
    except Exception:
        logger.exception("GraphStore wipe failed")
    try:
        span_graph_store.wipe()
    except Exception:
        logger.exception("SpanGraphStore wipe failed")
    try:
        claim_store.wipe()
    except Exception:
        logger.exception("ClaimStore wipe failed")

    # 4) Local disposable caches.
    #
    # Phase 09.3 removes required durable local state under ./data/**, but some
    # optional retrieval indexes (ColBERT) may still be stored locally.
    paths = [
        Path(app_settings.COLBERT_ROOT),
        Path(app_settings.COLBERT_INDEX_PATH),
    ]
    removed = 0
    for p in paths:
        try:
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
                removed += 1
        except Exception:
            pass
    return {
        "ok": True,
        "spine_tables_truncated": True,
        "s3_deleted_objects": int(deleted_objects),
        "removed_paths": int(removed),
    }


# ---------------------------------------------------------------------------
# Phase 10-01: Pipeline stage contract API (minimal POC surface)
# ---------------------------------------------------------------------------


class PipelineRunCreateRequest(BaseModel):
    work_id: str
    caps: Optional[dict] = None
    note: Optional[str] = None


@app.post("/pipeline/runs")
def post_pipeline_run(payload: PipelineRunCreateRequest):
    try:
        run = pipeline_contracts_service.create_run(
            work_id=str(payload.work_id),
            caps_override=dict(payload.caps or {})
            if payload.caps is not None
            else None,
            note=str(payload.note).strip() if payload.note is not None else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return run


class PipelineStagePutRequest(BaseModel):
    status: str
    data: dict = {}
    warnings: Optional[list[str]] = None
    error: Optional[dict] = None
    component: Optional[dict] = None
    caps: Optional[dict] = None


@app.put("/pipeline/runs/{run_id}/stages/{stage}")
def put_pipeline_stage(run_id: str, stage: str, payload: PipelineStagePutRequest):
    try:
        stored = pipeline_contracts_service.store_stage(
            run_id=str(run_id),
            stage=str(stage),
            status=str(payload.status),
            data=dict(payload.data or {}),
            warnings=list(payload.warnings or [])
            if payload.warnings is not None
            else None,
            error=dict(payload.error or {}) if payload.error is not None else None,
            component=dict(payload.component or {})
            if payload.component is not None
            else None,
            caps_override=dict(payload.caps or {})
            if payload.caps is not None
            else None,
        )
    except StageArtifactAlreadyExists as exc:
        return JSONResponse(
            status_code=409,
            content={
                "error": {
                    "code": "stage_artifact_exists",
                    "message": str(exc),
                }
            },
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return stored


@app.get("/pipeline/runs/{run_id}/stages/{stage}")
def get_pipeline_stage(run_id: str, stage: str):
    try:
        artifact = pipeline_contracts_service.fetch_stage(str(run_id), str(stage))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if artifact is None:
        raise HTTPException(status_code=404, detail="Stage artifact not found")
    return artifact


# ---------------------------------------------------------------------------
# Phase 10-02: Happy-path workflow API (runs + status + events)
# ---------------------------------------------------------------------------


class WorkflowStartRunRequest(BaseModel):
    reviewer_uid: str
    citing_doc_id: str


@app.post("/workflow/claimspans/{claim_id}/runs")
def workflow_start_run(claim_id: str, payload: WorkflowStartRunRequest):
    try:
        out = happy_path_orchestrator.start_run_for_claimspan(
            claim_id=str(claim_id),
            reviewer_uid=str(payload.reviewer_uid),
            citing_doc_id=str(payload.citing_doc_id),
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return out


class WorkflowAssessmentFinalizeRequest(BaseModel):
    reviewer_uid: str
    citing_doc_id: str
    assessed_at: str
    rollup_label: Optional[str] = None
    by_target: Optional[dict] = None
    judgment_snapshot: dict = Field(default_factory=dict)


@app.post("/workflow/claimspans/{claim_id}/assessment/finalize")
def workflow_finalize_assessment(
    claim_id: str, payload: WorkflowAssessmentFinalizeRequest
):
    cid = str(claim_id or "").strip()
    reviewer_uid = str(payload.reviewer_uid or "").strip() or "default"
    citing_doc_id = str(payload.citing_doc_id or "").strip()
    assessed_at = str(payload.assessed_at or "").strip()
    if not cid:
        raise HTTPException(status_code=422, detail="claim_id is required")
    if not citing_doc_id:
        raise HTTPException(status_code=422, detail="citing_doc_id is required")
    if not assessed_at:
        raise HTTPException(status_code=422, detail="assessed_at is required")

    run_id = run_scopes_spine.latest_run_id(
        scope_type="claimspan",
        scope_id=cid,
        reviewer_uid=reviewer_uid,
    )
    if run_id is None:
        run = pipeline_contracts_service.create_run(
            work_id=work_id_from_doc_id(citing_doc_id),
            note="assessment-only",
        )
        run_id = str(run.get("run_id") or "").strip() or None
        if not run_id:
            raise HTTPException(
                status_code=500, detail="create_run did not return run_id"
            )
        run_scopes_spine.insert_scope(
            scope_type="claimspan",
            scope_id=cid,
            reviewer_uid=reviewer_uid,
            run_id=run_id,
            citing_doc_id=citing_doc_id,
        )
        run_status_spine.upsert_run_status(
            run_id,
            scope_type="claimspan",
            scope_id=cid,
            reviewer_uid=reviewer_uid,
            citing_doc_id=citing_doc_id,
            state="queued"
            if not background_state.get_state().get("paused")
            else "blocked",
        )

    assessment_data = {
        "reviewer_uid": reviewer_uid,
        "scope_id": cid,
        "citing_doc_id": citing_doc_id,
        "assessed_at": assessed_at,
        "rollup_label": (
            str(payload.rollup_label).strip()
            if payload.rollup_label is not None
            else None
        )
        or None,
        "by_target": dict(payload.by_target or {}),
        "judgment_snapshot": dict(payload.judgment_snapshot or {}),
    }

    try:
        pipeline_contracts_service.store_stage(
            run_id=str(run_id),
            stage="assessment",
            status="complete",
            data=assessment_data,
        )
        already = False
    except StageArtifactAlreadyExists:
        already = True
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Best-effort: reflect assessment in per-target stage_state_json.
    try:
        for tgt in run_status_spine.list_target_status(str(run_id)):
            tid = str((tgt or {}).get("target_id") or "").strip()
            if not tid:
                continue
            run_status_spine.upsert_target_status(
                str(run_id),
                tid,
                state=str((tgt or {}).get("state") or "requested"),
                citation_index=(tgt or {}).get("citation_index"),
                reference_id=(tgt or {}).get("reference_id"),
                attachment_id=(tgt or {}).get("attachment_id"),
                stage_state={
                    "stages": {
                        "assessment": {
                            "state": "done",
                            "started_at": assessed_at,
                            "finished_at": assessed_at,
                            "message": "finalized",
                        }
                    }
                },
            )
    except Exception:
        pass

    return {"run_id": run_id, "already_stored": bool(already)}


@app.post("/workflow/runs/{run_id}/resume")
def workflow_resume_run(run_id: str):
    try:
        return happy_path_orchestrator.resume_run(str(run_id))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/workflow/claimspans/{claim_id}/runs/latest")
def workflow_latest_run(claim_id: str, reviewer_uid: str = Query(...)):
    try:
        rid = run_scopes_spine.latest_run_id(
            scope_type="claimspan",
            scope_id=str(claim_id),
            reviewer_uid=str(reviewer_uid),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"run_id": rid}


@app.get("/workflow/runs/{run_id}/status")
def workflow_get_run_status(run_id: str):
    try:
        run = run_status_spine.get_run_status(str(run_id))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")

    targets = run_status_spine.list_target_status(str(run_id))
    queue = [
        {
            "target_id": t.get("target_id"),
            "reference_id": t.get("reference_id"),
            "citation_index": t.get("citation_index"),
            "attachment_id": t.get("attachment_id"),
            "state": t.get("state"),
            "updated_at": t.get("updated_at"),
        }
        for t in targets
        if isinstance(t, dict)
    ]
    return {"run": run, "targets": targets, "queue": queue}


@app.post("/workflow/runs/{run_id}/targets/{target_id}/cancel")
def workflow_cancel_target(run_id: str, target_id: str):
    try:
        return happy_path_orchestrator.cancel_target(str(run_id), str(target_id))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/workflow/runs/{run_id}/events")
def workflow_events_sse(
    run_id: str,
    after_event_id: int = Query(0, ge=0),
    heartbeat_ms: int = Query(10000, ge=250, le=60000),
):
    rid = str(run_id or "").strip()
    if not rid:
        raise HTTPException(status_code=422, detail="run_id is required")

    async def _gen():
        last = int(after_event_id)
        heartbeat_s = max(0.25, float(heartbeat_ms) / 1000.0)
        while True:
            events = []
            try:
                events = run_status_spine.list_events(
                    rid, after_event_id=last, limit=200
                )
            except Exception:
                events = []

            if events:
                for ev in events:
                    if not isinstance(ev, dict):
                        continue
                    last = int(ev.get("event_id") or last)
                    et = str(ev.get("type") or "message").strip() or "message"
                    payload = json.dumps(ev, ensure_ascii=True, sort_keys=True)
                    yield f"event: {et}\n"
                    yield f"data: {payload}\n\n"
            else:
                yield ": heartbeat\n\n"
            await asyncio.sleep(heartbeat_s)

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.get(
    "/spans/lookup-citation-window",
    response_model=schemas.CitationSpanLookupResponse,
)
def lookup_citation_window_span(
    ingest_id: str,
    citation_index: int,
    target_id: Optional[str] = None,
):
    ingest = str(ingest_id or "").strip()
    if not ingest:
        raise HTTPException(status_code=422, detail="ingest_id is required")
    span = span_graph_store.find_citation_span(
        ingest_id=ingest,
        citation_index=int(citation_index),
        target_id=str(target_id).strip() or None,
    )
    return {
        "ingest_id": ingest,
        "citation_index": int(citation_index),
        "target_id": str(target_id).strip() or None,
        "span_id": (str(span.get("span_id")) if span else None),
    }


@app.get(
    "/graph/edge/{edge_id}/votes", response_model=schemas.ClaimGraphVoteListResponse
)
def get_claim_graph_edge_votes(edge_id: int):
    votes = graph_store.list_edge_votes(int(edge_id))
    return {
        "edge_id": int(edge_id),
        "votes": [
            schemas.ClaimGraphVote(
                reviewer_uid=str(v.get("reviewer_uid") or "default"),
                verdict=str(v.get("verdict") or "neutral"),
                confidence=v.get("confidence"),
                comment=v.get("comment"),
                updated_at=v.get("updated_at"),
            )
            for v in votes
        ],
    }


@app.put(
    "/graph/edge/{edge_id}/vote",
    response_model=schemas.ClaimGraphVoteUpsertResponse,
)
def put_claim_graph_edge_vote(
    edge_id: int,
    payload: schemas.ClaimGraphVoteUpsertRequest,
    reviewer_uid: str = "default",
):
    try:
        stored = graph_store.upsert_edge_vote(
            edge_id=int(edge_id),
            reviewer_uid=reviewer_uid,
            verdict=payload.verdict,
            confidence=payload.confidence,
            comment=payload.comment,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    aggregates = graph_store.edge_vote_aggregates(int(edge_id))
    return {
        "edge_id": int(edge_id),
        "vote": schemas.ClaimGraphVote(
            reviewer_uid=str(stored.get("reviewer_uid") or "default"),
            verdict=str(stored.get("verdict") or "neutral"),
            confidence=stored.get("confidence"),
            comment=stored.get("comment"),
            updated_at=stored.get("updated_at"),
        ),
        "aggregates": aggregates,
    }


@app.post("/graph/claim-link", response_model=schemas.ClaimLinkCreateResponse)
def post_claim_link(
    payload: schemas.ClaimLinkCreateRequest, reviewer_uid: str = "default"
):
    reviewer = str(reviewer_uid or "").strip() or "default"
    try:
        edge_id = graph_store.upsert_claim_link(
            source_claim_id=payload.source_claim_id,
            target_claim_id=payload.target_claim_id,
            source="manual",
            creator_uid=reviewer,
            explored_by=[reviewer],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    edge = None
    for cand in graph_store.list_claim_links_for_claim(
        claim_id=payload.source_claim_id, enabled_only=False
    ):
        if int(cand.get("edge_id") or 0) == int(edge_id):
            edge = cand
            break
    if edge is None:
        for cand in graph_store.list_claim_links_for_claim(
            claim_id=payload.target_claim_id, enabled_only=False
        ):
            if int(cand.get("edge_id") or 0) == int(edge_id):
                edge = cand
                break
    if edge is None:
        raise HTTPException(status_code=500, detail="Created edge not readable")

    return {
        "edge": _edge_payload(
            edge, aggregates=graph_store.edge_vote_aggregates(int(edge_id))
        )
    }


@app.delete(
    "/graph/claim-link/{edge_id}",
    response_model=schemas.ClaimLinkDeleteResponse,
)
def delete_claim_link(edge_id: int, reviewer_uid: str = "default"):
    try:
        graph_store.delete_claim_link(edge_id=int(edge_id), reviewer_uid=reviewer_uid)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"ok": True}


@app.get("/graph/claim-subgraph", response_model=schemas.ClaimSubgraphResponse)
def get_claim_subgraph(
    center_claim_id: str,
    hops: int = Query(1, ge=1, le=3),
    edge_cap: int = Query(25, ge=1, le=250),
    min_votes: int = Query(0, ge=0, le=1000),
    sources: str = "auto,manual,external_search",
):
    allowed_sources = {"auto", "manual", "external_search"}
    want_sources = {
        s.strip()
        for s in (sources or "").split(",")
        if s is not None and str(s).strip()
    }
    if not want_sources:
        want_sources = set(allowed_sources)
    unknown = sorted(want_sources - allowed_sources)
    if unknown:
        raise HTTPException(
            status_code=422, detail=f"Unknown sources: {', '.join(unknown)}"
        )

    center = str(center_claim_id or "").strip()
    if not center:
        raise HTTPException(status_code=422, detail="center_claim_id is required")
    if graph_store.get_claim_node(center) is None:
        raise HTTPException(status_code=404, detail="Claim not found")

    visited: set[str] = {center}
    frontier: list[str] = [center]
    edges_by_id: dict[int, tuple[dict, dict]] = {}

    for _depth in range(int(hops)):
        next_frontier: list[str] = []
        for claim_id in frontier:
            edges = graph_store.list_claim_links_for_claim(
                claim_id=claim_id,
                sources=sorted(want_sources),
                enabled_only=True,
            )
            if edge_cap and len(edges) > int(edge_cap):
                edges = edges[: int(edge_cap)]

            for edge in edges:
                edge_id = int(edge.get("edge_id") or 0)
                if edge_id <= 0:
                    continue
                if edge_id in edges_by_id:
                    continue
                aggs = graph_store.edge_vote_aggregates(edge_id)
                if int(aggs.get("n_total") or 0) < int(min_votes):
                    continue
                edges_by_id[edge_id] = (edge, aggs)

                src = str(edge.get("source_id") or "")
                tgt = str(edge.get("target_id") or "")
                neighbor = tgt if src == claim_id else src
                if neighbor and neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append(neighbor)

        if not next_frontier:
            break
        frontier = next_frontier

    nodes: list[schemas.ClaimGraphNode] = []
    if center in visited:
        visited.remove(center)
        visited_order = [center] + sorted(visited)
    else:
        visited_order = sorted(visited)
    for node_id in visited_order:
        node = graph_store.get_claim_node(node_id)
        if node is None:
            continue
        nodes.append(_node_payload(node))

    edges: list[schemas.ClaimGraphEdge] = []
    for edge_id in sorted(edges_by_id.keys()):
        edge, aggs = edges_by_id[edge_id]
        edges.append(_edge_payload(edge, aggregates=aggs))

    return {"center_claim_id": center, "nodes": nodes, "edges": edges}


@app.get(
    "/graph/claim/{claim_id}/candidates",
    response_model=schemas.ClaimCandidatesResponse,
)
def get_claim_candidates(claim_id: str, limit: int = Query(10, ge=1, le=100)):
    node = graph_store.get_claim_node(claim_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Claim not found")
    base_text = _claim_text(node)

    existing_neighbors: set[str] = set()
    for edge in graph_store.list_claim_links_for_claim(
        claim_id=claim_id, enabled_only=True
    ):
        src = str(edge.get("source_id") or "")
        tgt = str(edge.get("target_id") or "")
        if src and src != claim_id:
            existing_neighbors.add(src)
        if tgt and tgt != claim_id:
            existing_neighbors.add(tgt)

    scored: list[tuple[float, str, dict]] = []
    for cand in graph_store.list_claim_nodes():
        cand_id = str(cand.get("node_id") or "")
        if not cand_id or cand_id == claim_id:
            continue
        if cand_id in existing_neighbors:
            continue
        score = _candidate_score(base_text, _claim_text(cand))
        if score <= 0:
            continue
        scored.append((score, cand_id, cand))

    scored.sort(key=lambda t: (-t[0], t[1]))
    top = scored[: int(limit)]

    return {
        "claim_id": str(claim_id),
        "candidates": [
            {
                "target_claim_id": cand_id,
                "score": float(score),
                "node": _node_payload(cand),
            }
            for score, cand_id, cand in top
        ],
    }


# --- Span-first graph (Rebuild) ----------------------------------------------


@app.post("/spans/upsert", response_model=schemas.SpanUpsertResponse)
def upsert_span(payload: schemas.SpanUpsertRequest):
    try:
        span = span_graph_store.upsert_span(
            kind=str(payload.kind),
            selector=payload.selector.model_dump(),
            window_fingerprint=payload.window_fingerprint,
            ingest_id=payload.ingest_id,
            work_id=payload.work_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"span": span}


@app.post(
    "/spans/{span_id}/cites",
    response_model=schemas.SpanCitesUpsertResponse,
)
def upsert_span_cites(span_id: str, payload: schemas.SpanCitesUpsertRequest):
    inserted = span_graph_store.add_span_cites(
        span_id=str(span_id),
        cites=[c.model_dump() for c in (payload.cites or [])],
    )
    return {"span_id": str(span_id), "inserted": int(inserted)}


@app.put(
    "/spans/{span_id}/cites/{cited_work_id:path}/role",
    response_model=schemas.OkResponse,
)
def set_span_cite_role(
    span_id: str,
    cited_work_id: str,
    payload: schemas.SpanCiteRoleUpsertRequest,
):
    span_graph_store.set_span_cite_role(
        span_id=str(span_id),
        cited_work_id=str(cited_work_id),
        reviewer_uid=str(payload.reviewer_uid),
        role=str(payload.role),
    )
    return {"ok": True}


@app.post(
    "/spans/{span_id}/claim-spans",
    response_model=schemas.ClaimSpansUpsertResponse,
)
def upsert_claim_spans(span_id: str, payload: schemas.ClaimSpansUpsertRequest):
    items = []
    for cs in payload.claim_spans or []:
        selector = cs.selector.model_dump() if cs.selector is not None else None
        items.append({"order_index": int(cs.order_index), "selector": selector})
    claim_spans = span_graph_store.upsert_claim_spans(span_id=str(span_id), items=items)
    return {"span_id": str(span_id), "claim_spans": claim_spans}


def _topology_edge_payload(edge_id: int) -> schemas.TopologyEdgePayload:
    edge = graph_store.get_edge(int(edge_id)) or {}
    aggs = graph_store.edge_vote_aggregates(int(edge_id))
    return schemas.TopologyEdgePayload(
        edge_id=int(edge_id),
        kind=str(edge.get("kind") or ""),
        source_id=str(edge.get("source_id") or ""),
        target_id=str(edge.get("target_id") or ""),
        enabled=bool(edge.get("enabled")),
        properties=(edge.get("properties") or {}),
        aggregates=schemas.ClaimGraphEdgeAggregates(**aggs),
    )


@app.post("/claim-atoms", response_model=schemas.ClaimAtomCreateResponse)
def create_claim_atom(payload: schemas.ClaimAtomCreateRequest):
    reviewer_uid = str(payload.reviewer_uid or "default").strip() or "default"
    try:
        atom = span_graph_store.create_claim_atom(
            text=str(payload.text or ""),
            created_by=reviewer_uid,
            supersedes_id=str(payload.supersedes_id).strip()
            if payload.supersedes_id
            else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"atom": atom}


@app.get("/claim-atoms/{claim_atom_id}", response_model=schemas.ClaimAtomCreateResponse)
def get_claim_atom(claim_atom_id: str):
    atom = span_graph_store.get_claim_atom(str(claim_atom_id))
    if not atom:
        raise HTTPException(status_code=404, detail="Claim atom not found")
    return {"atom": atom}


@app.post(
    "/claim-spans/{claim_span_id}/atoms",
    response_model=schemas.TopologyEdgePayload,
)
def link_claim_span_atom(claim_span_id: str, payload: schemas.ClaimSpanAtomLinkRequest):
    reviewer_uid = str(payload.reviewer_uid or "default").strip() or "default"
    csid = str(claim_span_id)
    aid = str(payload.claim_atom_id or "").strip()
    if not aid:
        raise HTTPException(status_code=422, detail="claim_atom_id is required")
    try:
        span_graph_store.link_claim_span_atom(claim_span_id=csid, claim_atom_id=aid)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    edge_id = graph_store.upsert_topology_edge(
        source_id=csid,
        target_id=aid,
        kind="CLAIMSPAN_EXPRESSES_ATOM",
        source=str(payload.source or "manual"),
        creator_uid=reviewer_uid,
        enabled=True,
    )
    return _topology_edge_payload(edge_id)


@app.get(
    "/claim-spans/{claim_span_id}/atoms",
    response_model=schemas.ClaimSpanAtomsResponse,
)
def list_claim_span_atoms(claim_span_id: str):
    csid = str(claim_span_id)
    atoms = span_graph_store.list_claim_span_atoms(claim_span_id=csid)
    edges = graph_store.list_edges(
        kind="CLAIMSPAN_EXPRESSES_ATOM",
        source_id=csid,
        include_disabled=True,
    )
    payload_edges = []
    for e in edges:
        edge_id = int(e.get("edge_id") or 0)
        if not edge_id:
            continue
        payload_edges.append(_topology_edge_payload(edge_id))
    return {"claim_span_id": csid, "atoms": atoms, "edges": payload_edges}


@app.post(
    "/claim-atoms/{claim_atom_id}/align",
    response_model=schemas.TopologyEdgePayload,
)
def align_claim_atom_to_claim(
    claim_atom_id: str, payload: schemas.AtomAlignClaimRequest
):
    reviewer_uid = str(payload.reviewer_uid or "default").strip() or "default"
    aid = str(claim_atom_id)
    cid = str(payload.claim_id or "").strip()
    if not cid:
        raise HTTPException(status_code=422, detail="claim_id is required")

    edge_id = graph_store.upsert_topology_edge(
        source_id=aid,
        target_id=cid,
        kind="ATOM_ALIGNS_TO_CLAIM",
        source=str(payload.source or "manual"),
        creator_uid=reviewer_uid,
        enabled=True,
    )
    return _topology_edge_payload(edge_id)


@app.post("/topology/settle", response_model=schemas.TopologySettleResponse)
def topology_settle(payload: schemas.TopologySettleRequest):
    kind = str(payload.kind or "").strip()
    if not kind:
        raise HTTPException(status_code=422, detail="kind is required")

    policy = payload.policy
    edges = graph_store.list_edges_by_kind(kind=kind, include_disabled=True)
    evaluated = 0
    enabled_n = 0
    disabled_n = 0
    unchanged_n = 0

    for e in edges:
        edge_id = int(e.get("edge_id") or 0)
        if not edge_id:
            continue
        evaluated += 1
        props = e.get("properties") or {}
        aggs = graph_store.edge_vote_aggregates(edge_id)
        n_total = int(aggs.get("n_total") or 0)
        n_support = int(aggs.get("n_support") or 0)
        n_contra = int(aggs.get("n_contradict") or 0)

        if n_total < int(policy.min_total_votes or 1):
            unchanged_n += 1
            continue

        decision = None
        if bool(policy.contradict_veto) and n_contra > 0 and n_support == 0:
            decision = False
        elif (n_support - n_contra) >= int(
            policy.support_margin or 0
        ) and n_support >= int(policy.min_support or 0):
            decision = True
        elif (n_contra - n_support) >= int(
            policy.support_margin or 0
        ) and n_contra >= int(policy.min_contradict or 0):
            decision = False

        if decision is None:
            unchanged_n += 1
            continue

        current_enabled = bool(e.get("enabled"))
        if decision is False and current_enabled and bool(policy.manual_lock):
            if str(props.get("source") or "") == "manual":
                unchanged_n += 1
                continue

        if decision is True and not current_enabled:
            enabled_n += 1
        elif decision is False and current_enabled:
            disabled_n += 1
        else:
            unchanged_n += 1
            continue

        if not bool(payload.dry_run):
            graph_store.set_edge_enabled(
                edge_id=edge_id,
                enabled=bool(decision),
                merge_properties={
                    "settled_at": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "settled_by": "topology/settle",
                    "settled_kind": kind,
                    "settled_policy": policy.model_dump(),
                    "settled_counts": aggs,
                },
            )

    return {
        "kind": kind,
        "dry_run": bool(payload.dry_run),
        "evaluated": int(evaluated),
        "enabled": int(enabled_n),
        "disabled": int(disabled_n),
        "unchanged": int(unchanged_n),
    }


@app.post("/assertions", response_model=schemas.AssertionCreateResponse)
def create_assertion(payload: schemas.AssertionCreateRequest):
    try:
        assertion = span_graph_store.create_assertion(payload=payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"assertion": assertion}


@app.get(
    "/claim-spans/{claim_span_id}/assertions",
    response_model=schemas.ClaimSpanAssertionsResponse,
)
def list_claim_span_assertions(claim_span_id: str, reviewer_uid: Optional[str] = None):
    assertions = span_graph_store.list_assertions_for_claim_span(
        claim_span_id=str(claim_span_id),
        reviewer_uid=str(reviewer_uid) if reviewer_uid else None,
    )
    return {"claim_span_id": str(claim_span_id), "assertions": assertions}


@app.get(
    "/claim-spans/{claim_span_id}/status",
    response_model=schemas.ClaimSpanStatusResponse,
)
def get_claim_span_status(claim_span_id: str, reviewer_uid: str = "default"):
    return span_graph_store.claim_span_status(
        claim_span_id=str(claim_span_id),
        reviewer_uid=str(reviewer_uid),
    )


@app.get(
    "/spans/{span_id}/status",
    response_model=schemas.SpanStatusResponse,
)
def get_span_status(span_id: str, reviewer_uid: str = "default"):
    return span_graph_store.span_status(
        span_id=str(span_id), reviewer_uid=str(reviewer_uid)
    )


@app.get(
    "/spans/{span_id}/bundle",
    response_model=schemas.SpanBundleResponse,
)
def get_span_bundle(
    span_id: str,
    reviewer_uid: str = "default",
    include_history: bool = False,
):
    bundle = span_graph_store.span_bundle(
        span_id=str(span_id),
        reviewer_uid=str(reviewer_uid),
        include_history=bool(include_history),
    )
    if not bundle:
        raise HTTPException(status_code=404, detail="Span not found")
    return bundle


@app.post(
    "/maintenance/span-graph/compact",
    response_model=schemas.SpanGraphCompactResponse,
)
def compact_span_graph(payload: schemas.SpanGraphCompactRequest):
    return span_graph_store.compact_assertions(
        dry_run=bool(payload.dry_run),
        aggressive=bool(payload.aggressive),
    )


def _lexical_similarity(query: str, text: str) -> float:
    q = {t for t in re.findall(r"[a-z0-9]{3,}", (query or "").lower())}
    h = {t for t in re.findall(r"[a-z0-9]{3,}", (text or "").lower())}
    if not q or not h:
        return 0.0
    inter = len(q & h)
    denom = max(1, min(len(q), len(h)))
    return max(0.0, min(1.0, inter / denom))


@app.post(
    "/neighborhood/search",
    response_model=schemas.NeighborhoodSearchResponse,
)
def neighborhood_search(payload: schemas.NeighborhoodSearchRequest):
    span_id = str(payload.span_id or "").strip()
    reviewer_uid = str(payload.reviewer_uid or "default").strip() or "default"
    bundle = span_graph_store.span_bundle(
        span_id=span_id,
        reviewer_uid=reviewer_uid,
        include_history=False,
    )
    if not bundle:
        raise HTTPException(status_code=404, detail="Span not found")
    seeds = [
        c.get("cited_work_id")
        for c in (bundle.get("cites") or [])
        if c.get("cited_work_id")
    ]
    seeds = [str(s).strip() for s in seeds if str(s).strip()]
    if not seeds:
        run_id = span_graph_store.create_neighborhood_run(
            created_by=reviewer_uid,
            context_span_id=span_id,
            method="openalex_cited_by_intersection",
            params=payload.model_dump(),
        )
        return {
            "run_id": run_id,
            "span_id": span_id,
            "reviewer_uid": reviewer_uid,
            "candidates": [],
        }

    max_per_seed = int(payload.max_per_seed)
    counts: Dict[str, int] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for seed in seeds:
        try:
            citing = citation_graph.fetch_cited_by(seed, max_nodes=max_per_seed)
        except Exception:
            continue
        for work in citing:
            work_id = citation_graph._openalex_id(work.get("id")) or work.get("id")
            if not work_id:
                continue
            counts[work_id] = counts.get(work_id, 0) + 1
            meta.setdefault(work_id, work)

    # Score + filter.
    query_text = (payload.query_text or "").strip()
    scored = []
    for work_id, bib in counts.items():
        if bib < int(payload.min_bib_intersection):
            continue
        work = meta.get(work_id) or {}
        title = work.get("display_name") or work.get("title") or ""
        abstract = citation_graph.extract_abstract(work) or ""
        sim = (
            _lexical_similarity(query_text, f"{title} {abstract}")
            if query_text
            else 0.0
        )
        if query_text and sim < float(payload.min_abstract_score):
            continue
        scored.append((bib, sim, work_id, work, title))
    scored.sort(key=lambda row: (-row[0], -row[1], row[2]))

    candidates_payload = []
    for rank, (bib, sim, work_id, work, title) in enumerate(scored[:200], start=1):
        doi = work.get("doi")
        year = work.get("publication_year")
        span_graph_store.upsert_work(
            work_id=str(work_id),
            doi=str(doi) if doi else None,
            openalex_id=str(work_id),
            title=str(title) if title else None,
            year=str(year) if year else None,
            abstract=citation_graph.extract_abstract(work),
            abstract_source="openalex",
        )
        candidates_payload.append(
            {
                "work_id": str(work_id),
                "bib_intersection": int(bib),
                "abstract_score": float(sim) if query_text else None,
                "rank": int(rank),
                "title": str(title) if title else None,
                "doi": str(doi) if doi else None,
                "year": str(year) if year else None,
            }
        )

    run_id = span_graph_store.create_neighborhood_run(
        created_by=reviewer_uid,
        context_span_id=span_id,
        method="openalex_cited_by_intersection",
        params=payload.model_dump(),
    )
    span_graph_store.add_neighborhood_candidates(
        run_id=run_id,
        candidates=[
            {
                "work_id": c["work_id"],
                "bib_intersection": c["bib_intersection"],
                "abstract_score": c.get("abstract_score"),
                "rank": c["rank"],
                "detail": {},
            }
            for c in candidates_payload
        ],
    )
    return {
        "run_id": run_id,
        "span_id": span_id,
        "reviewer_uid": reviewer_uid,
        "candidates": candidates_payload,
    }


@app.get(
    "/neighborhood/{run_id}",
    response_model=schemas.NeighborhoodRunResponse,
)
def get_neighborhood_run(run_id: str, limit: int = 50):
    run = span_graph_store.get_neighborhood_run(run_id=str(run_id))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    candidates = span_graph_store.list_neighborhood_candidates(
        run_id=str(run_id), limit=int(limit)
    )
    out = []
    for c in candidates:
        work = span_graph_store.get_work(str(c.get("candidate_work_id"))) or {}
        out.append(
            {
                "work_id": str(c.get("candidate_work_id")),
                "bib_intersection": int(c.get("bib_intersection") or 0),
                "abstract_score": c.get("abstract_score"),
                "rank": int(c.get("rank") or 0),
                "title": work.get("title"),
                "doi": work.get("doi"),
                "year": work.get("year"),
            }
        )
    return {"run": run, "candidates": out}


@app.get(
    "/claims/{claim_id}/span-context",
    response_model=schemas.ClaimSpanContextResponse,
)
def get_claim_span_context(
    claim_id: str,
    target_id: Optional[str] = None,
):
    parsed = span_graph_store.parse_cite_claim_id(claim_id)
    if not parsed:
        raise HTTPException(status_code=422, detail="Unsupported claim id")

    ingest_id = str(parsed.get("document_id") or "").strip()
    citation_index = int(parsed.get("citation_index") or 0)
    reviewer_uid = str(parsed.get("reviewer_uid") or "default")
    order_index = parsed.get("order_index")
    if order_index is None:
        raise HTTPException(status_code=422, detail="Claim segment id not parseable")

    resolved_target = str(target_id or "").strip() or None
    if not resolved_target:
        for rec in attachment_store.list_attachments(claim_id=claim_id):
            tid = str(rec.get("target_id") or "").strip() or None
            if tid:
                resolved_target = tid
                break

    span = span_graph_store.find_citation_span(
        ingest_id=ingest_id,
        citation_index=citation_index,
        target_id=resolved_target,
    )
    if not span:
        raise HTTPException(status_code=404, detail="Span not found")
    claim_span = span_graph_store.get_claim_span(
        span_id=str(span["span_id"]),
        order_index=int(order_index),
    )
    if not claim_span:
        raise HTTPException(status_code=404, detail="Claim span not found")

    cited_work_id = None
    try:
        for cite in span_graph_store.list_span_cites(span_id=str(span["span_id"])):
            if resolved_target and str(cite.get("reference_id") or "").strip() != str(
                resolved_target
            ):
                continue
            cited_work_id = str(cite.get("cited_work_id") or "").strip() or None
            if cited_work_id:
                break
    except Exception:
        cited_work_id = None
    if not cited_work_id and resolved_target:
        cited_work_id = f"ref:{ingest_id}:{resolved_target}"
    return {
        "claim_id": str(claim_id),
        "reviewer_uid": reviewer_uid,
        "ingest_id": ingest_id,
        "citation_index": int(citation_index),
        "target_id": resolved_target,
        "span_id": str(span["span_id"]),
        "claim_span_id": str(claim_span["claim_span_id"]),
        "order_index": int(order_index),
        "cited_work_id": cited_work_id,
    }


@app.get(
    "/claims/{claim_id}/status",
    response_model=schemas.ClaimStatusResponse,
)
def get_claim_status(
    claim_id: str,
    target_id: Optional[str] = None,
):
    context = get_claim_span_context(claim_id=claim_id, target_id=target_id)
    status = span_graph_store.claim_span_status(
        claim_span_id=str(context["claim_span_id"]),
        reviewer_uid=str(context["reviewer_uid"]),
    )
    return {
        "claim_id": str(claim_id),
        "reviewer_uid": str(context["reviewer_uid"]),
        "span_id": str(context["span_id"]),
        "claim_span_id": str(context["claim_span_id"]),
        "status": str(status["status"]),
        "checked": bool(status.get("checked")),
    }


@app.post("/ingest/{doc_id}/extract", response_model=schemas.ExtractionResponse)
def extract_ingested_document(
    doc_id: str,
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)
    user_id = str(app_settings.DEFAULT_USER_ID)

    try:
        document = build_ingested_document_from_spine(
            work_id=str(doc_id),
            project_id=project_id,
            include_extraction_data=True,
        )
    except Exception as exc:
        logger.exception("Spine get failed for doc_id=%s", doc_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    work_id = work_id_from_doc_id(doc_id)
    extraction_stage = document.get("extraction") or {}
    extraction_data = (
        (extraction_stage.get("data") or {})
        if isinstance(extraction_stage, dict)
        else {}
    )
    extraction_status = (
        str(extraction_stage.get("status") or "").strip()
        if isinstance(extraction_stage, dict)
        else ""
    )
    if extraction_status == "running" and not extraction_data:
        return {"document_id": doc_id, "extraction": document.get("extraction")}
    if extraction_status == "complete" and extraction_data:
        return {"document_id": doc_id, "extraction": document.get("extraction")}

    try:
        attempt = get_latest_attempt_for_work(
            project_id=project_id,
            work_id=work_id,
            kind="primary",
        )
        if attempt and str(attempt.get("state") or "") in {"running"}:
            return {"document_id": doc_id, "extraction": {"status": "running"}}
    except Exception:
        pass

    job_id = spine_extract_pool.enqueue(
        doc_id=work_id,
        project_id=project_id,
        user_id=user_id,
        force_fallback=False,
    )
    if not job_id:
        raise HTTPException(status_code=503, detail="Unable to enqueue extraction")
    return {"document_id": doc_id, "extraction": {"status": "running"}}


@app.post("/ingest/{doc_id}/fallback-extract")
def fallback_extract_ingested_document(
    doc_id: str,
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    """Force a fallback extraction attempt (spine mode only)."""
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)
    user_id = str(app_settings.DEFAULT_USER_ID)

    work_id = str(doc_id or "").strip()
    if not work_id:
        raise HTTPException(status_code=400, detail="doc_id is required")

    job_id = spine_extract_pool.enqueue(
        doc_id=work_id,
        project_id=project_id,
        user_id=user_id,
        force_fallback=True,
    )
    if not job_id:
        raise HTTPException(
            status_code=503, detail="Unable to enqueue fallback extraction"
        )
    return {"document_id": work_id, "job_id": job_id}


@app.post("/ingest/{doc_id}/extract/cancel")
def cancel_extraction(
    doc_id: str,
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)
    wid = str(doc_id or "").strip()
    if not wid:
        raise HTTPException(status_code=400, detail="doc_id is required")

    attempt = get_latest_attempt_for_work(
        project_id=project_id, work_id=wid, kind="primary"
    )
    if not attempt or not attempt.get("attempt_id"):
        raise HTTPException(status_code=404, detail="No attempt found")
    try:
        mark_attempt_cancelled(str(attempt["attempt_id"]))
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"Cancel failed: {exc}")
    return {
        "document_id": wid,
        "attempt_id": str(attempt["attempt_id"]),
        "state": "cancelled",
    }


@app.get("/ingest/{doc_id}/extraction", response_model=schemas.ExtractionResult)
def get_ingested_extraction(
    doc_id: str,
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)
    try:
        doc = build_ingested_document_from_spine(
            work_id=str(doc_id),
            project_id=project_id,
            include_extraction_data=True,
        )
    except Exception as exc:
        logger.exception("Spine extraction read failed for doc_id=%s", doc_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc.get("extraction")


@app.get("/ingest/{doc_id}/spine")
def get_ingest_spine_debug(
    doc_id: str,
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    """Debug/verification view of spine attempt + artifacts.

    This is intentionally separate from schemas.IngestedDocument to keep the
    UI payload stable while we iterate on robustness + fallback contracts.
    """
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)
    wid = str(doc_id or "").strip()
    if not wid:
        raise HTTPException(status_code=400, detail="doc_id is required")

    doc = build_ingested_document_from_spine(
        work_id=wid,
        project_id=project_id,
        include_extraction_data=False,
    )
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    attempt = get_latest_attempt_for_work(
        project_id=project_id, work_id=wid, kind="primary"
    )
    artifacts = []
    jobs = []
    if attempt and attempt.get("attempt_id"):
        try:
            artifacts = list_artifacts_for_attempt(str(attempt["attempt_id"]))
        except Exception:
            artifacts = []
        try:
            jobs = list_jobs_for_attempt(str(attempt["attempt_id"]), limit=50)
        except Exception:
            jobs = []

    return {
        "work_id": wid,
        "project_id": project_id,
        "attempt": attempt,
        "artifacts": artifacts,
        "jobs": jobs,
    }


@app.get("/ingest/{doc_id}/body", response_model=schemas.DocumentBodyResponse)
def get_ingested_body(
    doc_id: str,
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)

    tei_xml: Optional[str] = get_tei_xml_from_spine(
        work_id=str(doc_id), project_id=project_id
    )

    tei_payload: Optional[dict] = None
    if tei_xml:
        tei_payload = tei_body.build_document_body(tei_xml)
        # If TEI exists but yields an empty body, prefer fallback artifacts when
        # available (common for refs-only or structure-poor parses).
        if tei_payload.get("paragraphs"):
            return {"document_id": doc_id, **tei_payload}

    # Fallback body from spine artifact.
    attempt = get_latest_attempt_for_work(
        project_id=project_id,
        work_id=str(doc_id),
        kind="primary",
    )
    if attempt and attempt.get("attempt_id"):
        art = None
        try:
            art = get_artifact_for_attempt(
                attempt_id=str(attempt["attempt_id"]),
                artifact_type="fallback.body.txt",
            )
        except Exception:
            art = None
        if art and art.get("object_key"):
            try:
                raw = object_store_s3.get_bytes(str(art["object_key"]))
                text = raw.decode("utf-8", errors="ignore")
                return fallback_body.build_body_from_plaintext(
                    document_id=str(doc_id),
                    text=text,
                )
            except Exception:
                pass

    # TEI exists but was empty and no fallback artifact was found.
    if tei_payload is not None:
        return {"document_id": doc_id, **tei_payload}

    raise HTTPException(status_code=404, detail="TEI XML not found")


@app.post("/ingest/{doc_id}/resolve", response_model=schemas.ResolutionResponse)
def resolve_ingested_references(
    doc_id: str,
    force: bool = Query(False),
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)
    user_id = str(app_settings.DEFAULT_USER_ID)

    try:
        document = build_ingested_document_from_spine(
            work_id=str(doc_id),
            project_id=project_id,
            include_extraction_data=True,
        )
    except Exception as exc:
        logger.exception("Spine get failed for doc_id=%s", doc_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    work_id = work_id_from_doc_id(doc_id)
    attempt_id: Optional[str] = None
    attempt = None
    try:
        attempt = get_latest_attempt_for_work(
            project_id=project_id,
            work_id=work_id,
            kind="primary",
        )
        if attempt and attempt.get("attempt_id"):
            attempt_id = str(attempt["attempt_id"])
    except Exception:
        attempt = None
        attempt_id = None

    extraction = document.get("extraction") or {}
    extraction_data = extraction.get("data") or {}
    references = extraction_data.get("references")
    if references is None:
        references = []
    if not isinstance(references, list):
        raise HTTPException(status_code=422, detail="Invalid extraction references")

    # Resolution gating policy (Phase 09.2-07).
    if attempt is None and attempt_id:
        try:
            attempt = get_latest_attempt_for_work(
                project_id=project_id,
                work_id=work_id,
                kind="primary",
            )
        except Exception:
            attempt = None
    quality = (attempt.get("quality_json") if isinstance(attempt, dict) else None) or {}
    primary = quality.get("primary") if isinstance(quality.get("primary"), dict) else {}
    fb = quality.get("fallback") if isinstance(quality.get("fallback"), dict) else {}

    needs_confirm = bool(
        fb.get("used")
        or fb.get("ocr_used")
        or primary.get("refs_only")
        or primary.get("header_failed")
    )
    if (not references) or needs_confirm:
        if not bool(force):
            reason = "no_references" if not references else "low_quality_references"
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Resolution gated ({reason}). "
                    "Re-run with ?force=true after review."
                ),
            )

    try:
        resolved = resolve_references(references)

        # Ensure we have a stable attempt id to attach artifacts to.
        if not attempt_id:
            try:
                attempt_id, _state = create_or_get_attempt(
                    project_id,
                    user_id,
                    work_id,
                    "primary",
                    settings_json={},
                )
            except Exception:
                attempt_id = None

        # Persist to the spine (S3 + artifacts table).
        if attempt_id:
            resolution_key = (
                f"resolve/{work_id}/attempts/{attempt_id}/primary/resolution.json"
            )
            resolution_bytes = json.dumps(
                resolved,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
            object_store_s3.put_bytes(
                resolution_key,
                resolution_bytes,
                content_type="application/json",
            )
            create_artifact(
                project_id,
                user_id,
                attempt_id,
                "resolution.json",
                resolution_key,
                len(resolution_bytes),
                "application/json",
            )

        try:
            graph_store.index_resolution(ingest_meta=document, resolution_data=resolved)
        except Exception:
            logger.exception("Graph index failed for resolution")
    except Exception as exc:
        logger.exception("Reference resolution failed for document %s", doc_id)
        raise HTTPException(
            status_code=500, detail=f"Reference resolution failed: {exc}"
        )

    # Spine-only / best-effort response.
    resolved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "document_id": doc_id,
        "resolution": {
            "status": "complete",
            "resolved_at": resolved_at,
            "data": resolved,
        },
    }


@app.get("/ingest/{doc_id}/resolution", response_model=schemas.ResolutionResult)
def get_ingested_resolution(
    doc_id: str,
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)

    try:
        data = get_resolution_from_spine(work_id=str(doc_id), project_id=project_id)
    except Exception as exc:
        logger.exception("Spine resolution read failed for doc_id=%s", doc_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if data is None:
        raise HTTPException(status_code=404, detail="Resolution data not found")

    resolved_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {"status": "complete", "resolved_at": resolved_at, "data": data}


@app.post(
    "/ingest/{doc_id}/resolution/{reference_id}/select",
    response_model=schemas.ResolutionResponse,
)
def select_resolution_source(
    doc_id: str, reference_id: str, payload: schemas.ResolutionSelectionRequest
):
    project_id = str(app_settings.DEFAULT_PROJECT_ID)
    user_id = str(app_settings.DEFAULT_USER_ID)

    resolution: dict = {}
    try:
        resolution_data = (
            get_resolution_from_spine(work_id=str(doc_id), project_id=project_id) or []
        )
    except Exception as exc:
        logger.exception("Spine resolution read failed for doc_id=%s", doc_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

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

    attempt = get_latest_attempt_for_work(
        project_id=project_id,
        work_id=str(doc_id),
        kind="primary",
    )
    attempt_id = str((attempt or {}).get("attempt_id") or "").strip() or None
    if not attempt_id:
        attempt_id, _state = create_or_get_attempt(
            project_id,
            user_id,
            str(doc_id),
            "primary",
            settings_json={},
        )

    resolution_key = f"resolve/{doc_id}/attempts/{attempt_id}/primary/resolution.json"
    resolution_bytes = json.dumps(
        updated_entries,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    object_store_s3.put_bytes(
        resolution_key,
        resolution_bytes,
        content_type="application/json",
    )
    create_artifact(
        project_id,
        user_id,
        attempt_id,
        "resolution.json",
        resolution_key,
        len(resolution_bytes),
        "application/json",
    )

    # Keep the claim graph's reference -> ingest anchoring up to date.
    try:
        graph_store.index_resolution(
            ingest_meta={"id": str(doc_id)},
            resolution_data=resolution_payload.get("data") or [],
        )
    except Exception:
        logger.exception("Graph index failed for resolution selection")
    return {"document_id": doc_id, "resolution": resolution_payload}


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


@app.post(
    "/attachments/{attachment_id}/promote-ingest",
    response_model=schemas.AttachmentResponse,
)
def promote_attachment_ingest(attachment_id: str):
    """Promote an attachment PDF into an ingested Work."""
    try:
        public_record = attachment_pipeline.promote_attachment_to_ingest(attachment_id)
    except attachment_store.AttachmentNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
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
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)

    try:
        document = build_ingested_document_from_spine(
            work_id=str(doc_id),
            project_id=project_id,
            include_extraction_data=True,
        )
    except Exception as exc:
        logger.exception("Spine document read failed for doc_id=%s", doc_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        tei_xml = get_tei_xml_from_spine(work_id=str(doc_id), project_id=project_id)
    except Exception:
        tei_xml = None

    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    if not tei_xml:
        return {"document_id": doc_id, "context": None}

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
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)

    try:
        document = build_ingested_document_from_spine(
            work_id=str(doc_id),
            project_id=project_id,
            include_extraction_data=True,
        )
    except Exception as exc:
        logger.exception("Spine document read failed for doc_id=%s", doc_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

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
    try:
        span_graph_store.index_claim_confirmation(payload.model_dump())
    except Exception:
        logger.exception("Span graph index failed for confirmed claims")
    return {"inserted": inserted}


@app.get(
    "/claims/{claim_id}/evidence",
    response_model=schemas.EvidenceListResponse,
)
def list_claim_evidence(
    claim_id: str,
    reviewer_uid: str = Query("default"),
    label: Optional[str] = None,
    offset: int = 0,
    limit: int = 10,
    include_neutral: bool = True,
    pinned_only: bool = False,
    claim_text: Optional[str] = None,
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    reviewer = str(reviewer_uid or "").strip() or "default"
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)

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

    # Phase 10-04: annotate evidence candidates with durable decision overlays.
    decisions = evidence_decisions_spine.get_projection(
        claim_id,
        reviewer,
        project_id=project_id,
    )
    targets_by_key = decisions.get("targets_by_key") or {}

    def _span_id_for_candidate(candidate: dict) -> Optional[str]:
        meta = candidate.get("metadata") or {}
        if isinstance(meta, dict):
            for key in ("span_id", "anchor_id"):
                value = str(meta.get(key) or "").strip()
                if value:
                    return value
        spans = candidate.get("spans") or []
        if isinstance(spans, list) and spans:
            first = spans[0]
            if isinstance(first, dict):
                for key in ("span_id", "id", "sentence_id"):
                    value = str(first.get(key) or "").strip()
                    if value:
                        return value
        return None

    annotated: list[dict] = []
    by_target: dict[str, dict] = {}
    for cand in payload.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        entry = dict(cand)
        attachment_id = str(entry.get("attachment_id") or "").strip()
        span_id = _span_id_for_candidate(entry)
        if attachment_id and span_id:
            tkey = f"{attachment_id}:{span_id}"
            state = targets_by_key.get(tkey) or {}
            pinned = bool(state.get("pinned"))
            triage = str(state.get("triage") or "none")
            entry["decision_target"] = {
                "attachment_id": attachment_id,
                "span_id": span_id,
                "target_key": tkey,
            }
            entry["decision_state"] = {
                "pinned": pinned,
                "triage": triage,
                "updated_at": state.get("updated_at"),
            }
            badges = list(entry.get("badges") or [])
            if pinned and "decision:pinned" not in badges:
                badges.append("decision:pinned")
            if triage == "accepted" and "decision:accepted" not in badges:
                badges.append("decision:accepted")
            if triage == "rejected" and "decision:rejected" not in badges:
                badges.append("decision:rejected")
            entry["badges"] = badges
            by_target[tkey] = entry
        annotated.append(entry)

    pinned_payloads: list[dict] = []
    for target in decisions.get("pinned_targets") or []:
        if not isinstance(target, dict):
            continue
        tkey = str(target.get("target_key") or "").strip()
        attachment_id = str(target.get("attachment_id") or "").strip()
        span_id = str(target.get("span_id") or "").strip()
        if not tkey or not attachment_id or not span_id:
            continue
        if tkey in by_target:
            pinned_payloads.append(by_target[tkey])
            continue

        state = targets_by_key.get(tkey) or {}
        snippet = str(state.get("snippet") or "").strip()
        text = snippet or "Pinned target not in current run."
        pinned_payloads.append(
            {
                "id": f"pinned:{attachment_id}:{span_id}",
                "claim_id": str(claim_id),
                "attachment_id": attachment_id,
                "label": "neutral",
                "text": text,
                "scores": {},
                "badges": ["decision:pinned"],
                "metadata": {
                    "not_in_current_run": True,
                    "span_id": span_id,
                },
                "spans": [],
                "highlights": [],
                "decision_target": {
                    "attachment_id": attachment_id,
                    "span_id": span_id,
                    "target_key": tkey,
                },
                "decision_state": {
                    "pinned": True,
                    "triage": str((state.get("triage") or "none")),
                    "updated_at": state.get("updated_at"),
                },
            }
        )

    if pinned_only:
        annotated = [
            cand
            for cand in annotated
            if bool((cand.get("decision_state") or {}).get("pinned"))
        ]

    return schemas.EvidenceListResponse(
        claim_id=claim_id,
        pinned=[schemas.EvidenceCandidatePayload(**cand) for cand in pinned_payloads],
        candidates=[schemas.EvidenceCandidatePayload(**cand) for cand in annotated],
        total=payload["total"],
        offset=payload["offset"],
        limit=payload["limit"],
        lock_state=payload["lock_state"],
        run=payload.get("run"),
        decisions={
            "reviewer_uid": reviewer,
            "version": int(decisions.get("version") or 0),
        },
    )


@app.get(
    "/claims/{claim_id}/evidence/decisions",
    response_model=schemas.EvidenceDecisionReadResponse,
)
def get_evidence_decisions(
    claim_id: str,
    reviewer_uid: str = Query("default"),
    events_limit: int = Query(20, ge=0, le=200),
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    reviewer = str(reviewer_uid or "").strip() or "default"
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)

    projection = evidence_decisions_spine.get_projection(
        claim_id,
        reviewer,
        project_id=project_id,
    )
    events = evidence_decisions_spine.list_recent_events(
        claim_id,
        reviewer,
        project_id=project_id,
        limit=int(events_limit),
    )

    targets: dict[str, Any] = {}
    for key, entry in (projection.get("targets_by_key") or {}).items():
        if not isinstance(entry, dict):
            continue
        targets[str(key)] = {
            "pinned": bool(entry.get("pinned")),
            "triage": str(entry.get("triage") or "none"),
            "updated_at": entry.get("updated_at"),
        }

    pinned_targets: list[dict[str, str]] = []
    for tgt in projection.get("pinned_targets") or []:
        if not isinstance(tgt, dict):
            continue
        attachment_id = str(tgt.get("attachment_id") or "").strip()
        span_id = str(tgt.get("span_id") or "").strip()
        tkey = str(tgt.get("target_key") or "").strip()
        if attachment_id and span_id and tkey:
            pinned_targets.append(
                {"attachment_id": attachment_id, "span_id": span_id, "target_key": tkey}
            )

    events_out: list[dict[str, Any]] = []
    for ev in events:
        action = str(ev.get("action") or "").strip().lower()
        attachment_id = str(ev.get("target_attachment_id") or "").strip() or None
        span_id = str(ev.get("target_span_id") or "").strip() or None
        target = None
        if action != "clear" and attachment_id and span_id:
            target = {
                "attachment_id": attachment_id,
                "span_id": span_id,
                "target_key": str(ev.get("target_key") or f"{attachment_id}:{span_id}"),
            }
        payload = (
            ev.get("payload_json") if isinstance(ev.get("payload_json"), dict) else {}
        )
        events_out.append(
            {
                "event_id": int(ev.get("event_id") or 0),
                "event_uid": str(ev.get("event_uid") or ""),
                "action": action,
                "target": target,
                "set": ev.get("set_value"),
                "created_at": str(ev.get("created_at") or ""),
                "payload": dict(payload or {}),
            }
        )

    return {
        "claim_id": str(claim_id),
        "reviewer_uid": reviewer,
        "version": int(projection.get("version") or 0),
        "targets": targets,
        "pinned_targets": pinned_targets,
        "events": events_out,
    }


@app.post(
    "/claims/{claim_id}/evidence/decisions/events",
    response_model=schemas.EvidenceDecisionAppendResponse,
)
def post_evidence_decision_event(
    claim_id: str,
    payload: schemas.EvidenceDecisionAppendRequest,
    reviewer_uid: str = Query("default"),
    x_project_id: Optional[str] = Header(None, alias="X-Project-Id"),
):
    reviewer = str(reviewer_uid or "").strip() or "default"
    project_id = str(x_project_id or "").strip() or str(app_settings.DEFAULT_PROJECT_ID)
    user_id = str(app_settings.DEFAULT_USER_ID)

    target = payload.target
    try:
        result = evidence_decisions_spine.append_event(
            claim_id,
            reviewer,
            expected_version=int(payload.expected_version),
            idempotency_key=str(payload.idempotency_key),
            action=str(payload.action),
            target_attachment_id=str(getattr(target, "attachment_id", None))
            if target is not None
            else None,
            target_span_id=str(getattr(target, "span_id", None))
            if target is not None
            else None,
            set_value=payload.set,
            payload=dict(payload.payload or {}),
            project_id=project_id,
            user_id=user_id,
        )
    except evidence_decisions_spine.EvidenceDecisionConflict as exc:
        body: dict[str, Any] = {"detail": str(exc.detail)}
        if exc.current_version is not None:
            body["current_version"] = int(exc.current_version)
        return JSONResponse(status_code=409, content=body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result


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

    # Best-effort: mirror evidence selections into the span-first graph as
    # reviewer-attributed assertions.
    try:
        parsed = span_graph_store.parse_cite_claim_id(claim_id)
        if parsed:
            reviewer_uid = str(parsed.get("reviewer_uid") or "default")
            order_index = parsed.get("order_index")
            ingest_id = str(parsed.get("document_id") or "").strip()
            cite_idx = int(parsed.get("citation_index") or 0)

            target_id = None
            source_ingest_id = None
            evidence_span_id = None
            if stored.primary is not None:
                evidence_span_id = stored.primary.span_id
                record = attachment_store.get_attachment(
                    stored.primary.attachment_id,
                    public=False,
                )
                if record:
                    target_id = (record.get("target_id") or "").strip() or None
                    source_ingest_id = (
                        record.get("source_ingest_id") or ""
                    ).strip() or None

            if ingest_id and order_index is not None:
                span = span_graph_store.find_citation_span(
                    ingest_id=ingest_id,
                    citation_index=cite_idx,
                    target_id=target_id,
                )
                if span:
                    claim_span = span_graph_store.get_claim_span(
                        span_id=span["span_id"],
                        order_index=int(order_index),
                    )
                    if claim_span:
                        claim_span_id = str(claim_span["claim_span_id"])
                        span_graph_store.mark_checked(
                            claim_span_id=claim_span_id,
                            reviewer_uid=reviewer_uid,
                        )

                        if stored.verdict != "none":
                            evidence_work_id = None
                            if source_ingest_id:
                                evidence_work_id = f"ingest:{source_ingest_id}"
                            elif target_id:
                                evidence_work_id = f"ref:{ingest_id}:{target_id}"
                        if evidence_work_id:
                            span_graph_store.upsert_work(work_id=evidence_work_id)
                            span_graph_store.upsert_selection_assertion(
                                claim_id=str(claim_id),
                                reviewer_uid=reviewer_uid,
                                verdict=str(stored.verdict),
                                claim_span_id=claim_span_id,
                                evidence_span_id=evidence_span_id,
                                evidence_work_id=evidence_work_id,
                                comment=stored.note,
                            )
    except Exception:
        logger.exception("Span graph mirror failed for evidence selection")
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
