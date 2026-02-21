"""Stage payload builders for the Phase 10-02 happy-path orchestrator.

These functions intentionally return plain dicts suitable for
`pipeline_contracts_service.store_stage(..., data=...)`.
"""

from __future__ import annotations

from typing import Any

from backend.settings import settings as app_settings
from backend.spine.ingest_view import build_ingested_document_from_spine
from backend import text_selectors
from backend.span_graph_store import SpanGraphStore


def build_extract_data(*, citing_doc_id: str) -> dict:
    doc_id = str(citing_doc_id or "").strip()
    if not doc_id:
        raise ValueError("citing_doc_id is required")

    project_id = str(
        getattr(app_settings, "DEFAULT_PROJECT_ID", "default") or "default"
    )
    doc = build_ingested_document_from_spine(
        work_id=doc_id,
        project_id=project_id,
        include_extraction_data=True,
    )
    if doc is None:
        raise KeyError("citing document not found")
    extraction = doc.get("extraction") or {}
    if str(extraction.get("status") or "").strip() != "complete":
        raise RuntimeError("citing document extraction is not complete")
    extraction_data = extraction.get("data")
    if not isinstance(extraction_data, dict):
        raise RuntimeError("extraction data missing")

    citations = extraction_data.get("citations")
    if not isinstance(citations, list):
        citations = []

    anchors: list[dict[str, Any]] = []
    for idx, cite in enumerate(citations):
        if not isinstance(cite, dict):
            continue
        target_id = str(cite.get("target_id") or "").strip() or None
        callout = str(cite.get("callout") or "").strip() or None
        window_text = str(cite.get("sentence") or "").strip() or None
        sentence_id = str(cite.get("sentence_id") or "").strip() or None
        anchors.append(
            {
                "citation_index": int(idx),
                "reference_id": target_id,
                "target_id": target_id,
                "callout": callout,
                "window_text": window_text,
                "sentence_id": sentence_id,
            }
        )

    structured_doc = dict(extraction_data)
    return {"structured_doc": structured_doc, "citation_anchors": anchors}


def build_citespans_data(
    *,
    span_graph_store: SpanGraphStore,
    citing_doc_id: str,
    citation_anchors: list[dict],
) -> dict:
    doc_id = str(citing_doc_id or "").strip()
    if not doc_id:
        raise ValueError("citing_doc_id is required")
    if not isinstance(citation_anchors, list):
        raise ValueError("citation_anchors must be a list")

    by_target: dict[str, dict[str, Any]] = {}
    for anchor in citation_anchors:
        if not isinstance(anchor, dict):
            continue
        citation_index = anchor.get("citation_index")
        if citation_index is None:
            continue
        try:
            ci = int(citation_index)
        except Exception:
            continue

        target_id = str(
            anchor.get("target_id") or anchor.get("reference_id") or ""
        ).strip()
        if not target_id:
            continue

        window_text = str(anchor.get("window_text") or "").strip()
        selectors = text_selectors.build_anchor_quote(window_text)
        window_fp = text_selectors.fingerprint(window_text)
        span = span_graph_store.upsert_span(
            kind="citation_window",
            selector=dict(selectors),
            window_fingerprint=str(window_fp),
            ingest_id=doc_id,
        )
        span_id = str((span or {}).get("span_id") or "").strip()
        if span_id:
            span_graph_store.upsert_citation_span_index(
                ingest_id=doc_id,
                citation_index=ci,
                target_id=target_id,
                span_id=span_id,
            )

        entry = by_target.setdefault(target_id, {"anchors": []})
        entry["anchors"].append(
            {
                "citation_index": int(ci),
                "span_id": span_id,
                "selectors": dict(selectors),
                "window_fingerprint": str(window_fp),
            }
        )

    return {"by_target": by_target}


def build_candidate_stage_data_from_evidence(
    *,
    claim_id: str,
    run_id: str,
    target_ids: list[str],
    attachment_ids_by_target: dict[str, list[str]],
    stage: str,
) -> dict:
    """MVP placeholder for candidate stages.

    This plan's verifier allows targets to remain requested/blocked. For now we
    return an empty payload when no runnable targets exist.
    """
    _ = (claim_id, run_id, target_ids, attachment_ids_by_target, stage)
    return {"by_target": {}}


__all__ = [
    "build_extract_data",
    "build_citespans_data",
    "build_candidate_stage_data_from_evidence",
]
