"""Background processing pipeline for persisted attachments."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Iterable, List, Optional
from uuid import uuid4

from lxml import etree

from backend import attachment_store, background_state, extraction, grobid_client, utils
from backend.evidence_matching.service import evidence_service
from backend.graph_store import GraphStore
from backend.ingestion_store import (
    create_ingested_document,
    get_ingested_document,
    store_extraction,
)
from backend.settings import settings


logger = logging.getLogger(__name__)


graph_store = GraphStore(settings.GRAPH_DB_PATH)


def _maybe_ingest_matched_attachment(record: dict) -> None:
    """Ensure a placed attachment's PDF is ingested as a Work.

    Rationale: a placed/synced PDF should not remain "synced but not ingested".
    Surfing and work-level graphs depend on ingested docs (with extraction) to
    form stable work-work links and resolve reference targets.

    This function is best-effort and safe under duplicates (ingestion_store
    de-dupes by sha256).
    """
    if not isinstance(record, dict):
        return
    if str(record.get("source_ingest_id") or "").strip():
        return
    doc_id = str(record.get("doc_id") or "").strip()
    target_id = str(record.get("target_id") or "").strip()
    if not doc_id or not target_id:
        return

    try:
        pdf_path = Path(str(record.get("file_path") or ""))
    except Exception:
        return
    if not pdf_path.exists():
        return

    file_bytes = pdf_path.read_bytes()
    filename = str(record.get("filename") or pdf_path.name)
    metadata = create_ingested_document(file_bytes, filename)
    ingest_id = str(metadata.get("id") or "").strip()
    if not ingest_id:
        return

    try:
        graph_store.index_ingest_upload(metadata)
    except Exception:
        logger.exception("Graph index failed for attachment ingest upload")

    # Reuse TEI artifacts produced by attachment processing if extraction isn't
    # complete.
    existing = get_ingested_document(ingest_id) or {}
    extraction_stage = existing.get("extraction") or {}
    has_extraction = bool(
        (extraction_stage.get("data") or {})
        if isinstance(extraction_stage, dict)
        else {}
    )
    if not has_extraction:
        artifacts = record.get("artifacts") or {}
        tei_xml_path = artifacts.get("tei_xml")
        tei_json_path = artifacts.get("tei_json")
        if tei_xml_path and tei_json_path:
            try:
                tei_xml = Path(str(tei_xml_path)).read_text(
                    encoding="utf-8", errors="ignore"
                )
                extraction_payload = json.loads(
                    Path(str(tei_json_path)).read_text(encoding="utf-8")
                )
                stored = store_extraction(ingest_id, tei_xml, extraction_payload)
                try:
                    graph_store.index_extraction(
                        ingest_meta=stored,
                        extraction_data=extraction_payload,
                    )
                except Exception:
                    logger.exception("Graph index failed for attachment extraction")
            except Exception:
                logger.exception("Unable to reuse attachment TEI artifacts for ingest")

    try:
        attachment_store.update_attachment(
            str(record.get("id")),
            source_ingest_id=ingest_id,
            timeline_event="ingested",
            timeline_detail=f"Promoted to ingest:{ingest_id}",
        )
    except Exception:
        logger.exception("Unable to stamp source_ingest_id on attachment")


def promote_attachment_to_ingest(attachment_id: str) -> dict:
    """Promote an existing attachment PDF into an ingested document.

    Returns the updated attachment record (public view).
    """
    rec = attachment_store.get_attachment(str(attachment_id))
    if rec is None:
        raise attachment_store.AttachmentNotFound(
            f"Attachment {attachment_id} not found"
        )
    _maybe_ingest_matched_attachment(rec)
    return attachment_store.public_status(str(attachment_id)) or {}


def _normalize_ws(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _section_label(node: etree._Element) -> str:
    """Best-effort section label for a TEI node."""
    try:
        if node.xpath("boolean(ancestor::tei:note)", namespaces=extraction.NS):
            return "Footnote"
        if node.xpath(
            "boolean(ancestor::tei:listBibl | ancestor::tei:biblStruct)",
            namespaces=extraction.NS,
        ):
            return "References"
        head = node.xpath(
            "string((ancestor::tei:div[tei:head][1]/tei:head)[1])",
            namespaces=extraction.NS,
        )
        cleaned = _normalize_ws(head)
        if cleaned:
            return cleaned[:80]
    except Exception:
        return "Body"
    return "Body"


def _sentence_nodes(tei_xml: str) -> List[etree._Element]:
    root = etree.fromstring(
        tei_xml.encode("utf-8"), parser=etree.XMLParser(recover=True)
    )
    nodes: List[etree._Element] = root.xpath(
        "//tei:text//tei:body//tei:s",
        namespaces=extraction.NS,
    )
    return nodes


def _extract_sentences(tei_xml: str) -> List[dict]:
    sentences: List[dict] = []
    for idx, node in enumerate(_sentence_nodes(tei_xml)):
        text = extraction._text_content(node).strip()  # type: ignore[attr-defined]
        if not text:
            continue
        sentence_id = (
            node.get(f"{{{extraction.XML_NS}}}id")
            or node.get("xml:id")
            or f"sent-{uuid4()}"
        )
        page = None
        page_nodes = node.xpath("ancestor::tei:pb[1]/@n", namespaces=extraction.NS)
        if page_nodes:
            page = page_nodes[0]
        section = _section_label(node)
        sentences.append(
            {
                "sentence_id": sentence_id,
                "text": text,
                "page": page,
                "section": section,
                "position": idx,
            }
        )
    if sentences:
        return sentences

    # Fallback when TEI lacks <s> nodes (common for some GROBID outputs):
    # extract paragraph-level text from the TEI body.
    root = etree.fromstring(
        tei_xml.encode("utf-8"), parser=etree.XMLParser(recover=True)
    )
    para_xpath = (
        "//tei:text//tei:body//*[self::tei:p or self::tei:item or self::tei:cell]"
    )
    para_nodes: List[etree._Element] = list(
        root.xpath(para_xpath, namespaces=extraction.NS)
    )
    fallback: List[dict] = []
    for idx, para in enumerate(para_nodes):
        text = extraction._text_content(para).strip()  # type: ignore[attr-defined]
        if not text:
            continue
        sentence_id = (
            para.get(f"{{{extraction.XML_NS}}}id")
            or para.get("xml:id")
            or f"auto-sent-{idx}"
        )
        page = None
        page_nodes = para.xpath("ancestor::tei:pb[1]/@n", namespaces=extraction.NS)
        if page_nodes:
            page = page_nodes[0]
        section = _section_label(para)
        fallback.append(
            {
                "sentence_id": sentence_id,
                "text": text,
                "page": page,
                "section": section,
                "position": idx,
            }
        )
    if fallback:
        return fallback

    # Last resort: extract body text as a single passage.
    body_nodes: List[etree._Element] = list(
        root.xpath("//tei:text//tei:body", namespaces=extraction.NS)
    )
    if body_nodes:
        # extraction._text_content is a private helper, but stable in this codebase.
        body_text = extraction._text_content(body_nodes[0])
        text = body_text.strip()
        if text:
            return [
                {
                    "sentence_id": f"sent-{uuid4()}",
                    "text": text,
                    "page": None,
                    "position": 0,
                }
            ]
    return []


def _merge_embeddings(
    sentences: List[dict], embeddings: Iterable[Iterable[float]]
) -> List[dict]:
    merged: List[dict] = []
    for sentence, vector in zip(sentences, embeddings):
        payload = dict(sentence)
        payload["embedding"] = list(vector)
        merged.append(payload)
    if len(merged) < len(sentences):
        # Pad remaining sentences with empty embeddings to preserve ordering
        for sentence in sentences[len(merged) :]:
            payload = dict(sentence)
            payload["embedding"] = []
            merged.append(payload)
    return merged


def process_attachment(
    attachment_id: str, *, max_attempts: Optional[int] = None
) -> None:
    record = attachment_store.get_attachment(attachment_id)
    if record is None:
        logger.warning("Attachment %s no longer exists", attachment_id)
        return

    max_attempts = (
        max_attempts
        or record.get("max_attempts")
        or attachment_store.DEFAULT_MAX_ATTEMPTS
    )
    attempt = record.get("attempts", 0)

    while attempt < max_attempts:
        attempt += 1
        attachment_store.mark_converting(attachment_id, attempt)
        attachment_store.mark_parsing(attachment_id, attempt)
        try:
            pdf_path = Path(record["file_path"])
            tei_xml = grobid_client.extract_tei(pdf_path)
            tei_json = extraction.parse_tei(tei_xml)
            sentences = _extract_sentences(tei_xml)
            texts = [sentence["text"] for sentence in sentences]
            embeddings = (
                utils.embed(texts, model_name=settings.EMBED_MODEL, mode="passage")
                if texts
                else []
            )
            enriched = _merge_embeddings(sentences, embeddings)
            artifacts = attachment_store.save_artifacts(
                attachment_id,
                tei_xml=tei_xml,
                tei_json=tei_json,
                sentences=enriched,
            )
            attachment_store.mark_matched(attachment_id, artifacts)
            logger.info("Attachment %s processed successfully", attachment_id)

            # If this attachment is placed, promote it to an ingested Work so
            # work-level graphs can resolve cited works automatically.
            try:
                latest = attachment_store.get_attachment(attachment_id)
                if latest:
                    _maybe_ingest_matched_attachment(latest)
            except Exception:
                logger.exception("Unable to promote matched attachment to ingested doc")
            try:
                record = attachment_store.get_attachment(attachment_id)
                raw_claim_id = (record or {}).get("claim_id")
                claim_id = str(raw_claim_id) if raw_claim_id else None
                claim_text = (record or {}).get("claim_text")
                if claim_id:
                    evidence_service.trigger_auto_rerun(claim_id, claim_text=claim_text)
            except (
                Exception
            ):  # pragma: no cover - rerun failures shouldn't block pipeline
                logger.exception(
                    "Unable to enqueue evidence rerun for claim %s",
                    (record or {}).get("claim_id"),
                )
            return
        except (
            Exception
        ) as exc:  # pragma: no cover - log path exercised in tests via mark_error
            logger.exception("Attachment %s failed: %s", attachment_id, exc)
            if attempt >= max_attempts:
                attachment_store.mark_error(attachment_id, str(exc))
                return
            attachment_store.update_attachment(
                attachment_id,
                status=attachment_store.STATUS_PENDING,
                error=str(exc),
                attempts=attempt,
                timeline_event="retrying",
                timeline_detail=str(exc),
            )
            record = attachment_store.get_attachment(attachment_id)
            if record is None:
                return


def enqueue_processing(attachment_id: str) -> None:
    state = background_state.get_state()
    if state.get("paused"):
        try:
            attachment_store.update_attachment(
                attachment_id,
                status=attachment_store.STATUS_PENDING,
                timeline_event="paused",
                timeline_detail=state.get("reason"),
            )
        except Exception:  # pragma: no cover - best effort
            logger.exception("Unable to mark attachment %s as paused", attachment_id)
        return
    thread = threading.Thread(
        target=process_attachment,
        args=(attachment_id,),
        daemon=True,
        name=f"attachment-{attachment_id}",
    )
    thread.start()
