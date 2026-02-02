"""Background processing pipeline for persisted attachments."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Iterable, List, Optional
from uuid import uuid4

from lxml import etree

from backend import attachment_store, extraction, grobid_client, utils
from backend.evidence_matching.service import evidence_service
from backend.settings import settings


logger = logging.getLogger(__name__)


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
            try:
                record = attachment_store.get_attachment(attachment_id)
                claim_id = str(record.get("claim_id")) if record else None
                claim_text = record.get("claim_text") if record else None
                if claim_id:
                    evidence_service.trigger_auto_rerun(claim_id, claim_text=claim_text)
            except (
                Exception
            ):  # pragma: no cover - rerun failures shouldn't block pipeline
                logger.exception(
                    "Unable to enqueue evidence rerun for claim %s",
                    record.get("claim_id"),
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
    thread = threading.Thread(
        target=process_attachment,
        args=(attachment_id,),
        daemon=True,
        name=f"attachment-{attachment_id}",
    )
    thread.start()
