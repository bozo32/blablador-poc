"""Background processing pipeline for persisted attachments."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Iterable, List, Optional
from uuid import uuid4

from lxml import etree

import hashlib

from backend import attachment_store, background_state, extraction, grobid_client, utils
from backend.evidence_matching.service import evidence_service
from backend.graph_store import GraphStore
from backend.object_store import s3 as object_store_s3
from backend.settings import settings
from backend.spine.artifacts import create_artifact
from backend.spine.attempts import create_or_get_attempt
from backend.spine.documents import (
    ensure_project_document,
    get_document_version_by_sha256,
    get_or_create_document,
    upsert_document_version,
)
from backend.spine.ids import pdf_object_key_for_work_pdf
from backend.spine.ingest_view import build_ingested_document_from_spine
from backend.spine.works import upsert_work_from_pdf


logger = logging.getLogger(__name__)


def _maybe_ingest_matched_attachment(record: dict) -> None:
    """Ensure a placed attachment's PDF is ingested as a Work.

    Rationale: a placed/synced PDF should not remain "synced but not ingested".
    Surfing and work-level graphs depend on ingested docs (with extraction) to
    form stable work-work links and resolve reference targets.

    This function is best-effort and safe under duplicates (spine de-dupes by
    sha256).
    """
    if not isinstance(record, dict):
        return
    if str(record.get("source_ingest_id") or "").strip():
        return

    pdf_object_key = str(record.get("pdf_object_key") or "").strip()
    if not pdf_object_key:
        return
    try:
        file_bytes = object_store_s3.get_bytes(pdf_object_key)
    except Exception:
        logger.exception("Unable to download attachment PDF from object store")
        return
    filename = (
        str(record.get("filename") or "attachment.pdf").strip() or "attachment.pdf"
    )
    sha256 = hashlib.sha256(file_bytes).hexdigest().lower()
    size_bytes = int(len(file_bytes))

    project_id = str(settings.DEFAULT_PROJECT_ID)
    user_id = str(settings.DEFAULT_USER_ID)

    existing = get_document_version_by_sha256(sha256=sha256)
    if existing is not None:
        ingest_id = str(existing.get("document_id") or "").strip()
        pdf_object_key = str(existing.get("pdf_object_key") or "").strip()
        if not ingest_id or not pdf_object_key:
            return
    else:
        ingest_id = str(uuid4())
        pdf_object_key = pdf_object_key_for_work_pdf(ingest_id, sha256)

    try:
        if not object_store_s3.exists(pdf_object_key):
            object_store_s3.put_bytes(
                pdf_object_key,
                file_bytes,
                content_type="application/pdf",
            )
    except Exception:
        logger.exception("Object store upload failed for attachment ingest")
        return

    try:
        upsert_work_from_pdf(
            work_id=ingest_id,
            project_id=project_id,
            created_by_user_id=user_id,
            filename=filename,
            sha256=sha256,
            size_bytes=size_bytes,
            pdf_object_key=pdf_object_key,
        )
        get_or_create_document(ingest_id, created_by_user_id=user_id)
        upsert_document_version(
            ingest_id,
            document_id=ingest_id,
            sha256=sha256,
            size_bytes=size_bytes,
            pdf_object_key=pdf_object_key,
            filename=filename,
            created_by_user_id=user_id,
        )
        ensure_project_document(
            project_id,
            document_id=ingest_id,
            added_by_user_id=user_id,
        )
    except Exception:
        logger.exception("Spine upsert failed for attachment ingest")
        return

    graph_store = GraphStore(settings=settings)
    ingest_meta = build_ingested_document_from_spine(
        work_id=ingest_id,
        project_id=project_id,
        include_extraction_data=False,
    ) or {"id": ingest_id, "project_id": project_id, "sha256": sha256}

    try:
        graph_store.index_ingest_upload(ingest_meta)
    except Exception:
        logger.exception("Graph index failed for attachment ingest upload")

    # Reuse TEI artifacts produced by attachment processing.
    artifacts = record.get("artifacts") or {}
    tei_xml_key = str(artifacts.get("tei_xml") or "").strip()
    tei_json_key = str(artifacts.get("tei_json") or "").strip()
    if tei_xml_key and tei_json_key:
        try:
            tei_xml = object_store_s3.get_bytes(tei_xml_key).decode(
                "utf-8", errors="ignore"
            )
            extraction_payload = json.loads(
                object_store_s3.get_bytes(tei_json_key).decode("utf-8")
            )

            attempt_id, _state = create_or_get_attempt(
                project_id,
                user_id,
                ingest_id,
                "primary",
                settings_json={},
            )
            tei_key = f"extract/{ingest_id}/attempts/{attempt_id}/primary/tei.xml"
            tei_bytes = tei_xml.encode("utf-8", errors="ignore")
            object_store_s3.put_bytes(
                tei_key, tei_bytes, content_type="application/xml"
            )
            create_artifact(
                project_id,
                user_id,
                attempt_id,
                "tei.xml",
                tei_key,
                len(tei_bytes),
                "application/xml",
            )

            extraction_key = (
                f"extract/{ingest_id}/attempts/{attempt_id}/primary/extraction.json"
            )
            extraction_bytes = json.dumps(
                extraction_payload,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
            object_store_s3.put_bytes(
                extraction_key,
                extraction_bytes,
                content_type="application/json",
            )
            create_artifact(
                project_id,
                user_id,
                attempt_id,
                "extraction.json",
                extraction_key,
                len(extraction_bytes),
                "application/json",
            )

            try:
                graph_store.index_extraction(
                    ingest_meta=ingest_meta,
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
            pdf_object_key = str(record.get("pdf_object_key") or "").strip()
            if not pdf_object_key:
                raise RuntimeError("Attachment is missing pdf_object_key")

            tmp_pdf_path = None
            try:
                tmp_pdf_path = Path(f"/tmp/attach-{attachment_id}.pdf")
                tmp_pdf_path.write_bytes(object_store_s3.get_bytes(pdf_object_key))
                tei_xml = grobid_client.extract_tei(tmp_pdf_path)
            finally:
                try:
                    if tmp_pdf_path and tmp_pdf_path.exists():
                        tmp_pdf_path.unlink()
                except Exception:
                    pass
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
