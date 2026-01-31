"""Attachment TEI span helpers for evidence review.

This module builds a deterministic index over a Grobid TEI body so the API can:
- jump to a stable anchor (span_id == sentence_id)
- return a bounded excerpt window that never crosses paragraph boundaries
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lxml import etree

from backend import attachment_store, extraction


TEI_NS = extraction.NS


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


def _text_content(elem: etree._Element) -> str:
    return _collapse_ws("".join(elem.itertext()))


def _xml_id(elem: etree._Element) -> Optional[str]:
    return elem.get(f"{{{extraction.XML_NS}}}id") or elem.get("xml:id")


def _page_label(elem: etree._Element) -> str:
    # Spec prefers nearest ancestor page break.
    nodes = elem.xpath("ancestor::tei:pb[1]/@n", namespaces=TEI_NS)
    if nodes:
        return str(nodes[0])

    # Grobid commonly emits <pb> as a sibling milestone; fall back to nearest
    # preceding page break to provide useful page labels.
    nodes = elem.xpath("preceding::tei:pb[1]/@n", namespaces=TEI_NS)
    if nodes:
        return str(nodes[0])

    return "Unknown"


def _section_path(elem: etree._Element) -> str:
    # Build outer-to-inner headings from ancestor <div> blocks.
    divs: List[etree._Element] = list(
        elem.xpath("ancestor::tei:div", namespaces=TEI_NS)
    )
    labels: List[str] = []
    for div in divs:
        heads = div.xpath("./tei:head[1]", namespaces=TEI_NS)
        if not heads:
            continue
        label = _text_content(heads[0]).strip()
        if label:
            labels.append(label)
    if not labels:
        return "Body"
    return " > ".join(labels)


@dataclass(frozen=True)
class _Sentence:
    sentence_id: str
    text: str
    page: str
    section_path: str
    paragraph_id: str
    sentence_index: int


@dataclass(frozen=True)
class _Paragraph:
    paragraph_id: str
    sentence_ids: Tuple[str, ...]


_INDEX_CACHE: Dict[str, Tuple[float, "AttachmentSpanIndex"]] = {}


class AttachmentSpanIndex:
    """Cache-friendly TEI body index for excerpt + jump lookups."""

    def __init__(
        self,
        *,
        attachment_id: str,
        paragraphs: List[_Paragraph],
        sentences: Dict[str, _Sentence],
    ) -> None:
        """Create an index instance for a specific attachment."""
        self.attachment_id = attachment_id
        self._paragraphs = list(paragraphs)
        self._sentences = dict(sentences)
        self._paragraph_lookup: Dict[str, _Paragraph] = {
            paragraph.paragraph_id: paragraph for paragraph in paragraphs
        }

    @classmethod
    def for_attachment(
        cls, attachment_id: str, *, use_cache: bool = True
    ) -> "AttachmentSpanIndex":
        record = attachment_store.get_attachment(attachment_id)
        if record is None:
            raise attachment_store.AttachmentNotFound(
                f"Attachment {attachment_id} not found"
            )
        if not attachment_store.is_ready(record):
            raise RuntimeError(
                "Attachment %s is not ready (status=%s)"
                % (attachment_id, record.get("status"))
            )

        artifacts = record.get("artifacts") or {}
        tei_path = artifacts.get("tei_xml")
        if not tei_path:
            raise FileNotFoundError(
                f"Attachment {attachment_id} is missing persisted TEI XML"
            )
        path = Path(str(tei_path))
        if not path.exists():
            raise FileNotFoundError(path)

        mtime = path.stat().st_mtime
        if use_cache:
            cached = _INDEX_CACHE.get(attachment_id)
            if cached and cached[0] == mtime:
                return cached[1]

        tei_xml = path.read_text(encoding="utf-8")
        index = cls._build_from_tei_xml(attachment_id, tei_xml)
        _INDEX_CACHE[attachment_id] = (mtime, index)
        return index

    @staticmethod
    def _build_from_tei_xml(
        attachment_id: str, tei_xml: str | bytes
    ) -> "AttachmentSpanIndex":
        if isinstance(tei_xml, bytes):
            tei_xml = tei_xml.decode("utf-8", errors="ignore")
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(str(tei_xml).encode("utf-8"), parser=parser)

        paragraph_xpath = (
            "//tei:text//tei:body//*"
            "[self::tei:p or self::tei:item or self::tei:cell]"
        )
        para_nodes: List[etree._Element] = list(
            root.xpath(paragraph_xpath, namespaces=TEI_NS)
        )

        paragraphs: List[_Paragraph] = []
        sentences: Dict[str, _Sentence] = {}
        global_sentence_idx = 0

        for para_idx, para in enumerate(para_nodes):
            paragraph_id = _xml_id(para) or f"auto-par-{para_idx}"
            sent_nodes: List[etree._Element] = list(
                para.xpath(".//tei:s", namespaces=TEI_NS)
            )

            sentence_ids: List[str] = []
            if sent_nodes:
                for sent_idx, sent in enumerate(sent_nodes):
                    text = _text_content(sent).strip()
                    if not text:
                        continue
                    sentence_id = _xml_id(sent) or f"auto-sent-{global_sentence_idx}"
                    global_sentence_idx += 1
                    sentence_ids.append(sentence_id)
                    sentences[sentence_id] = _Sentence(
                        sentence_id=sentence_id,
                        text=text,
                        page=_page_label(sent),
                        section_path=_section_path(sent),
                        paragraph_id=paragraph_id,
                        sentence_index=sent_idx,
                    )
            else:
                text = _text_content(para).strip()
                if text:
                    sentence_id = f"auto-sent-{global_sentence_idx}"
                    global_sentence_idx += 1
                    sentence_ids.append(sentence_id)
                    sentences[sentence_id] = _Sentence(
                        sentence_id=sentence_id,
                        text=text,
                        page=_page_label(para),
                        section_path=_section_path(para),
                        paragraph_id=paragraph_id,
                        sentence_index=0,
                    )

            if sentence_ids:
                paragraphs.append(
                    _Paragraph(
                        paragraph_id=paragraph_id, sentence_ids=tuple(sentence_ids)
                    )
                )

        return AttachmentSpanIndex(
            attachment_id=attachment_id,
            paragraphs=paragraphs,
            sentences=sentences,
        )

    def jump(self, span_id: str) -> dict:
        sentence = self._sentences.get(span_id)
        if sentence is None:
            raise KeyError(span_id)
        return {
            "page": sentence.page,
            "section_path": sentence.section_path,
            "paragraph_id": sentence.paragraph_id,
            "sentence_id": sentence.sentence_id,
            "sentence_index": sentence.sentence_index,
        }

    def excerpt(self, span_id: str, *, before: int = 2, after: int = 1) -> List[dict]:
        sentence = self._sentences.get(span_id)
        if sentence is None:
            raise KeyError(span_id)

        paragraph = self._paragraph_lookup.get(sentence.paragraph_id)
        if paragraph is None:
            raise KeyError(sentence.paragraph_id)

        ids = list(paragraph.sentence_ids)
        idx = sentence.sentence_index
        start = max(0, idx - max(0, int(before)))
        end = min(len(ids), idx + max(0, int(after)) + 1)

        payload: List[dict] = []
        for sent_id in ids[start:end]:
            sent = self._sentences.get(sent_id)
            if sent is None:
                continue
            payload.append(
                {
                    "sentence_id": sent.sentence_id,
                    "text": sent.text,
                    "is_highlight": sent.sentence_id == span_id,
                    "page": sent.page,
                    "section_path": sent.section_path,
                    "paragraph_id": sent.paragraph_id,
                }
            )
        return payload


def clear_attachment_span_cache(attachment_id: Optional[str] = None) -> None:
    if attachment_id is None:
        _INDEX_CACHE.clear()
        return
    _INDEX_CACHE.pop(attachment_id, None)


__all__ = ["AttachmentSpanIndex", "clear_attachment_span_cache"]
