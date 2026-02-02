from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional

from lxml import etree


NS = {"tei": "http://www.tei-c.org/ns/1.0"}
XML_NS = "http://www.w3.org/XML/1998/namespace"


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


def _text_content(elem: etree._Element) -> str:
    return _collapse_ws("".join(elem.itertext()))


def _parse_tei_xml(tei_xml: str | bytes) -> etree._Element:
    if isinstance(tei_xml, bytes):
        tei_xml = tei_xml.decode("utf-8", errors="ignore")
    parser = etree.XMLParser(recover=True)
    return etree.fromstring(tei_xml.encode("utf-8"), parser=parser)


def _get_xml_id(elem: etree._Element) -> Optional[str]:
    return elem.get(f"{{{XML_NS}}}id") or elem.get("xml:id")


def _find_sentence_id(elem: etree._Element) -> Optional[str]:
    sent = elem.xpath("ancestor::tei:s[1]", namespaces=NS)
    if sent:
        return _get_xml_id(sent[0])
    para = elem.xpath("ancestor::tei:p[1]", namespaces=NS)
    if para:
        return _get_xml_id(para[0])
    return None


def _label_callout(callout: str) -> str:
    label = (callout or "").strip()
    label = label.strip("()[]")
    label = re.sub(r"[;,]\s*$", "", label)
    label = re.sub(r"\s+", " ", label).strip()
    return label or "citation"


def build_document_body(tei_xml: str | bytes) -> Dict[str, Any]:
    root = _parse_tei_xml(tei_xml)
    tree = root.getroottree()

    refs = root.xpath(
        "//tei:text//tei:body//tei:ref[@type='bibr']",
        namespaces=NS,
    )
    path_to_index: Dict[str, int] = {}
    for idx, ref in enumerate(refs):
        path_to_index[tree.getpath(ref)] = idx

    paragraphs: List[Dict[str, Any]] = []
    para_nodes = root.xpath("//tei:text//tei:body//tei:p", namespaces=NS)

    cite_re = re.compile(r"__CITE_(\d+)__")

    def flatten_segments(segments: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        cite_idx = 0
        for seg in segments:
            if seg.get("type") == "text":
                parts.append(seg.get("text") or "")
            elif seg.get("type") == "citation":
                parts.append(f"__CITE_{cite_idx}__")
                cite_idx += 1
        return _collapse_ws(" ".join(p for p in parts if p)).strip()

    def stable_sentence_id(
        paragraph_id: Optional[str], index: int, sentence_text: str
    ) -> str:
        base = paragraph_id or "para"
        norm = _collapse_ws(sentence_text).strip()
        digest = hashlib.sha1(norm.encode("utf-8", errors="ignore")).hexdigest()[:10]
        return f"{base}-fb-{index}-{digest}"

    def fallback_segment_paragraph(
        paragraph_id: Optional[str], segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        stream_parts: List[str] = []
        for seg in segments:
            if seg.get("type") == "text":
                stream_parts.append(seg.get("text") or "")
            elif seg.get("type") == "citation":
                placeholder = f"__CITE_{len(citations)}__"
                citations.append(seg)
                stream_parts.append(placeholder)

        stream = _collapse_ws(" ".join(p for p in stream_parts if p)).strip()
        if not stream:
            return []

        sentence_texts = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", stream) if s and s.strip()
        ]
        if not sentence_texts:
            sentence_texts = [stream]

        out: List[Dict[str, Any]] = []
        for idx, sentence_text in enumerate(sentence_texts):
            sentence_id = stable_sentence_id(paragraph_id, idx, sentence_text)
            sentence_segments: List[Dict[str, Any]] = []

            cursor = 0
            for match in cite_re.finditer(sentence_text):
                before = sentence_text[cursor : match.start()]
                before_text = _collapse_ws(before)
                if before_text:
                    sentence_segments.append({"type": "text", "text": before_text})

                cite_idx = int(match.group(1))
                original = citations[cite_idx]
                cite_seg = dict(original)
                cite_seg["sentence_id"] = sentence_id
                sentence_segments.append(cite_seg)
                cursor = match.end()

            after = sentence_text[cursor:]
            after_text = _collapse_ws(after)
            if after_text:
                sentence_segments.append({"type": "text", "text": after_text})

            citation_indices = [
                seg.get("citation_index")
                for seg in sentence_segments
                if seg.get("type") == "citation"
                and seg.get("citation_index") is not None
            ]
            out.append(
                {
                    "sentence_id": sentence_id,
                    "segments": sentence_segments,
                    "citation_indices": citation_indices,
                }
            )

        return out

    def walk(elem: etree._Element, segments: List[Dict[str, Any]]) -> None:
        if elem.text:
            text = _collapse_ws(elem.text)
            if text:
                segments.append({"type": "text", "text": text})
        for child in elem:
            if (
                child.tag.endswith("ref")
                and (child.get("type") or "").strip() == "bibr"
            ):
                callout = _text_content(child)
                target = (child.get("target") or "").lstrip("#") or None
                citation_index = path_to_index.get(tree.getpath(child))
                segments.append(
                    {
                        "type": "citation",
                        "citation_index": citation_index,
                        "target_id": target,
                        "callout": callout,
                        "label": _label_callout(callout),
                        "sentence_id": _find_sentence_id(child),
                    }
                )
            else:
                walk(child, segments)
            if child.tail:
                tail = _collapse_ws(child.tail)
                if tail:
                    segments.append({"type": "text", "text": tail})

    for para in para_nodes:
        paragraph_id = _get_xml_id(para)
        sentence_nodes = para.xpath(".//tei:s", namespaces=NS)
        sentences: List[Dict[str, Any]] = []

        paragraph_segments: List[Dict[str, Any]] = []
        walk(para, paragraph_segments)

        if sentence_nodes:
            sent_segments_list: List[List[Dict[str, Any]]] = []
            sent_text_lens: List[int] = []
            for sent in sentence_nodes:
                segs: List[Dict[str, Any]] = []
                walk(sent, segs)
                sent_segments_list.append(segs)
                sent_text_lens.append(len(flatten_segments(segs)))

            paragraph_text_len = len(flatten_segments(paragraph_segments))
            suspicious = any(length > 480 for length in sent_text_lens) or (
                len(sentence_nodes) == 1 and paragraph_text_len > 600
            )

            if suspicious:
                sentences = fallback_segment_paragraph(paragraph_id, paragraph_segments)
            else:
                for sent, segs in zip(sentence_nodes, sent_segments_list):
                    sentence_id = _get_xml_id(sent) or paragraph_id
                    citation_indices = [
                        seg.get("citation_index")
                        for seg in segs
                        if seg.get("type") == "citation"
                        and seg.get("citation_index") is not None
                    ]
                    sentences.append(
                        {
                            "sentence_id": sentence_id,
                            "segments": segs,
                            "citation_indices": citation_indices,
                        }
                    )
        else:
            sentences = fallback_segment_paragraph(paragraph_id, paragraph_segments)
            if not sentences:
                sentence_id = paragraph_id
                citation_indices = [
                    seg.get("citation_index")
                    for seg in paragraph_segments
                    if seg.get("type") == "citation"
                    and seg.get("citation_index") is not None
                ]
                sentences.append(
                    {
                        "sentence_id": sentence_id,
                        "segments": paragraph_segments,
                        "citation_indices": citation_indices,
                    }
                )

        paragraphs.append({"paragraph_id": paragraph_id, "sentences": sentences})

    return {"paragraphs": paragraphs}
