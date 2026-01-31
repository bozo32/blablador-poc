from __future__ import annotations

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

        if sentence_nodes:
            for sent in sentence_nodes:
                sentence_id = _get_xml_id(sent) or paragraph_id
                segments: List[Dict[str, Any]] = []
                walk(sent, segments)
                citation_indices = [
                    seg.get("citation_index")
                    for seg in segments
                    if seg.get("type") == "citation"
                    and seg.get("citation_index") is not None
                ]
                sentences.append(
                    {
                        "sentence_id": sentence_id,
                        "segments": segments,
                        "citation_indices": citation_indices,
                    }
                )
        else:
            # Fallback: paragraph has no explicit sentence nodes.
            sentence_id = paragraph_id
            segments = []
            walk(para, segments)
            citation_indices = [
                seg.get("citation_index")
                for seg in segments
                if seg.get("type") == "citation"
                and seg.get("citation_index") is not None
            ]
            sentences.append(
                {
                    "sentence_id": sentence_id,
                    "segments": segments,
                    "citation_indices": citation_indices,
                }
            )

        paragraphs.append({"paragraph_id": paragraph_id, "sentences": sentences})

    return {"paragraphs": paragraphs}
