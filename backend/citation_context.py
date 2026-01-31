from __future__ import annotations

import re
from typing import Dict, List, Optional

from lxml import etree

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


def _text_content(elem: etree._Element) -> str:
    return _collapse_ws("".join(elem.itertext()))


def _parse_tei_xml(tei_xml: str | bytes) -> etree._Element:
    if isinstance(tei_xml, bytes):
        tei_xml = tei_xml.decode("utf-8", errors="ignore")
    parser = etree.XMLParser(recover=True)
    return etree.fromstring(tei_xml.encode("utf-8"), parser=parser)


def _sentence_index(root: etree._Element) -> Dict[etree._Element, int]:
    sentences = root.xpath("//tei:text//tei:body//tei:s", namespaces=TEI_NS)
    return {sentence: index for index, sentence in enumerate(sentences)}


def _callouts(root: etree._Element) -> List[Dict[str, Optional[str]]]:
    callouts: List[Dict[str, Optional[str]]] = []
    refs = root.xpath(
        "//tei:text//tei:body//tei:ref[@type='bibr']",
        namespaces=TEI_NS,
    )
    for ref in refs:
        target_id = (ref.get("target") or "").lstrip("#") or None
        callouts.append(
            {
                "target_id": target_id,
                "callout": _text_content(ref),
                "ref_elem": ref,
                "sentence_elem": ref.xpath("ancestor::tei:s[1]", namespaces=TEI_NS),
            }
        )
    return callouts


def _text_with_marker(
    root: etree._Element, marker_elem: etree._Element, token: str
) -> str:
    parts: List[str] = []

    def walk(elem: etree._Element) -> None:
        if elem.text:
            parts.append(elem.text)
        for child in elem:
            if child is marker_elem:
                parts.append(token)
            else:
                walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(root)
    return _collapse_ws("".join(parts))


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def get_citation_context(
    tei_xml: str | bytes,
    citation_index: int,
    target_id: Optional[str] = None,
) -> Optional[Dict[str, Optional[str]]]:
    root = _parse_tei_xml(tei_xml)
    sentence_lookup = _sentence_index(root)
    callouts = _callouts(root)

    callout = None
    if 0 <= citation_index < len(callouts):
        callout = callouts[citation_index]
    elif target_id:
        for item in callouts:
            if item.get("target_id") == target_id:
                callout = item
                break

    if callout is None:
        return None
    ref_elem = callout.get("ref_elem")
    sentence_elem_list = callout.get("sentence_elem") or []
    sentence_elem = sentence_elem_list[0] if sentence_elem_list else None

    if sentence_elem is None or sentence_elem not in sentence_lookup:
        return {
            "target_id": callout.get("target_id"),
            "callout": callout.get("callout"),
            "sentence": None,
            "citing_sentence": None,
            "previous_sentence": None,
            "next_sentence": None,
            "sentence_id": None,
        }

    sentence_idx = sentence_lookup[sentence_elem]
    sentences = list(sentence_lookup.keys())
    prev_sentence = sentences[sentence_idx - 1] if sentence_idx > 0 else None
    next_sentence = (
        sentences[sentence_idx + 1] if sentence_idx + 1 < len(sentences) else None
    )
    sentence_id = sentence_elem.get(
        "{http://www.w3.org/XML/1998/namespace}id"
    ) or sentence_elem.get("xml:id")

    citing_sentence = None
    snippet = None
    if ref_elem is not None:
        token = "<<<CITATION_MARKER>>>"
        marked = _text_with_marker(sentence_elem, ref_elem, token)
        candidates = _split_sentences(marked)
        for candidate in candidates:
            if token in candidate:
                citing_sentence = candidate.replace(token, callout.get("callout") or "")
                snippet = candidate.replace(token, "[citation]")
                break
        if citing_sentence is None:
            citing_sentence = marked.replace(token, callout.get("callout") or "")
            snippet = marked.replace(token, "[citation]")

    return {
        "target_id": callout.get("target_id"),
        "callout": callout.get("callout"),
        "sentence": _text_content(sentence_elem),
        "citing_sentence": citing_sentence,
        "citing_snippet": snippet,
        "previous_sentence": _text_content(prev_sentence) if prev_sentence else None,
        "next_sentence": _text_content(next_sentence) if next_sentence else None,
        "sentence_id": sentence_id,
    }
