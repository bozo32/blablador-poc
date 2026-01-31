from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from lxml import etree

NS = {"tei": "http://www.tei-c.org/ns/1.0"}
XML_NS = "http://www.w3.org/XML/1998/namespace"
EXTRACTION_VERSION = "v1"


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


def _text_content(elem: etree._Element) -> str:
    return _collapse_ws("".join(elem.itertext()))


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


def _first_text(root: etree._Element, xpath: str) -> Optional[str]:
    matches = root.xpath(xpath, namespaces=NS)
    for match in matches:
        if isinstance(match, etree._Element):
            text = _text_content(match)
        else:
            text = str(match).strip()
        if text:
            return text
    return None


def _extract_year(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    match = re.search(r"\d{4}", value)
    return match.group(0) if match else None


def _parse_reference_authors(entry: etree._Element) -> List[str]:
    authors: List[str] = []
    for author in entry.xpath(".//tei:author", namespaces=NS):
        forename = _first_text(author, ".//tei:forename")
        surname = _first_text(author, ".//tei:surname")
        name_parts = [part for part in (forename, surname) if part]
        name = " ".join(name_parts) if name_parts else _text_content(author)
        if name:
            authors.append(name)
    return authors


def _parse_reference_title(entry: etree._Element) -> Optional[str]:
    return _first_text(entry, ".//tei:analytic//tei:title") or _first_text(
        entry, ".//tei:monogr//tei:title"
    )


def _parse_reference_container(
    entry: etree._Element,
) -> tuple[Optional[str], Optional[str]]:
    journal = _first_text(entry, ".//tei:monogr//tei:title[@level='j']")
    container = _first_text(
        entry, ".//tei:monogr//tei:title[@level='m']"
    ) or _first_text(entry, ".//tei:monogr//tei:title")
    return journal, container


def _parse_reference_year(entry: etree._Element) -> Optional[str]:
    date_value = _first_text(entry, ".//tei:imprint//tei:date/@when") or _first_text(
        entry, ".//tei:imprint//tei:date"
    )
    return _extract_year(date_value)


def _parse_grobid_reference(entry: etree._Element) -> Dict[str, Any]:
    title = _parse_reference_title(entry)
    authors = _parse_reference_authors(entry)
    journal, container = _parse_reference_container(entry)
    year = _parse_reference_year(entry)
    doi = _first_text(entry, ".//tei:idno[@type='DOI']")
    url = _first_text(entry, ".//tei:idno[@type='URL']")
    if url is None:
        url = _first_text(entry, ".//tei:ptr/@target")
    return {
        "title": title,
        "authors": authors,
        "year": year,
        "journal": journal,
        "container": container,
        "doi": doi,
        "url": url,
    }


def parse_metadata(tei_root: etree._Element) -> Dict[str, Any]:
    title = _first_text(
        tei_root,
        "//tei:teiHeader//tei:fileDesc//tei:titleStmt//tei:title",
    )

    authors: List[str] = []
    for author in tei_root.xpath("//tei:teiHeader//tei:author", namespaces=NS):
        forename = _first_text(author, ".//tei:forename")
        surname = _first_text(author, ".//tei:surname")
        name_parts = [part for part in (forename, surname) if part]
        name = " ".join(name_parts) if name_parts else _text_content(author)
        if name:
            authors.append(name)

    journal = _first_text(
        tei_root,
        "//tei:teiHeader//tei:sourceDesc//tei:biblStruct"
        "//tei:monogr//tei:title[@level='j']",
    )
    container = _first_text(
        tei_root,
        "//tei:teiHeader//tei:sourceDesc//tei:biblStruct"
        "//tei:monogr//tei:title[@level='m']",
    ) or _first_text(
        tei_root,
        "//tei:teiHeader//tei:sourceDesc//tei:biblStruct//tei:monogr//tei:title",
    )

    date_value = (
        _first_text(
            tei_root,
            "//tei:teiHeader//tei:sourceDesc//tei:biblStruct"
            "//tei:monogr//tei:imprint//tei:date/@when",
        )
        or _first_text(
            tei_root,
            "//tei:teiHeader//tei:publicationStmt//tei:date/@when",
        )
        or _first_text(
            tei_root,
            "//tei:teiHeader//tei:sourceDesc//tei:biblStruct"
            "//tei:monogr//tei:imprint//tei:date",
        )
    )

    return {
        "title": title,
        "authors": authors,
        "year": _extract_year(date_value),
        "journal": journal,
        "container": container,
    }


def parse_citations(tei_root: etree._Element) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    refs = tei_root.xpath(
        "//tei:text//tei:body//tei:ref[@type='bibr']",
        namespaces=NS,
    )
    for ref in refs:
        target = (ref.get("target") or "").lstrip("#") or None
        callout = _text_content(ref)

        # Drop footnote-like numeric callouts without a resolvable target.
        if target is None and callout.strip().isdigit():
            continue

        sentence_elem = ref.xpath("ancestor::tei:s[1]", namespaces=NS)
        paragraph_elem = ref.xpath("ancestor::tei:p[1]", namespaces=NS)
        context_elem = sentence_elem[0] if sentence_elem else None
        if context_elem is None and paragraph_elem:
            context_elem = paragraph_elem[0]

        context_text = None
        if context_elem is not None:
            token = "<<<CITATION_MARKER>>>"
            marked = _text_with_marker(context_elem, ref, token)
            candidates = _split_sentences(marked)
            for candidate in candidates:
                if token in candidate:
                    context_text = candidate.replace(token, callout)
                    break
            if context_text is None:
                context_text = marked.replace(token, callout)

        sentence_id = None
        if context_elem is not None:
            sentence_id = context_elem.get(f"{{{XML_NS}}}id") or context_elem.get(
                "xml:id"
            )
        citations.append(
            {
                "target_id": target,
                "callout": callout,
                "sentence": context_text,
                "sentence_id": sentence_id,
            }
        )
    return citations


def parse_bibliography(tei_root: etree._Element) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for entry in tei_root.xpath("//tei:listBibl//tei:biblStruct", namespaces=NS):
        entry_id = entry.get("{http://www.w3.org/XML/1998/namespace}id")
        raw_reference = _text_content(entry)
        doi = _first_text(entry, ".//tei:idno[@type='DOI']")
        url = _first_text(entry, ".//tei:idno[@type='URL']")
        if url is None:
            url = _first_text(entry, ".//tei:ptr/@target")
        grobid = _parse_grobid_reference(entry)
        entries.append(
            {
                "id": entry_id,
                "raw_reference": raw_reference,
                "doi": doi,
                "url": url,
                "grobid": grobid,
            }
        )
    return entries


def parse_tei(tei_xml: str) -> Dict[str, Any]:
    if isinstance(tei_xml, bytes):
        tei_xml = tei_xml.decode("utf-8", errors="ignore")

    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(tei_xml.encode("utf-8"), parser=parser)

    return {
        "metadata": parse_metadata(root),
        "citations": parse_citations(root),
        "references": parse_bibliography(root),
        "extraction_version": EXTRACTION_VERSION,
    }
