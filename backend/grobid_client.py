from __future__ import annotations

import random
import threading
import time
from pathlib import Path
from typing import Union

import requests
from lxml import etree

from backend.settings import settings


class GrobidError(RuntimeError):
    pass


_SEM = threading.BoundedSemaphore(
    max(1, int(getattr(settings, "GROBID_MAX_CONCURRENT", 1) or 1))
)


def _post_grobid(*, endpoint: str, pdf_path: Path) -> str:
    url = f"{settings.GROBID_URL.rstrip('/')}{endpoint}"
    consolidate_citations = bool(
        getattr(settings, "GROBID_CONSOLIDATE_CITATIONS", False)
    )
    consolidate_header = bool(getattr(settings, "GROBID_CONSOLIDATE_HEADER", False))
    retries = max(0, int(getattr(settings, "GROBID_RETRY_503", 0) or 0))

    attempt = 0
    last_resp = None
    while True:
        attempt += 1
        try:
            with _SEM:
                with pdf_path.open("rb") as pdf_file:
                    files = {"input": (pdf_path.name, pdf_file, "application/pdf")}
                    last_resp = requests.post(
                        url,
                        files=files,
                        data={
                            "consolidateCitations": "1"
                            if consolidate_citations
                            else "0",
                            "consolidateHeader": "1" if consolidate_header else "0",
                        },
                        timeout=settings.GROBID_TIMEOUT,
                    )
        except requests.RequestException as exc:
            if attempt <= retries:
                backoff = min(8.0, (2 ** (attempt - 1)) * 0.5) + random.random() * 0.25
                time.sleep(backoff)
                continue
            raise GrobidError(f"GROBID request failed ({url}): {exc}") from exc

        if last_resp.status_code == 503 and attempt <= retries:
            backoff = min(8.0, (2 ** (attempt - 1)) * 0.5) + random.random() * 0.25
            time.sleep(backoff)
            continue
        break

    if last_resp is None or last_resp.status_code != 200:
        message = (last_resp.text.strip() if last_resp is not None else "").strip()
        status = last_resp.status_code if last_resp is not None else "no_response"
        raise GrobidError(f"GROBID extraction failed ({status}): {message}")

    return last_resp.text


def extract_tei_fulltext(pdf_path: Union[str, Path]) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    return _post_grobid(endpoint="/api/processFulltextDocument", pdf_path=path)


def extract_tei_header(pdf_path: Union[str, Path]) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    return _post_grobid(endpoint="/api/processHeaderDocument", pdf_path=path)


def extract_tei_references(pdf_path: Union[str, Path]) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    return _post_grobid(endpoint="/api/processReferences", pdf_path=path)


def merge_header_and_references_tei(*, header_xml: str, references_xml: str) -> str:
    """Merge header TEI into a references TEI document."""
    parser = etree.XMLParser(recover=True)
    header_root = etree.fromstring(header_xml.encode("utf-8"), parser=parser)
    refs_root = etree.fromstring(references_xml.encode("utf-8"), parser=parser)

    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    header_node = header_root.xpath("//tei:teiHeader", namespaces=ns)
    refs_node = refs_root.xpath("//tei:teiHeader", namespaces=ns)
    if header_node:
        new_header = header_node[0]
        if refs_node:
            parent = refs_node[0].getparent()
            if parent is not None:
                parent.replace(refs_node[0], new_header)
        else:
            refs_root.insert(0, new_header)

    return etree.tostring(refs_root, encoding="utf-8", xml_declaration=False).decode(
        "utf-8", errors="ignore"
    )


def extract_tei(pdf_path: Union[str, Path]) -> str:
    """Backward-compatible: fulltext TEI only."""
    return extract_tei_fulltext(pdf_path)
