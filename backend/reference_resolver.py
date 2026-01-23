from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from backend.settings import settings


_THROTTLE_SECONDS = 0.2


def _extract_year(message: Dict[str, Any]) -> Optional[str]:
    for field in ("issued", "published-print", "published-online", "created"):
        date_info = message.get(field, {})
        date_parts = date_info.get("date-parts") or []
        if date_parts and date_parts[0]:
            return str(date_parts[0][0])
    return None


def _extract_title(message: Dict[str, Any]) -> Optional[str]:
    titles = message.get("title") or []
    return titles[0] if titles else None


def _resolve_by_doi(doi: str) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    response = requests.get(
        f"{settings.CROSSREF_API_URL}/{doi}",
        params={"mailto": settings.CROSSREF_MAILTO},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json().get("message")
    return payload, 1.0


def _resolve_by_query(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    response = requests.get(
        settings.CROSSREF_API_URL,
        params={
            "query.bibliographic": raw,
            "rows": 1,
            "mailto": settings.CROSSREF_MAILTO,
        },
        timeout=10,
    )
    response.raise_for_status()
    message = response.json().get("message", {})
    items = message.get("items") or []
    if not items:
        return None, None
    top = items[0]
    return top, top.get("score")


def _normalize_entry(
    entry: Dict[str, Any], index: int
) -> Tuple[str, str, Optional[str]]:
    reference_id = entry.get("id") or entry.get("reference_id") or f"ref-{index + 1}"
    raw = entry.get("raw_reference") or entry.get("raw") or ""
    doi = entry.get("doi")
    return reference_id, raw, doi


def resolve_references(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not settings.CROSSREF_MAILTO:
        raise RuntimeError("CROSSREF_MAILTO must be configured to use Crossref")

    resolved: List[Dict[str, Any]] = []
    for index, entry in enumerate(entries):
        reference_id, raw, doi = _normalize_entry(entry, index)
        if not raw and not doi:
            resolved.append(
                {
                    "reference_id": reference_id,
                    "raw": raw,
                    "doi": None,
                    "title": None,
                    "publisher": None,
                    "year": None,
                    "source": "missing",
                    "confidence": None,
                }
            )
            continue

        if doi:
            message, confidence = _resolve_by_doi(doi)
            source = "doi"
        else:
            message, confidence = _resolve_by_query(raw)
            source = "query"

        resolved.append(
            {
                "reference_id": reference_id,
                "raw": raw,
                "doi": (message or {}).get("DOI") or doi,
                "title": _extract_title(message or {}),
                "publisher": (message or {}).get("publisher"),
                "year": _extract_year(message or {}),
                "source": source,
                "confidence": confidence,
            }
        )
        time.sleep(_THROTTLE_SECONDS)

    return resolved
