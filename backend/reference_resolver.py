from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
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


def _normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    normalized = doi.strip().lower()
    normalized = re.sub(r"^https?://(dx\.)?doi\.org/", "", normalized)
    normalized = normalized.replace("doi:", "").strip()
    return normalized or None


def _normalize_title(title: Optional[str]) -> Optional[str]:
    if not title:
        return None
    normalized = re.sub(r"\s+", " ", title).strip().lower()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return normalized or None


def _candidates_match(
    left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]
) -> bool:
    if not left or not right:
        return False
    left_doi = _normalize_doi(left.get("doi"))
    right_doi = _normalize_doi(right.get("doi"))
    if left_doi and right_doi:
        return left_doi == right_doi
    left_title = _normalize_title(left.get("title"))
    right_title = _normalize_title(right.get("title"))
    if left_title and right_title:
        return left_title == right_title
    return False


def compare_candidates(
    grobid: Optional[Dict[str, Any]],
    crossref: Optional[Dict[str, Any]],
    openalex: Optional[Dict[str, Any]],
) -> Tuple[str, Optional[str]]:
    if not crossref and not openalex:
        return "missing", "no_external_candidates"
    if crossref and openalex:
        crossref_match = _candidates_match(crossref, openalex)
        grobid_crossref = grobid and _candidates_match(grobid, crossref)
        grobid_openalex = grobid and _candidates_match(grobid, openalex)
        if crossref_match and (not grobid or (grobid_crossref and grobid_openalex)):
            return "match", None
        reasons = []
        if not crossref_match:
            reasons.append("crossref_openalex_disagree")
        if grobid and not grobid_crossref:
            reasons.append("grobid_crossref_disagree")
        if grobid and not grobid_openalex:
            reasons.append("grobid_openalex_disagree")
        return "mismatch", ", ".join(reasons) if reasons else "mismatch"
    candidate = crossref or openalex
    if grobid and _candidates_match(grobid, candidate):
        return "match", None
    return "needs_review", "single_source_disagreement" if grobid else "single_source"


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


def _openalex_params() -> Dict[str, str]:
    params: Dict[str, str] = {}
    if settings.OPENALEX_API_KEY:
        params["api_key"] = settings.OPENALEX_API_KEY
    if settings.CROSSREF_MAILTO:
        params["mailto"] = settings.CROSSREF_MAILTO
    return params


def _openalex_doi(work: Dict[str, Any]) -> Optional[str]:
    doi = work.get("doi")
    if not doi:
        doi = (work.get("ids") or {}).get("doi")
    return _normalize_doi(doi)


def _resolve_openalex_by_doi(
    doi: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    normalized = _normalize_doi(doi)
    if not normalized:
        return None, None
    params = _openalex_params()
    params["filter"] = f"doi:{normalized}"
    params["per-page"] = 1
    response = requests.get(settings.OPENALEX_API_URL, params=params, timeout=10)
    response.raise_for_status()
    results = response.json().get("results") or []
    if not results:
        return None, None
    return results[0], 1.0


def _resolve_openalex_by_title(
    query: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    if not query:
        return None, None
    params = _openalex_params()
    params["search"] = query
    params["per-page"] = 1
    response = requests.get(settings.OPENALEX_API_URL, params=params, timeout=10)
    response.raise_for_status()
    results = response.json().get("results") or []
    if not results:
        return None, None
    top = results[0]
    return top, top.get("relevance_score")


def _normalize_entry(
    entry: Dict[str, Any], index: int
) -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    reference_id = entry.get("id") or entry.get("reference_id") or f"ref-{index + 1}"
    raw = entry.get("raw_reference") or entry.get("raw") or ""
    doi = entry.get("doi")
    grobid = entry.get("grobid") or {}
    return reference_id, raw, doi, grobid


def _normalize_crossref_candidate(
    message: Optional[Dict[str, Any]],
    fallback_doi: Optional[str],
    source: str,
    confidence: Optional[float],
) -> Optional[Dict[str, Any]]:
    if not message and not fallback_doi:
        return None
    return {
        "doi": _normalize_doi((message or {}).get("DOI") or fallback_doi),
        "title": _extract_title(message or {}),
        "publisher": (message or {}).get("publisher"),
        "year": _extract_year(message or {}),
        "source": "crossref",
        "confidence": confidence,
        "via": source,
    }


def _normalize_openalex_candidate(
    work: Dict[str, Any], confidence: Optional[float]
) -> Dict[str, Any]:
    year = work.get("publication_year")
    return {
        "doi": _openalex_doi(work),
        "title": work.get("title") or work.get("display_name"),
        "year": str(year) if year else None,
        "source": "openalex",
        "confidence": confidence,
    }


def _resolve_crossref_candidate(
    raw: str, doi: Optional[str]
) -> Optional[Dict[str, Any]]:
    if doi:
        message, confidence = _resolve_by_doi(doi)
        source = "doi"
    else:
        message, confidence = _resolve_by_query(raw)
        source = "query"
    return _normalize_crossref_candidate(message, doi, source, confidence)


def _resolve_openalex_candidate(
    raw: str, doi: Optional[str]
) -> Optional[Dict[str, Any]]:
    try:
        if doi:
            work, confidence = _resolve_openalex_by_doi(doi)
        else:
            work, confidence = _resolve_openalex_by_title(raw)
    except requests.RequestException:
        return None
    if not work:
        return None
    return _normalize_openalex_candidate(work, confidence)


def _select_default_source(
    crossref: Optional[Dict[str, Any]],
    openalex: Optional[Dict[str, Any]],
    grobid: Optional[Dict[str, Any]],
) -> Optional[str]:
    if crossref:
        return "crossref"
    if openalex:
        return "openalex"
    if grobid:
        return "grobid"
    return None


def apply_resolution_selection(
    entry: Dict[str, Any],
    selected_source: str,
    *,
    override_status: bool = False,
) -> Dict[str, Any]:
    candidates = {
        "grobid": entry.get("grobid"),
        "crossref": entry.get("crossref"),
        "openalex": entry.get("openalex"),
    }
    candidate = candidates.get(selected_source)
    if not candidate:
        raise ValueError(f"Unknown or missing candidate source: {selected_source}")
    updated = dict(entry)
    updated.update(
        {
            "selected_source": selected_source,
            "doi": candidate.get("doi"),
            "title": candidate.get("title"),
            "year": candidate.get("year"),
            "source": candidate.get("source") or selected_source,
            "confidence": candidate.get("confidence"),
            "publisher": candidate.get("publisher"),
        }
    )
    if override_status:
        updated["status"] = "match"
        updated["mismatch_reason"] = None
    return updated


def resolve_references(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not settings.CROSSREF_MAILTO:
        raise RuntimeError("CROSSREF_MAILTO must be configured to use Crossref")

    resolved: List[Dict[str, Any]] = []
    for index, entry in enumerate(entries):
        reference_id, raw, doi, grobid = _normalize_entry(entry, index)
        query = grobid.get("title") or raw
        if not query and not doi:
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
                    "grobid": grobid,
                    "crossref": None,
                    "openalex": None,
                    "status": "missing",
                    "mismatch_reason": "no_reference_data",
                    "selected_source": None,
                }
            )
            continue

        with ThreadPoolExecutor(max_workers=2) as executor:
            crossref_future = executor.submit(_resolve_crossref_candidate, query, doi)
            openalex_future = executor.submit(_resolve_openalex_candidate, query, doi)
            crossref = crossref_future.result()
            openalex = openalex_future.result()

        status, reason = compare_candidates(grobid, crossref, openalex)
        selected_source = _select_default_source(crossref, openalex, grobid)
        base_entry = {
            "reference_id": reference_id,
            "raw": raw,
            "grobid": grobid,
            "crossref": crossref,
            "openalex": openalex,
            "status": status,
            "mismatch_reason": reason,
            "selected_source": selected_source,
        }
        if selected_source:
            resolved_entry = apply_resolution_selection(
                base_entry, selected_source, override_status=False
            )
        else:
            resolved_entry = {
                **base_entry,
                "doi": None,
                "title": None,
                "publisher": None,
                "year": None,
                "source": None,
                "confidence": None,
            }
        resolved.append(resolved_entry)
        time.sleep(_THROTTLE_SECONDS)

    return resolved
