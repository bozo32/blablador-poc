"""Utilities for building reference retrieval dossiers.

These helpers consolidate information from extraction (GROBID), resolution
payloads (Crossref/OpenAlex), and ingestion metadata to generate an actionable
retrieval dossier for reviewers. The dossier powers the
`GET /references/{doc_id}/{reference_id}/retrieval` endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Optional

from backend import schemas
from backend.settings import settings
from backend.spine.ingest_view import build_ingested_document_from_spine


@dataclass
class RetrievalSource:
    """Normalized representation of one metadata source used for retrieval."""

    label: str
    title: Optional[str]
    doi: Optional[str]
    url: Optional[str]
    confidence: Optional[float]
    source: Optional[str]


def _load_document(doc_id: str) -> Dict[str, Any]:
    did = str(doc_id or "").strip()
    if not did:
        raise FileNotFoundError("Document id is required")
    project_id = str(settings.DEFAULT_PROJECT_ID)
    document = build_ingested_document_from_spine(
        work_id=did,
        project_id=project_id,
        include_extraction_data=True,
    )
    if not document:
        raise FileNotFoundError(f"Document {did} not found")
    return document


def _find_reference_entry(
    document: Dict[str, Any], reference_id: str
) -> Dict[str, Any]:
    extraction = (document.get("extraction") or {}).get("data") or {}
    references = extraction.get("references") or []
    for ref in references:
        if ref.get("id") == reference_id:
            return ref
    raise FileNotFoundError(
        "Reference {ref} missing in extraction data for doc {doc}".format(
            ref=reference_id,
            doc=document.get("id"),
        )
    )


def _find_resolution_entry(
    document: Dict[str, Any], reference_id: str
) -> Optional[Dict[str, Any]]:
    resolution = document.get("resolution") or {}
    for entry in resolution.get("data") or []:
        if entry.get("reference_id") == reference_id:
            return entry
    return None


def _normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    normalized = doi.strip()
    normalized = re.sub(r"^https?://(dx\.)?doi\.org/", "", normalized, flags=re.I)
    normalized = normalized.replace("doi:", "").strip()
    return normalized or None


def _provider_url(source: Dict[str, Any]) -> Optional[str]:
    for key in ("publisher_url", "url"):
        value = source.get(key)
        if value:
            return value
    return None


def _select_best_url(entry: Dict[str, Any], reference: Dict[str, Any]) -> Optional[str]:
    # Prefer explicit publisher URL from the selected entry
    primary = _provider_url(entry)
    if primary:
        return primary

    selected_source = entry.get("selected_source")
    if selected_source:
        provider_entry = entry.get(selected_source) or {}
        provider_url = _provider_url(provider_entry)
        if provider_url:
            return provider_url

    for provider in ("crossref", "openalex", "grobid"):
        provider_entry = entry.get(provider) or {}
        provider_url = _provider_url(provider_entry)
        if provider_url:
            return provider_url

    ref_url = reference.get("url")
    if ref_url:
        return ref_url
    grobid_url = _provider_url(reference.get("grobid") or {})
    if grobid_url:
        return grobid_url

    doi = _normalize_doi((entry or {}).get("doi") or reference.get("doi"))
    if doi:
        return f"https://doi.org/{doi}"

    return None


def _canonical_citation(
    reference: Dict[str, Any], entry: Optional[Dict[str, Any]]
) -> str:
    grobid = reference.get("grobid") or (entry or {}).get("grobid") or {}
    authors = grobid.get("authors")
    year = grobid.get("year") or grobid.get("date")
    title = grobid.get("title") or entry.get("title") if entry else None

    author_text = _format_authors(authors)
    parts = []
    if author_text:
        parts.append(author_text)
    if year:
        parts.append(f"({year})")
    if title:
        parts.append(title)
    if not parts:
        return reference.get("raw_reference") or "Citation metadata unavailable"
    return " ".join(parts)


def _format_authors(authors: Any) -> Optional[str]:  # type: ignore[override]
    if not authors:
        return None
    if isinstance(authors, str):
        return authors
    if isinstance(authors, list):
        names = []
        for author in authors:
            if isinstance(author, str):
                names.append(author)
            elif isinstance(author, dict):
                full = author.get("full_name") or author.get("name")
                if full:
                    names.append(full)
                else:
                    parts = [
                        author.get("given_name") or author.get("first_name"),
                        author.get("surname") or author.get("last_name"),
                    ]
                    full = " ".join(filter(None, parts)).strip()
                    if full:
                        names.append(full)
        return "; ".join(names) if names else None
    return None


def _fallback_instructions(
    reference: Dict[str, Any], entry: Optional[Dict[str, Any]]
) -> str:
    doi = _normalize_doi((entry or {}).get("doi") or reference.get("doi"))
    raw = reference.get("raw_reference")
    lines = [
        "No direct link available. Use the bibliography entry to search your "
        "library catalog or Google Scholar.",
    ]
    if raw:
        lines.append(raw)
    if doi:
        lines.append(f"DOI: https://doi.org/{doi}")
    lines.append(
        (
            "If unavailable online, request the PDF via your library or the "
            "corresponding author."
        )
    )
    return "\n".join(lines)


def _source_from_entry(
    label: str, entry: Optional[Dict[str, Any]]
) -> Optional[RetrievalSource]:
    if not entry:
        return None
    return RetrievalSource(
        label=label,
        title=entry.get("title"),
        doi=entry.get("doi"),
        url=entry.get("url") or entry.get("publisher_url"),
        confidence=entry.get("confidence"),
        source=entry.get("source") or label.lower(),
    )


def _resolve_sources(
    entry: Optional[Dict[str, Any]], reference: Dict[str, Any]
) -> list[RetrievalSource]:
    sources: list[RetrievalSource] = []
    if entry:
        for label in ("grobid", "crossref", "openalex"):
            candidate = entry.get(label)
            source_obj = _source_from_entry(label.capitalize(), candidate)
            if source_obj:
                sources.append(source_obj)
    else:
        grobid_source = _source_from_entry("Grobid", reference.get("grobid"))
        if grobid_source:
            sources.append(grobid_source)
    return sources


def build_retrieval_dossier(
    doc_id: str, reference_id: str
) -> schemas.ReferenceRetrievalResponse:
    document = _load_document(doc_id)
    reference = _find_reference_entry(document, reference_id)
    resolution_entry = _find_resolution_entry(document, reference_id)

    canonical = _canonical_citation(reference, resolution_entry)
    doi = _normalize_doi((resolution_entry or {}).get("doi") or reference.get("doi"))
    primary_url = _select_best_url(resolution_entry or {}, reference)
    manual_notes = None
    if not primary_url:
        manual_notes = _fallback_instructions(reference, resolution_entry)

    sources = _resolve_sources(resolution_entry, reference)

    payload = schemas.ReferenceRetrievalResponse(
        document_id=document.get("id", doc_id),
        reference_id=reference_id,
        canonical_citation=canonical,
        doi=doi,
        primary_url=primary_url,
        manual_instructions=manual_notes,
        resolver_status=(resolution_entry or {}).get("status"),
        resolver_confidence=(resolution_entry or {}).get("confidence"),
        sources=[
            schemas.ReferenceRetrievalSource(
                label=src.label,
                title=src.title,
                doi=src.doi,
                url=src.url,
                confidence=src.confidence,
                source=src.source,
            )
            for src in sources
        ],
    )
    return payload


__all__ = ["build_retrieval_dossier"]
