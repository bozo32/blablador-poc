"""Serializer utilities for evidence matching outputs."""

from __future__ import annotations

from typing import Sequence

from .types import EvidenceCandidate

DEFAULT_SNIPPET_LIMIT = 600


def serialize_candidates(
    candidates: Sequence[EvidenceCandidate],
    *,
    limit: int | None = None,
    max_text: int = DEFAULT_SNIPPET_LIMIT,
) -> list[dict]:
    """Serialize candidates into JSON-friendly payloads."""
    window = candidates if limit is None else candidates[:limit]
    return [serialize_candidate(candidate, max_text=max_text) for candidate in window]


def serialize_candidate(
    candidate: EvidenceCandidate, *, max_text: int = DEFAULT_SNIPPET_LIMIT
) -> dict:
    """Serialize a single candidate with trimmed snippet and metadata."""
    metadata = dict(candidate.metadata or {})
    metadata.setdefault("page", candidate.primary_page)
    metadata.setdefault("section", metadata.get("section") or "Unknown")
    metadata.setdefault("attachment_id", candidate.attachment_id)
    metadata["bbox_count"] = _bbox_count(candidate)

    payload = {
        "id": candidate.id,
        "claim_id": candidate.claim_id,
        "attachment_id": candidate.attachment_id,
        "label": candidate.label.value,
        "text": _truncate(candidate.text, max_text),
        "scores": candidate.scores.to_dict(),
        "badges": list(candidate.badges),
        "metadata": metadata,
        "spans": [span.to_dict() for span in candidate.spans],
    }
    if candidate.token_saliencies is not None:
        payload["token_saliencies"] = candidate.token_saliencies
    if candidate.highlights:
        payload["highlights"] = list(candidate.highlights)
    else:
        payload["highlights"] = _build_highlights(candidate)
    return payload


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    trimmed = text[: max(limit - 1, 1)].rstrip()
    return f"{trimmed}…"


def _bbox_count(candidate: EvidenceCandidate) -> int:
    total = 0
    for span in candidate.spans:
        total += len(span.bbox)
    return total


def _build_highlights(candidate: EvidenceCandidate) -> list[dict]:
    highlights = []
    for span in candidate.spans:
        highlights.append(
            {
                "text": span.text,
                "page": span.page,
                "section": span.section,
            }
        )
    return highlights


__all__ = ["serialize_candidate", "serialize_candidates"]
