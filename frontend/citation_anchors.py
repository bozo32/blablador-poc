from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional


def normalize_cited_work_id(value: Optional[str]) -> Optional[str]:
    """Normalize a cited-work identifier to a stable string."""
    text = str(value or "").strip()
    if not text:
        return None
    lower = text.lower()
    if lower.startswith("https://doi.org/"):
        return text
    if lower.startswith("doi:"):
        tail = text[4:].strip()
        return f"https://doi.org/{tail}" if tail else None
    if lower.startswith("10."):
        return f"https://doi.org/{text}"
    if "openalex.org/" in lower:
        return text.rsplit("/", 1)[-1]
    return text


def build_anchor_quote(window_text: str) -> dict:
    """Create a compact quote selector near the end of a window."""
    norm = " ".join(str(window_text or "").split()).strip()
    tokens = [t for t in norm.split(" ") if t]
    if not tokens:
        return {"exact": None, "prefix": None, "suffix": None}
    exact_tokens = tokens[-14:]
    prefix_tokens = tokens[-28:-14]
    return {
        "exact": " ".join(exact_tokens).strip() or None,
        "prefix": " ".join(prefix_tokens).strip() or None,
        "suffix": None,
    }


def maybe_attach_citation_anchor(
    *,
    provenance: dict,
    context: Optional[dict],
) -> dict:
    """Attach stable citation anchoring fields to a provenance dict."""
    if not isinstance(context, dict):
        return provenance

    resolution = (
        context.get("resolution") if isinstance(context.get("resolution"), dict) else {}
    )
    reference = (
        context.get("reference") if isinstance(context.get("reference"), dict) else {}
    )

    cited_work_id = normalize_cited_work_id(
        (resolution or {}).get("doi")
        or (resolution or {}).get("openalex_id")
        or (resolution or {}).get("openalex_work_id")
        or (reference or {}).get("doi")
    )

    prev_sentence = str(context.get("previous_sentence") or "").strip()
    citing_prefix = str(context.get("citing_prefix") or "").strip()
    window_text = " ".join(
        part for part in [prev_sentence, citing_prefix] if part
    ).strip()
    window_norm = " ".join(window_text.split()).strip()

    if not cited_work_id and not window_norm:
        return provenance

    window_fingerprint = (
        hashlib.sha256(window_norm.encode("utf-8")).hexdigest() if window_norm else None
    )
    anchor_quote = (
        build_anchor_quote(window_norm)
        if window_norm
        else {"exact": None, "prefix": None, "suffix": None}
    )

    anchor: Dict[str, Any] = {
        "version": 1,
        "citing_doc_id": provenance.get("doc_id"),
        "cited_work_id": cited_work_id,
        "citation_index": provenance.get("citation_index"),
        "target_id": provenance.get("target_id"),
        "window_policy": {
            "kind": "preceding_text",
            "parts": ["previous_sentence", "citing_prefix"],
        },
        "window_fingerprint": window_fingerprint,
        "anchor_quote": anchor_quote,
    }

    merged = dict(provenance)
    merged["cited_work_id"] = cited_work_id
    merged["citation_anchor"] = anchor
    return merged
