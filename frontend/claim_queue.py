"""Claim queue helpers for retrieval instructions and attachments."""

from __future__ import annotations

import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Optional

import streamlit as st

from frontend import evidence_store
from frontend.clipboard import render_copy_to_clipboard
from frontend.ingestion_api import get_reference_retrieval


CLAIM_REGISTRY_KEY = "claim_queue_records"
CLAIM_REGISTRY_ORDER_KEY = "claim_queue_order"
CLAIM_TIMELINE_KEY = "claim_queue_timelines"
TIMELINE_MAX_EVENTS = 5


def _registry() -> Dict[str, dict]:
    return st.session_state.setdefault(CLAIM_REGISTRY_KEY, {})


def _registry_order() -> List[str]:
    return st.session_state.setdefault(CLAIM_REGISTRY_ORDER_KEY, [])


def init_claim_registry() -> None:
    _ = _registry()
    _ = _registry_order()
    st.session_state.setdefault(CLAIM_TIMELINE_KEY, {})


def register_claim(claim_id: Optional[str] = None, **metadata) -> dict:
    init_claim_registry()
    claim_id = claim_id or metadata.pop("id", None)
    if not claim_id:
        raise ValueError("register_claim requires a claim_id or id field")
    registry = _registry()
    record = registry.get(claim_id, {})
    record.update(
        {
            "id": claim_id,
            "claim": metadata.get("claim", record.get("claim")),
            "callout": metadata.get("callout", record.get("callout")),
            "doc_id": metadata.get("doc_id", record.get("doc_id")),
            "reference_id": metadata.get("reference_id", record.get("reference_id")),
            "doi": metadata.get("doi", record.get("doi")),
            "author": metadata.get("author", record.get("author")),
            "year": metadata.get("year", record.get("year")),
            "reference_hint": metadata.get(
                "reference_hint", record.get("reference_hint", {})
            ),
        }
    )
    registry[claim_id] = record
    order = _registry_order()
    if claim_id not in order:
        order.append(claim_id)
    _sync_evidence_store_metadata(record)
    return record


def sync_claims_from_results(results: Optional[dict]) -> None:
    """Populate registry entries from segmentation or backend results."""
    if not isinstance(results, dict):
        return
    for row_id, payload in results.items():
        segments = (payload or {}).get("segments") or []
        doc_id = (payload or {}).get("document_id")
        reference_id = (payload or {}).get("reference_id")
        for segment in segments:
            segment_id = segment.get("segment_id") or segment.get("id") or "segment"
            claim_id = f"{row_id}:{segment_id}"
            reference_hint = {
                "callout": segment.get("callout"),
                "reference_id": segment.get("reference_id") or reference_id,
                "doi": segment.get("doi"),
                "author": segment.get("author"),
                "year": segment.get("year"),
            }
            register_claim(
                claim_id,
                claim=segment.get("claim") or segment.get("text"),
                callout=segment.get("callout"),
                doc_id=doc_id,
                reference_id=segment.get("reference_id") or reference_id,
                doi=segment.get("doi"),
                author=segment.get("author"),
                year=segment.get("year"),
                reference_hint=reference_hint,
            )


def ensure_demo_claims() -> None:
    """Seed the registry with sample claims when none exist."""
    init_claim_registry()
    if _registry():
        return
    demo_claims = [
        {
            "id": "demo:1a",
            "claim": "1a. Renewable energy adoption reduces peak grid demand",
            "callout": "(Smith 2019)",
            "reference_id": "smith2019",
            "author": "Smith",
            "year": "2019",
        },
        {
            "id": "demo:1b",
            "claim": "1b. Battery storage offsets intermittency",
            "callout": "(Lopez et al. 2021)",
            "reference_id": "lopez2021",
            "author": "Lopez",
            "year": "2021",
        },
        {
            "id": "demo:2a",
            "claim": "2a. Retrofitting buildings cuts heating demand by 30%",
            "callout": "[2a]",
            "reference_id": "retrofit2020",
            "author": "Nguyen",
            "year": "2020",
        },
    ]
    for entry in demo_claims:
        register_claim(**entry)


def get_claim_records() -> List[dict]:
    init_claim_registry()
    registry = _registry()
    order = _registry_order()
    return [registry[key] for key in order if key in registry]


def get_claim_record(claim_id: str) -> Optional[dict]:
    init_claim_registry()
    return _registry().get(claim_id)


def get_claim_options(max_claim_chars: int = 60) -> List[dict]:
    options: List[dict] = []
    for record in get_claim_records():
        claim_id = record.get("id")
        if not claim_id:
            continue
        label = _claim_option_label(record, max_claim_chars=max_claim_chars)
        options.append({"id": claim_id, "label": label})
    return options


def set_active_claim(claim_id: Optional[str]) -> Optional[str]:
    if claim_id:
        init_claim_registry()
        if claim_id not in _registry():
            raise KeyError(f"Unknown claim: {claim_id}")
    return evidence_store.set_active_claim(claim_id)


def _claim_option_label(record: dict, *, max_claim_chars: int = 60) -> str:
    callout = record.get("callout") or "Unlabeled citation"
    claim_id = str(record.get("id") or "")
    suffix = claim_id.split(":")[-1] if ":" in claim_id else ""
    claim_text = (record.get("claim") or "Untitled claim").strip()
    if len(claim_text) > max_claim_chars:
        claim_text = claim_text[: max_claim_chars - 1].rstrip() + "…"
    if suffix:
        return f"{callout} ({suffix}) • {claim_text}"
    return f"{callout} • {claim_text}"


def record_timeline_event(
    claim_id: str, event: str, detail: Optional[dict] = None
) -> None:
    init_claim_registry()
    timelines = st.session_state.setdefault(CLAIM_TIMELINE_KEY, {})
    entries = timelines.setdefault(claim_id, [])
    entries.insert(
        0, {"event": event, "detail": detail, "at": datetime.utcnow().isoformat() + "Z"}
    )
    del entries[TIMELINE_MAX_EVENTS:]
    _notify_evidence_refresh(claim_id, event, detail)


def get_timeline(claim_id: str) -> List[dict]:
    init_claim_registry()
    timelines = st.session_state.setdefault(CLAIM_TIMELINE_KEY, {})
    return timelines.get(claim_id, [])


def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def _tokenize(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return re.findall(r"[a-z0-9]+", value.lower())


def _filename_tokens(queue_item: dict) -> List[str]:
    filename = queue_item.get("filename", "")
    name, *_ = filename.split(".")
    return _tokenize(name)


def auto_match_claim(queue_item: dict) -> List[dict]:
    """Return possible claim matches for a queued attachment."""
    records = get_claim_records()
    candidates: List[dict] = []
    for record in records:
        score = _score_claim_match(record, queue_item)
        if score <= 0:
            continue
        candidates.append(
            {"id": record["id"], "score": round(score, 3), "record": record}
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:3]


def _score_claim_match(record: dict, queue_item: dict) -> float:
    score = 0.0
    if not record:
        return score
    reference_hint = queue_item.get("reference_hint") or {}
    record_hint = record.get("reference_hint") or {}
    filename_tokens = set(_filename_tokens(queue_item))
    claim_tokens = set(_tokenize(record.get("claim")))
    callout_tokens = set(_tokenize(record.get("callout")))
    if queue_item.get("claim_id") == record.get("id"):
        score += 0.5
    if reference_hint.get("reference_id") and reference_hint.get(
        "reference_id"
    ) == record.get("reference_id"):
        score += 0.4
    if reference_hint.get("doi") and reference_hint.get("doi") == record.get("doi"):
        score += 0.3
    if reference_hint.get("author") and reference_hint.get("author") == record.get(
        "author"
    ):
        score += 0.2
    if reference_hint.get("year") and reference_hint.get("year") == record.get("year"):
        score += 0.1
    overlap = len(filename_tokens & (claim_tokens or callout_tokens))
    if overlap:
        score += min(0.2, 0.05 * overlap)
    if record_hint.get("callout") and reference_hint.get("callout"):
        ratio = SequenceMatcher(
            a=_normalize_text(record_hint.get("callout")),
            b=_normalize_text(reference_hint.get("callout")),
        ).ratio()
        score += 0.3 * ratio
    return min(score, 1.0)


def _cache_key(doc_id: str, reference_id: str) -> str:
    return f"retrieval__{doc_id}__{reference_id}"


def _get_cached_dossier(
    api_url: str,
    doc_id: str,
    reference_id: str,
) -> Optional[dict]:
    cache: Dict[str, dict] = st.session_state.setdefault("retrieval_cache", {})
    key = _cache_key(doc_id, reference_id)
    if key not in cache:
        cache[key] = get_reference_retrieval(api_url, doc_id, reference_id)
    return cache.get(key)


def render_retrieval_instructions(
    api_url: str,
    doc_id: str,
    reference_id: str,
    *,
    resolved_ingest_id: Optional[str] = None,
    key_prefix: str = "",
) -> None:
    dossier = _get_cached_dossier(api_url, doc_id, reference_id)
    if not dossier:
        st.error("No retrieval data available yet. Resolve references to continue.")
        return

    canonical = dossier.get("canonical_citation") or "Citation metadata unavailable"
    st.markdown(f"**{canonical}**")

    doi = dossier.get("doi")
    if doi:
        st.caption(f"DOI: https://doi.org/{doi}")

    primary = dossier.get("primary_url")
    manual = dossier.get("manual_instructions")

    col_open, col_copy = st.columns(2)
    copy_result: Optional[dict] = None
    key_suffix = key_prefix.strip() or "default"
    open_source_key = f"open_source::{doc_id}::{reference_id}::{key_suffix}"
    copy_key = f"retrieval_copy::{doc_id}::{reference_id}::{key_suffix}"
    with col_open:
        if primary:
            try:
                st.link_button(
                    "Open source",
                    primary,
                    width="stretch",
                    help="Opens the publisher/best available link",
                    key=open_source_key,
                )
            except TypeError:
                st.link_button(
                    "Open source",
                    primary,
                    width="stretch",
                    help="Opens the publisher/best available link",
                )
        else:
            st.button(
                "Open source",
                disabled=True,
                help="No direct link resolved yet",
                key=open_source_key,
            )
    with col_copy:
        copy_result = render_copy_to_clipboard(
            "Copy instructions",
            manual,
            key=copy_key,
            toast=f"Copied retrieval instructions for {reference_id}",
            help_text="Copies the full retrieval instructions to your clipboard.",
        )

    if copy_result and copy_result.get("copied") and copy_result.get("snippet"):
        st.success(f"Copied: {copy_result['snippet']}")

    copied_at = (copy_result or {}).get("copied_at")

    resolved_ingest = str(resolved_ingest_id or "").strip() or None

    if resolved_ingest:
        st.success("Linked to uploaded source in this project.")
    elif manual:
        st.code(manual.strip(), language="text")
        if copied_at:
            st.caption(f"Copied at {copied_at}")
        else:
            st.caption(
                "Review or copy the instructions above when contacting a library "
                "or visiting the publisher site."
            )
    else:
        st.warning("No direct links available. Use the fallback steps above.")

    sources = dossier.get("sources") or []
    if sources:
        with st.expander("Metadata sources", expanded=False):
            for source in sources:
                label = source.get("label") or "Unknown source"
                title = source.get("title") or "Untitled work"
                confidence = source.get("confidence")
                st.markdown(f"- **{label}:** {title}")
                if confidence is not None:
                    st.caption(f"Confidence: {confidence:.2f}")


def _notify_evidence_refresh(
    claim_id: Optional[str], event: str, detail: Optional[dict]
) -> None:
    if not claim_id or not _should_refresh_evidence(event, detail):
        return
    evidence_store.mark_claim_stale(claim_id, reason=event)
    if evidence_store.get_active_claim_id() == claim_id:
        evidence_store.sync_for_claim(
            claim_id, claim_text=_claim_text_for(claim_id), force=True
        )


def _should_refresh_evidence(event: str, detail: Optional[dict]) -> bool:
    watched = {"attached", "assigned", "auto-matched", "detached", "dropped", "matched"}
    if event in watched:
        return True
    status = (detail or {}).get("status") if detail else None
    return status == "matched"


def _claim_text_for(claim_id: str) -> Optional[str]:
    record = get_claim_record(claim_id)
    if not record:
        return None
    return record.get("claim")


def _sync_evidence_store_metadata(record: dict) -> None:
    claim_id = record.get("id")
    if not claim_id:
        return
    metadata = {
        "claim_text": record.get("claim"),
        "callout": record.get("callout"),
        "reference_hint": record.get("reference_hint"),
        "doc_id": record.get("doc_id"),
        "flags": record.get("flags"),
    }
    cleaned = {key: value for key, value in metadata.items() if value is not None}
    if cleaned:
        evidence_store.ensure_claim_state(claim_id, **cleaned)
