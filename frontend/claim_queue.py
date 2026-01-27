"""Claim queue helpers for retrieval instructions."""

from __future__ import annotations

from typing import Dict, Optional

import streamlit as st

from frontend.ingestion_api import get_reference_retrieval


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

    copy_disabled = manual is None

    col_open, col_copy = st.columns(2)
    with col_open:
        if primary:
            st.link_button(
                "Open source",
                primary,
                use_container_width=True,
                help="Opens the publisher/best available link",
            )
        else:
            st.button(
                "Open source",
                disabled=True,
                help="No direct link resolved yet",
                key=f"open_source_{reference_id}",
            )
    with col_copy:
        if st.button(
            "Copy instructions",
            disabled=copy_disabled,
            key=f"copy_instr_{reference_id}",
        ):
            st.session_state["retrieval_clipboard"] = manual
            st.success("Instructions copied to clipboard placeholder")

    if manual:
        st.info(manual)
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
