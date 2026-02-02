"""Chasing-claims panel (context edit, segmentation, retrieval)."""

from __future__ import annotations

from typing import Callable, Optional

import streamlit as st

from frontend.state_keys import (
    canonical_context_edit_key,
    canonical_segments_key,
    scoped,
)


def render(
    *,
    doc_id: str,
    citation_index: int,
    target_id: Optional[str],
    scope: str,
    get_context_cached: Callable[[int, Optional[str]], dict],
    seg_via_llm: Callable[[str, int, str], list[str]],
    to_segment_dict: Callable[[str], dict],
    claim_queue_register: Callable[..., None],
    format_reference_summary: Callable[[dict, dict], str],
    render_retrieval_instructions: Callable[..., None],
    api_url: str,
    selected_model: Optional[str],
    rerun: Callable[[], None],
) -> None:
    context = get_context_cached(int(citation_index), target_id)
    cite_text = (
        context.get("citing_prefix")
        or context.get("citing_sentence")
        or context.get("sentence")
        or ""
    )
    if context.get("error"):
        st.error(f"Failed to load context: {context.get('error')}")
        return

    st.markdown("**Context**")
    canonical_edit = canonical_context_edit_key(
        doc_id=doc_id,
        citation_index=citation_index,
    )
    widget_edit = scoped(scope=scope, canonical=canonical_edit)
    st.session_state.setdefault(canonical_edit, cite_text)
    if widget_edit not in st.session_state:
        st.session_state[widget_edit] = str(st.session_state.get(canonical_edit) or "")
    edited = st.text_area(
        "Context (editable)",
        key=widget_edit,
        height=140,
        help=(
            "Edit the claim text to segment; defaults to the text preceding the "
            "citation."
        ),
    )
    st.session_state[canonical_edit] = str(edited or "")
    cite_text = (edited or "").strip()

    st.markdown("**Parsing**")
    canonical_segments = canonical_segments_key(citation_index=citation_index)
    # One shared revision per (doc_id, citation_index) so rail/tab stay in sync.
    rev_key = f"segments-rev::{doc_id}::{int(citation_index)}"
    st.session_state.setdefault(rev_key, 0)
    rev = int(st.session_state.get(rev_key) or 0)
    widget_segments = scoped(scope=scope, canonical=f"{canonical_segments}::v{rev}")
    stored = st.session_state.get("citation_sentence_segments", {}).get(
        str(int(citation_index)),
        [],
    )
    st.session_state.setdefault(canonical_segments, "\n".join(stored))

    if st.button(
        "Segment sentence",
        key=scoped(scope=scope, canonical=f"segment-sentence-inline-{citation_index}"),
        disabled=not selected_model or not cite_text,
    ):
        segments = seg_via_llm(cite_text, int(citation_index) + 1, selected_model or "")
        st.session_state.setdefault("citation_sentence_segments", {})[
            str(int(citation_index))
        ] = segments
        st.session_state[canonical_segments] = "\n".join(segments)
        st.session_state[rev_key] = int(st.session_state.get(rev_key) or 0) + 1
        rerun()

    seg_text = st.text_area(
        "Parsed claims (one per line)",
        value=str(st.session_state.get(canonical_segments) or ""),
        key=widget_segments,
        height=140,
    )
    st.session_state[canonical_segments] = seg_text

    if st.button(
        "Save claims",
        key=scoped(scope=scope, canonical=f"save-claims-inline-{citation_index}"),
    ):
        lines = [ln.strip() for ln in seg_text.splitlines() if ln.strip()]
        st.session_state.setdefault("citation_sentence_segments", {})[
            str(int(citation_index))
        ] = lines
        st.session_state[canonical_segments] = "\n".join(lines)
        st.session_state[rev_key] = int(st.session_state.get(rev_key) or 0) + 1

        primary_callout = context.get("callout") or "citation"
        reference_hint = {
            "callout": primary_callout,
            "reference_id": target_id,
        }
        for idx, line in enumerate(lines):
            parsed = to_segment_dict(line)
            segment_id = parsed.get("segment_id") or f"seg-{idx+1}"
            claim_text = parsed.get("claim") or line
            claim_id = f"cite:{doc_id}:{int(citation_index)}:{segment_id}"
            claim_queue_register(
                claim_id,
                claim=claim_text,
                callout=primary_callout,
                doc_id=doc_id,
                reference_id=target_id,
                reference_hint=reference_hint,
            )
        st.success(f"Saved {len(lines)} claim(s) to the workspace.")

    if st.button(
        "Chase",
        key=scoped(scope=scope, canonical=f"chase-inline-{citation_index}"),
        type="primary",
    ):
        # Set a lightweight intention flag for the main UI to honor.
        st.session_state["chase_intent"] = {
            "doc_id": doc_id,
            "citation_index": int(citation_index),
            "target_id": target_id,
        }
        rerun()

    st.markdown("**Retrieving**")
    reference_id = target_id or (context.get("reference") or {}).get("id")
    if not reference_id:
        st.info("No target ID available for this citation.")
        return
    reference = context.get("reference") or {}
    resolution = context.get("resolution") or {}
    summary = format_reference_summary(reference, resolution)
    if summary:
        st.markdown("**Reference summary**")
        st.markdown(summary)
    with st.expander("Retrieval instructions", expanded=False):
        render_retrieval_instructions(
            api_url=api_url,
            doc_id=doc_id,
            reference_id=reference_id,
            key_prefix=f"{scope}-{int(citation_index)}",
        )
