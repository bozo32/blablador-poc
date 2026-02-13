"""Chasing-claims panel (context edit, segmentation, retrieval)."""

from __future__ import annotations

import html
import re
from typing import Callable, Optional

import streamlit as st

from frontend import claim_queue as claim_queue_state
from frontend import evidence_store
from frontend.state_keys import (
    canonical_context_edit_key,
    canonical_segments_key,
    scoped,
)
from frontend.ingestion_api import auto_place_claim_source, confirm_claims
from frontend.citation_anchors import maybe_attach_citation_anchor
from frontend import judgment_api


def _segment_locally(text: str, base_index: int) -> list[str]:
    """Deterministic fallback when no LLM model is configured."""
    import re

    raw = (text or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", raw) if p.strip()]
    if not parts:
        parts = [raw]

    segments: list[str] = []
    for idx, part in enumerate(parts):
        letter = chr(ord("a") + idx)
        segments.append(f"{base_index}{letter}. {part}")
    return segments


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
    active_reviewer_uid: Optional[str],
    reviewers: list[str],
    rerun: Callable[[], None],
) -> None:
    def _segment_to_claim_index(segment_id: str) -> Optional[int]:
        text = str(segment_id or "").strip()
        m = re.match(r"^\d+([a-z])$", text, flags=re.IGNORECASE)
        if not m:
            return None
        letter = m.group(1).lower()
        return 1 + (ord(letter) - ord("a"))

    def _normalize_reviewer(value: Optional[str]) -> str:
        text = str(value or "").strip()
        text = " ".join(text.split())
        return text or "default"

    def _resolve_sentence_id() -> Optional[str]:
        # Prefer explicit sentence_id from the selected callout tuple derived
        # from the loaded document body.
        selected = st.session_state.get("selected_callout_tuple")
        if isinstance(selected, dict):
            try:
                idx = int(selected.get("citation_index"))
            except Exception:
                idx = None
            if (
                str(selected.get("doc_id") or "").strip() == str(doc_id)
                and idx is not None
                and idx == int(citation_index)
                and str(selected.get("target_id") or "").strip()
                == str(target_id or "").strip()
            ):
                value = str(selected.get("sentence_id") or "").strip()
                if value:
                    return value

        # Fallbacks.
        value = str(st.session_state.get("citation_selected_sentence_id") or "").strip()
        if value:
            return value
        value = str((context or {}).get("sentence_id") or "").strip()
        return value or None

    def _persist_confirmed_claims(lines: list[str]) -> None:
        # Graph tab expects backend claim nodes of the form:
        # claim:{doc_id}:{sentence_id}:{claim_index}
        sentence_id = _resolve_sentence_id()
        sentence_text = (
            context.get("citing_sentence")
            or context.get("sentence")
            or context.get("citing_prefix")
            or cite_text
            or ""
        )
        sentence_text = str(sentence_text or "").strip()

        if not sentence_id or not sentence_text:
            return

        confirmed: list[dict] = []
        for idx, line in enumerate(lines):
            parsed = to_segment_dict(line)
            segment_id = parsed.get("segment_id") or ""
            claim_text = (parsed.get("claim") or line or "").strip()
            if not claim_text:
                continue
            claim_index = _segment_to_claim_index(segment_id) or (idx + 1)
            confirmed.append(
                {
                    "claim_index": int(claim_index),
                    "parsed_text": claim_text,
                }
            )

        if not confirmed:
            return

        try:
            prov = {
                "doc_id": str(doc_id),
                "citation_index": int(citation_index),
                "target_id": str(target_id) if target_id else None,
            }
            prov = maybe_attach_citation_anchor(provenance=prov, context=context)
            confirm_claims(
                api_url,
                document_id=str(doc_id),
                sentence_id=str(sentence_id),
                sentence_text=sentence_text,
                citation_index=int(citation_index),
                target_id=str(target_id) if target_id else None,
                reviewer_uid=reviewer_label,
                confirmed_claims=confirmed,
                segmentation_model=(selected_model or "local"),
                cited_work_id=prov.get("cited_work_id"),
                citation_anchor=prov.get("citation_anchor"),
            )
        except RuntimeError:
            # Non-blocking: evidence can still run without graph indexing.
            return

    # NOTE: We key reviewer-scoped state by casefolded name so "Alice" and
    # "alice" don't fork independent drafts.
    reviewer_label = _normalize_reviewer(active_reviewer_uid)
    reviewer_state = reviewer_label.casefold()
    cite_key = f"{doc_id}::{int(citation_index)}::{target_id or ''}"
    segments_by_reviewer = st.session_state.setdefault(
        "citation_segments_by_reviewer", {}
    )
    if not isinstance(segments_by_reviewer, dict):
        segments_by_reviewer = {}
        st.session_state["citation_segments_by_reviewer"] = segments_by_reviewer
    reviewer_segments = segments_by_reviewer.setdefault(reviewer_state, {})

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

    st.markdown("**Context (editable)**")
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
        label_visibility="collapsed",
    )
    st.session_state[canonical_edit] = str(edited or "")
    cite_text = (edited or "").strip()

    st.markdown("**Segments**")

    callout_judgment_id = f"callout:{doc_id}:{int(citation_index)}:{target_id or ''}"
    if cite_key not in reviewer_segments:
        # Best-effort restore saved segments from backend judgment drafts so
        # browser reloads can resume without re-segmentation.
        try:
            stored = judgment_api.get_judgment(
                callout_judgment_id,
                reviewer_uid=str(active_reviewer_uid or "default"),
            )
        except Exception:
            stored = {}
        span_selectors = (stored or {}).get("span_selectors")
        if isinstance(span_selectors, list) and span_selectors:
            restored_lines: list[str] = []
            for entry in span_selectors:
                if not isinstance(entry, dict):
                    continue
                text = str(entry.get("text") or "").strip()
                if text:
                    restored_lines.append(text)
            if restored_lines:
                reviewer_segments[cite_key] = restored_lines
    canonical_segments = (
        f"{canonical_segments_key(citation_index=citation_index)}::{reviewer_state}"
    )
    # One shared revision per (doc_id, citation_index, target_id, reviewer) so
    # rail/tab stay in sync while different reviewers keep independent drafts.
    rev_key = (
        "segments-rev::"
        f"{doc_id}::{int(citation_index)}::{target_id or ''}::{reviewer_state}"
    )
    st.session_state.setdefault(rev_key, 0)
    rev = int(st.session_state.get(rev_key) or 0)
    widget_segments = scoped(scope=scope, canonical=f"{canonical_segments}::v{rev}")

    accepted = reviewer_segments.get(cite_key)
    if accepted is None:
        accepted = []

    # If this reviewer has not accepted a segmentation yet but other reviewers
    # have, show those as tentative suggestions.
    suggestion_options: list[str] = []
    suggestion_map: dict[str, list[str]] = {}
    for raw in reviewers or []:
        name = _normalize_reviewer(raw)
        key = name.casefold()
        if key == reviewer_state:
            continue
        other = segments_by_reviewer.get(key) or {}
        if not isinstance(other, dict):
            continue
        lines = other.get(cite_key)
        if not isinstance(lines, list) or not any(str(ln).strip() for ln in lines):
            continue
        suggestion_options.append(name)
        suggestion_map[name] = [str(ln).strip() for ln in lines if str(ln).strip()]

    has_accepted = bool([ln for ln in accepted if str(ln).strip()])
    mode_key = scoped(
        scope=scope, canonical=f"segments-mode::{cite_key}::{reviewer_state}"
    )
    mode = str(st.session_state.get(mode_key) or "").strip().lower()
    if (not has_accepted) and suggestion_options and mode != "own":
        st.caption(
            "Another reviewer already segmented this citing span. Review their "
            "suggestions and accept them to start your own work."
        )
        choice_key = scoped(
            scope=scope, canonical=f"segments-suggestion-src::{cite_key}"
        )
        st.session_state.setdefault(choice_key, suggestion_options[0])
        chosen = st.selectbox(
            "Suggested segments from",
            options=suggestion_options,
            key=choice_key,
        )
        suggested_lines = suggestion_map.get(chosen) or []
        suggested_text = "\n".join(suggested_lines)

        # Streamlit's disabled text_area can render suggested text too faint,
        # especially with dense workspace CSS. Render suggestions in a readable
        # grey block (still clearly "not yet accepted").
        suggested_html = html.escape(suggested_text or "(No suggested segments found.)")
        st.markdown(
            "<div class='ws-suggested-segments'><pre>"
            + suggested_html
            + "</pre></div>",
            unsafe_allow_html=True,
        )

        if st.button(
            "Accept suggestions",
            key=scoped(
                scope=scope,
                canonical=f"accept-segments-inline-{citation_index}-{reviewer_state}",
            ),
            type="primary",
        ):
            lines = [ln.strip() for ln in suggested_text.splitlines() if ln.strip()]
            reviewer_segments[cite_key] = lines
            st.session_state[canonical_segments] = "\n".join(lines)
            st.session_state[rev_key] = int(st.session_state.get(rev_key) or 0) + 1

            _persist_confirmed_claims(lines)

            primary_callout = context.get("callout") or "citation"
            reference_hint = {
                "callout": primary_callout,
                "reference_id": target_id,
            }
            placed = 0
            for idx, line in enumerate(lines):
                parsed = to_segment_dict(line)
                segment_id = parsed.get("segment_id") or f"seg-{idx+1}"
                claim_text = parsed.get("claim") or line
                # Reviewer-scoped claim IDs prevent one reviewer from overwriting
                # another's claim text in the local registry.
                claim_id = (
                    f"cite:{doc_id}:{int(citation_index)}:{reviewer_state}:{segment_id}"
                )

                previous = claim_queue_state.get_claim_record(claim_id) or {}
                previous_text = (previous.get("claim") or "").strip()
                claim_queue_register(
                    claim_id,
                    claim=claim_text,
                    callout=primary_callout,
                    doc_id=doc_id,
                    reference_id=target_id,
                    reference_hint=reference_hint,
                )

                if target_id:
                    try:
                        resp = auto_place_claim_source(
                            api_url,
                            claim_id=str(claim_id),
                            doc_id=str(doc_id),
                            citation_index=int(citation_index),
                            target_id=str(target_id),
                        )
                    except RuntimeError:
                        resp = None
                    if (resp or {}).get("attachment"):
                        placed += 1

                if claim_text.strip() and claim_text.strip() != previous_text:
                    evidence_store.queue_rerun(
                        claim_id,
                        claim_text=claim_text,
                        note="auto-claim-save",
                        quiet=True,
                    )

            st.caption(
                f"Accepted {len(lines)} claim(s). "
                f"Auto-placed sources for {placed} claim(s)."
            )
            st.session_state["chase_intent"] = {
                "doc_id": doc_id,
                "citation_index": int(citation_index),
                "target_id": target_id,
            }
            st.session_state[mode_key] = "own"
            rerun()

        if st.button(
            "Start my own",
            key=scoped(
                scope=scope,
                canonical=(
                    f"start-own-segments-inline-{citation_index}-" f"{reviewer_state}"
                ),
            ),
        ):
            st.session_state[mode_key] = "own"
            st.session_state[canonical_segments] = ""
            st.session_state[rev_key] = int(st.session_state.get(rev_key) or 0) + 1
            rerun()

        # Hide the editable editor until the reviewer explicitly accepts
        # suggestions or chooses to start their own version.
        return

    st.session_state.setdefault(canonical_segments, "\n".join(accepted))

    if st.button(
        "Segment",
        key=scoped(
            scope=scope,
            canonical=f"segment-sentence-inline-{citation_index}-{reviewer_state}",
        ),
        disabled=not cite_text,
    ):
        base_index = int(citation_index) + 1
        if selected_model:
            segments = seg_via_llm(cite_text, base_index, selected_model)
        else:
            segments = _segment_locally(cite_text, base_index)
        # Segmentation results are drafts until saved; do not overwrite any
        # reviewer-accepted segmentation.
        st.session_state[canonical_segments] = "\n".join(segments)
        st.session_state[mode_key] = "own"
        st.session_state[rev_key] = int(st.session_state.get(rev_key) or 0) + 1
        rerun()

    seg_value = str(st.session_state.get(canonical_segments) or "")
    line_count = max(1, len([ln for ln in seg_value.splitlines() if ln.strip()]))
    seg_height = min(320, max(110, 26 * (line_count + 1)))
    seg_text = st.text_area(
        "Segments",
        value=seg_value,
        key=widget_segments,
        height=seg_height,
        label_visibility="collapsed",
        placeholder="One per line",
    )
    st.session_state[canonical_segments] = seg_text

    if st.button(
        "Save",
        key=scoped(
            scope=scope,
            canonical=f"save-claims-inline-{citation_index}-{reviewer_state}",
        ),
        type="primary",
    ):
        lines = [ln.strip() for ln in seg_text.splitlines() if ln.strip()]
        reviewer_segments[cite_key] = lines
        st.session_state[canonical_segments] = "\n".join(lines)
        st.session_state[rev_key] = int(st.session_state.get(rev_key) or 0) + 1

        _persist_confirmed_claims(lines)

        # Persist segmentation lines to backend as a draft callout judgment.
        try:
            prov = maybe_attach_citation_anchor(
                provenance={
                    "doc_id": str(doc_id),
                    "citation_index": int(citation_index),
                    "target_id": str(target_id) if target_id else None,
                },
                context=context,
            )
            judgment_api.put_judgment(
                callout_judgment_id,
                {
                    "status": "draft",
                    "verdict": None,
                    "notes": None,
                    "doc_id": str(doc_id),
                    "citation_index": int(citation_index),
                    "target_id": str(target_id) if target_id else None,
                    "sentence_id": _resolve_sentence_id() or context.get("sentence_id"),
                    "callout": context.get("callout"),
                    "reference_id": str(target_id) if target_id else None,
                    "cited_work_id": prov.get("cited_work_id"),
                    "citation_anchor": prov.get("citation_anchor"),
                    "span_selectors": [
                        {"segment_id": None, "text": ln} for ln in lines
                    ],
                },
                reviewer_uid=str(active_reviewer_uid or "default"),
            )
        except Exception:
            pass

        primary_callout = context.get("callout") or "citation"
        reference_hint = {
            "callout": primary_callout,
            "reference_id": target_id,
        }
        placed = 0
        for idx, line in enumerate(lines):
            parsed = to_segment_dict(line)
            segment_id = parsed.get("segment_id") or f"seg-{idx+1}"
            claim_text = parsed.get("claim") or line
            claim_id = (
                f"cite:{doc_id}:{int(citation_index)}:{reviewer_state}:{segment_id}"
            )

            previous = claim_queue_state.get_claim_record(claim_id) or {}
            previous_text = (previous.get("claim") or "").strip()
            claim_queue_register(
                claim_id,
                claim=claim_text,
                callout=primary_callout,
                doc_id=doc_id,
                reference_id=target_id,
                reference_hint=reference_hint,
            )

            # Best-effort: if the cited PDF is uploaded + merged, auto-place it now
            # so any rerun captures a non-empty attachments_state.
            if target_id:
                try:
                    resp = auto_place_claim_source(
                        api_url,
                        claim_id=str(claim_id),
                        doc_id=str(doc_id),
                        citation_index=int(citation_index),
                        target_id=str(target_id),
                    )
                except RuntimeError:
                    resp = None
                if (resp or {}).get("attachment"):
                    placed += 1

            if claim_text.strip() and claim_text.strip() != previous_text:
                evidence_store.queue_rerun(
                    claim_id,
                    claim_text=claim_text,
                    note="auto-claim-save",
                    quiet=True,
                )
        st.caption(
            f"Saved {len(lines)} claim(s). Auto-placed sources for {placed} claim(s)."
        )

        # Automatically enter chase mode after saving claims.
        st.session_state["chase_intent"] = {
            "doc_id": doc_id,
            "citation_index": int(citation_index),
            "target_id": target_id,
        }
        st.session_state[mode_key] = "own"
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
