"""Right-rail chase queue UI.

This module renders a VSCode-like collapsible stack of followed citations.
It avoids Streamlit widget key collisions by requiring a `scope` prefix.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import streamlit as st

try:  # Optional dependency for timed polling
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # pragma: no cover
    st_autorefresh = None

import requests

from frontend import workflow_api


def entry_key(citation_index: int, target_id: Optional[str]) -> str:
    return f"{int(citation_index)}:{target_id or ''}"


def truncate_two_lines(text: str, *, max_chars: int = 110) -> str:
    value = " ".join(str(text or "").split()).strip()
    if len(value) > max_chars:
        value = value[: max_chars - 1].rstrip() + "…"
    # Approximate a two-line label inside a button.
    if len(value) <= 58:
        return value
    cut = value.rfind(" ", 0, 58)
    if cut <= 20:
        cut = 58
    return value[:cut].rstrip() + "\n" + value[cut:].lstrip()


def render(
    *,
    title: str,
    caption: str,
    followed: List[dict],
    selected_index: Optional[int],
    selected_target: Optional[str],
    get_label: Callable[[dict], str],
    get_status: Callable[[int, Optional[str]], Dict[str, bool]],
    on_open: Callable[[int, Optional[str]], None],
    on_drop: Callable[[int, Optional[str]], None],
    render_panel: Callable[[int, Optional[str], str], None],
    rerun: Callable[[], None],
    scope: str = "rail",
    open_state_key: str = "chase_queue_open",
) -> None:
    """Render chase queue.

    `render_panel(cite_idx, tgt, scope)` is called for the open item.
    """
    st.markdown("#### " + title)
    st.caption(caption)

    if not followed:
        st.info("Click a citation in Document text to add it here.")
        return

    def _sort_key(entry: dict) -> tuple[int, str]:
        try:
            cite_idx = int(entry.get("citation_index") or 0)
        except Exception:
            cite_idx = 0
        tgt = entry.get("target_id")
        return (cite_idx, "" if tgt is None else str(tgt))

    # Render in stable document order; opening/activating an item must not
    # reshuffle the list.
    ordered_followed = sorted(list(followed), key=_sort_key)

    selected_entry_key = None
    if selected_index is not None:
        selected_entry_key = entry_key(int(selected_index), selected_target)

    open_key = str(st.session_state.get(open_state_key) or "")
    if not open_key and selected_entry_key:
        # Default to the currently selected citation.
        open_key = selected_entry_key
        st.session_state[open_state_key] = open_key

    for entry in ordered_followed:
        try:
            cite_idx = int(entry.get("citation_index") or 0)
        except Exception:
            cite_idx = 0
        tgt = entry.get("target_id")
        ek = entry_key(cite_idx, tgt)
        is_open = open_key == ek
        # Avoid literal leading letters (e.g., "v") in labels; use ASCII-only
        # disclosure markers so the citation label stays clean.
        caret = "[-]" if is_open else "[+]"

        label = truncate_two_lines(get_label(entry))
        status = get_status(cite_idx, tgt)
        has_saved = bool(status.get("has_saved"))
        processed = bool(status.get("processed"))

        if st.button(
            f"{caret} {label}",
            key=f"{scope}::row::{ek}",
            use_container_width=True,
        ):
            st.session_state[open_state_key] = ek if not is_open else ""
            on_open(cite_idx, tgt)
            rerun()

        action = st.columns([1, 1], gap="small")
        with action[0]:
            if not has_saved:
                if st.button(
                    "Segment",
                    key=f"{scope}::segment::{ek}",
                    use_container_width=True,
                ):
                    st.session_state[open_state_key] = ek
                    on_open(cite_idx, tgt)
                    rerun()
            else:
                if st.button(
                    "Chase",
                    key=f"{scope}::chase::{ek}",
                    type="primary" if processed else "secondary",
                    use_container_width=True,
                ):
                    # Keep selection but leave routing to caller.
                    on_open(cite_idx, tgt)
                    rerun()

        with action[1]:
            if st.button(
                "Drop",
                key=f"{scope}::drop::{ek}",
                use_container_width=True,
            ):
                on_drop(cite_idx, tgt)
                if str(st.session_state.get(open_state_key) or "") == ek:
                    st.session_state[open_state_key] = ""
                rerun()

        if is_open:
            st.markdown("---")
            render_panel(cite_idx, tgt, scope)
            st.markdown("---")


def render_requested_works_queue(
    *,
    api_url: str,
    run_id: str,
    claim_id: str,
    citing_doc_id: str,
    reviewer_uid: str,
    poll_ms: int = 1500,
    scope: str = "queue",
) -> None:
    """Render the requested-works queue for a workflow run (polling-first)."""
    rid = str(run_id or "").strip()
    if not rid:
        st.info("No workflow run yet.")
        return

    data = {}
    try:
        data = workflow_api.get_run_status(api_url, run_id=rid)
    except Exception as exc:
        st.warning(f"Workflow status unavailable: {exc}")
        return

    run = (data or {}).get("run") or {}
    run_state = str((run or {}).get("state") or "").strip()

    is_terminal = run_state in {"complete", "partial", "error", "cancelled"}
    if not is_terminal and callable(st_autorefresh):  # pragma: no cover
        st_autorefresh(
            interval=int(poll_ms),
            key=f"{scope}::autorefresh::{rid}",
        )

    queue = (data or {}).get("queue")
    if not isinstance(queue, list):
        queue = []

    st.markdown("**Requested works**")
    if not queue:
        st.caption("No targets yet (citation mapping missing).")
        return

    try:
        attachments_payload = requests.get(
            f"{str(api_url).rstrip('/')}/attachments?archived=false",
            timeout=15,
        )
        attachments_payload.raise_for_status()
        attachments = (attachments_payload.json() or {}).get("attachments") or []
    except Exception:
        attachments = []
    unassigned = [
        a
        for a in attachments
        if isinstance(a, dict)
        and not a.get("doc_id")
        and not a.get("target_id")
        and str(a.get("status") or "") == "matched"
    ]

    def _chip(state: str) -> str:
        val = str(state or "").strip().lower()
        mapping = {
            "requested": "Requested",
            "available": "Available",
            "processing": "Processing",
            "done": "Done",
            "error": "Error",
            "cancelled": "Cancelled",
            "blocked": "Blocked",
        }
        return mapping.get(val, state or "Unknown")

    for entry in queue:
        if not isinstance(entry, dict):
            continue
        target_id = str(entry.get("target_id") or "").strip()
        reference_id = str(entry.get("reference_id") or "").strip()
        state = str(entry.get("state") or "").strip().lower()
        attachment_id = str(entry.get("attachment_id") or "").strip() or None

        label = reference_id or target_id or "(unknown target)"
        cols = st.columns([5, 2], gap="small")
        with cols[0]:
            st.markdown(f"`{label}`")
        with cols[1]:
            st.caption(_chip(state))

        if state == "requested" and unassigned:
            options = [
                f"{str(a.get('id'))} • {str(a.get('filename') or 'attachment.pdf')}"
                for a in unassigned
                if a.get("id")
            ]
            choice = st.selectbox(
                "Assign from Source Bin",
                options,
                key=f"{scope}::assign::{rid}::{target_id}",
                label_visibility="collapsed",
            )
            if st.button(
                "Assign",
                key=f"{scope}::assign-btn::{rid}::{target_id}",
                use_container_width=True,
            ):
                attachment_id = str(choice).split(" • ", 1)[0].strip()
                try:
                    resp = requests.patch(
                        f"{str(api_url).rstrip('/')}/attachments/{attachment_id}",
                        json={
                            "claim_id": str(claim_id),
                            "doc_id": str(citing_doc_id),
                            "citation_index": int(str(claim_id).split(":")[2])
                            if ":" in str(claim_id)
                            else None,
                            "target_id": target_id,
                        },
                        timeout=30,
                    )
                    resp.raise_for_status()
                    _ = workflow_api.resume_run(api_url, run_id=rid)
                    st.caption("Assigned; resuming run...")
                except Exception as exc:
                    st.warning(f"Assign failed: {exc}")

        if attachment_id:
            st.caption(f"attachment_id={attachment_id}")
