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

from frontend import scope_lock


def _project_id() -> str:
    return str(scope_lock.get_applied_project_id() or "").strip()


def _active_reviewer_uid() -> str:
    return str(scope_lock.get_applied_uid() or "").strip()


def _project_headers() -> Dict[str, str]:
    project_id = _project_id()
    reviewer_uid = _active_reviewer_uid()
    if not project_id:
        raise RuntimeError("Attachment mutation requires project_id scope")
    if not reviewer_uid:
        raise RuntimeError("Attachment mutation requires active reviewer identity")
    return {
        "X-Project-Id": project_id,
        "X-User-Id": reviewer_uid,
        "X-Reviewer-Uid": reviewer_uid,
    }


from frontend import workflow_api


def entry_key(
    citation_index: int,
    target_id: Optional[str],
    *,
    span_key: Optional[str] = None,
    order: Optional[int] = None,
) -> str:
    sid = str(span_key or "").strip()
    if sid:
        return f"{int(citation_index)}:{target_id or ''}:{sid}"
    if order is not None:
        return f"{int(citation_index)}:{target_id or ''}:ord{int(order)}"
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
    # Header is rendered by the caller pane.

    if not followed:
        st.info("Click a citation in Document text to add it here.")
        return

    def _sort_key(entry: dict) -> tuple[int, int, str, str]:
        try:
            cite_idx = int(entry.get("citation_index") or 0)
        except Exception:
            cite_idx = 0
        try:
            order = int(entry.get("order") or 0)
        except Exception:
            order = 0
        tgt = entry.get("target_id")
        sid = str(entry.get("span_key") or "")
        return (cite_idx, order, "" if tgt is None else str(tgt), sid)

    # Render in stable document order; opening/activating an item must not
    # reshuffle the list.
    ordered_followed = sorted(list(followed), key=_sort_key)

    seen_keys: set[str] = set()
    for idx, entry in enumerate(ordered_followed):
        try:
            cite_idx = int(entry.get("citation_index") or 0)
        except Exception:
            cite_idx = 0
        tgt = entry.get("target_id")
        ek = entry_key(
            cite_idx,
            tgt,
            span_key=str(entry.get("span_key") or "") or None,
            order=int(entry.get("order") or 0),
        )
        if ek in seen_keys:
            continue
        seen_keys.add(ek)
        label = truncate_two_lines(get_label(entry), max_chars=44)
        status = get_status(cite_idx, tgt)
        is_complete = bool(status.get("processed"))
        status_chip = ":material/check_circle:" if is_complete else ":material/incomplete_circle:"
        header = f"{label} [{status_chip}]"

        with st.expander(f"> {header}", expanded=False):
            span_key = str(entry.get("span_key") or "").strip()
            if span_key:
                st.session_state["active_queue_span_key"] = span_key
            if st.button(
                ":material/open_in_browser: Focus in center",
                key=f"{scope}::focus::{ek}::{idx}",
                use_container_width=False,
            ):
                if span_key:
                    st.session_state["pending_scroll_span_key"] = span_key
                    st.session_state["active_queue_span_key"] = span_key
                on_open(cite_idx, tgt)
                rerun()
            render_panel(cite_idx, tgt, scope)


def render_requested_works_queue(
    *,
    api_url: str,
    run_id: str,
    project_id: str,
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
        data = workflow_api.get_run_status(
            api_url,
            run_id=rid,
            project_id=project_id,
        )
    except Exception as exc:
        st.warning(f"Workflow status unavailable: {exc}")
        return

    run = (data or {}).get("run") or {}
    run_state = str((run or {}).get("state") or "").strip()

    # Disable in-panel autorefresh to prevent duplicate-element key collisions
    # when multiple citespan panels render in one Streamlit pass.

    queue = (data or {}).get("queue")
    if not isinstance(queue, list):
        queue = []

    st.markdown("**Available works**")
    if not queue:
        st.caption("No targets yet (citation mapping missing).")
        return

    try:
        attachments_payload = requests.get(
            f"{str(api_url).rstrip('/')}/attachments?archived=false",
            headers=_project_headers(),
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
        if len(label) >= 30 and label.count("-") >= 3:
            label = "Unresolved work"
        cols = st.columns([5, 2], gap="small")
        with cols[0]:
            st.markdown(f"`{label}`")
        with cols[1]:
            st.caption(_chip(state))

        if state == "requested" and unassigned:
            options = [
                str(a.get("filename") or "attachment.pdf")
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
                    src_id = str(choice).split(" • ", 1)[0].strip()
                    headers = _project_headers()

                    selected = next(
                        (
                            a
                            for a in unassigned
                            if str(a.get("id") or "").strip() == src_id
                        ),
                        None,
                    )

                    # Ensure source_ingest_id exists when possible.
                    try:
                        if not str(
                            (selected or {}).get("source_ingest_id") or ""
                        ).strip():
                            promote_url = (
                                f"{str(api_url).rstrip('/')}/attachments/"
                                f"{src_id}/promote-ingest"
                            )
                            promote = requests.post(
                                promote_url,
                                headers=headers,
                                timeout=30,
                            )
                            promote.raise_for_status()
                            promoted = (promote.json() or {}).get("attachment") or {}
                            if isinstance(promoted, dict):
                                selected = promoted
                    except Exception:
                        # Best-effort.
                        pass

                    payload: Dict[str, object] = {
                        "claim_id": str(claim_id),
                        "doc_id": str(citing_doc_id),
                        "citation_index": entry.get("citation_index"),
                        "target_id": target_id,
                        "reference_hint": {"reference_id": reference_id}
                        if reference_id
                        else None,
                    }
                    payload = {k: v for k, v in payload.items() if v is not None}

                    clone = requests.post(
                        f"{str(api_url).rstrip('/')}/attachments/{src_id}/clone",
                        headers=headers,
                        json=payload,
                        timeout=30,
                    )
                    clone.raise_for_status()
                    attachment_id = (
                        str(
                            ((clone.json() or {}).get("attachment") or {}).get("id")
                            or ""
                        ).strip()
                        or attachment_id
                    )

                    _ = workflow_api.resume_run(
                        api_url,
                        run_id=rid,
                        project_id=project_id,
                    )
                    st.caption("Assigned via clone; resuming run...")
                except Exception as exc:
                    st.warning(f"Assign failed: {exc}")

        if attachment_id:
            st.caption(f"attachment_id={attachment_id}")
