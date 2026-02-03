"""Right-rail chase queue UI.

This module renders a VSCode-like collapsible stack of followed citations.
It avoids Streamlit widget key collisions by requiring a `scope` prefix.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import streamlit as st


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
