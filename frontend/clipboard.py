"""Clipboard utilities for Streamlit interactions."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components


def _session_prefix(key: str) -> str:
    return f"clipboard::{key}"


def _truncate(value: str, *, limit: int = 120) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def render_copy_to_clipboard(
    label: str,
    payload: Optional[str],
    *,
    key: str,
    toast: Optional[str] = None,
    help_text: Optional[str] = None,
) -> dict:
    """Render a button that copies ``payload`` to the system clipboard.

    Returns a dict containing ``copied`` (bool), ``copied_at`` (ISO string or None),
    ``payload`` (the sanitized text), and ``snippet`` for quick display.
    """
    payload_text = (payload or "").strip()
    session_key = _session_prefix(key)
    event_state_key = f"{session_key}::event"
    fallback_state_key = f"{session_key}::fallback"
    copied_key = f"{session_key}::copied_at"

    st.session_state.setdefault(fallback_state_key, False)
    st.session_state.setdefault(event_state_key, None)

    help_attr = f' title="{help_text}"' if help_text else ""

    if not payload_text:
        st.button(
            label,
            disabled=True,
            key=f"{session_key}::disabled",
            help=help_text or "No instructions available yet",
        )
        return {
            "copied": False,
            "copied_at": None,
            "payload": "",
            "snippet": "",
        }

    component_value = components.html(
        f"""
<div class='clipboard-control'>
  <button class='clipboard-button' onclick="copyPayload_{key}()"{help_attr}>
    {label}
  </button>
</div>
<script>
const payload_{key} = {json.dumps(payload_text)};
function copyPayload_{key}() {{
  if (!navigator.clipboard) {{
    Streamlit.setComponentValue("fallback");
    return;
  }}
  navigator.clipboard.writeText(payload_{key}).then(
    () => Streamlit.setComponentValue("success"),
    () => Streamlit.setComponentValue("fallback")
  );
}}
</script>
        """,
        height=70,
        key=f"{session_key}::component",
    )

    last_event = st.session_state.get(event_state_key)
    new_event = component_value if component_value != last_event else None
    if component_value is not None:
        st.session_state[event_state_key] = component_value

    copied_at: Optional[str] = st.session_state.get(copied_key)
    copied = False

    if new_event == "success":
        copied = True
        copied_at = datetime.utcnow().isoformat() + "Z"
        st.session_state[copied_key] = copied_at
        st.session_state[fallback_state_key] = False
        if toast:
            st.toast(toast)
    elif new_event == "fallback":
        st.session_state[fallback_state_key] = True
        st.info(
            "Clipboard access was blocked. Use the manual copy field below.",
            icon="⚠",
        )

    if st.session_state.get(fallback_state_key):
        st.text_area(
            "Manual copy",
            payload_text,
            key=f"{session_key}::fallback_text",
            help="Select all (⌘/Ctrl+A) then copy (⌘/Ctrl+C)",
            height=160,
        )

    return {
        "copied": copied,
        "copied_at": copied_at,
        "payload": payload_text,
        "snippet": _truncate(payload_text),
    }
