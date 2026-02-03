"""Clipboard utilities for Streamlit interactions."""

from __future__ import annotations

import os
import json
import re
import subprocess
import sys
from datetime import datetime
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components


def copy_text(text: str) -> bool:
    """Copy plain text to the local OS clipboard (best effort).

    This is intended for local-dev Streamlit flows where the server and user are
    on the same machine.
    """
    payload = (text or "").strip()
    if not payload:
        return False

    try:
        if sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=payload, text=True, check=True)
            return True
        if sys.platform.startswith("win"):
            # Windows clipboard expects UTF-16LE via `clip`.
            subprocess.run(["clip"], input=payload, text=True, check=True)
            return True
        # Linux / other: try xclip then xsel.
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=payload,
            text=True,
            check=True,
        )
        return True
    except Exception:
        try:
            subprocess.run(
                ["xsel", "--clipboard", "--input"],
                input=payload,
                text=True,
                check=True,
            )
            return True
        except Exception:
            return False


def open_file(path: str) -> Optional[str]:
    """Open a file path on the local machine (best effort).

    Returns an error string on failure, otherwise None.
    """
    resolved = str(path or "").strip()
    if not resolved:
        return "Missing local file path"
    if not os.path.exists(resolved):
        return f"File not found: {resolved}"

    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", resolved])
            return None
        if sys.platform.startswith("win"):
            os.startfile(resolved)  # type: ignore[attr-defined]
            return None
        subprocess.Popen(["xdg-open", resolved])
        return None
    except Exception as exc:
        return str(exc)


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
    js_key = re.sub(r"[^A-Za-z0-9_]", "_", key)
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

    html_payload = f"""
<div class='clipboard-control'>
  <button class='clipboard-button' onclick="copyPayload_{js_key}()"{help_attr}>
    {label}
  </button>
</div>
<script>
const payload_{js_key} = {json.dumps(payload_text)};
function copyPayload_{js_key}() {{
  if (!navigator.clipboard) {{
    Streamlit.setComponentValue("fallback");
    return;
  }}
  navigator.clipboard.writeText(payload_{js_key}).then(
    () => Streamlit.setComponentValue("success"),
    () => Streamlit.setComponentValue("fallback")
  );
}}
</script>
    """

    # Streamlit versions differ: some accept `key=`, older ones don't.
    try:
        component_value = components.html(
            html_payload,
            height=70,
            key=f"{session_key}::component",
        )
    except TypeError:
        component_value = components.html(
            html_payload,
            height=70,
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
