"""Centralized Streamlit session/widget key helpers."""

from __future__ import annotations

from typing import Optional


# Workspace shell (Phase 08)
WORKSPACE_DENSE_MODE = "workspace_dense_mode"
WORKSPACE_SETTINGS_OPEN = "workspace_settings_open"
WORKSPACE_ACTIVE_TAB = "workspace_active_tab"

WORKSPACE_TAB_DOCUMENT = "Document"
WORKSPACE_TAB_REVIEW = "Review"


def canonical_context_edit_key(*, doc_id: str, citation_index: int) -> str:
    return f"context-edit::{doc_id}::{int(citation_index)}"


def canonical_segments_key(*, citation_index: int) -> str:
    return f"citation-segments-{int(citation_index)}"


def scoped(*, scope: str, canonical: str) -> str:
    return f"{scope}::{canonical}"


def citation_entry_key(*, citation_index: int, target_id: Optional[str]) -> str:
    return f"{int(citation_index)}:{target_id or ''}"
