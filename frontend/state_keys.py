"""Centralized Streamlit session/widget key helpers."""

from __future__ import annotations

from typing import Optional


def canonical_context_edit_key(*, doc_id: str, citation_index: int) -> str:
    return f"context-edit::{doc_id}::{int(citation_index)}"


def canonical_segments_key(*, citation_index: int) -> str:
    return f"citation-segments-{int(citation_index)}"


def scoped(*, scope: str, canonical: str) -> str:
    return f"{scope}::{canonical}"


def citation_entry_key(*, citation_index: int, target_id: Optional[str]) -> str:
    return f"{int(citation_index)}:{target_id or ''}"
