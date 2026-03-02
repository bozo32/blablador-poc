from __future__ import annotations

from typing import Optional

import streamlit as st

from frontend.state_keys import (
    SCOPE_APPLIED_PROJECT_ID,
    SCOPE_APPLIED_UID,
    SCOPE_DRAFT_PROJECT_ID,
    SCOPE_DRAFT_UID,
)


def _clean(value: Optional[str]) -> str:
    return str(value or "").strip()


def ensure_seeded() -> None:
    """Ensure scope keys exist without auto-applying defaults."""
    st.session_state.setdefault(SCOPE_DRAFT_UID, "")
    st.session_state.setdefault(SCOPE_DRAFT_PROJECT_ID, "")
    st.session_state.setdefault(SCOPE_APPLIED_UID, "")
    st.session_state.setdefault(SCOPE_APPLIED_PROJECT_ID, "")

    # Backward-compat migration: keep existing visible values as draft only.
    if not _clean(st.session_state.get(SCOPE_DRAFT_UID)):
        legacy_uid = _clean(st.session_state.get("active_reviewer_uid"))
        if legacy_uid:
            st.session_state[SCOPE_DRAFT_UID] = legacy_uid
    if not _clean(st.session_state.get(SCOPE_DRAFT_PROJECT_ID)):
        legacy_project_id = _clean(st.session_state.get("project_id"))
        if legacy_project_id:
            st.session_state[SCOPE_DRAFT_PROJECT_ID] = legacy_project_id


def get_draft_uid() -> str:
    ensure_seeded()
    return _clean(st.session_state.get(SCOPE_DRAFT_UID))


def get_draft_project_id() -> str:
    ensure_seeded()
    return _clean(st.session_state.get(SCOPE_DRAFT_PROJECT_ID))


def get_applied_uid() -> str:
    ensure_seeded()
    return _clean(st.session_state.get(SCOPE_APPLIED_UID))


def get_applied_project_id() -> str:
    ensure_seeded()
    return _clean(st.session_state.get(SCOPE_APPLIED_PROJECT_ID))


def has_applied_scope() -> bool:
    return bool(get_applied_uid() and get_applied_project_id())


def apply_draft_scope() -> tuple[str, str]:
    """Commit draft uid+project into applied scope and legacy keys."""
    uid = get_draft_uid()
    project_id = get_draft_project_id()
    if not uid:
        raise ValueError("Scope apply requires user id")
    if not project_id:
        raise ValueError("Scope apply requires project id")

    st.session_state[SCOPE_APPLIED_UID] = uid
    st.session_state[SCOPE_APPLIED_PROJECT_ID] = project_id

    # Mirror to existing callsites that still read legacy keys.
    st.session_state["active_reviewer_uid"] = uid
    st.session_state["project_id"] = project_id

    meta = st.session_state.get("project_meta")
    if isinstance(meta, dict):
        next_meta = dict(meta)
        next_meta["active_reviewer_uid"] = uid
        st.session_state["project_meta"] = next_meta
    return uid, project_id
