from __future__ import annotations

from typing import Optional

import streamlit as st

from frontend import project_api
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
    st.session_state.setdefault("scope_sync_error", None)


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


def sync_from_backend(*, preferred_user: Optional[str] = None) -> tuple[str, str] | None:
    ensure_seeded()
    candidates = [
        _clean(preferred_user),
        get_draft_uid(),
        get_applied_uid(),
    ]
    user_id = next((candidate for candidate in candidates if candidate), "")
    if not user_id:
        st.session_state["scope_sync_error"] = None
        return None

    try:
        session = project_api.get_scope_session(user_id=user_id)
    except project_api.ProjectApiError as exc:
        st.session_state["scope_sync_error"] = str(exc)
        return None

    project_id = _clean(session.get("active_project_id"))
    uid = _clean(session.get("active_reviewer_uid")) or _clean(session.get("user_id")) or user_id
    if not (uid and project_id):
        st.session_state["scope_sync_error"] = None
        return None

    st.session_state[SCOPE_DRAFT_UID] = uid
    st.session_state[SCOPE_DRAFT_PROJECT_ID] = project_id
    st.session_state[SCOPE_APPLIED_UID] = uid
    st.session_state[SCOPE_APPLIED_PROJECT_ID] = project_id

    meta = st.session_state.get("project_meta")
    if isinstance(meta, dict):
        next_meta = dict(meta)
        next_meta["active_reviewer_uid"] = uid
        st.session_state["project_meta"] = next_meta
    st.session_state["scope_sync_error"] = None
    return uid, project_id


def apply_draft_scope(*, persist_backend: bool = False) -> tuple[str, str]:
    """Commit draft uid+project into applied scope."""
    uid = get_draft_uid()
    project_id = get_draft_project_id()
    if not uid:
        raise ValueError("Scope apply requires user id")
    if not project_id:
        raise ValueError("Scope apply requires project id")

    if persist_backend:
        project_api.put_scope_session(
            user_id=uid,
            active_project_id=project_id,
            active_reviewer_uid=uid,
        )

    st.session_state[SCOPE_APPLIED_UID] = uid
    st.session_state[SCOPE_APPLIED_PROJECT_ID] = project_id

    meta = st.session_state.get("project_meta")
    if isinstance(meta, dict):
        next_meta = dict(meta)
        next_meta["active_reviewer_uid"] = uid
        st.session_state["project_meta"] = next_meta
    return uid, project_id
