from __future__ import annotations

import types

import pytest

from frontend import scope_lock


@pytest.fixture(autouse=True)
def _stub_streamlit(monkeypatch: pytest.MonkeyPatch):
    stub = types.SimpleNamespace(session_state={})
    monkeypatch.setattr(scope_lock, "st", stub)
    yield stub
    stub.session_state.clear()


def test_scope_seed_migrates_legacy_values_to_draft_only(_stub_streamlit) -> None:
    _stub_streamlit.session_state["active_reviewer_uid"] = "reviewer-a"
    _stub_streamlit.session_state["project_id"] = "project-a"

    scope_lock.ensure_seeded()

    assert scope_lock.get_draft_uid() == "reviewer-a"
    assert scope_lock.get_draft_project_id() == "project-a"
    assert scope_lock.get_applied_uid() == ""
    assert scope_lock.get_applied_project_id() == ""
    assert scope_lock.has_applied_scope() is False


def test_apply_requires_both_uid_and_project(_stub_streamlit) -> None:
    _stub_streamlit.session_state[scope_lock.SCOPE_DRAFT_UID] = ""
    _stub_streamlit.session_state[scope_lock.SCOPE_DRAFT_PROJECT_ID] = "project-a"
    with pytest.raises(ValueError, match="user id"):
        scope_lock.apply_draft_scope()

    _stub_streamlit.session_state[scope_lock.SCOPE_DRAFT_UID] = "reviewer-a"
    _stub_streamlit.session_state[scope_lock.SCOPE_DRAFT_PROJECT_ID] = ""
    with pytest.raises(ValueError, match="project id"):
        scope_lock.apply_draft_scope()


def test_apply_commits_draft_to_applied_and_legacy_keys(_stub_streamlit) -> None:
    _stub_streamlit.session_state[scope_lock.SCOPE_DRAFT_UID] = "reviewer-a"
    _stub_streamlit.session_state[scope_lock.SCOPE_DRAFT_PROJECT_ID] = "project-a"
    _stub_streamlit.session_state["project_meta"] = {"name": "A"}

    uid, project_id = scope_lock.apply_draft_scope()

    assert (uid, project_id) == ("reviewer-a", "project-a")
    assert scope_lock.get_applied_uid() == "reviewer-a"
    assert scope_lock.get_applied_project_id() == "project-a"
    assert scope_lock.has_applied_scope() is True
    assert _stub_streamlit.session_state["active_reviewer_uid"] == "reviewer-a"
    assert _stub_streamlit.session_state["project_id"] == "project-a"
    assert _stub_streamlit.session_state["project_meta"]["active_reviewer_uid"] == "reviewer-a"
