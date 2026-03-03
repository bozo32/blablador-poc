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


def test_scope_seed_starts_unapplied_without_legacy_bootstrap(_stub_streamlit) -> None:
    scope_lock.ensure_seeded()

    assert scope_lock.get_draft_uid() == ""
    assert scope_lock.get_draft_project_id() == ""
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


def test_apply_commits_draft_to_applied(_stub_streamlit) -> None:
    _stub_streamlit.session_state[scope_lock.SCOPE_DRAFT_UID] = "reviewer-a"
    _stub_streamlit.session_state[scope_lock.SCOPE_DRAFT_PROJECT_ID] = "project-a"
    _stub_streamlit.session_state["project_meta"] = {"name": "A"}

    uid, project_id = scope_lock.apply_draft_scope()

    assert (uid, project_id) == ("reviewer-a", "project-a")
    assert scope_lock.get_applied_uid() == "reviewer-a"
    assert scope_lock.get_applied_project_id() == "project-a"
    assert scope_lock.has_applied_scope() is True
    assert _stub_streamlit.session_state["project_meta"]["active_reviewer_uid"] == "reviewer-a"


def test_apply_with_backend_persistence_calls_scope_session_api(
    _stub_streamlit,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}
    _stub_streamlit.session_state[scope_lock.SCOPE_DRAFT_UID] = "reviewer-a"
    _stub_streamlit.session_state[scope_lock.SCOPE_DRAFT_PROJECT_ID] = "project-a"

    def _fake_put_scope_session(**kwargs):
        captured.update({k: str(v) for k, v in kwargs.items()})
        return {
            "user_id": "reviewer-a",
            "active_project_id": "project-a",
            "active_reviewer_uid": "reviewer-a",
        }

    monkeypatch.setattr(scope_lock.project_api, "put_scope_session", _fake_put_scope_session)

    scope_lock.apply_draft_scope(persist_backend=True)
    assert captured == {
        "user_id": "reviewer-a",
        "active_project_id": "project-a",
        "active_reviewer_uid": "reviewer-a",
    }


def test_sync_from_backend_hydrates_applied_scope(
    _stub_streamlit,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_streamlit.session_state[scope_lock.SCOPE_DRAFT_UID] = "reviewer-a"
    _stub_streamlit.session_state["project_meta"] = {"name": "A"}

    monkeypatch.setattr(
        scope_lock.project_api,
        "get_scope_session",
        lambda **_kwargs: {
            "user_id": "reviewer-a",
            "active_project_id": "project-a",
            "active_reviewer_uid": "reviewer-a",
        },
    )

    synced = scope_lock.sync_from_backend()
    assert synced == ("reviewer-a", "project-a")
    assert scope_lock.has_applied_scope() is True
    assert _stub_streamlit.session_state["project_meta"]["active_reviewer_uid"] == "reviewer-a"


def test_sync_from_backend_preserves_failure_signal(
    _stub_streamlit,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_streamlit.session_state[scope_lock.SCOPE_DRAFT_UID] = "reviewer-a"

    monkeypatch.setattr(
        scope_lock.project_api,
        "get_scope_session",
        lambda **_kwargs: (_ for _ in ()).throw(scope_lock.project_api.ProjectApiError("boom")),
    )

    synced = scope_lock.sync_from_backend()
    assert synced is None
    assert _stub_streamlit.session_state["scope_sync_error"] == "boom"
    assert scope_lock.has_applied_scope() is False
