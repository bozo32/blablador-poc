from __future__ import annotations

from backend.spine import project_membership, scope_session


def test_create_list_and_select_active_project() -> None:
    created = project_membership.create_project_for_user(
        user_id="user-a",
        actor_user_id="user-a",
        project_id="proj-a",
    )
    assert created["project_id"] == "proj-a"
    assert created["role"] == "owner"

    changed = project_membership.set_active_project_for_user(
        user_id="user-a",
        project_id="proj-a",
        actor_user_id="user-a",
    )
    assert changed is True

    active = project_membership.get_active_project_for_user(user_id="user-a")
    assert active == "proj-a"

    rows = project_membership.list_projects_for_user(user_id="user-a")
    assert len(rows) == 1
    assert rows[0]["project_id"] == "proj-a"
    assert rows[0]["is_active"] is True


def test_set_active_requires_membership() -> None:
    ok = project_membership.set_active_project_for_user(
        user_id="user-a",
        project_id="proj-missing",
        actor_user_id="user-a",
    )
    assert ok is False
    assert project_membership.get_active_project_for_user(user_id="user-a") is None


def test_create_project_generates_id_when_missing() -> None:
    created = project_membership.create_project_for_user(
        user_id="user-b",
        actor_user_id="user-b",
        project_id=None,
    )
    assert created["project_id"]
    assert len(created["project_id"]) >= 8


def test_list_known_user_ids_includes_scope_session_only_users() -> None:
    project_membership.create_project_for_user(
        user_id="member-user",
        actor_user_id="member-user",
        project_id="proj-a",
    )
    scope_session.set_scope_session_for_user(
        user_id="scope-only-user",
        actor_user_id="scope-only-user",
        active_project_id=None,
        active_reviewer_uid="scope-only-user",
    )

    users = project_membership.list_known_user_ids(limit=200)
    assert "member-user" in users
    assert "scope-only-user" in users
