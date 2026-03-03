from __future__ import annotations

from fastapi.testclient import TestClient

import backend.main as backend_main


def test_projects_endpoints_require_x_user_id() -> None:
    client = TestClient(backend_main.app)

    assert client.get("/projects").status_code == 400
    assert client.post("/projects", json={"name": "A"}).status_code == 400
    assert client.post("/projects/select", json={"project_id": "proj-a"}).status_code == 400
    assert client.get("/projects/active").status_code == 400
    assert client.get("/scope/session").status_code == 400
    assert client.put("/scope/session", json={"active_project_id": "proj-a"}).status_code == 400


def test_projects_users_list(monkeypatch) -> None:
    monkeypatch.setattr(backend_main, "list_known_user_ids", lambda **_kwargs: ["user-a", "user-b"])
    client = TestClient(backend_main.app)

    resp = client.get("/projects/users")
    assert resp.status_code == 200
    assert resp.json() == {"users": ["user-a", "user-b"]}


def test_projects_users_list_includes_scope_session_only_users() -> None:
    backend_main.create_project_for_user(
        user_id="member-user",
        actor_user_id="member-user",
        project_id="proj-a",
    )
    backend_main.set_scope_session_for_user(
        user_id="scope-only-user",
        actor_user_id="scope-only-user",
        active_project_id=None,
        active_reviewer_uid="scope-only-user",
    )

    client = TestClient(backend_main.app)
    resp = client.get("/projects/users")
    assert resp.status_code == 200
    users = set(resp.json().get("users") or [])
    assert "member-user" in users
    assert "scope-only-user" in users


def test_projects_list_and_active(monkeypatch) -> None:
    monkeypatch.setattr(
        backend_main,
        "list_projects_for_user",
        lambda **kwargs: [
            {
                "project_id": "proj-a",
                "role": "owner",
                "joined_at": "2026-03-01T00:00:00Z",
                "updated_at": "2026-03-01T00:00:00Z",
                "is_active": True,
            }
        ],
    )
    monkeypatch.setattr(
        backend_main,
        "get_active_project_for_user",
        lambda **kwargs: "proj-a",
    )

    client = TestClient(backend_main.app)
    resp = client.get("/projects", headers={"X-User-Id": "user-a"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["active_project_id"] == "proj-a"
    assert body["projects"][0]["project_id"] == "proj-a"

    active_resp = client.get("/projects/active", headers={"X-User-Id": "user-a"})
    assert active_resp.status_code == 200
    assert active_resp.json() == {"active_project_id": "proj-a"}


def test_create_project_and_select_project(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        backend_main,
        "create_project_for_user",
        lambda **kwargs: {
            "project_id": "proj-new",
            "role": "owner",
            "joined_at": "2026-03-01T00:00:00Z",
            "updated_at": "2026-03-01T00:00:00Z",
        },
    )
    monkeypatch.setattr(
        backend_main,
        "get_or_create_project_meta",
        lambda **kwargs: {"name": "proj-new"},
    )

    def _fake_update_project_meta(**kwargs):
        captured["patch"] = kwargs["patch"]
        return {"name": kwargs["patch"].get("name")}

    monkeypatch.setattr(backend_main, "update_project_meta", _fake_update_project_meta)
    monkeypatch.setattr(backend_main, "set_active_project_for_user", lambda **kwargs: True)

    client = TestClient(backend_main.app)
    create_resp = client.post(
        "/projects",
        json={"name": "Project New"},
        headers={"X-User-Id": "user-a"},
    )
    assert create_resp.status_code == 200
    assert create_resp.json()["active_project_id"] == "proj-new"
    assert captured["patch"] == {"name": "Project New"}

    select_resp = client.post(
        "/projects/select",
        json={"project_id": "proj-new"},
        headers={"X-User-Id": "user-a"},
    )
    assert select_resp.status_code == 200
    assert select_resp.json() == {"ok": True, "active_project_id": "proj-new"}


def test_select_project_returns_403_without_membership(monkeypatch) -> None:
    monkeypatch.setattr(backend_main, "set_active_project_for_user", lambda **kwargs: False)
    client = TestClient(backend_main.app)

    resp = client.post(
        "/projects/select",
        json={"project_id": "proj-a"},
        headers={"X-User-Id": "user-a"},
    )
    assert resp.status_code == 403


def test_scope_session_endpoints_read_and_write(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        backend_main,
        "get_scope_session_for_user",
        lambda **kwargs: {
            "user_id": kwargs["user_id"],
            "active_project_id": "proj-a",
            "active_reviewer_uid": "reviewer-a",
            "updated_at": "2026-03-03T00:00:00Z",
        },
    )

    def _fake_set_scope_session_for_user(**kwargs):
        captured.update(kwargs)
        return {
            "user_id": kwargs["user_id"],
            "active_project_id": kwargs.get("active_project_id"),
            "active_reviewer_uid": kwargs.get("active_reviewer_uid") or kwargs["user_id"],
            "updated_at": "2026-03-03T00:00:00Z",
        }

    monkeypatch.setattr(backend_main, "set_scope_session_for_user", _fake_set_scope_session_for_user)

    client = TestClient(backend_main.app)

    get_resp = client.get("/scope/session", headers={"X-User-Id": "reviewer-a"})
    assert get_resp.status_code == 200
    assert get_resp.json()["active_project_id"] == "proj-a"

    put_resp = client.put(
        "/scope/session",
        json={"active_project_id": "proj-b", "active_reviewer_uid": "reviewer-b"},
        headers={"X-User-Id": "reviewer-a"},
    )
    assert put_resp.status_code == 200
    assert put_resp.json()["active_project_id"] == "proj-b"
    assert captured["user_id"] == "reviewer-a"
    assert captured["active_project_id"] == "proj-b"
    assert captured["active_reviewer_uid"] == "reviewer-b"


def test_scope_session_put_requires_payload_values() -> None:
    client = TestClient(backend_main.app)
    resp = client.put("/scope/session", json={}, headers={"X-User-Id": "reviewer-a"})
    assert resp.status_code == 422
