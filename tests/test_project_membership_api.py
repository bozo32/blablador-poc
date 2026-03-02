from __future__ import annotations

from fastapi.testclient import TestClient

import backend.main as backend_main


def test_projects_endpoints_require_x_user_id() -> None:
    client = TestClient(backend_main.app)

    assert client.get("/projects").status_code == 400
    assert client.post("/projects", json={"name": "A"}).status_code == 400
    assert client.post("/projects/select", json={"project_id": "proj-a"}).status_code == 400
    assert client.get("/projects/active").status_code == 400


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
