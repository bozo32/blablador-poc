from __future__ import annotations

import io
import json
import zipfile

from fastapi.testclient import TestClient

import backend.main as backend_main


def _project_meta(name: str = "proj") -> dict:
    return {
        "version": 1,
        "name": name,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": None,
        "reviewers": [],
        "active_reviewer_uid": None,
        "compare_reviewer_a": None,
        "compare_reviewer_b": None,
        "graph_settings": {},
    }


def test_project_get_and_put_thread_x_user_id(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def _fake_get_or_create(*, project_id: str, user_id: str = "local") -> dict:
        captured["get_project_id"] = project_id
        captured["get_user_id"] = user_id
        return _project_meta(project_id)

    def _fake_update(*, project_id: str, user_id: str, patch: dict) -> dict:
        captured["put_project_id"] = project_id
        captured["put_user_id"] = user_id
        captured["patch"] = json.dumps(patch, sort_keys=True)
        return _project_meta(project_id)

    monkeypatch.setattr(backend_main, "get_or_create_project_meta", _fake_get_or_create)
    monkeypatch.setattr(backend_main, "update_project_meta", _fake_update)

    client = TestClient(backend_main.app)
    resp_get = client.get(
        "/project",
        headers={"X-Project-Id": "proj-a", "X-User-Id": "reviewer-a"},
    )
    assert resp_get.status_code == 200
    assert captured["get_project_id"] == "proj-a"
    assert captured["get_user_id"] == "reviewer-a"

    resp_put = client.put(
        "/project",
        json={"name": "renamed"},
        headers={"X-Project-Id": "proj-a", "X-User-Id": "reviewer-a"},
    )
    assert resp_put.status_code == 200
    assert captured["put_project_id"] == "proj-a"
    assert captured["put_user_id"] == "reviewer-a"
    assert captured["patch"] == '{"name": "renamed"}'


def test_project_export_and_import_require_user_scope(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def _fake_get_or_create(*, project_id: str, user_id: str = "local") -> dict:
        captured["meta_project_id"] = project_id
        captured["meta_user_id"] = user_id
        return _project_meta(project_id)

    def _fake_export(*, project_id: str) -> bytes:
        captured["export_project_id"] = project_id
        return b"{}\n"

    def _fake_import_meta(*, project_id: str, user_id: str, meta: dict) -> dict:
        captured["import_project_id"] = project_id
        captured["import_user_id"] = user_id
        return _project_meta(project_id)

    class _DecisionsStub:
        @staticmethod
        def export_events_ndjson(*, project_id: str) -> bytes:
            captured["decisions_export_project_id"] = project_id
            return b""

        @staticmethod
        def import_events_ndjson(*, project_id: str, user_id: str, blob: bytes, overwrite: bool):
            captured["decisions_import_user_id"] = user_id

    class _OpinionStub:
        @staticmethod
        def export_events_ndjson(*, project_id: str, owner_uid: str | None, include_private: bool) -> bytes:
            captured["opinion_export_owner_uid"] = str(owner_uid)
            return b""

        @staticmethod
        def import_events_ndjson(*, project_id: str, user_id: str, blob: bytes, overwrite: bool):
            captured["opinion_import_user_id"] = user_id

    monkeypatch.setattr(backend_main, "get_or_create_project_meta", _fake_get_or_create)
    monkeypatch.setattr(backend_main, "export_project_meta_json", _fake_export)
    monkeypatch.setattr(backend_main, "import_project_meta_json", _fake_import_meta)
    monkeypatch.setattr(backend_main, "evidence_decisions_spine", _DecisionsStub())
    monkeypatch.setattr(backend_main, "opinion_events", _OpinionStub())

    client = TestClient(backend_main.app)

    export_resp = client.get(
        "/project/export",
        headers={"X-Project-Id": "proj-a", "X-User-Id": "reviewer-a"},
    )
    assert export_resp.status_code == 200
    assert captured["meta_user_id"] == "reviewer-a"
    assert captured["opinion_export_owner_uid"] == "None"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.json", json.dumps(_project_meta("proj-a")))
        zf.writestr("evidence_decision_events.ndjson", "")
        zf.writestr("opinion_events.ndjson", "")

    import_resp = client.post(
        "/project/import?overwrite=true",
        headers={"X-Project-Id": "proj-a", "X-User-Id": "reviewer-a"},
        files={"file": ("project.zip", buf.getvalue(), "application/zip")},
    )
    assert import_resp.status_code == 200
    assert captured["import_user_id"] == "reviewer-a"
