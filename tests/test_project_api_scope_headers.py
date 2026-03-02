from __future__ import annotations

from typing import Any

import pytest

from frontend import project_api


class _FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        status_code: int = 200,
        content: bytes = b"",
    ):
        self._payload = payload or {}
        self.status_code = status_code
        self.text = "{}"
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def json(self) -> dict[str, Any]:
        return self._payload


def test_project_api_headers_include_project_and_optional_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_get(url: str, **kwargs):
        captured["get"] = {"url": url, **kwargs}
        return _FakeResponse({"name": "proj"}, content=b"zip")

    def _fake_put(url: str, **kwargs):
        captured["put"] = {"url": url, **kwargs}
        return _FakeResponse({"name": "proj"})

    def _fake_post(url: str, **kwargs):
        captured["post"] = {"url": url, **kwargs}
        return _FakeResponse({"ok": True})

    monkeypatch.setattr(project_api, "_api_root", lambda: "http://api")
    monkeypatch.setattr(project_api.requests, "get", _fake_get)
    monkeypatch.setattr(project_api.requests, "put", _fake_put)
    monkeypatch.setattr(project_api.requests, "post", _fake_post)

    meta = project_api.get_meta(project_id="proj-a", user_id="user-a")
    assert meta["name"] == "proj"
    assert captured["get"]["headers"] == {
        "X-Project-Id": "proj-a",
        "X-User-Id": "user-a",
    }

    updated = project_api.put_meta(
        {"name": "next"},
        project_id="proj-a",
        user_id="user-a",
    )
    assert updated["name"] == "proj"
    assert captured["put"]["headers"] == {
        "X-Project-Id": "proj-a",
        "X-User-Id": "user-a",
    }

    blob = project_api.export_zip(project_id="proj-a", user_id="user-a")
    assert blob == b"zip"
    assert captured["get"]["url"].endswith("/project/export")

    imported = project_api.import_zip(
        b"zip-data",
        overwrite=True,
        project_id="proj-a",
        user_id="user-a",
    )
    assert imported == {"ok": True}
    assert captured["post"]["headers"] == {
        "X-Project-Id": "proj-a",
        "X-User-Id": "user-a",
    }


def test_project_api_headers_require_project_id() -> None:
    with pytest.raises(project_api.ProjectApiError, match="project_id"):
        project_api.put_meta({"name": "x"}, project_id=None)


def test_project_api_headers_omit_user_when_not_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_get(url: str, **kwargs):
        captured.update(kwargs)
        return _FakeResponse({"name": "proj"})

    monkeypatch.setattr(project_api, "_api_root", lambda: "http://api")
    monkeypatch.setattr(project_api.requests, "get", _fake_get)

    project_api.get_meta(project_id="proj-a", user_id=None)
    assert captured["headers"] == {"X-Project-Id": "proj-a"}
