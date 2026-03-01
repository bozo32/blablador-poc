from __future__ import annotations

import pytest
from fastapi import HTTPException

import backend.main as backend_main


def test_scope_guard_uses_headers_when_present() -> None:
    project_id, user_id, sources = backend_main._resolve_scope_guard(
        x_project_id=" proj-1 ",
        x_user_id=" user-1 ",
        require_project=True,
        include_user=True,
    )

    assert project_id == "proj-1"
    assert user_id == "user-1"
    assert sources == {"project": "header", "user": "header"}


def test_scope_guard_rejects_missing_required_project_without_dev_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ALLOW_DEFAULT_PROJECT_ID_FOR_DEV", raising=False)

    with pytest.raises(HTTPException) as exc_info:
        backend_main._resolve_scope_guard(
            x_project_id=None,
            require_project=True,
            allow_dev_project_default=True,
            include_user=False,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "X-Project-Id header is required"


def test_scope_guard_allows_dev_default_for_required_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALLOW_DEFAULT_PROJECT_ID_FOR_DEV", "1")

    project_id, user_id, sources = backend_main._resolve_scope_guard(
        x_project_id=None,
        require_project=True,
        allow_dev_project_default=True,
        include_user=False,
    )

    assert project_id == str(backend_main.app_settings.DEFAULT_PROJECT_ID)
    assert user_id is None
    assert sources == {"project": "dev-default-project"}


def test_scope_guard_preserves_non_strict_default_behavior() -> None:
    project_id, user_id, sources = backend_main._resolve_scope_guard(
        x_project_id=None,
        x_user_id=None,
        require_project=False,
        include_user=True,
    )

    assert project_id == str(backend_main.app_settings.DEFAULT_PROJECT_ID)
    assert user_id == str(backend_main.app_settings.DEFAULT_USER_ID)
    assert sources == {"project": "default-project", "user": "default-user"}


def test_require_scope_for_upload_returns_project_and_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALLOW_DEFAULT_PROJECT_ID_FOR_DEV", "true")

    project_id, user_id = backend_main._require_scope_for_upload(
        x_project_id=None,
        endpoint="/ingest",
    )

    assert project_id == str(backend_main.app_settings.DEFAULT_PROJECT_ID)
    assert user_id == str(backend_main.app_settings.DEFAULT_USER_ID)
