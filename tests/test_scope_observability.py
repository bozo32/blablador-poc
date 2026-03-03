from __future__ import annotations

import logging

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import backend.main as backend_main


def test_scope_observability_logs_project_scope_without_fallback(caplog) -> None:
    backend_main._reset_scope_observability_for_tests()
    client = TestClient(backend_main.app)

    with caplog.at_level(logging.INFO, logger="backend.main"):
        resp = client.get(
            "/project",
            headers={"X-Project-Id": "proj-a", "X-User-Id": "user-a"},
        )

    assert resp.status_code == 200

    counters = client.get("/scope/observability/fallbacks")
    assert counters.status_code == 200
    payload = counters.json()
    assert int(payload["fallback_counts_by_endpoint"].get("/project") or 0) == 0

    assert any("resolved_user_id" in record.getMessage() for record in caplog.records)
    assert any("resolved_project_id" in record.getMessage() for record in caplog.records)
    assert any("scope_source" in record.getMessage() for record in caplog.records)
    assert not any("scope.observability.fallback" in record.getMessage() for record in caplog.records)


def test_scope_observability_graph_resolve_requires_explicit_scope_headers() -> None:
    backend_main._reset_scope_observability_for_tests()
    client = TestClient(backend_main.app)

    resp = client.post(
        "/graph/resolve-references",
        json={
            "citing_doc_id": "doc-1",
            "reference_ids": ["ref-a", "ref-b"],
        },
    )
    assert resp.status_code == 422

    counters = client.get("/scope/observability/fallbacks")
    assert counters.status_code == 200
    payload = counters.json()
    assert int(payload["fallback_counts_by_endpoint"].get("/graph/resolve-references") or 0) == 0


def test_require_project_id_for_upload_keeps_strict_error_without_dev_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ALLOW_DEFAULT_PROJECT_ID_FOR_DEV", raising=False)

    with pytest.raises(HTTPException) as exc_info:
        backend_main._require_project_id_for_upload(None, endpoint="/ingest")

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "X-Project-Id header is required"


def test_require_project_id_for_upload_supports_dev_default_and_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_main._reset_scope_observability_for_tests()
    monkeypatch.setenv("ALLOW_DEFAULT_PROJECT_ID_FOR_DEV", "true")

    resolved = backend_main._require_project_id_for_upload(
        None,
        endpoint="/test/upload",
    )
    assert resolved == str(backend_main.app_settings.DEFAULT_PROJECT_ID)

    snapshot = backend_main._scope_fallback_snapshot()
    assert int(snapshot.get("/test/upload") or 0) == 1


def test_scope_observability_runtime_stamp_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_main.app_settings, "API_RUNTIME_STAMP", "api-runtime-2026")
    monkeypatch.setattr(backend_main.app_settings, "API_GIT_SHA", "abcdef1234567890")
    monkeypatch.setattr(backend_main.app_settings, "API_IMAGE_TAG", "app-api:test")

    client = TestClient(backend_main.app)
    resp = client.get("/scope/observability/runtime-stamp")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["api_runtime_stamp"] == "api-runtime-2026"
    assert payload["api_git_sha"] == "abcdef1234567890"
    assert payload["api_image_tag"] == "app-api:test"
