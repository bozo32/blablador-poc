from __future__ import annotations

from typing import Any

import pytest

from frontend import graph_api, ingestion_api, ledger_api, opinion_api


class _FakeResponse:
    def __init__(self, payload: dict[str, Any] | None = None, *, status_code: int = 200):
        self._payload = payload or {}
        self.status_code = status_code
        self.text = "{}"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def json(self) -> dict[str, Any]:
        return self._payload


def test_ledger_mutation_fails_fast_without_project_or_user() -> None:
    with pytest.raises(RuntimeError, match="project_id"):
        ledger_api.set_outgoing("http://api", 1, [2], project_id=None, user_id="u1")

    with pytest.raises(RuntimeError, match="user_id"):
        ledger_api.set_outgoing("http://api", 1, [2], project_id="p1", user_id=None)


def test_ledger_mutation_sends_strict_scope_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_patch(url: str, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse({"rows": [], "options": []})

    monkeypatch.setattr(ledger_api.requests, "patch", _fake_patch)

    payload = ledger_api.set_outgoing(
        "http://api",
        7,
        [8, 9],
        project_id="proj-a",
        user_id="reviewer-a",
    )
    assert payload == {"rows": [], "options": []}
    assert captured["headers"] == {
        "X-Project-Id": "proj-a",
        "X-User-Id": "reviewer-a",
    }


def test_opinion_append_follow_fails_fast_without_scope() -> None:
    with pytest.raises(RuntimeError, match="project_id"):
        opinion_api.append_follow(
            "http://api",
            "",
            "reviewer-a",
            "doc-1",
            0,
            None,
            "span-1",
            "follow",
        )

    with pytest.raises(RuntimeError, match="reviewer_uid"):
        opinion_api.append_follow(
            "http://api",
            "proj-a",
            "",
            "doc-1",
            0,
            None,
            "span-1",
            "follow",
        )


def test_opinion_append_follow_sends_identity_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_request(method: str, url: str, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse({"event_id": 1})

    monkeypatch.setattr(opinion_api.requests, "request", _fake_request)

    resp = opinion_api.append_follow(
        "http://api",
        "proj-a",
        "reviewer-a",
        "doc-1",
        3,
        "ref-9",
        "span-1",
        "follow",
    )
    assert resp == {"event_id": 1}
    assert captured["headers"]["X-Project-Id"] == "proj-a"
    assert captured["headers"]["X-User-Id"] == "reviewer-a"
    assert captured["headers"]["X-Reviewer-Uid"] == "reviewer-a"


def test_ingestion_mutation_fails_fast_without_project_or_user() -> None:
    with pytest.raises(RuntimeError, match="project_id"):
        ingestion_api.trigger_resolution(
            "http://api",
            "doc-1",
            project_id=None,
            user_id="reviewer-a",
        )

    with pytest.raises(RuntimeError, match="user_id"):
        ingestion_api.trigger_resolution(
            "http://api",
            "doc-1",
            project_id="proj-a",
            user_id=None,
        )


def test_ingestion_mutation_sends_scope_identity_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_post(url: str, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse({"document_id": "doc-1", "resolution": {"status": "running"}})

    monkeypatch.setattr(ingestion_api.requests, "post", _fake_post)

    payload = ingestion_api.trigger_resolution(
        "http://api",
        "doc-1",
        project_id="proj-a",
        user_id="reviewer-a",
    )
    assert payload["document_id"] == "doc-1"
    assert captured["headers"] == {
        "X-Project-Id": "proj-a",
        "X-User-Id": "reviewer-a",
    }


def test_graph_mutation_fails_fast_without_scope_identity() -> None:
    with pytest.raises(graph_api.GraphApiError, match="project_id"):
        graph_api.reindex_docs(project_id=None, user_id="reviewer-a")

    with pytest.raises(graph_api.GraphApiError, match="user_id"):
        graph_api.reindex_docs(project_id="proj-a", user_id=None)


def test_graph_reindex_sends_scope_identity_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_post(url: str, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse({"ok": True})

    monkeypatch.setattr(graph_api, "_api_root", lambda: "http://api")
    monkeypatch.setattr(graph_api.requests, "post", _fake_post)

    payload = graph_api.reindex_docs(project_id="proj-a", user_id="reviewer-a")
    assert payload == {"ok": True}
    assert captured["headers"] == {
        "X-Project-Id": "proj-a",
        "X-User-Id": "reviewer-a",
    }
