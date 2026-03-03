from __future__ import annotations

from typing import Any

import pytest

from frontend import (
    evidence_api,
    graph_api,
    ingestion_api,
    judgment_api,
    ledger_api,
    opinion_api,
    project_api,
    ui,
)


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


def test_opinion_read_fails_fast_without_project_or_reviewer() -> None:
    with pytest.raises(RuntimeError, match="project_id"):
        opinion_api.list_follows_by_doc("http://api", "", "reviewer-a", "doc-1")

    with pytest.raises(RuntimeError, match="reviewer_uid"):
        opinion_api.get_follow_for_target("http://api", "proj-a", "", "citespan:1")


def test_opinion_read_sends_strict_scope_identity_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_request(method: str, url: str, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse({"follows": [{"span_id": "span-1"}]})

    monkeypatch.setattr(opinion_api.requests, "request", _fake_request)
    follows = opinion_api.list_follows_by_doc(
        "http://api",
        "proj-a",
        "reviewer-a",
        "doc-1",
    )
    assert follows == [{"span_id": "span-1"}]
    assert captured["headers"] == {
        "X-Project-Id": "proj-a",
        "X-User-Id": "reviewer-a",
        "X-Reviewer-Uid": "reviewer-a",
    }


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


def test_ingestion_citation_context_fails_fast_without_project_or_user() -> None:
    with pytest.raises(RuntimeError, match="project_id"):
        ingestion_api.get_citation_context(
            "http://api",
            "doc-1",
            0,
            project_id=None,
            user_id="reviewer-a",
        )

    with pytest.raises(RuntimeError, match="user_id"):
        ingestion_api.get_citation_context(
            "http://api",
            "doc-1",
            0,
            project_id="proj-a",
            user_id=None,
        )


def test_ingestion_citation_context_sends_scope_identity_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_get(url: str, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse({"context": {"citing_sentence": "x"}})

    monkeypatch.setattr(ingestion_api.requests, "get", _fake_get)

    payload = ingestion_api.get_citation_context(
        "http://api",
        "doc-1",
        2,
        target_id="ref-1",
        project_id="proj-a",
        user_id="reviewer-a",
    )
    assert payload["context"]["citing_sentence"] == "x"
    assert captured["headers"] == {
        "X-Project-Id": "proj-a",
        "X-User-Id": "reviewer-a",
    }


def test_ingestion_citation_graph_fails_fast_without_project_or_user() -> None:
    with pytest.raises(RuntimeError, match="project_id"):
        ingestion_api.get_citation_graph(
            "http://api",
            "doc-1",
            "ref-1",
            1,
            25,
            project_id=None,
            user_id="reviewer-a",
        )

    with pytest.raises(RuntimeError, match="user_id"):
        ingestion_api.get_citation_graph(
            "http://api",
            "doc-1",
            "ref-1",
            1,
            25,
            project_id="proj-a",
            user_id=None,
        )


def test_ingestion_citation_graph_sends_scope_identity_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_get(url: str, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse({"nodes": [], "edges": []})

    monkeypatch.setattr(ingestion_api.requests, "get", _fake_get)

    payload = ingestion_api.get_citation_graph(
        "http://api",
        "doc-1",
        "ref-1",
        1,
        25,
        project_id="proj-a",
        user_id="reviewer-a",
    )
    assert payload == {"nodes": [], "edges": []}
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


def test_project_api_fails_fast_without_project_scope() -> None:
    with pytest.raises(project_api.ProjectApiError, match="project_id"):
        project_api.get_meta(project_id=None)


def test_project_api_fails_fast_without_user_scope() -> None:
    with pytest.raises(project_api.ProjectApiError, match="user_id"):
        project_api.get_meta(project_id="proj-a", user_id=None)


def test_evidence_client_fails_fast_without_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = type("_Stub", (), {"session_state": {}})()
    monkeypatch.setattr(evidence_api, "st", stub)
    with pytest.raises(evidence_api.EvidenceApiError, match="project_id"):
        evidence_api.list_evidence("claim-1", reviewer_uid="reviewer-a")


def test_evidence_client_sends_strict_scope_identity_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    stub = type("_Stub", (), {"session_state": {"api_url": "http://api"}})()
    monkeypatch.setattr(evidence_api, "st", stub)
    monkeypatch.setattr(
        evidence_api.scope_lock,
        "get_applied_project_id",
        lambda: "proj-a",
    )
    monkeypatch.setattr(
        evidence_api.scope_lock,
        "get_applied_uid",
        lambda: "reviewer-a",
    )

    def _fake_request(method: str, path: str, **kwargs):
        captured["method"] = method
        captured["path"] = path
        captured.update(kwargs)
        return {"candidates": [], "total": 0, "offset": 0, "limit": 5}

    monkeypatch.setattr(evidence_api, "_request", _fake_request)
    evidence_api.list_evidence("claim-1", reviewer_uid="reviewer-a")
    assert captured["headers"] == {
        "X-Project-Id": "proj-a",
        "X-User-Id": "reviewer-a",
        "X-Reviewer-Uid": "reviewer-a",
    }


def test_judgment_client_sends_strict_scope_identity_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    stub = type(
        "_Stub",
        (),
        {
            "session_state": {
                "api_url": "http://api",
            },
            "toast": lambda *_args, **_kwargs: None,
            "error": lambda *_args, **_kwargs: None,
        },
    )()
    monkeypatch.setattr(judgment_api, "st", stub)
    monkeypatch.setattr(
        judgment_api.scope_lock,
        "get_applied_project_id",
        lambda: "proj-a",
    )
    monkeypatch.setattr(
        judgment_api.scope_lock,
        "get_applied_uid",
        lambda: "reviewer-a",
    )

    def _fake_request(method: str, url: str, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse({"claim_id": "claim-1"})

    monkeypatch.setattr(judgment_api.requests, "request", _fake_request)
    payload = judgment_api.get_judgment("claim-1", reviewer_uid="reviewer-a")
    assert payload["claim_id"] == "claim-1"
    assert captured["headers"] == {
        "X-Project-Id": "proj-a",
        "X-User-Id": "reviewer-a",
        "X-Reviewer-Uid": "reviewer-a",
    }


def test_ui_prefers_canonical_ledger_stage_fields() -> None:
    row = {
        "canonical_extraction_status": "running",
        "extraction_status": "complete",
    }
    snapshot = {"canonical_extraction_status": "error"}
    assert (
        ui._ledger_canonical_stage(
            row,
            snapshot,
            stage="extraction",
            field="status",
        )
        == "running"
    )
