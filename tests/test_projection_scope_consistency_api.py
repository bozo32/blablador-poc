from __future__ import annotations

from fastapi.testclient import TestClient

import backend.main as backend_main


def test_ledger_place_reference_uses_scoped_graph_store_and_reconcile(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeGraphStore:
        def link_reference_to_ingest(self, **kwargs):
            captured["link_kwargs"] = dict(kwargs)
            return True

    monkeypatch.setattr(
        backend_main,
        "_require_scope_for_write_pilot",
        lambda **_kwargs: ("proj-scope", "user-scope"),
    )
    monkeypatch.setattr(
        backend_main,
        "_scoped_graph_store",
        lambda project_id: (captured.__setitem__("project_id", project_id) or _FakeGraphStore()),
    )
    monkeypatch.setattr(
        backend_main,
        "_build_ledger_response",
        lambda **kwargs: {"rows": [], "options": [], "_meta": kwargs},
    )

    client = TestClient(backend_main.app)
    resp = client.post(
        "/ledger/place-reference",
        headers={"X-Project-Id": "proj-scope", "X-User-Id": "user-scope"},
        json={
            "citing_doc_id": "doc-1",
            "reference_id": "ref-1",
            "cited_ingest_id": "doc-2",
            "canonical": False,
        },
    )
    assert resp.status_code == 200
    assert captured["project_id"] == "proj-scope"
    assert captured["link_kwargs"] == {
        "citing_doc_id": "doc-1",
        "reference_id": "ref-1",
        "cited_ingest_id": "doc-2",
        "project_id": "proj-scope",
    }


def test_attachment_clone_uses_scoped_graph_writes(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeGraphStore:
        def link_reference_to_ingest(self, **kwargs):
            captured["link_kwargs"] = dict(kwargs)
            return True

        def reconcile_ingest_projection(self, **kwargs):
            captured["reconcile_kwargs"] = dict(kwargs)
            return 0

    monkeypatch.setattr(
        backend_main,
        "_require_scope_for_write_pilot",
        lambda **_kwargs: ("proj-scope", "user-scope"),
    )
    monkeypatch.setattr(
        backend_main,
        "_scoped_graph_store",
        lambda project_id: (captured.__setitem__("project_id", project_id) or _FakeGraphStore()),
    )
    monkeypatch.setattr(backend_main, "_list_spine_ingests_by_id", lambda _project_id: {})
    monkeypatch.setattr(
        backend_main.attachment_store,
        "clone_attachment",
        lambda *_args, **_kwargs: {"id": "att-1", "source_ingest_id": "doc-2"},
    )
    monkeypatch.setattr(
        backend_main.attachment_store,
        "public_status_for_project",
        lambda *_args, **_kwargs: {
            "id": "att-1",
            "filename": "doc.pdf",
            "status": "pending",
            "uploaded_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        },
    )
    monkeypatch.setattr(backend_main.evidence_service, "trigger_auto_rerun", lambda *_a, **_k: None)

    client = TestClient(backend_main.app)
    resp = client.post(
        "/attachments/att-0/clone",
        headers={"X-Project-Id": "proj-scope", "X-User-Id": "user-scope"},
        json={"doc_id": "doc-1", "target_id": "ref-7", "claim_id": "claim-1"},
    )
    assert resp.status_code == 200
    assert captured["project_id"] == "proj-scope"
    assert captured["link_kwargs"] == {
        "citing_doc_id": "doc-1",
        "reference_id": "ref-7",
        "cited_ingest_id": "doc-2",
        "project_id": "proj-scope",
    }
    assert captured["reconcile_kwargs"]["project_id"] == "proj-scope"
