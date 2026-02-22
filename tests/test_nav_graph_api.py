from __future__ import annotations

from fastapi.testclient import TestClient

import backend.main as backend_main
from backend.span_graph_store import SpanGraphStore


def _seed_one_citing_span(*, client: TestClient) -> dict:
    resp = client.post(
        "/claims/confirm",
        json={
            "document_id": "doc-1",
            "sentence_id": "s1",
            "sentence_text": "Seed sentence with a citation.",
            "citation_index": 9,
            "target_id": "b4",
            "segmentation_model": "local",
            "reviewer_uid": "alice",
            "cited_work_id": "ref:doc-1:b4",
            "confirmed_claims": [
                {"claim_index": 1, "parsed_text": "Seed snippet for context."}
            ],
        },
    )
    assert resp.status_code == 200
    return resp.json()


def test_nav_work_contexts_and_graph(tmp_path, monkeypatch):
    store = SpanGraphStore(tmp_path / "graph.db")
    monkeypatch.setattr(backend_main, "span_graph_store", store)
    client = TestClient(backend_main.app)

    _ = _seed_one_citing_span(client=client)

    span = client.get(
        "/spans/lookup-citation-window",
        params={"ingest_id": "doc-1", "citation_index": 9, "target_id": "b4"},
    )
    assert span.status_code == 200
    span_id = str((span.json() or {}).get("span_id") or "")
    assert span_id.startswith("span:")

    contexts = client.get(
        "/nav/works/ref:doc-1:b4/contexts", params={"reviewer_uid": "alice"}
    )
    assert contexts.status_code == 200
    payload = contexts.json()
    rows = payload.get("contexts") or []
    assert isinstance(rows, list)
    assert len(rows) >= 1
    first = rows[0]
    assert "citing_doc_id" in first
    assert "citation_index" in first
    assert "reference_id" in first
    assert "sentence_id" in first
    assert "snippet" in first

    graph = client.get(
        "/nav/graph",
        params={
            "reviewer_uid": "alice",
            "focus_type": "citespan",
            "focus_id": span_id,
            "show_claimspans": "true",
        },
    )
    assert graph.status_code == 200
    graph_payload = graph.json()
    elements = graph_payload.get("elements") or []
    assert isinstance(elements, list)
    assert len(elements) >= 1
    for el in elements:
        assert isinstance(el, dict)
        data = el.get("data")
        assert isinstance(data, dict)
        assert str(data.get("id") or "").strip()
        assert str(data.get("selectable_type") or "").strip()
        assert str(data.get("state") or "").strip()


def test_nav_contexts_missing_optional_data_never_500(tmp_path, monkeypatch):
    store = SpanGraphStore(tmp_path / "graph.db")
    monkeypatch.setattr(backend_main, "span_graph_store", store)
    client = TestClient(backend_main.app)

    resp = client.get(
        "/nav/works/unknown-work/contexts", params={"reviewer_uid": "alice"}
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload.get("contexts") == []


def test_nav_graph_work_focus_show_claimspans_toggle(tmp_path, monkeypatch):
    store = SpanGraphStore(tmp_path / "graph.db")
    monkeypatch.setattr(backend_main, "span_graph_store", store)
    client = TestClient(backend_main.app)

    _ = _seed_one_citing_span(client=client)

    with_claims = client.get(
        "/nav/graph",
        params={
            "reviewer_uid": "alice",
            "focus_type": "work",
            "focus_id": "doc-1",
            "show_claimspans": "true",
        },
    )
    assert with_claims.status_code == 200
    elements = (with_claims.json() or {}).get("elements") or []
    assert any(
        isinstance(el, dict)
        and isinstance(el.get("data"), dict)
        and (el.get("data") or {}).get("selectable_type") == "claimspan"
        for el in elements
    )

    without_claims = client.get(
        "/nav/graph",
        params={
            "reviewer_uid": "alice",
            "focus_type": "work",
            "focus_id": "doc-1",
            "show_claimspans": "false",
        },
    )
    assert without_claims.status_code == 200
    elements2 = (without_claims.json() or {}).get("elements") or []
    assert not any(
        isinstance(el, dict)
        and isinstance(el.get("data"), dict)
        and (el.get("data") or {}).get("selectable_type") == "claimspan"
        for el in elements2
    )
