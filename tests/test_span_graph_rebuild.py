from __future__ import annotations

from fastapi.testclient import TestClient

import backend.main as backend_main
from backend.span_graph_store import SpanGraphStore


def test_span_claimspan_assertion_roundtrip(tmp_path, monkeypatch):
    store = SpanGraphStore(tmp_path / "graph.db")
    monkeypatch.setattr(backend_main, "span_graph_store", store)
    client = TestClient(backend_main.app)

    created = client.post(
        "/spans/upsert",
        json={
            "kind": "citation_window",
            "selector": {"exact": "See also", "prefix": None, "suffix": None},
            "window_fingerprint": "abc",
            "ingest_id": "doc-1",
        },
    )
    assert created.status_code == 200
    span_id = created.json()["span"]["span_id"]
    assert span_id.startswith("span:")

    cited_work_id = "doi:10.1234/example"
    cites = client.post(
        f"/spans/{span_id}/cites",
        json={
            "cites": [
                {
                    "cited_work_id": cited_work_id,
                    "reference_id": "b4",
                    "citation_index": 9,
                }
            ]
        },
    )
    assert cites.status_code == 200
    assert cites.json()["inserted"] >= 1

    role = client.put(
        f"/spans/{span_id}/cites/{cited_work_id}/role",
        json={"reviewer_uid": "alice", "role": "evidentiary"},
    )
    assert role.status_code == 200
    assert role.json()["ok"] is True

    claim_spans = client.post(
        f"/spans/{span_id}/claim-spans",
        json={"claim_spans": [{"order_index": 1}, {"order_index": 2}]},
    )
    assert claim_spans.status_code == 200
    rows = claim_spans.json()["claim_spans"]
    assert len(rows) == 2
    claim_span_id = rows[0]["claim_span_id"]
    assert claim_span_id.startswith("claimspan:")

    assertion = client.post(
        "/assertions",
        json={
            "reviewer_uid": "alice",
            "verdict": "support",
            "claim_span_id": claim_span_id,
            "evidence_work_id": cited_work_id,
            "comment": "Looks consistent.",
        },
    )
    assert assertion.status_code == 200
    payload = assertion.json()["assertion"]
    assert payload["reviewer_uid"] == "alice"
    assert payload["verdict"] == "support"
    assert payload["claim_span_id"] == claim_span_id
    assert payload["evidence_work_id"] == cited_work_id

    listed = client.get(f"/claim-spans/{claim_span_id}/assertions")
    assert listed.status_code == 200
    assert listed.json()["claim_span_id"] == claim_span_id
    assert len(listed.json()["assertions"]) == 1
