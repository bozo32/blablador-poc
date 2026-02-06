from __future__ import annotations

from fastapi.testclient import TestClient

import backend.main as backend_main
from backend.graph_store import GraphStore


def _seed_two_claims(store: GraphStore) -> tuple[str, str]:
    store.index_ingest_upload(
        {
            "id": "doc-1",
            "sha256": "a" * 64,
            "filename": "doc-1.pdf",
        }
    )
    store.index_confirmed_claims(
        {
            "document_id": "doc-1",
            "sentence_id": "s1",
            "sentence_text": "Seed",
            "citation_index": 0,
            "target_id": "t1",
            "reviewer_uid": "default",
            "confirmed_claims": [
                {"claim_index": 0, "parsed_text": "Alpha claim."},
                {"claim_index": 1, "parsed_text": "Beta claim."},
            ],
        }
    )
    return (
        "claim:doc-1:s1:0",
        "claim:doc-1:s1:1",
    )


def test_claim_subgraph_includes_manual_claim_link(tmp_path, monkeypatch):
    store = GraphStore(tmp_path / "graph.db")
    monkeypatch.setattr(backend_main, "graph_store", store)
    client = TestClient(backend_main.app)

    a, b = _seed_two_claims(store)

    created = client.post(
        "/graph/claim-link",
        params={"reviewer_uid": "alice"},
        json={"source_claim_id": a, "target_claim_id": b},
    )
    assert created.status_code == 200
    edge_id = created.json()["edge"]["edge_id"]

    sub = client.get(
        "/graph/claim-subgraph",
        params={
            "center_claim_id": a,
            "hops": 1,
            "edge_cap": 10,
            "min_votes": 0,
            "sources": "manual",
        },
    )
    assert sub.status_code == 200
    payload = sub.json()
    assert payload["center_claim_id"] == a
    assert {n["id"] for n in payload["nodes"]} >= {a, b}
    assert any(e["edge_id"] == edge_id for e in payload["edges"])

    edge = next(e for e in payload["edges"] if e["edge_id"] == edge_id)
    assert edge["properties"]["source"] == "manual"
    assert edge["properties"]["creator_uid"] == "alice"


def test_edge_votes_and_aggregates(tmp_path, monkeypatch):
    store = GraphStore(tmp_path / "graph.db")
    monkeypatch.setattr(backend_main, "graph_store", store)
    client = TestClient(backend_main.app)

    a, b = _seed_two_claims(store)
    created = client.post(
        "/graph/claim-link",
        params={"reviewer_uid": "alice"},
        json={"source_claim_id": a, "target_claim_id": b},
    )
    edge_id = created.json()["edge"]["edge_id"]

    v1 = client.put(
        f"/graph/edge/{edge_id}/vote",
        params={"reviewer_uid": "alice"},
        json={"verdict": "support"},
    )
    assert v1.status_code == 200
    assert v1.json()["aggregates"]["n_support"] == 1
    assert v1.json()["aggregates"]["n_total"] == 1

    v2 = client.put(
        f"/graph/edge/{edge_id}/vote",
        params={"reviewer_uid": "bob"},
        json={"verdict": "contradict"},
    )
    assert v2.status_code == 200
    assert v2.json()["aggregates"]["n_support"] == 1
    assert v2.json()["aggregates"]["n_contradict"] == 1
    assert v2.json()["aggregates"]["n_total"] == 2

    listed = client.get(f"/graph/edge/{edge_id}/votes")
    assert listed.status_code == 200
    votes = listed.json()["votes"]
    assert {v["reviewer_uid"] for v in votes} == {"alice", "bob"}
    by_user = {v["reviewer_uid"]: v for v in votes}
    assert by_user["alice"]["verdict"] == "support"
    assert by_user["bob"]["verdict"] == "contradict"


def test_manual_edge_delete_is_creator_scoped(tmp_path, monkeypatch):
    store = GraphStore(tmp_path / "graph.db")
    monkeypatch.setattr(backend_main, "graph_store", store)
    client = TestClient(backend_main.app)

    a, b = _seed_two_claims(store)
    created = client.post(
        "/graph/claim-link",
        params={"reviewer_uid": "alice"},
        json={"source_claim_id": a, "target_claim_id": b},
    )
    edge_id = created.json()["edge"]["edge_id"]

    denied = client.delete(
        f"/graph/claim-link/{edge_id}", params={"reviewer_uid": "bob"}
    )
    assert denied.status_code == 403

    ok = client.delete(f"/graph/claim-link/{edge_id}", params={"reviewer_uid": "alice"})
    assert ok.status_code == 200
    assert ok.json()["ok"] is True

    sub = client.get(
        "/graph/claim-subgraph",
        params={"center_claim_id": a, "hops": 1, "sources": "manual"},
    )
    assert sub.status_code == 200
    assert all(e["edge_id"] != edge_id for e in sub.json()["edges"])
