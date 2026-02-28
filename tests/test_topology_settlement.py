from __future__ import annotations

from fastapi.testclient import TestClient

import backend.main as backend_main
from backend.graph_store import GraphStore
from backend.span_graph_store import SpanGraphStore

# Test project ID constant
PROJECT_ID = "default"


def _seed_claim(store: GraphStore) -> str:
    store.index_ingest_upload(
        {
            "id": "doc-1",
            "sha256": "b" * 64,
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
            "confirmed_claims": [{"claim_index": 0, "parsed_text": "Alpha."}],
        }
    )
    return "claim:doc-1:s1:0"


def test_topology_edges_can_be_voted_and_settled(tmp_path, monkeypatch):
    graph_store = GraphStore(tmp_path / "graph.db")
    span_store = SpanGraphStore(tmp_path / "graph.db")
    monkeypatch.setattr(backend_main, "graph_store", graph_store)
    monkeypatch.setattr(backend_main, "span_graph_store", span_store)
    client = TestClient(backend_main.app)

    # Create a citation-window span + claimspan.
    created = client.post(
        "/spans/upsert",
        json={
            "kind": "citation_window",
            "selector": {"exact": "x", "prefix": None, "suffix": None},
            "window_fingerprint": "fp",
            "ingest_id": "doc-1",
        },
        headers={"X-Project-Id": PROJECT_ID},
    )
    assert created.status_code == 200
    span_id = created.json()["span"]["span_id"]

    claim_spans = client.post(
        f"/spans/{span_id}/claim-spans",
        json={"claim_spans": [{"order_index": 1}]},
        headers={"X-Project-Id": PROJECT_ID},
    )
    assert claim_spans.status_code == 200
    claim_span_id = claim_spans.json()["claim_spans"][0]["claim_span_id"]

    # Create a ClaimAtom.
    atom = client.post(
        "/claim-atoms",
        json={"reviewer_uid": "alice", "text": "The sky is blue."},
    )
    assert atom.status_code == 200
    atom_id = atom.json()["atom"]["claim_atom_id"]

    # Link ClaimSpan -> ClaimAtom as an auto/topology edge.
    link = client.post(
        f"/claim-spans/{claim_span_id}/atoms",
        json={"reviewer_uid": "alice", "claim_atom_id": atom_id, "source": "auto"},
    )
    assert link.status_code == 200
    edge_id = int(link.json()["edge_id"])

    # Disable it, then vote to re-enable via settlement.
    graph_store.set_edge_enabled(edge_id=edge_id, enabled=False)
    assert graph_store.get_edge(edge_id)["enabled"] is False

    v = client.put(
        f"/graph/edge/{edge_id}/vote",
        params={"reviewer_uid": "alice"},
        json={"verdict": "support"},
    )
    assert v.status_code == 200
    assert v.json()["aggregates"]["n_support"] == 1

    settled = client.post(
        "/topology/settle",
        json={"kind": "CLAIMSPAN_EXPRESSES_ATOM", "dry_run": False},
    )
    assert settled.status_code == 200
    assert graph_store.get_edge(edge_id)["enabled"] is True

    # Create another edge and vote it down.
    atom2 = client.post(
        "/claim-atoms",
        json={"reviewer_uid": "alice", "text": "Grass is red."},
    )
    atom2_id = atom2.json()["atom"]["claim_atom_id"]
    link2 = client.post(
        f"/claim-spans/{claim_span_id}/atoms",
        json={"reviewer_uid": "alice", "claim_atom_id": atom2_id, "source": "auto"},
    )
    edge2_id = int(link2.json()["edge_id"])

    v2 = client.put(
        f"/graph/edge/{edge2_id}/vote",
        params={"reviewer_uid": "bob"},
        json={"verdict": "contradict"},
    )
    assert v2.status_code == 200
    assert v2.json()["aggregates"]["n_contradict"] == 1

    settled2 = client.post(
        "/topology/settle",
        json={"kind": "CLAIMSPAN_EXPRESSES_ATOM", "dry_run": False},
    )
    assert settled2.status_code == 200
    assert graph_store.get_edge(edge2_id)["enabled"] is False

    # Atom -> Claim alignment edge follows same pattern.
    claim_id = _seed_claim(graph_store)
    align = client.post(
        f"/claim-atoms/{atom_id}/align",
        json={"reviewer_uid": "alice", "claim_id": claim_id, "source": "auto"},
    )
    assert align.status_code == 200
    align_edge_id = int(align.json()["edge_id"])

    graph_store.set_edge_enabled(edge_id=align_edge_id, enabled=False)
    client.put(
        f"/graph/edge/{align_edge_id}/vote",
        params={"reviewer_uid": "alice"},
        json={"verdict": "support"},
    )
    client.post(
        "/topology/settle",
        json={"kind": "ATOM_ALIGNS_TO_CLAIM", "dry_run": False},
    )
    assert graph_store.get_edge(align_edge_id)["enabled"] is True
