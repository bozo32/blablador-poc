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

    status = client.get(
        f"/claim-spans/{claim_span_id}/status",
        params={"reviewer_uid": "alice"},
    )
    assert status.status_code == 200
    assert status.json()["status"] == "supported"


def test_claim_confirm_indexes_span_graph(tmp_path, monkeypatch):
    span_store = SpanGraphStore(tmp_path / "graph.db")
    monkeypatch.setattr(backend_main, "span_graph_store", span_store)

    class _NoopClaimStore:
        def persist_confirmed_claims(self, payload):
            return len(payload.confirmed_claims or [])

    class _NoopGraphStore:
        def index_confirmed_claims(self, payload):
            return None

    monkeypatch.setattr(backend_main, "claim_store", _NoopClaimStore())
    monkeypatch.setattr(backend_main, "graph_store", _NoopGraphStore())

    client = TestClient(backend_main.app)
    resp = client.post(
        "/claims/confirm",
        json={
            "document_id": "doc-1",
            "sentence_id": "s1",
            "sentence_text": (
                "Notably, formal external peer review is a modern invention."
            ),
            "citation_index": 9,
            "target_id": "b4",
            "segmentation_model": "local",
            "reviewer_uid": "alice",
            "cited_work_id": "https://doi.org/10.1234/example",
            "citation_anchor": {
                "version": 1,
                "window_fingerprint": "fp-123",
                "anchor_quote": {
                    "exact": "see also",
                    "prefix": "only since",
                    "suffix": None,
                },
                "cited_work_id": "https://doi.org/10.1234/example",
            },
            "confirmed_claims": [
                {"claim_index": 1, "parsed_text": "Claim one."},
                {"claim_index": 2, "parsed_text": "Claim two."},
            ],
        },
    )
    assert resp.status_code == 200

    span = span_store.find_citation_span(
        ingest_id="doc-1", citation_index=9, target_id="b4"
    )
    assert span is not None
    assert span.get("window_fingerprint") == "fp-123"
    cs1 = span_store.get_claim_span(span_id=span["span_id"], order_index=1)
    cs2 = span_store.get_claim_span(span_id=span["span_id"], order_index=2)
    assert cs1 is not None
    assert cs2 is not None

    mapped = span_store.get_citation_span_id(
        ingest_id="doc-1",
        citation_index=9,
        target_id="b4",
    )
    assert mapped == span["span_id"]


def test_claim_confirm_normalizes_0_based_indexes(tmp_path, monkeypatch):
    span_store = SpanGraphStore(tmp_path / "graph.db")
    monkeypatch.setattr(backend_main, "span_graph_store", span_store)

    class _NoopClaimStore:
        def persist_confirmed_claims(self, payload):
            return len(payload.confirmed_claims or [])

    class _NoopGraphStore:
        def index_confirmed_claims(self, payload):
            return None

    monkeypatch.setattr(backend_main, "claim_store", _NoopClaimStore())
    monkeypatch.setattr(backend_main, "graph_store", _NoopGraphStore())

    client = TestClient(backend_main.app)
    resp = client.post(
        "/claims/confirm",
        json={
            "document_id": "doc-1",
            "sentence_id": "s1",
            "sentence_text": "Seed sentence.",
            "citation_index": 0,
            "target_id": "t1",
            "segmentation_model": "local",
            "reviewer_uid": "alice",
            "confirmed_claims": [
                {"claim_index": 0, "parsed_text": "Zero."},
                {"claim_index": 1, "parsed_text": "One."},
            ],
        },
    )
    assert resp.status_code == 200
    span = span_store.find_citation_span(
        ingest_id="doc-1", citation_index=0, target_id="t1"
    )
    assert span is not None
    # Shifted to 1-based.
    assert span_store.get_claim_span(span_id=span["span_id"], order_index=1) is not None
    assert span_store.get_claim_span(span_id=span["span_id"], order_index=2) is not None


def test_evidence_selection_mirrors_to_assertion(tmp_path, monkeypatch):
    span_store = SpanGraphStore(tmp_path / "graph.db")
    monkeypatch.setattr(backend_main, "span_graph_store", span_store)

    class _NoopClaimStore:
        def persist_confirmed_claims(self, payload):
            return len(payload.confirmed_claims or [])

    class _NoopGraphStore:
        def index_confirmed_claims(self, payload):
            return None

    monkeypatch.setattr(backend_main, "claim_store", _NoopClaimStore())
    monkeypatch.setattr(backend_main, "graph_store", _NoopGraphStore())

    # Keep evidence selection writes under tmp.
    from backend.evidence_selection_store import EvidenceSelectionStore

    monkeypatch.setattr(
        EvidenceSelectionStore,
        "root_dir",
        property(lambda self: tmp_path / "evidence_selections"),
    )

    # Seed span graph via /claims/confirm.
    client = TestClient(backend_main.app)
    resp = client.post(
        "/claims/confirm",
        json={
            "document_id": "doc-1",
            "sentence_id": "s1",
            "sentence_text": "Example sentence with a citation.",
            "citation_index": 9,
            "target_id": "b4",
            "segmentation_model": "local",
            "reviewer_uid": "alice",
            "citation_anchor": {
                "version": 1,
                "window_fingerprint": "fp-1",
                "anchor_quote": {"exact": "x", "prefix": None, "suffix": None},
            },
            "confirmed_claims": [
                {"claim_index": 1, "parsed_text": "Claim one."},
            ],
        },
    )
    assert resp.status_code == 200

    # Patch attachment lookup so /evidence/selection can resolve target_id.
    monkeypatch.setattr(
        backend_main.attachment_store,
        "get_attachment",
        lambda attachment_id, public=False: {
            "id": attachment_id,
            "doc_id": "doc-1",
            "target_id": "b4",
            "source_ingest_id": "cited-1",
        },
    )

    claim_id = "cite:doc-1:9:alice:10a"
    sel = client.put(
        f"/claims/{claim_id}/evidence/selection",
        json={
            "verdict": "support",
            "primary": {
                "candidate_id": "cand-1",
                "attachment_id": "att-1",
                "span_id": "evspan-1",
            },
            "secondary": [],
            "note": "ok",
        },
    )
    assert sel.status_code == 200

    # Second save should replace (not append) the mirrored selection assertion.
    sel2 = client.put(
        f"/claims/{claim_id}/evidence/selection",
        json={
            "verdict": "support",
            "primary": {
                "candidate_id": "cand-1",
                "attachment_id": "att-1",
                "span_id": "evspan-1",
            },
            "secondary": [],
            "note": "ok",
        },
    )
    assert sel2.status_code == 200

    span = span_store.find_citation_span(
        ingest_id="doc-1", citation_index=9, target_id="b4"
    )
    assert span is not None
    claim_span = span_store.get_claim_span(span_id=span["span_id"], order_index=1)
    assert claim_span is not None

    listed = span_store.list_assertions_for_claim_span(
        claim_span_id=str(claim_span["claim_span_id"]), reviewer_uid="alice"
    )
    assert len(listed) == 1
    assert listed[0]["verdict"] == "support"
    assert listed[0]["evidence_span_id"] == "evspan-1"
    assert listed[0]["evidence_work_id"] == "ingest:cited-1"
    assert span_store.has_checked(
        claim_span_id=str(claim_span["claim_span_id"]), reviewer_uid="alice"
    )


def test_claim_span_status_unknown_vs_not_supported(tmp_path):
    store = SpanGraphStore(tmp_path / "graph.db")
    span = store.upsert_span(
        kind="citation_window",
        selector={"exact": "x", "prefix": None, "suffix": None},
        window_fingerprint="fp",
        ingest_id="doc-1",
    )
    claim_spans = store.upsert_claim_spans(
        span_id=str(span["span_id"]),
        items=[{"order_index": 1, "selector": None}],
    )
    claim_span_id = claim_spans[0]["claim_span_id"]

    s1 = store.claim_span_status(claim_span_id=claim_span_id, reviewer_uid="alice")
    assert s1["status"] == "unknown"
    assert s1["checked"] is False

    store.mark_checked(claim_span_id=claim_span_id, reviewer_uid="alice")
    s2 = store.claim_span_status(claim_span_id=claim_span_id, reviewer_uid="alice")
    assert s2["status"] == "not_supported"
    assert s2["checked"] is True


def test_claim_span_context_endpoint(tmp_path, monkeypatch):
    span_store = SpanGraphStore(tmp_path / "graph.db")
    monkeypatch.setattr(backend_main, "span_graph_store", span_store)

    class _NoopClaimStore:
        def persist_confirmed_claims(self, payload):
            return len(payload.confirmed_claims or [])

    class _NoopGraphStore:
        def index_confirmed_claims(self, payload):
            return None

    monkeypatch.setattr(backend_main, "claim_store", _NoopClaimStore())
    monkeypatch.setattr(backend_main, "graph_store", _NoopGraphStore())

    client = TestClient(backend_main.app)
    resp = client.post(
        "/claims/confirm",
        json={
            "document_id": "doc-1",
            "sentence_id": "s1",
            "sentence_text": "Seed sentence.",
            "citation_index": 9,
            "target_id": "b4",
            "segmentation_model": "local",
            "reviewer_uid": "alice",
            "confirmed_claims": [
                {"claim_index": 1, "parsed_text": "Claim one."},
            ],
        },
    )
    assert resp.status_code == 200

    claim_id = "cite:doc-1:9:alice:10a"
    ctx = client.get(
        f"/claims/{claim_id}/span-context",
        params={"target_id": "b4"},
    )
    assert ctx.status_code == 200
    payload = ctx.json()
    assert payload["claim_id"] == claim_id
    assert payload["span_id"].startswith("span:")
    assert payload["claim_span_id"].startswith("claimspan:")
    assert payload["order_index"] == 1


def test_span_status_endpoint(tmp_path):
    store = SpanGraphStore(tmp_path / "graph.db")
    span = store.upsert_span(
        kind="citation_window",
        selector={"exact": "x", "prefix": None, "suffix": None},
        window_fingerprint="fp",
        ingest_id="doc-1",
    )
    claim_spans = store.upsert_claim_spans(
        span_id=str(span["span_id"]),
        items=[
            {"order_index": 1, "selector": None},
            {"order_index": 2, "selector": None},
        ],
    )
    cs1 = claim_spans[0]["claim_span_id"]
    cs2 = claim_spans[1]["claim_span_id"]
    store.mark_checked(claim_span_id=cs1, reviewer_uid="alice")
    store.create_assertion(
        payload={
            "reviewer_uid": "alice",
            "verdict": "support",
            "claim_span_id": cs2,
            "evidence_work_id": "ref:x",
        }
    )
    # One checked/no-evidence + one supported => not_supported dominates.
    status = store.span_status(span_id=str(span["span_id"]), reviewer_uid="alice")
    assert status["status"] == "not_supported"
    assert status["n_claim_spans"] == 2
    assert status["n_supported"] == 1
    assert status["n_not_supported"] == 1


def test_span_bundle_endpoint(tmp_path, monkeypatch):
    store = SpanGraphStore(tmp_path / "graph.db")
    monkeypatch.setattr(backend_main, "span_graph_store", store)
    client = TestClient(backend_main.app)

    created = client.post(
        "/spans/upsert",
        json={
            "kind": "citation_window",
            "selector": {"exact": "x", "prefix": None, "suffix": None},
            "window_fingerprint": "fp",
            "ingest_id": "doc-1",
        },
    )
    assert created.status_code == 200
    span_id = created.json()["span"]["span_id"]

    client.post(
        f"/spans/{span_id}/cites",
        json={
            "cites": [
                {
                    "cited_work_id": "ref:doc-1:b4",
                    "reference_id": "b4",
                    "citation_index": 9,
                }
            ]
        },
    )
    client.put(
        f"/spans/{span_id}/cites/ref:doc-1:b4/role",
        json={"reviewer_uid": "alice", "role": "evidentiary"},
    )
    client.post(
        f"/spans/{span_id}/claim-spans",
        json={"claim_spans": [{"order_index": 1}]},
    )

    bundle = client.get(
        f"/spans/{span_id}/bundle",
        params={"reviewer_uid": "alice"},
    )
    assert bundle.status_code == 200
    payload = bundle.json()
    assert payload["span"]["span_id"] == span_id
    assert payload["reviewer_uid"] == "alice"
    assert len(payload["cites"]) == 1
    assert payload["cites"][0]["role"] == "evidentiary"
    assert len(payload["claim_spans"]) == 1


def test_compact_assertions_removes_duplicates(tmp_path):
    store = SpanGraphStore(tmp_path / "graph.db")
    span = store.upsert_span(
        kind="citation_window",
        selector={"exact": "x", "prefix": None, "suffix": None},
        window_fingerprint="fp",
        ingest_id="doc-1",
    )
    cs = store.upsert_claim_spans(
        span_id=str(span["span_id"]),
        items=[{"order_index": 1, "selector": None}],
    )[0]["claim_span_id"]

    store.create_assertion(
        payload={
            "reviewer_uid": "alice",
            "verdict": "support",
            "claim_span_id": cs,
            "evidence_work_id": "ref:x",
        }
    )
    store.create_assertion(
        payload={
            "reviewer_uid": "alice",
            "verdict": "support",
            "claim_span_id": cs,
            "evidence_work_id": "ref:x",
        }
    )
    assert len(store.list_assertions_for_claim_span(claim_span_id=cs)) == 2

    result = store.compact_assertions(dry_run=False, aggressive=False)
    assert result["after"] == result["before"] - 1
    assert len(store.list_assertions_for_claim_span(claim_span_id=cs)) == 1
