from __future__ import annotations

from fastapi.testclient import TestClient

import backend.main as backend_main


def _base_headers() -> dict[str, str]:
    return {
        "X-Project-Id": "proj-pilot",
        "X-User-Id": "reviewer-1",
        "X-Reviewer-Uid": "reviewer-1",
    }


def _ledger_payload(path: str) -> dict:
    if path.endswith("/assign"):
        return {"assigned": True}
    if path.endswith("/place"):
        return {
            "source_num": 1,
            "target_num": 2,
            "relation": "is cited by",
            "canonical": False,
            "reviewer_uid": "reviewer-1",
        }
    if path.endswith("/place-reference"):
        return {
            "citing_doc_id": "doc-1",
            "reference_id": "ref-1",
            "cited_ingest_id": "doc-2",
            "canonical": False,
            "reviewer_uid": "reviewer-1",
        }
    return {"targets": [2, 3]}


def test_write_pilot_requires_project_and_user_headers() -> None:
    client = TestClient(backend_main.app)

    ledger_paths = [
        ("PATCH", "/ledger/1/outgoing"),
        ("PATCH", "/ledger/1/incoming"),
        ("PATCH", "/ledger/1/assign"),
        ("POST", "/ledger/place"),
        ("POST", "/ledger/place-reference"),
    ]

    for method, path in ledger_paths:
        payload = _ledger_payload(path)

        missing_project = client.request(method, path, json=payload)
        assert missing_project.status_code == 400
        assert missing_project.json()["detail"] == "X-Project-Id header is required"

        missing_user = client.request(
            method,
            path,
            json=payload,
            headers={"X-Project-Id": "proj-pilot"},
        )
        assert missing_user.status_code == 400
        assert missing_user.json()["detail"] == "X-User-Id header is required"


def test_attachment_write_pilot_requires_project_and_user_headers() -> None:
    client = TestClient(backend_main.app)

    # upload
    missing_upload_scope = client.post(
        "/attachments/upload",
        files={"file": ("doc.pdf", b"%PDF-1.4\n", "application/pdf")},
    )
    assert missing_upload_scope.status_code == 400
    assert missing_upload_scope.json()["detail"] == "X-Project-Id header is required"

    missing_upload_user = client.post(
        "/attachments/upload",
        headers={"X-Project-Id": "proj-pilot"},
        files={"file": ("doc.pdf", b"%PDF-1.4\n", "application/pdf")},
    )
    assert missing_upload_user.status_code == 400
    assert missing_upload_user.json()["detail"] == "X-User-Id header is required"

    # mutation endpoints with id path
    for method, path, payload in [
        ("PATCH", "/attachments/a-1", {"archived": True}),
        ("POST", "/attachments/a-1/clone", {"claim_id": "c1", "doc_id": "d1"}),
        ("POST", "/attachments/a-1/promote-ingest", None),
        ("POST", "/attachments/a-1/retry", None),
    ]:
        kwargs = {"json": payload} if payload is not None else {}

        missing_project = client.request(method, path, **kwargs)
        assert missing_project.status_code == 400
        assert missing_project.json()["detail"] == "X-Project-Id header is required"

        missing_user = client.request(
            method,
            path,
            headers={"X-Project-Id": "proj-pilot"},
            **kwargs,
        )
        assert missing_user.status_code == 400
        assert missing_user.json()["detail"] == "X-User-Id header is required"


def test_opinion_event_requires_project_user_and_reviewer_identity() -> None:
    client = TestClient(backend_main.app)
    payload = {
        "kind": "follow",
        "target_key": "citespan:span-1",
        "payload": {"status": "follow"},
    }

    missing_project = client.post("/opinions/events", json=payload)
    assert missing_project.status_code == 400
    assert missing_project.json()["detail"] == "X-Project-Id header is required"

    missing_user = client.post(
        "/opinions/events",
        json=payload,
        headers={"X-Project-Id": "proj-pilot"},
    )
    assert missing_user.status_code == 400
    assert missing_user.json()["detail"] == "X-User-Id header is required"

    missing_reviewer = client.post(
        "/opinions/events",
        json=payload,
        headers={"X-Project-Id": "proj-pilot", "X-User-Id": "reviewer-1"},
    )
    assert missing_reviewer.status_code == 400
    assert missing_reviewer.json()["detail"] == "X-Reviewer-Uid header is required"


def test_opinion_event_reviewer_mismatch_rejected(monkeypatch) -> None:
    class _StubOpinionEvents:
        def append_event(self, **kwargs):  # pragma: no cover - should not be called
            return kwargs

    monkeypatch.setattr(backend_main, "opinion_events", _StubOpinionEvents())
    client = TestClient(backend_main.app)

    payload = {
        "kind": "follow",
        "target_key": "citespan:span-1",
        "payload": {"status": "follow"},
    }
    resp = client.post(
        "/opinions/events?reviewer_uid=reviewer-2",
        json=payload,
        headers=_base_headers(),
    )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "X-Reviewer-Uid must match reviewer_uid parameter"
