from __future__ import annotations

from fastapi.testclient import TestClient

import backend.main as backend_main


def test_project_and_graph_resolve_require_strict_scope_headers() -> None:
    client = TestClient(backend_main.app)

    missing_project = client.get("/project", headers={"X-User-Id": "user-a"})
    assert missing_project.status_code == 422

    missing_user = client.put("/project", json={"name": "A"}, headers={"X-Project-Id": "proj-a"})
    assert missing_user.status_code == 422

    missing_mutation_user = client.post(
        "/graph/resolve-references",
        json={"citing_doc_id": "doc-1", "reference_ids": ["r1"]},
        headers={"X-Project-Id": "proj-a"},
    )
    assert missing_mutation_user.status_code == 422


def test_ingest_control_routes_require_mutation_user_and_project_headers() -> None:
    client = TestClient(backend_main.app)

    for path in [
        "/ingest/doc-1/extract",
        "/ingest/doc-1/fallback-extract",
        "/ingest/doc-1/extract/cancel",
        "/ingest/doc-1/resolve",
        "/ingest/doc-1/resolution/ref-1/select",
    ]:
        missing_project = client.post(path, json={"selected_source": "crossref"} if path.endswith("/select") else None)
        assert missing_project.status_code == 422

        missing_user = client.post(
            path,
            json={"selected_source": "crossref"} if path.endswith("/select") else None,
            headers={"X-Project-Id": "proj-a"},
        )
        assert missing_user.status_code == 422


def test_opinion_reviewer_scoped_reads_require_reviewer_identity_header() -> None:
    client = TestClient(backend_main.app)

    missing_reviewer = client.get(
        "/opinions/events",
        params={"reviewer_uid": "reviewer-a"},
        headers={"X-Project-Id": "proj-a"},
    )
    assert missing_reviewer.status_code == 422

    mismatch = client.get(
        "/opinions/follow/by-doc",
        params={"doc_id": "doc-1", "reviewer_uid": "reviewer-b"},
        headers={"X-Project-Id": "proj-a", "X-Reviewer-Uid": "reviewer-a"},
    )
    assert mismatch.status_code == 403
    assert mismatch.json()["detail"] == "X-Reviewer-Uid must match reviewer_uid parameter"
