from __future__ import annotations

import json

from fastapi.testclient import TestClient

import backend.main as backend_main
from backend.object_store import s3 as object_store_s3
from backend.spine import artifacts as spine_artifacts
from backend.spine import attempts as spine_attempts
from backend.spine import documents as spine_documents
from backend.spine import works as spine_works


def _seed_spine_doc_with_extraction(*, doc_id: str) -> None:
    project_id = "default"
    user_id = "local"

    sha256 = "0" * 64
    pdf_object_key = f"pdf/{doc_id}/{sha256}.pdf"
    spine_works.upsert_work_from_pdf(
        work_id=doc_id,
        project_id=project_id,
        created_by_user_id=user_id,
        filename="doc.pdf",
        sha256=sha256,
        size_bytes=1,
        pdf_object_key=pdf_object_key,
    )
    spine_documents.get_or_create_document(doc_id, created_by_user_id=user_id)
    spine_documents.upsert_document_version(
        doc_id,
        document_id=doc_id,
        sha256=sha256,
        size_bytes=1,
        pdf_object_key=pdf_object_key,
        filename="doc.pdf",
        created_by_user_id=user_id,
    )
    spine_documents.ensure_project_document(
        project_id,
        document_id=doc_id,
        added_by_user_id=user_id,
    )

    attempt_id, _ = spine_attempts.create_or_get_attempt(
        project_id,
        user_id,
        doc_id,
        "primary",
        settings_json={},
    )
    spine_attempts.mark_attempt_running(attempt_id)
    spine_attempts.mark_attempt_succeeded(attempt_id)

    extraction = {
        "metadata": {"title": "Test"},
        "citations": [
            {
                "target_id": "ref1",
                "callout": "[1]",
                "sentence": "A claim with [1]",
                "sentence_id": "s1",
            }
        ],
        "references": [{"id": "ref1", "raw_reference": "Ref 1"}],
        "extraction_version": 1,
    }
    key = f"tests/{attempt_id}/extraction.json"
    blob = json.dumps(extraction, ensure_ascii=True).encode("utf-8")
    object_store_s3.put_bytes(key, blob, content_type="application/json")
    spine_artifacts.create_artifact(
        project_id,
        user_id,
        attempt_id,
        "extraction.json",
        key,
        bytes=len(blob),
        content_type="application/json",
    )


def test_workflow_status_api_includes_stage_state_and_cancel() -> None:
    client = TestClient(backend_main.app)

    doc_id = "doc-workflow-test"
    _seed_spine_doc_with_extraction(doc_id=doc_id)

    claim_id = f"cite:{doc_id}:0:default:1a"
    resp = client.post(
        f"/workflow/claimspans/{claim_id}/runs",
        json={"reviewer_uid": "default", "citing_doc_id": doc_id},
    )
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    status1 = client.get(f"/workflow/runs/{run_id}/status")
    assert status1.status_code == 200
    payload = status1.json()
    assert "run" in payload
    assert "targets" in payload
    assert "queue" in payload
    targets = payload["targets"]
    assert isinstance(targets, list)
    assert targets

    first = targets[0]
    assert "stage_state_json" in first
    ss = first["stage_state_json"]
    assert isinstance(ss, dict)
    assert isinstance(ss.get("stages"), dict)
    for stage_name, entry in ss.get("stages", {}).items():
        assert isinstance(stage_name, str)
        assert isinstance(entry, dict)
        assert "state" in entry
        assert "updated_at" in entry

    target_id = first["target_id"]
    cancel = client.post(f"/workflow/runs/{run_id}/targets/{target_id}/cancel")
    assert cancel.status_code == 200
    assert cancel.json().get("state") == "cancelled"

    status2 = client.get(f"/workflow/runs/{run_id}/status")
    assert status2.status_code == 200
    targets2 = status2.json()["targets"]
    assert [t.get("target_id") for t in targets2] == [
        t.get("target_id") for t in targets
    ]
    assert any(t.get("state") == "cancelled" for t in targets2)
