from __future__ import annotations

import hashlib
import json
from uuid import uuid4

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
    _seed_spine_doc_with_extraction_for_project(
        doc_id=doc_id,
        project_id=project_id,
        user_id=user_id,
    )


def _seed_spine_doc_with_extraction_for_project(
    *,
    doc_id: str,
    project_id: str,
    user_id: str = "local",
) -> None:
    sha256 = hashlib.sha256(str(doc_id).encode("utf-8")).hexdigest()
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

    doc_id = f"doc-workflow-test-{uuid4().hex[:8]}"
    project_id = "proj-workflow-status"
    _seed_spine_doc_with_extraction_for_project(doc_id=doc_id, project_id=project_id)

    claim_id = f"cite:{doc_id}:0:default:1a"
    resp = client.post(
        f"/workflow/claimspans/{claim_id}/runs",
        json={"reviewer_uid": "default", "citing_doc_id": doc_id},
        headers={"X-Project-Id": project_id},
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
    cancel = client.post(
        f"/workflow/runs/{run_id}/targets/{target_id}/cancel",
        headers={"X-Project-Id": project_id},
    )
    assert cancel.status_code == 200
    assert cancel.json().get("state") == "cancelled"

    status2 = client.get(f"/workflow/runs/{run_id}/status")
    assert status2.status_code == 200
    targets2 = status2.json()["targets"]
    assert [t.get("target_id") for t in targets2] == [
        t.get("target_id") for t in targets
    ]
    assert any(t.get("state") == "cancelled" for t in targets2)


def test_workflow_latest_run_isolated_by_project() -> None:
    client = TestClient(backend_main.app)

    doc_id = f"doc-workflow-scope-{uuid4().hex[:8]}"
    reviewer_uid = "reviewer-a"
    claim_id = f"cite:{doc_id}:0:{reviewer_uid}:1a"
    project_a = "proj-alpha"
    project_b = "proj-beta"

    _seed_spine_doc_with_extraction_for_project(doc_id=doc_id, project_id=project_a)
    _seed_spine_doc_with_extraction_for_project(doc_id=doc_id, project_id=project_b)

    first = client.post(
        f"/workflow/claimspans/{claim_id}/runs",
        json={"reviewer_uid": reviewer_uid, "citing_doc_id": doc_id},
        headers={"X-Project-Id": project_a},
    )
    assert first.status_code == 200
    run_a = str(first.json().get("run_id") or "")
    assert run_a

    second = client.post(
        f"/workflow/claimspans/{claim_id}/runs",
        json={"reviewer_uid": reviewer_uid, "citing_doc_id": doc_id},
        headers={"X-Project-Id": project_b},
    )
    assert second.status_code == 200
    run_b = str(second.json().get("run_id") or "")
    assert run_b
    assert run_b != run_a

    latest_a = client.get(
        f"/workflow/claimspans/{claim_id}/runs/latest",
        params={"reviewer_uid": reviewer_uid},
        headers={"X-Project-Id": project_a},
    )
    assert latest_a.status_code == 200
    assert latest_a.json().get("run_id") == run_a

    latest_b = client.get(
        f"/workflow/claimspans/{claim_id}/runs/latest",
        params={"reviewer_uid": reviewer_uid},
        headers={"X-Project-Id": project_b},
    )
    assert latest_b.status_code == 200
    assert latest_b.json().get("run_id") == run_b


def test_workflow_public_trace_reads_and_write_scope_guards() -> None:
    client = TestClient(backend_main.app)

    doc_id = f"doc-workflow-public-trace-{uuid4().hex[:8]}"
    project_id = "proj-public"
    _seed_spine_doc_with_extraction_for_project(doc_id=doc_id, project_id=project_id)
    claim_id = f"cite:{doc_id}:0:default:1a"

    started = client.post(
        f"/workflow/claimspans/{claim_id}/runs",
        json={"reviewer_uid": "default", "citing_doc_id": doc_id},
        headers={"X-Project-Id": project_id},
    )
    assert started.status_code == 200
    run_id = str(started.json().get("run_id") or "")
    assert run_id

    status_public = client.get(f"/workflow/runs/{run_id}/status")
    assert status_public.status_code == 200

    status_mismatch = client.get(
        f"/workflow/runs/{run_id}/status",
        headers={"X-Project-Id": "proj-other"},
    )
    assert status_mismatch.status_code == 409

    events_mismatch = client.get(
        f"/workflow/runs/{run_id}/events",
        headers={"X-Project-Id": "proj-other"},
    )
    assert events_mismatch.status_code == 409

    resume_missing_header = client.post(f"/workflow/runs/{run_id}/resume")
    assert resume_missing_header.status_code == 400

    resume_mismatch = client.post(
        f"/workflow/runs/{run_id}/resume",
        headers={"X-Project-Id": "proj-other"},
    )
    assert resume_mismatch.status_code == 409

    resume_ok = client.post(
        f"/workflow/runs/{run_id}/resume",
        headers={"X-Project-Id": project_id},
    )
    assert resume_ok.status_code == 200

    run_payload = status_public.json()
    first_target = ((run_payload.get("targets") or [])[0] or {}).get("target_id")
    assert first_target

    cancel_missing_header = client.post(
        f"/workflow/runs/{run_id}/targets/{first_target}/cancel"
    )
    assert cancel_missing_header.status_code == 400

    cancel_mismatch = client.post(
        f"/workflow/runs/{run_id}/targets/{first_target}/cancel",
        headers={"X-Project-Id": "proj-other"},
    )
    assert cancel_mismatch.status_code == 409

    cancel_ok = client.post(
        f"/workflow/runs/{run_id}/targets/{first_target}/cancel",
        headers={"X-Project-Id": project_id},
    )
    assert cancel_ok.status_code == 200
