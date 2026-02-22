from __future__ import annotations

from fastapi.testclient import TestClient

import backend.main as backend_main


def test_workflow_assessment_finalize_writes_immutable_artifact() -> None:
    client = TestClient(backend_main.app)

    claim_id = "cite:doc-assess:0:default:1a"
    body = {
        "reviewer_uid": "default",
        "citing_doc_id": "doc-assess",
        "assessed_at": "2026-02-22T00:00:00Z",
        "rollup_label": "supports",
        "by_target": {},
        "judgment_snapshot": {"status": "final", "verdict": "support"},
    }
    first = client.post(
        f"/workflow/claimspans/{claim_id}/assessment/finalize",
        json=body,
    )
    assert first.status_code == 200
    run_id = first.json()["run_id"]

    fetched = client.get(f"/pipeline/runs/{run_id}/stages/assessment")
    assert fetched.status_code == 200
    artifact = fetched.json()
    assert artifact["stage"] == "assessment"
    assert artifact["status"] == "complete"
    assert artifact["data"]["judgment_snapshot"]["verdict"] == "support"

    second = client.post(
        f"/workflow/claimspans/{claim_id}/assessment/finalize",
        json={
            **body,
            "judgment_snapshot": {"status": "final", "verdict": "contradict"},
        },
    )
    assert second.status_code == 200
    assert second.json().get("already_stored") is True

    fetched2 = client.get(f"/pipeline/runs/{run_id}/stages/assessment")
    assert fetched2.status_code == 200
    artifact2 = fetched2.json()
    assert artifact2["data"]["judgment_snapshot"]["verdict"] == "support"
