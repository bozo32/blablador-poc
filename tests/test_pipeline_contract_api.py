from __future__ import annotations

from datetime import datetime, timezone
from fastapi.testclient import TestClient

import backend.main as backend_main
from backend.contracts.pipeline_v1 import stage_object_key
from backend.object_store import s3 as object_store_s3


def _utcnow_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def test_pipeline_contract_endpoints_round_trip_and_wipe() -> None:
    client = TestClient(backend_main.app)

    resp = client.post("/pipeline/runs", json={"work_id": "work:api-test"})
    assert resp.status_code == 200
    run = resp.json()
    run_id = str(run["run_id"])

    put1 = client.put(
        f"/pipeline/runs/{run_id}/stages/extract",
        json={
            "status": "complete",
            "data": {"structured_doc": {"a": 1}, "citation_anchors": []},
            "warnings": [],
        },
    )
    assert put1.status_code == 200
    stored1 = put1.json()
    assert stored1["run_id"] == run_id
    assert stored1["stage"] == "extract"
    assert stored1["schema_version"] == 1
    assert stored1.get("created_at", "").endswith("Z")

    get1 = client.get(f"/pipeline/runs/{run_id}/stages/extract")
    assert get1.status_code == 200
    fetched1 = get1.json()
    assert fetched1["data"]["structured_doc"]["a"] == 1

    put2 = client.put(
        f"/pipeline/runs/{run_id}/stages/extract",
        json={
            "status": "complete",
            "data": {"structured_doc": {"a": 2}, "citation_anchors": []},
            "warnings": ["second"],
        },
    )
    assert put2.status_code == 409

    get2 = client.get(f"/pipeline/runs/{run_id}/stages/extract")
    assert get2.status_code == 200
    fetched2 = get2.json()
    assert fetched2["data"]["structured_doc"]["a"] == 1

    key = stage_object_key(run_id, "extract")
    assert object_store_s3.exists(key)

    wiped = client.post("/dev/wipe", json={"confirm": "WIPE"})
    assert wiped.status_code == 200

    after = client.get(f"/pipeline/runs/{run_id}/stages/extract")
    assert after.status_code == 404
    assert object_store_s3.exists(key) is False
