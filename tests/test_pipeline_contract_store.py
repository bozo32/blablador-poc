from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backend.contracts.pipeline_v1 import artifact_type_for, stage_object_key
from backend.db.pg import connect
from backend.object_store import s3 as object_store_s3
from backend.spine import pipeline_artifacts, pipeline_runs


def _utcnow_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def test_put_and_get_stage_payload_is_write_once() -> None:
    run = pipeline_runs.create_run(
        "work:test",
        caps={"candidates": 200, "rerank": 50, "nli": 12},
        settings_json={},
        input_fingerprint="fp-test",
        note=None,
    )
    run_id = str(run["run_id"])

    payload1 = {
        "schema_version": 1,
        "artifact_type": artifact_type_for("extract", 1),
        "run_id": run_id,
        "stage": "extract",
        "work_id": "work:test",
        "created_at": _utcnow_z(),
        "status": "complete",
        "warnings": [],
        "caps": {"candidates": 200, "rerank": 50, "nli": 12},
        "input_fingerprint": "fp-test",
        "data": {"structured_doc": {"ok": True}, "citation_anchors": []},
    }

    stored = pipeline_artifacts.put_stage_payload(payload_dict=payload1)
    assert stored["run_id"] == run_id
    assert stored["stage"] == "extract"
    assert stored["schema_version"] == 1

    key = stage_object_key(run_id, "extract")
    assert object_store_s3.exists(key)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT artifact_type, object_key
                FROM pipeline_stage_artifacts
                WHERE run_id=%s AND stage=%s
                """,
                (run_id, "extract"),
            )
            row = cur.fetchone()
    assert row is not None
    assert str(row[0]) == artifact_type_for("extract", 1)
    assert str(row[1]) == key

    loaded = pipeline_artifacts.get_stage_payload(run_id, "extract")
    assert loaded is not None
    assert loaded["run_id"] == run_id
    assert loaded["stage"] == "extract"
    assert loaded["work_id"] == "work:test"
    assert loaded["data"]["structured_doc"]["ok"] is True

    first_bytes = object_store_s3.get_bytes(key)
    payload2 = dict(payload1)
    payload2["data"] = {"structured_doc": {"ok": False}, "citation_anchors": []}

    with pytest.raises(pipeline_artifacts.StageArtifactAlreadyExists):
        pipeline_artifacts.put_stage_payload(payload_dict=payload2)

    assert object_store_s3.get_bytes(key) == first_bytes
    loaded2 = pipeline_artifacts.get_stage_payload(run_id, "extract")
    assert loaded2 is not None
    assert loaded2["data"]["structured_doc"]["ok"] is True
