"""Spine-backed stage artifact persistence (Postgres pointers + S3 JSON)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

from backend.contracts.upgrade import validate_contract_payload
from backend.contracts.pipeline_v1 import deterministic_json_bytes, stage_object_key
from backend.db import connect
from backend.object_store import s3 as object_store_s3


@dataclass(frozen=True)
class StageArtifactAlreadyExists(Exception):
    run_id: str
    stage: str

    def __str__(self) -> str:  # pragma: no cover
        """Human-friendly message for logging/HTTP mapping."""
        return (
            f"Stage artifact already exists for run_id={self.run_id} stage={self.stage}"
        )


def put_stage_payload(*, payload_dict: dict) -> dict:
    """Validate and persist a stage payload.

    Write-once per (run_id, stage): a second store attempt is rejected without
    overwriting the existing S3 object.
    """
    validated = validate_contract_payload(dict(payload_dict or {}))
    run_id = str(validated.get("run_id") or "").strip()
    stage = str(validated.get("stage") or "").strip()
    if not run_id or not stage:
        raise ValueError("Validated payload missing run_id/stage")

    schema_version = int(validated.get("schema_version") or 0)
    artifact_type = str(validated.get("artifact_type") or "").strip()
    object_key = stage_object_key(run_id, stage)
    content_type = "application/json"
    artifact_id = str(uuid4())

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pipeline_stage_artifacts(
                  artifact_id,
                  run_id,
                  project_id,
                  created_by_user_id,
                  stage,
                  schema_version,
                  artifact_type,
                  object_key,
                  bytes,
                  content_type
                )
                VALUES (
                  %s,
                  %s,
                  'default',
                  'local',
                  %s,
                  %s,
                  %s,
                  %s,
                  NULL,
                  %s
                )
                ON CONFLICT (run_id, stage) DO NOTHING
                RETURNING artifact_id
                """,
                (
                    artifact_id,
                    run_id,
                    stage,
                    schema_version,
                    artifact_type,
                    object_key,
                    content_type,
                ),
            )
            row = cur.fetchone()
        conn.commit()

    if row is None:
        raise StageArtifactAlreadyExists(run_id=run_id, stage=stage)

    payload_bytes = deterministic_json_bytes(validated)
    try:
        object_store_s3.put_bytes(object_key, payload_bytes, content_type=content_type)
    except Exception:
        with connect(autocommit=True) as conn2:
            with conn2.cursor() as cur2:
                cur2.execute(
                    "DELETE FROM pipeline_stage_artifacts WHERE artifact_id=%s",
                    (artifact_id,),
                )
        raise

    with connect(autocommit=True) as conn3:
        with conn3.cursor() as cur3:
            cur3.execute(
                """
                UPDATE pipeline_stage_artifacts
                   SET bytes=%s,
                       content_type=%s
                 WHERE artifact_id=%s
                """,
                (int(len(payload_bytes)), content_type, artifact_id),
            )

    return dict(validated)


def get_stage_payload(run_id: str, stage: str) -> Optional[dict]:
    rid = str(run_id or "").strip()
    st = str(stage or "").strip()
    if not rid:
        raise ValueError("run_id is required")
    if not st:
        raise ValueError("stage is required")

    object_key: str | None = None
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT object_key
                FROM pipeline_stage_artifacts
                WHERE run_id=%s AND stage=%s
                LIMIT 1
                """,
                (rid, st),
            )
            row = cur.fetchone()
            if row is None:
                return None
            object_key = str(row[0] or "").strip() or None

    if not object_key:
        return None

    raw = object_store_s3.get_bytes(object_key)
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Stage artifact JSON must be an object")
    return validate_contract_payload(data)


__all__ = [
    "StageArtifactAlreadyExists",
    "put_stage_payload",
    "get_stage_payload",
]
