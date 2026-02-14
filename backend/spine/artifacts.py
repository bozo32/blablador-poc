"""Postgres helpers for stored artifacts (V2 ingestion spine)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from backend.db import connect


def create_artifact(
    project_id: str,
    created_by_user_id: str,
    attempt_id: str,
    artifact_type: str,
    object_key: str,
    bytes: Optional[int],
    content_type: Optional[str],
) -> str:
    pid = str(project_id or "").strip()
    uid = str(created_by_user_id or "").strip()
    aid = str(attempt_id or "").strip()
    at = str(artifact_type or "").strip()
    key = str(object_key or "").lstrip("/")
    if not pid:
        raise ValueError("project_id is required")
    if not uid:
        raise ValueError("created_by_user_id is required")
    if not aid:
        raise ValueError("attempt_id is required")
    if not at:
        raise ValueError("artifact_type is required")
    if not key:
        raise ValueError("object_key is required")

    artifact_id = str(uuid4())
    size = int(bytes) if bytes is not None else None
    ctype = str(content_type).strip() if content_type is not None else None

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT artifact_id
                  FROM artifacts
                 WHERE attempt_id = %s
                   AND artifact_type = %s
                   AND object_key = %s
                 ORDER BY created_at DESC
                 LIMIT 1
                """,
                (aid, at, key),
            )
            row = cur.fetchone()
            if row is not None:
                conn.commit()
                return str(row[0])

            cur.execute(
                """
                INSERT INTO artifacts (
                  artifact_id,
                  attempt_id,
                  project_id,
                  created_by_user_id,
                  artifact_type,
                  object_key,
                  bytes,
                  content_type
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (artifact_id, aid, pid, uid, at, key, size, ctype),
            )
        conn.commit()

    return artifact_id


def list_artifacts_for_attempt(attempt_id: str) -> List[Dict[str, Any]]:
    aid = str(attempt_id or "").strip()
    if not aid:
        raise ValueError("attempt_id is required")

    cols = (
        "artifact_id",
        "attempt_id",
        "project_id",
        "created_by_user_id",
        "artifact_type",
        "object_key",
        "bytes",
        "content_type",
        "created_at",
    )
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT artifact_id, attempt_id, project_id, created_by_user_id,
                       artifact_type, object_key,
                       bytes, content_type, created_at
                  FROM artifacts
                 WHERE attempt_id = %s
                 ORDER BY created_at ASC
                """,
                (aid,),
            )
            rows = cur.fetchall() or []
            return [{str(k): v for k, v in zip(cols, row)} for row in rows]
