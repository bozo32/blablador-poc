"""Postgres helpers for stored artifacts (V2 ingestion spine)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from backend.db import connect


def create_artifact(
    attempt_id: str,
    artifact_type: str,
    object_key: str,
    bytes: Optional[int],
    content_type: Optional[str],
) -> str:
    aid = str(attempt_id or "").strip()
    at = str(artifact_type or "").strip()
    key = str(object_key or "").lstrip("/")
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
                INSERT INTO artifacts (
                  artifact_id,
                  attempt_id,
                  artifact_type,
                  object_key,
                  bytes,
                  content_type
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (artifact_id, aid, at, key, size, ctype),
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
                SELECT artifact_id, attempt_id, artifact_type, object_key,
                       bytes, content_type, created_at
                  FROM artifacts
                 WHERE attempt_id = %s
                 ORDER BY created_at ASC
                """,
                (aid,),
            )
            rows = cur.fetchall() or []
            return [dict(zip(cols, row)) for row in rows]
