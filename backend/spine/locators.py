"""Stable locator primitives (project-scoped).

Locators are durable addressing objects that can later be referenced by:
- evidence spans
- entailment edges
- annotations/comments

They intentionally store opaque JSON payloads so multiple locator strategies can
co-exist (page+bbox, quote anchors, char offsets, etc.).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from backend.db import connect


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


_LOCATOR_COLUMNS = (
    "locator_id",
    "project_id",
    "created_by_user_id",
    "document_version_id",
    "type",
    "payload_json",
    "created_at",
)


def create_locator(
    *,
    project_id: str,
    created_by_user_id: str,
    document_version_id: str,
    type: str,
    payload_json: dict,
) -> Dict[str, Any]:
    pid = str(project_id or "").strip()
    uid = str(created_by_user_id or "").strip()
    dvid = str(document_version_id or "").strip()
    lt = str(type or "").strip()
    if not pid:
        raise ValueError("project_id is required")
    if not uid:
        raise ValueError("created_by_user_id is required")
    if not dvid:
        raise ValueError("document_version_id is required")
    if not lt:
        raise ValueError("type is required")

    locator_id = str(uuid4())
    blob = _json_dumps(payload_json or {})

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO locators (
                  locator_id,
                  project_id,
                  created_by_user_id,
                  document_version_id,
                  type,
                  payload_json
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                RETURNING locator_id, project_id, created_by_user_id,
                          document_version_id, type, payload_json, created_at
                """,
                (locator_id, pid, uid, dvid, lt, blob),
            )
            row = cur.fetchone()
        conn.commit()

    if row is None:
        raise RuntimeError("locator insert failed")
    payload = row[5] if isinstance(row[5], dict) else json.loads(row[5])
    out: Dict[str, Any] = {
        "locator_id": row[0],
        "project_id": row[1],
        "created_by_user_id": row[2],
        "document_version_id": row[3],
        "type": row[4],
        "payload_json": payload,
        "created_at": row[6],
    }
    return out


def get_locator(*, locator_id: str, project_id: str) -> Optional[Dict[str, Any]]:
    lid = str(locator_id or "").strip()
    pid = str(project_id or "").strip()
    if not lid:
        raise ValueError("locator_id is required")
    if not pid:
        raise ValueError("project_id is required")

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT locator_id, project_id, created_by_user_id, document_version_id,
                       type, payload_json, created_at
                  FROM locators
                 WHERE locator_id = %s
                   AND project_id = %s
                 LIMIT 1
                """,
                (lid, pid),
            )
            row = cur.fetchone()
            if row is None:
                return None
            payload = row[5] if isinstance(row[5], dict) else json.loads(row[5])
            return {
                "locator_id": row[0],
                "project_id": row[1],
                "created_by_user_id": row[2],
                "document_version_id": row[3],
                "type": row[4],
                "payload_json": payload,
                "created_at": row[6],
            }


def list_locators_for_document_version(
    *, project_id: str, document_version_id: str, limit: int = 200
) -> List[Dict[str, Any]]:
    pid = str(project_id or "").strip()
    dvid = str(document_version_id or "").strip()
    n = int(limit)
    if not pid:
        raise ValueError("project_id is required")
    if not dvid:
        raise ValueError("document_version_id is required")
    if n <= 0:
        return []

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT locator_id, project_id, created_by_user_id, document_version_id,
                       type, payload_json, created_at
                  FROM locators
                 WHERE project_id = %s
                   AND document_version_id = %s
                 ORDER BY created_at DESC
                 LIMIT %s
                """,
                (pid, dvid, n),
            )
            rows = cur.fetchall() or []
            out: List[Dict[str, Any]] = []
            for row in rows:
                payload = row[5] if isinstance(row[5], dict) else json.loads(row[5])
                out.append(
                    {
                        "locator_id": row[0],
                        "project_id": row[1],
                        "created_by_user_id": row[2],
                        "document_version_id": row[3],
                        "type": row[4],
                        "payload_json": payload,
                        "created_at": row[6],
                    }
                )
            return out
