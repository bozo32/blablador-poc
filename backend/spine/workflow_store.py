"""Postgres-backed versioned workflow definitions.

This stores workflow graphs for future Orange/Langflow-like UI.
Execution is out of scope here; this is persistence + versioning.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from backend.db import connect


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def list_workflows(*, project_id: str) -> List[Dict[str, Any]]:
    pid = str(project_id or "").strip()
    if not pid:
        raise ValueError("project_id is required")

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT workflow_id, project_id, name, created_at, created_by_user_id
                  FROM workflow_definitions
                 WHERE project_id = %s
                 ORDER BY created_at DESC
                """,
                (pid,),
            )
            rows = cur.fetchall() or []
            return [
                {
                    "workflow_id": r[0],
                    "project_id": r[1],
                    "name": r[2],
                    "created_at": r[3],
                    "created_by_user_id": r[4],
                }
                for r in rows
            ]


def create_workflow_definition(
    *, project_id: str, name: str, created_by_user_id: str
) -> Dict[str, Any]:
    pid = str(project_id or "").strip()
    nm = str(name or "").strip()
    uid = str(created_by_user_id or "").strip()
    if not pid:
        raise ValueError("project_id is required")
    if not nm:
        raise ValueError("name is required")
    if not uid:
        raise ValueError("created_by_user_id is required")

    workflow_id = str(uuid4())
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO workflow_definitions (
                  workflow_id,
                  project_id,
                  name,
                  created_by_user_id
                )
                VALUES (%s, %s, %s, %s)
                RETURNING workflow_id, project_id, name, created_at, created_by_user_id
                """,
                (workflow_id, pid, nm, uid),
            )
            row = cur.fetchone()
        conn.commit()

    if row is None:
        raise RuntimeError("workflow definition insert failed")
    return {
        "workflow_id": row[0],
        "project_id": row[1],
        "name": row[2],
        "created_at": row[3],
        "created_by_user_id": row[4],
    }


def append_workflow_version(
    *,
    workflow_id: str,
    project_id: str,
    created_by_user_id: str,
    graph_json: dict,
) -> Dict[str, Any]:
    wid = str(workflow_id or "").strip()
    pid = str(project_id or "").strip()
    uid = str(created_by_user_id or "").strip()
    if not wid:
        raise ValueError("workflow_id is required")
    if not pid:
        raise ValueError("project_id is required")
    if not uid:
        raise ValueError("created_by_user_id is required")

    version_id = str(uuid4())
    blob = _json_dumps(graph_json or {})

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"workflow_versions:{wid}",),
            )
            cur.execute(
                """
                SELECT COALESCE(MAX(version), 0)
                  FROM workflow_versions
                 WHERE workflow_id = %s
                """,
                (wid,),
            )
            next_v = int((cur.fetchone() or [0])[0]) + 1
            cur.execute(
                """
                INSERT INTO workflow_versions (
                  workflow_version_id, workflow_id, project_id, version,
                  graph_json, created_by_user_id
                )
                VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                RETURNING workflow_version_id, workflow_id, project_id, version,
                          graph_json, created_at, created_by_user_id, locked
                """,
                (version_id, wid, pid, next_v, blob, uid),
            )
            row = cur.fetchone()
        conn.commit()

    if row is None:
        raise RuntimeError("workflow version insert failed")
    graph = row[4] if isinstance(row[4], dict) else json.loads(row[4])
    return {
        "workflow_version_id": row[0],
        "workflow_id": row[1],
        "project_id": row[2],
        "version": row[3],
        "graph_json": graph,
        "created_at": row[5],
        "created_by_user_id": row[6],
        "locked": bool(row[7]),
    }


def get_latest_workflow_version(
    *, workflow_id: str, project_id: str
) -> Optional[Dict[str, Any]]:
    wid = str(workflow_id or "").strip()
    pid = str(project_id or "").strip()
    if not wid:
        raise ValueError("workflow_id is required")
    if not pid:
        raise ValueError("project_id is required")

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT workflow_version_id, workflow_id, project_id, version,
                       graph_json, created_at, created_by_user_id, locked
                  FROM workflow_versions
                 WHERE workflow_id = %s
                   AND project_id = %s
                 ORDER BY version DESC
                 LIMIT 1
                """,
                (wid, pid),
            )
            row = cur.fetchone()
            if row is None:
                return None
            graph = row[4] if isinstance(row[4], dict) else json.loads(row[4])
            return {
                "workflow_version_id": row[0],
                "workflow_id": row[1],
                "project_id": row[2],
                "version": row[3],
                "graph_json": graph,
                "created_at": row[5],
                "created_by_user_id": row[6],
                "locked": bool(row[7]),
            }
