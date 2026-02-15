"""Postgres-backed versioned settings bundles.

This is a persistence layer for future:
- power-user settings panel (inspect + diff + rollback)
- safe-edit surface for noobs (via policies)
- workflow-builder configuration binding
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional
from uuid import uuid4

from backend.db import connect


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def get_or_create_settings_bundle(
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

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT bundle_id, project_id, name, created_at, created_by_user_id
                  FROM settings_bundles
                 WHERE project_id = %s
                   AND name = %s
                 LIMIT 1
                """,
                (pid, nm),
            )
            row = cur.fetchone()
            if row is not None:
                return {
                    "bundle_id": row[0],
                    "project_id": row[1],
                    "name": row[2],
                    "created_at": row[3],
                    "created_by_user_id": row[4],
                }

            bundle_id = str(uuid4())
            cur.execute(
                """
                INSERT INTO settings_bundles (
                  bundle_id,
                  project_id,
                  name,
                  created_by_user_id
                )
                VALUES (%s, %s, %s, %s)
                RETURNING bundle_id, project_id, name, created_at, created_by_user_id
                """,
                (bundle_id, pid, nm, uid),
            )
            row2 = cur.fetchone()
        conn.commit()

    if row2 is None:
        raise RuntimeError("settings bundle insert failed")
    return {
        "bundle_id": row2[0],
        "project_id": row2[1],
        "name": row2[2],
        "created_at": row2[3],
        "created_by_user_id": row2[4],
    }


def append_settings_version(
    *,
    bundle_id: str,
    project_id: str,
    created_by_user_id: str,
    config_json: dict,
    schema_version: int = 1,
) -> Dict[str, Any]:
    bid = str(bundle_id or "").strip()
    pid = str(project_id or "").strip()
    uid = str(created_by_user_id or "").strip()
    if not bid:
        raise ValueError("bundle_id is required")
    if not pid:
        raise ValueError("project_id is required")
    if not uid:
        raise ValueError("created_by_user_id is required")

    version_id = str(uuid4())
    blob = _json_dumps(config_json or {})
    schema_v = int(schema_version)
    if schema_v <= 0:
        raise ValueError("schema_version must be >= 1")

    with connect() as conn:
        with conn.cursor() as cur:
            # Serialize version increments per bundle.
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"settings_versions:{bid}",),
            )
            cur.execute(
                """
                SELECT COALESCE(MAX(version), 0)
                  FROM settings_versions
                 WHERE bundle_id = %s
                """,
                (bid,),
            )
            next_v = int((cur.fetchone() or [0])[0]) + 1
            cur.execute(
                """
                INSERT INTO settings_versions (
                  version_id, bundle_id, project_id, version, schema_version,
                  config_json, created_by_user_id
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                RETURNING version_id, bundle_id, project_id, version, schema_version,
                          config_json, created_at, created_by_user_id, locked
                """,
                (version_id, bid, pid, next_v, schema_v, blob, uid),
            )
            row = cur.fetchone()
        conn.commit()

    if row is None:
        raise RuntimeError("settings version insert failed")
    return {
        "version_id": row[0],
        "bundle_id": row[1],
        "project_id": row[2],
        "version": row[3],
        "schema_version": row[4],
        "config_json": row[5] if isinstance(row[5], dict) else json.loads(row[5]),
        "created_at": row[6],
        "created_by_user_id": row[7],
        "locked": bool(row[8]),
    }


def get_latest_settings_version(
    *, bundle_id: str, project_id: str
) -> Optional[Dict[str, Any]]:
    bid = str(bundle_id or "").strip()
    pid = str(project_id or "").strip()
    if not bid:
        raise ValueError("bundle_id is required")
    if not pid:
        raise ValueError("project_id is required")

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT version_id, bundle_id, project_id, version, schema_version,
                       config_json, created_at, created_by_user_id, locked
                  FROM settings_versions
                 WHERE bundle_id = %s
                   AND project_id = %s
                 ORDER BY version DESC
                 LIMIT 1
                """,
                (bid, pid),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cfg = row[5] if isinstance(row[5], dict) else json.loads(row[5])
            return {
                "version_id": row[0],
                "bundle_id": row[1],
                "project_id": row[2],
                "version": row[3],
                "schema_version": row[4],
                "config_json": cfg,
                "created_at": row[6],
                "created_by_user_id": row[7],
                "locked": bool(row[8]),
            }
