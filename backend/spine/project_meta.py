from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from backend.db.pg import connect
from backend import schemas


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_meta(*, project_id: str) -> dict:
    now = _now()
    return {
        "version": 1,
        "name": str(project_id or "default").strip() or "default",
        "created_at": now,
        "updated_at": None,
        "reviewers": [],
        "active_reviewer_uid": None,
        "compare_reviewer_a": None,
        "compare_reviewer_b": None,
        "graph_settings": {},
    }


def get_or_create_project_meta(
    *, project_id: str, user_id: str = "local"
) -> dict[str, Any]:
    pid = str(project_id or "").strip() or "default"
    uid = str(user_id or "").strip() or "local"
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT meta_json FROM project_meta WHERE project_id=%s",
                (pid,),
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                meta = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            else:
                meta = _default_meta(project_id=pid)
                cur.execute(
                    """
                    INSERT INTO project_meta(
                      project_id,
                      meta_json,
                      updated_at,
                      updated_by_user_id
                    )
                    VALUES (%s, %s::jsonb, now(), %s)
                    ON CONFLICT(project_id) DO NOTHING
                    """,
                    (pid, json.dumps(meta, ensure_ascii=True), uid),
                )
    # Normalize via schema to keep defaults stable.
    return schemas.ProjectMeta(**meta).model_dump(mode="json")


def update_project_meta(
    *, project_id: str, user_id: str, patch: dict
) -> dict[str, Any]:
    pid = str(project_id or "").strip() or "default"
    uid = str(user_id or "").strip() or "local"
    existing = get_or_create_project_meta(project_id=pid, user_id=uid)
    merged = dict(existing)
    for k, v in (patch or {}).items():
        if v is None:
            continue
        merged[k] = v
    merged["updated_at"] = _now()

    normalized = schemas.ProjectMeta(**merged).model_dump(mode="json")
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO project_meta(
                  project_id,
                  meta_json,
                  updated_at,
                  updated_by_user_id
                )
                VALUES (%s, %s::jsonb, now(), %s)
                ON CONFLICT(project_id) DO UPDATE
                  SET meta_json=excluded.meta_json,
                      updated_at=excluded.updated_at,
                      updated_by_user_id=excluded.updated_by_user_id
                """,
                (pid, json.dumps(normalized, ensure_ascii=True), uid),
            )
    return normalized


def export_project_meta_json(*, project_id: str) -> bytes:
    meta = get_or_create_project_meta(project_id=project_id)
    blob = json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=True)
    return (blob + "\n").encode("utf-8")


def import_project_meta_json(
    *, project_id: str, user_id: str, meta: dict
) -> dict[str, Any]:
    pid = str(project_id or "").strip() or "default"
    uid = str(user_id or "").strip() or "local"
    normalized = schemas.ProjectMeta(
        **(meta or _default_meta(project_id=pid))
    ).model_dump(mode="json")
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO project_meta(
                  project_id,
                  meta_json,
                  updated_at,
                  updated_by_user_id
                )
                VALUES (%s, %s::jsonb, now(), %s)
                ON CONFLICT(project_id) DO UPDATE
                  SET meta_json=excluded.meta_json,
                      updated_at=excluded.updated_at,
                      updated_by_user_id=excluded.updated_by_user_id
                """,
                (pid, json.dumps(normalized, ensure_ascii=True), uid),
            )
    return normalized


__all__ = [
    "get_or_create_project_meta",
    "update_project_meta",
    "export_project_meta_json",
    "import_project_meta_json",
]
