from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from backend.db import connect


def _clean_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value or "").strip()
    return text or None


def list_projects_for_user(*, user_id: str) -> list[dict[str, Any]]:
    uid = _clean_text(user_id, field="user_id")
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.project_id,
                       COALESCE(pm.meta_json->>'name', '') AS project_name,
                       m.role,
                       m.created_at,
                       m.updated_at,
                       a.project_id AS active_project_id
                  FROM user_project_memberships m
             LEFT JOIN project_meta pm
                    ON pm.project_id = m.project_id
             LEFT JOIN user_active_projects a
                    ON a.user_id = m.user_id
                 WHERE m.user_id = %s
               ORDER BY m.updated_at DESC, m.created_at DESC, m.project_id ASC
                """,
                (uid,),
            )
            rows = cur.fetchall() or []

    out: list[dict[str, Any]] = []
    for row in rows:
        active_project_id = str(row[5] or "").strip() if row[5] is not None else None
        project_id = str(row[0] or "").strip()
        project_name = str(row[1] or "").strip() or None
        out.append(
            {
                "project_id": project_id,
                "name": project_name,
                "role": str(row[2] or "member").strip() or "member",
                "joined_at": _iso(row[3]),
                "updated_at": _iso(row[4]),
                "is_active": bool(active_project_id and active_project_id == project_id),
            }
        )
    return out


def list_known_user_ids(*, limit: int = 200) -> list[str]:
    max_rows = max(1, min(int(limit), 1000))
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id
                  FROM (
                        SELECT user_id, MAX(updated_at) AS touched_at
                          FROM user_project_memberships
                         GROUP BY user_id
                        UNION ALL
                        SELECT user_id, MAX(updated_at) AS touched_at
                          FROM user_scope_sessions
                         GROUP BY user_id
                       ) users
                 GROUP BY user_id
                 ORDER BY MAX(touched_at) DESC NULLS LAST, user_id ASC
                 LIMIT %s
                """,
                (max_rows,),
            )
            rows = cur.fetchall() or []

    out: list[str] = []
    for row in rows:
        user_id = str((row or [""])[0] or "").strip()
        if user_id:
            out.append(user_id)
    return out


def get_active_project_for_user(*, user_id: str) -> str | None:
    uid = _clean_text(user_id, field="user_id")
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT project_id
                  FROM user_active_projects
                 WHERE user_id = %s
                 LIMIT 1
                """,
                (uid,),
            )
            row = cur.fetchone()
    if not row:
        return None
    project_id = str(row[0] or "").strip()
    return project_id or None


def create_project_for_user(
    *,
    user_id: str,
    actor_user_id: str,
    project_id: str | None = None,
    role: str = "owner",
) -> dict[str, Any]:
    uid = _clean_text(user_id, field="user_id")
    actor_uid = _clean_text(actor_user_id, field="actor_user_id")
    pid = str(project_id or "").strip() or str(uuid4())
    membership_role = str(role or "owner").strip().lower() or "owner"

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_project_memberships (
                  user_id,
                  project_id,
                  role,
                  created_by_user_id
                )
                VALUES (%s, %s, %s, %s)
                ON CONFLICT(user_id, project_id) DO UPDATE
                  SET role = EXCLUDED.role,
                      updated_at = now(),
                      created_by_user_id = EXCLUDED.created_by_user_id
                RETURNING user_id, project_id, role, created_at, updated_at
                """,
                (uid, pid, membership_role, actor_uid),
            )
            row = cur.fetchone()
        conn.commit()

    if row is None:
        raise RuntimeError("project membership upsert failed")
    return {
        "user_id": row[0],
        "project_id": row[1],
        "role": row[2],
        "joined_at": _iso(row[3]),
        "updated_at": _iso(row[4]),
    }


def set_active_project_for_user(*, user_id: str, project_id: str, actor_user_id: str) -> bool:
    uid = _clean_text(user_id, field="user_id")
    pid = _clean_text(project_id, field="project_id")
    actor_uid = _clean_text(actor_user_id, field="actor_user_id")

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                  FROM user_project_memberships
                 WHERE user_id = %s
                   AND project_id = %s
                 LIMIT 1
                """,
                (uid, pid),
            )
            allowed = cur.fetchone() is not None
            if not allowed:
                conn.rollback()
                return False

            cur.execute(
                """
                INSERT INTO user_active_projects (
                  user_id,
                  project_id,
                  updated_by_user_id
                )
                VALUES (%s, %s, %s)
                ON CONFLICT(user_id) DO UPDATE
                  SET project_id = EXCLUDED.project_id,
                      updated_at = now(),
                      updated_by_user_id = EXCLUDED.updated_by_user_id
                """,
                (uid, pid, actor_uid),
            )
        conn.commit()

    return True


def has_project_membership(*, user_id: str, project_id: str) -> bool:
    uid = _clean_text(user_id, field="user_id")
    pid = _clean_text(project_id, field="project_id")

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                  FROM user_project_memberships
                 WHERE user_id = %s
                   AND project_id = %s
                 LIMIT 1
                """,
                (uid, pid),
            )
            return cur.fetchone() is not None
