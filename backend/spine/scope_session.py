from __future__ import annotations

from datetime import datetime
from typing import Any

from backend.db import connect


def _clean_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _clean_optional(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value or "").strip()
    return text or None


def get_scope_session_for_user(*, user_id: str) -> dict[str, Any]:
    uid = _clean_text(user_id, field="user_id")
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.project_id,
                       s.reviewer_uid,
                       s.updated_at,
                       a.project_id AS active_project_id
                  FROM user_scope_sessions s
             LEFT JOIN user_active_projects a
                    ON a.user_id = s.user_id
                 WHERE s.user_id = %s
                 LIMIT 1
                """,
                (uid,),
            )
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    """
                    SELECT project_id
                      FROM user_active_projects
                     WHERE user_id = %s
                     LIMIT 1
                    """,
                    (uid,),
                )
                fallback = cur.fetchone()
                return {
                    "user_id": uid,
                    "active_project_id": _clean_optional((fallback or [None])[0]),
                    "active_reviewer_uid": uid,
                    "updated_at": None,
                }

    active_project_id = _clean_optional(row[0]) or _clean_optional(row[3])
    reviewer_uid = _clean_optional(row[1]) or uid
    return {
        "user_id": uid,
        "active_project_id": active_project_id,
        "active_reviewer_uid": reviewer_uid,
        "updated_at": _iso(row[2]),
    }


def set_scope_session_for_user(
    *,
    user_id: str,
    actor_user_id: str,
    active_project_id: str | None,
    active_reviewer_uid: str | None,
) -> dict[str, Any]:
    uid = _clean_text(user_id, field="user_id")
    actor_uid = _clean_text(actor_user_id, field="actor_user_id")
    project_id = _clean_optional(active_project_id)
    reviewer_uid = _clean_optional(active_reviewer_uid) or uid

    with connect() as conn:
        with conn.cursor() as cur:
            if project_id:
                cur.execute(
                    """
                    SELECT 1
                      FROM user_project_memberships
                     WHERE user_id = %s
                       AND project_id = %s
                     LIMIT 1
                    """,
                    (uid, project_id),
                )
                if cur.fetchone() is None:
                    conn.rollback()
                    raise ValueError("User is not a member of this project")

            cur.execute(
                """
                INSERT INTO user_scope_sessions (
                  user_id,
                  project_id,
                  reviewer_uid,
                  updated_by_user_id
                )
                VALUES (%s, %s, %s, %s)
                ON CONFLICT(user_id) DO UPDATE
                  SET project_id = EXCLUDED.project_id,
                      reviewer_uid = EXCLUDED.reviewer_uid,
                      updated_at = now(),
                      updated_by_user_id = EXCLUDED.updated_by_user_id
                """,
                (uid, project_id, reviewer_uid, actor_uid),
            )

            if project_id:
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
                    (uid, project_id, actor_uid),
                )
        conn.commit()

    return get_scope_session_for_user(user_id=uid)
