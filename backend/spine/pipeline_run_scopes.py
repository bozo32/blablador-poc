"""Mutable workflow run scope history (Phase 10-02).

`pipeline_run_scopes` is an append-only index so the UI can find the latest run
for a reviewer-scoped entity (e.g. claimspan).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.db import connect
from backend.settings import settings as app_settings


def _project_id() -> str:
    return str(getattr(app_settings, "DEFAULT_PROJECT_ID", "default") or "default")


def _user_id() -> str:
    return str(getattr(app_settings, "DEFAULT_USER_ID", "local") or "local")


def _to_iso(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    return value


def insert_scope(
    *,
    scope_type: str,
    scope_id: str,
    reviewer_uid: str,
    run_id: str,
    citing_doc_id: str,
) -> None:
    st = str(scope_type or "").strip()
    sid = str(scope_id or "").strip()
    rid = str(run_id or "").strip()
    ruid = str(reviewer_uid or "").strip()
    doc = str(citing_doc_id or "").strip()
    if not st:
        raise ValueError("scope_type is required")
    if not sid:
        raise ValueError("scope_id is required")
    if not ruid:
        raise ValueError("reviewer_uid is required")
    if not rid:
        raise ValueError("run_id is required")
    if not doc:
        raise ValueError("citing_doc_id is required")

    pid = _project_id()
    uid = _user_id()

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pipeline_run_scopes(
                  scope_type,
                  scope_id,
                  reviewer_uid,
                  run_id,
                  citing_doc_id,
                  project_id,
                  created_by_user_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (st, sid, ruid, rid, doc, pid, uid),
            )
        conn.commit()


def latest_run_id(
    *, scope_type: str, scope_id: str, reviewer_uid: str
) -> Optional[str]:
    st = str(scope_type or "").strip()
    sid = str(scope_id or "").strip()
    ruid = str(reviewer_uid or "").strip()
    if not st:
        raise ValueError("scope_type is required")
    if not sid:
        raise ValueError("scope_id is required")
    if not ruid:
        raise ValueError("reviewer_uid is required")

    pid = _project_id()

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_id
                  FROM pipeline_run_scopes
                 WHERE project_id = %s
                   AND scope_type = %s
                   AND scope_id = %s
                   AND reviewer_uid = %s
                 ORDER BY created_at DESC
                 LIMIT 1
                """,
                (pid, st, sid, ruid),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return str(row[0] or "").strip() or None


def list_runs(
    *, scope_type: str, scope_id: str, reviewer_uid: str, limit: int = 25
) -> List[Dict[str, Any]]:
    st = str(scope_type or "").strip()
    sid = str(scope_id or "").strip()
    ruid = str(reviewer_uid or "").strip()
    if not st:
        raise ValueError("scope_type is required")
    if not sid:
        raise ValueError("scope_id is required")
    if not ruid:
        raise ValueError("reviewer_uid is required")

    lim = int(limit)
    if lim <= 0:
        lim = 25
    if lim > 250:
        lim = 250

    pid = _project_id()

    cols = (
        "scope_type",
        "scope_id",
        "reviewer_uid",
        "run_id",
        "citing_doc_id",
        "project_id",
        "created_by_user_id",
        "created_at",
    )
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT scope_type, scope_id, reviewer_uid, run_id, citing_doc_id,
                       project_id, created_by_user_id, created_at
                  FROM pipeline_run_scopes
                 WHERE project_id = %s
                   AND scope_type = %s
                   AND scope_id = %s
                   AND reviewer_uid = %s
                 ORDER BY created_at DESC
                 LIMIT %s
                """,
                (pid, st, sid, ruid, lim),
            )
            rows = cur.fetchall() or []

    out: List[Dict[str, Any]] = []
    for row in rows:
        d = {str(k): _to_iso(v) for k, v in zip(cols, row)}
        out.append(d)
    return out


__all__ = ["insert_scope", "latest_run_id", "list_runs"]
