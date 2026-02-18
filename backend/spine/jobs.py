"""Postgres helpers for worker jobs (V2 ingestion spine)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from backend.db import connect


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def create_job(
    project_id: str,
    created_by_user_id: str,
    attempt_id: str,
    worker: str,
    *,
    state: str = "running",
    progress_json: Optional[dict] = None,
) -> str:
    pid = str(project_id or "").strip()
    uid = str(created_by_user_id or "").strip()
    aid = str(attempt_id or "").strip()
    w = str(worker or "").strip()
    if not pid:
        raise ValueError("project_id is required")
    if not uid:
        raise ValueError("created_by_user_id is required")
    if not aid:
        raise ValueError("attempt_id is required")
    if not w:
        raise ValueError("worker is required")

    job_id = str(uuid4())
    progress_blob = _json_dumps(progress_json or {})
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO jobs (
                  job_id,
                  attempt_id,
                  project_id,
                  created_by_user_id,
                  worker,
                  state,
                  progress_json,
                  heartbeat_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                """,
                (
                    job_id,
                    aid,
                    pid,
                    uid,
                    w,
                    str(state or "running"),
                    progress_blob,
                    _utc_now(),
                ),
            )
        conn.commit()
    return job_id


def update_job_progress(
    job_id: str,
    progress_json: dict,
    *,
    heartbeat_at: Optional[datetime] = None,
) -> None:
    jid = str(job_id or "").strip()
    if not jid:
        raise ValueError("job_id is required")

    progress_blob = _json_dumps(progress_json or {})
    hb = heartbeat_at or _utc_now()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE jobs
                   SET progress_json = %s::jsonb,
                       heartbeat_at = %s
                 WHERE job_id = %s
                """,
                (progress_blob, hb, jid),
            )
        conn.commit()


def set_job_state(
    job_id: str,
    state: str,
    *,
    progress_json: Optional[dict] = None,
    heartbeat_at: Optional[datetime] = None,
) -> None:
    jid = str(job_id or "").strip()
    st = str(state or "").strip()
    if not jid:
        raise ValueError("job_id is required")
    if not st:
        raise ValueError("state is required")

    hb = heartbeat_at or _utc_now()
    with connect() as conn:
        with conn.cursor() as cur:
            if progress_json is None:
                cur.execute(
                    """
                    UPDATE jobs
                       SET state = %s,
                           heartbeat_at = %s
                     WHERE job_id = %s
                    """,
                    (st, hb, jid),
                )
            else:
                progress_blob = _json_dumps(progress_json or {})
                cur.execute(
                    """
                    UPDATE jobs
                       SET state = %s,
                           progress_json = %s::jsonb,
                           heartbeat_at = %s
                     WHERE job_id = %s
                    """,
                    (st, progress_blob, hb, jid),
                )
        conn.commit()


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    jid = str(job_id or "").strip()
    if not jid:
        raise ValueError("job_id is required")

    cols = (
        "job_id",
        "attempt_id",
        "project_id",
        "created_by_user_id",
        "worker",
        "state",
        "progress_json",
        "heartbeat_at",
        "created_at",
    )
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT job_id, attempt_id, project_id, created_by_user_id,
                       worker, state, progress_json,
                       heartbeat_at, created_at
                  FROM jobs
                 WHERE job_id = %s
                """,
                (jid,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            out: Dict[str, Any] = {str(k): v for k, v in zip(cols, row)}
            if isinstance(out.get("progress_json"), str):
                try:
                    out["progress_json"] = json.loads(out["progress_json"]) or {}
                except Exception:
                    pass
            return out


def list_jobs_for_attempt(attempt_id: str, *, limit: int = 50) -> list[Dict[str, Any]]:
    aid = str(attempt_id or "").strip()
    if not aid:
        raise ValueError("attempt_id is required")
    lim = max(1, min(500, int(limit or 50)))

    cols = (
        "job_id",
        "attempt_id",
        "project_id",
        "created_by_user_id",
        "worker",
        "state",
        "progress_json",
        "heartbeat_at",
        "created_at",
    )
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT job_id, attempt_id, project_id, created_by_user_id,
                       worker, state, progress_json,
                       heartbeat_at, created_at
                  FROM jobs
                 WHERE attempt_id = %s
                 ORDER BY created_at DESC
                 LIMIT %s
                """,
                (aid, lim),
            )
            rows = cur.fetchall() or []

    out: list[Dict[str, Any]] = []
    for row in rows:
        item: Dict[str, Any] = {str(k): v for k, v in zip(cols, row)}
        if isinstance(item.get("progress_json"), str):
            try:
                item["progress_json"] = json.loads(item["progress_json"]) or {}
            except Exception:
                pass
        out.append(item)
    return out
