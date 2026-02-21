"""Spine-backed helpers for Phase 10 pipeline runs."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from backend.db import connect
from backend.settings import settings as app_settings


_RUN_COLS = (
    "run_id",
    "project_id",
    "created_by_user_id",
    "work_id",
    "created_at",
    "input_fingerprint",
    "caps_json",
    "settings_json",
    "note",
)


def _project_id() -> str:
    return str(getattr(app_settings, "DEFAULT_PROJECT_ID", "default") or "default")


def _user_id() -> str:
    return str(getattr(app_settings, "DEFAULT_USER_ID", "local") or "local")


def create_run(
    work_id: str,
    *,
    caps: dict,
    settings_json: dict,
    input_fingerprint: str,
    note: str | None = None,
) -> dict:
    wid = str(work_id or "").strip()
    if not wid:
        raise ValueError("work_id is required")
    fp = str(input_fingerprint or "").strip()
    if not fp:
        raise ValueError("input_fingerprint is required")

    pid = _project_id()
    uid = _user_id()
    run_id = str(uuid4())

    caps_blob = json.dumps(caps or {}, ensure_ascii=True, sort_keys=True)
    settings_blob = json.dumps(settings_json or {}, ensure_ascii=True, sort_keys=True)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pipeline_runs(
                  run_id,
                  project_id,
                  created_by_user_id,
                  work_id,
                  input_fingerprint,
                  caps_json,
                  settings_json,
                  note
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
                RETURNING run_id, project_id, created_by_user_id, work_id,
                          created_at, input_fingerprint, caps_json, settings_json, note
                """,
                (
                    run_id,
                    pid,
                    uid,
                    wid,
                    fp,
                    caps_blob,
                    settings_blob,
                    str(note).strip()
                    if note is not None and str(note).strip()
                    else None,
                ),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("pipeline_runs insert did not return a row")
        conn.commit()

    out = dict(zip(_RUN_COLS, row))
    created_at = out.get("created_at")
    if isinstance(created_at, datetime):
        out["created_at"] = created_at.isoformat().replace("+00:00", "Z")
    if isinstance(out.get("caps_json"), str):
        out["caps_json"] = json.loads(out["caps_json"] or "{}")
    if isinstance(out.get("settings_json"), str):
        out["settings_json"] = json.loads(out["settings_json"] or "{}")
    return out


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    rid = str(run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_id, project_id, created_by_user_id, work_id,
                       created_at, input_fingerprint, caps_json, settings_json, note
                FROM pipeline_runs
                WHERE run_id=%s
                """,
                (rid,),
            )
            row = cur.fetchone()
    if row is None:
        return None

    out = dict(zip(_RUN_COLS, row))
    created_at = out.get("created_at")
    if isinstance(created_at, datetime):
        out["created_at"] = created_at.isoformat().replace("+00:00", "Z")
    if isinstance(out.get("caps_json"), str):
        out["caps_json"] = json.loads(out["caps_json"] or "{}")
    if isinstance(out.get("settings_json"), str):
        out["settings_json"] = json.loads(out["settings_json"] or "{}")
    return out


__all__ = ["create_run", "get_run"]
