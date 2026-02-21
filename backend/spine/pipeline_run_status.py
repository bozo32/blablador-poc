"""Mutable workflow run status + per-target status + append-only events.

Phase 10-02.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.db import connect
from backend.settings import settings as app_settings


RUN_STATES = {
    "queued",
    "running",
    "blocked",
    "complete",
    "partial",
    "error",
    "cancelled",
}
TARGET_STATES = {
    "requested",
    "available",
    "processing",
    "done",
    "error",
    "cancelled",
    "blocked",
}


DEFAULT_STAGES = ("extract", "citespans", "retrieval", "filter", "rerank", "nli")


def _project_id() -> str:
    return str(getattr(app_settings, "DEFAULT_PROJECT_ID", "default") or "default")


def _user_id() -> str:
    return str(getattr(app_settings, "DEFAULT_USER_ID", "local") or "local")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat().replace("+00:00", "Z")


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _json_loads(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return value
    return value


def _to_iso(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    return value


def _validate_state(value: str, *, allowed: set[str], label: str) -> str:
    v = str(value or "").strip()
    if not v:
        raise ValueError(f"{label} is required")
    if v not in allowed:
        raise ValueError(f"invalid {label}: {v}")
    return v


def default_stage_state_json(*, include_assessment: bool = False) -> Dict[str, Any]:
    stages = list(DEFAULT_STAGES)
    if include_assessment and "assessment" not in stages:
        stages.append("assessment")

    now = _iso_now()
    out: Dict[str, Any] = {"stages": {}}
    for name in stages:
        out["stages"][name] = {
            "state": "requested",
            "progress": None,
            "started_at": None,
            "finished_at": None,
            "updated_at": now,
            "message": None,
        }
    return out


def _normalize_stage_state_json(value: Any, *, ensure_defaults: bool) -> Dict[str, Any]:
    data = _json_loads(value) or {}
    if not isinstance(data, dict):
        data = {}
    stages_obj = data.get("stages")
    if not isinstance(stages_obj, dict):
        stages_obj = {}

    out: Dict[str, Any] = {"stages": {}}
    for k, v in stages_obj.items():
        if isinstance(k, str) and k.strip():
            out["stages"][k] = v
    if ensure_defaults:
        for name in DEFAULT_STAGES:
            out["stages"].setdefault(name, {})

    now = _iso_now()
    for stage_name, stage_entry in list(out["stages"].items()):
        if not isinstance(stage_entry, dict):
            stage_entry = {}
        entry: Dict[str, Any] = dict(stage_entry)
        entry.setdefault("state", "requested")
        entry.setdefault("progress", None)
        entry.setdefault("started_at", None)
        entry.setdefault("finished_at", None)
        entry.setdefault("message", None)
        entry["updated_at"] = str(entry.get("updated_at") or "").strip() or now
        out["stages"][stage_name] = entry
    return out


def normalize_stage_state_json(value: Any) -> Dict[str, Any]:
    """Ensure stage_state_json matches the stable minimal schema."""
    return _normalize_stage_state_json(value, ensure_defaults=True)


def merge_stage_state_json(current: Any, patch: Any) -> Dict[str, Any]:
    base = _normalize_stage_state_json(current, ensure_defaults=True)
    incoming = _normalize_stage_state_json(patch, ensure_defaults=False)
    now = _iso_now()
    for stage, entry in (incoming.get("stages") or {}).items():
        prev = base["stages"].get(stage)
        if not isinstance(prev, dict):
            prev = {}
        merged = dict(prev)
        for k, v in (entry or {}).items():
            if k == "updated_at":
                continue
            merged[k] = v
        merged["updated_at"] = now
        base["stages"][stage] = merged
    return base


def upsert_run_status(
    run_id: str,
    *,
    scope_type: str,
    scope_id: str,
    reviewer_uid: str,
    citing_doc_id: str,
    state: str,
    started_at: str | None = None,
    finished_at: str | None = None,
    error_json: dict | None = None,
    metrics_json: dict | None = None,
) -> Dict[str, Any]:
    rid = str(run_id or "").strip()
    st = str(scope_type or "").strip()
    sid = str(scope_id or "").strip()
    ruid = str(reviewer_uid or "").strip()
    doc = str(citing_doc_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")
    if not st:
        raise ValueError("scope_type is required")
    if not sid:
        raise ValueError("scope_id is required")
    if not ruid:
        raise ValueError("reviewer_uid is required")
    if not doc:
        raise ValueError("citing_doc_id is required")
    run_state = _validate_state(state, allowed=set(RUN_STATES), label="state")

    pid = _project_id()
    uid = _user_id()

    started_blob = str(started_at).strip() if started_at is not None else None
    finished_blob = str(finished_at).strip() if finished_at is not None else None
    err_blob = _json_dumps(error_json) if error_json is not None else None
    metrics_blob = _json_dumps(metrics_json) if metrics_json is not None else None

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pipeline_run_status(
                  run_id,
                  scope_type,
                  scope_id,
                  reviewer_uid,
                  citing_doc_id,
                  project_id,
                  created_by_user_id,
                  state,
                  started_at,
                  finished_at,
                  error_json,
                  metrics_json
                )
                VALUES (
                  %s, %s, %s, %s, %s,
                  %s, %s,
                  %s,
                  %s::timestamptz,
                  %s::timestamptz,
                  COALESCE(%s::jsonb, '{}'::jsonb),
                  COALESCE(%s::jsonb, '{}'::jsonb)
                )
                ON CONFLICT (run_id)
                DO UPDATE SET
                  scope_type = EXCLUDED.scope_type,
                  scope_id = EXCLUDED.scope_id,
                  reviewer_uid = EXCLUDED.reviewer_uid,
                  citing_doc_id = EXCLUDED.citing_doc_id,
                  project_id = EXCLUDED.project_id,
                  created_by_user_id = EXCLUDED.created_by_user_id,
                  state = EXCLUDED.state,
                  started_at = COALESCE(
                    EXCLUDED.started_at,
                    pipeline_run_status.started_at
                  ),
                  finished_at = COALESCE(
                    EXCLUDED.finished_at,
                    pipeline_run_status.finished_at
                  ),
                  error_json = COALESCE(%s::jsonb, pipeline_run_status.error_json),
                  metrics_json = COALESCE(%s::jsonb, pipeline_run_status.metrics_json),
                  updated_at = now()
                """,
                (
                    rid,
                    st,
                    sid,
                    ruid,
                    doc,
                    pid,
                    uid,
                    run_state,
                    started_blob,
                    finished_blob,
                    err_blob,
                    metrics_blob,
                    err_blob,
                    metrics_blob,
                ),
            )
        conn.commit()

    updated = get_run_status(rid)
    if updated is None:
        raise RuntimeError("failed to upsert pipeline_run_status")
    return updated


def upsert_target_status(
    run_id: str,
    target_id: str,
    *,
    state: str,
    citation_index: int | None,
    reference_id: str | None,
    attachment_id: str | None,
    stage_state: dict | None = None,
    attempts: dict | None = None,
    last_error: dict | None = None,
) -> Dict[str, Any]:
    rid = str(run_id or "").strip()
    tid = str(target_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")
    if not tid:
        raise ValueError("target_id is required")
    tgt_state = _validate_state(state, allowed=set(TARGET_STATES), label="state")

    pid = _project_id()

    ref = str(reference_id).strip() if reference_id is not None else None
    att = str(attachment_id).strip() if attachment_id is not None else None
    cit = int(citation_index) if citation_index is not None else None

    stage_blob: str | None = None
    attempts_blob = _json_dumps(attempts) if attempts is not None else None
    last_error_blob = _json_dumps(last_error) if last_error is not None else None

    # Merge stage state as a patch so only touched stages update updated_at.
    if stage_state is not None:
        existing = get_target_status(rid, tid)
        if existing is None:
            merged = merge_stage_state_json(default_stage_state_json(), stage_state)
        else:
            merged = merge_stage_state_json(
                existing.get("stage_state_json"), stage_state
            )
        stage_blob = _json_dumps(merged)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pipeline_target_status(
                  run_id,
                  target_id,
                  reference_id,
                  citation_index,
                  attachment_id,
                  project_id,
                  state,
                  stage_state_json,
                  attempts_json,
                  last_error_json
                )
                VALUES (
                  %s, %s, %s, %s, %s,
                  %s,
                  %s,
                  COALESCE(%s::jsonb, %s::jsonb),
                  COALESCE(%s::jsonb, '{}'::jsonb),
                  COALESCE(%s::jsonb, '{}'::jsonb)
                )
                ON CONFLICT (run_id, target_id)
                DO UPDATE SET
                  reference_id = COALESCE(
                    EXCLUDED.reference_id,
                    pipeline_target_status.reference_id
                  ),
                  citation_index = COALESCE(
                    EXCLUDED.citation_index,
                    pipeline_target_status.citation_index
                  ),
                  attachment_id = COALESCE(
                    EXCLUDED.attachment_id,
                    pipeline_target_status.attachment_id
                  ),
                  project_id = EXCLUDED.project_id,
                  state = EXCLUDED.state,
                  stage_state_json = COALESCE(
                    %s::jsonb,
                    pipeline_target_status.stage_state_json
                  ),
                  attempts_json = COALESCE(
                    %s::jsonb,
                    pipeline_target_status.attempts_json
                  ),
                  last_error_json = COALESCE(
                    %s::jsonb,
                    pipeline_target_status.last_error_json
                  ),
                  updated_at = now()
                """,
                (
                    rid,
                    tid,
                    ref,
                    cit,
                    att,
                    pid,
                    tgt_state,
                    stage_blob,
                    _json_dumps(default_stage_state_json()),
                    attempts_blob,
                    last_error_blob,
                    stage_blob,
                    attempts_blob,
                    last_error_blob,
                ),
            )
        conn.commit()

    updated = get_target_status(rid, tid)
    if updated is None:
        raise RuntimeError("failed to upsert pipeline_target_status")
    return updated


def get_run_status(run_id: str) -> Optional[Dict[str, Any]]:
    rid = str(run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")

    pid = _project_id()

    cols = (
        "run_id",
        "scope_type",
        "scope_id",
        "reviewer_uid",
        "citing_doc_id",
        "project_id",
        "created_by_user_id",
        "state",
        "started_at",
        "finished_at",
        "updated_at",
        "error_json",
        "metrics_json",
    )
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_id, scope_type, scope_id, reviewer_uid, citing_doc_id,
                       project_id, created_by_user_id,
                       state, started_at, finished_at,
                       updated_at, error_json, metrics_json
                  FROM pipeline_run_status
                 WHERE project_id = %s
                   AND run_id = %s
                  LIMIT 1
                """,
                (pid, rid),
            )
            row = cur.fetchone()
    if row is None:
        return None

    out = {str(k): _to_iso(v) for k, v in zip(cols, row)}
    out["error_json"] = _json_loads(out.get("error_json")) or {}
    out["metrics_json"] = _json_loads(out.get("metrics_json")) or {}
    return out


def get_target_status(run_id: str, target_id: str) -> Optional[Dict[str, Any]]:
    rid = str(run_id or "").strip()
    tid = str(target_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")
    if not tid:
        raise ValueError("target_id is required")

    pid = _project_id()

    cols = (
        "run_id",
        "target_id",
        "reference_id",
        "citation_index",
        "attachment_id",
        "project_id",
        "updated_at",
        "state",
        "stage_state_json",
        "attempts_json",
        "last_error_json",
    )
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_id, target_id, reference_id, citation_index, attachment_id,
                       project_id, updated_at, state,
                       stage_state_json, attempts_json, last_error_json
                  FROM pipeline_target_status
                 WHERE project_id = %s
                   AND run_id = %s
                   AND target_id = %s
                  LIMIT 1
                """,
                (pid, rid, tid),
            )
            row = cur.fetchone()
    if row is None:
        return None

    out = {str(k): _to_iso(v) for k, v in zip(cols, row)}
    out["stage_state_json"] = normalize_stage_state_json(out.get("stage_state_json"))
    out["attempts_json"] = _json_loads(out.get("attempts_json")) or {}
    out["last_error_json"] = _json_loads(out.get("last_error_json")) or {}
    return out


def list_target_status(run_id: str) -> List[Dict[str, Any]]:
    rid = str(run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")

    pid = _project_id()

    cols = (
        "run_id",
        "target_id",
        "reference_id",
        "citation_index",
        "attachment_id",
        "project_id",
        "updated_at",
        "state",
        "stage_state_json",
        "attempts_json",
        "last_error_json",
    )
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_id, target_id, reference_id, citation_index, attachment_id,
                       project_id, updated_at, state,
                       stage_state_json, attempts_json, last_error_json
                  FROM pipeline_target_status
                 WHERE project_id = %s
                   AND run_id = %s
                  ORDER BY COALESCE(citation_index, 2147483647) ASC, target_id ASC
                """,
                (pid, rid),
            )
            rows = cur.fetchall() or []

    out: List[Dict[str, Any]] = []
    for row in rows:
        d = {str(k): _to_iso(v) for k, v in zip(cols, row)}
        d["stage_state_json"] = normalize_stage_state_json(d.get("stage_state_json"))
        d["attempts_json"] = _json_loads(d.get("attempts_json")) or {}
        d["last_error_json"] = _json_loads(d.get("last_error_json")) or {}
        out.append(d)
    return out


def append_event(
    run_id: str,
    *,
    type: str,
    target_id: str | None = None,
    payload: dict | None = None,
) -> int:
    rid = str(run_id or "").strip()
    et = str(type or "").strip()
    tid = str(target_id).strip() if target_id is not None else None
    if not rid:
        raise ValueError("run_id is required")
    if not et:
        raise ValueError("type is required")

    pid = _project_id()
    payload_blob = _json_dumps(payload or {})

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pipeline_run_events(
                  run_id,
                  target_id,
                  project_id,
                  type,
                  payload_json
                )
                VALUES (%s, %s, %s, %s, %s::jsonb)
                RETURNING event_id
                """,
                (rid, tid, pid, et, payload_blob),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("pipeline_run_events insert did not return event_id")
            event_id = int(row[0])
        conn.commit()
    return event_id


def list_events(
    run_id: str, *, after_event_id: int = 0, limit: int = 200
) -> List[Dict[str, Any]]:
    rid = str(run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")

    after = int(after_event_id)
    lim = int(limit)
    if lim <= 0:
        lim = 200
    if lim > 1000:
        lim = 1000

    pid = _project_id()

    cols = (
        "event_id",
        "run_id",
        "target_id",
        "project_id",
        "type",
        "payload_json",
        "created_at",
    )
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT event_id, run_id, target_id, project_id,
                       type, payload_json, created_at
                  FROM pipeline_run_events
                 WHERE project_id = %s
                   AND run_id = %s
                   AND event_id > %s
                 ORDER BY event_id ASC
                 LIMIT %s
                """,
                (pid, rid, after, lim),
            )
            rows = cur.fetchall() or []

    out: List[Dict[str, Any]] = []
    for row in rows:
        d = {str(k): _to_iso(v) for k, v in zip(cols, row)}
        d["payload_json"] = _json_loads(d.get("payload_json")) or {}
        out.append(d)
    return out


__all__ = [
    "RUN_STATES",
    "TARGET_STATES",
    "DEFAULT_STAGES",
    "default_stage_state_json",
    "normalize_stage_state_json",
    "merge_stage_state_json",
    "upsert_run_status",
    "upsert_target_status",
    "get_run_status",
    "get_target_status",
    "list_target_status",
    "append_event",
    "list_events",
]
