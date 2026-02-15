"""Postgres helpers for extraction attempts (V2 ingestion spine).

Attempts are durable, idempotent execution records keyed by:
  (work_id, kind, settings_hash)

This module intentionally stays small and dependency-light.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Tuple
from uuid import uuid4

from backend.db import connect


ATTEMPT_STATES = {
    "queued",
    "running",
    "succeeded",
    "partial",
    "failed",
    "cancelled",
}

TERMINAL_STATES = {"succeeded", "partial", "failed", "cancelled"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


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


def settings_hash_for_attempt(work_id: str, kind: str, settings_json: Any) -> str:
    """Compute deterministic idempotency hash.

    Contract: sha256(json.dumps({work_id, kind, settings_json}, sort_keys=True)).
    """
    wid = str(work_id or "").strip()
    k = str(kind or "").strip()
    if not wid:
        raise ValueError("work_id is required")
    if not k:
        raise ValueError("kind is required")

    payload = {"work_id": wid, "kind": k, "settings_json": settings_json}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _advisory_lock_id(settings_hash: str) -> int:
    """Return a stable signed bigint for pg_advisory_xact_lock."""
    h = str(settings_hash or "").strip().lower()
    if not h:
        raise ValueError("settings_hash is required")
    n = int(h[:16], 16)
    if n >= 2**63:
        n -= 2**64
    return n


def create_or_get_attempt(
    project_id: str,
    created_by_user_id: str,
    work_id: str,
    kind: str,
    settings_json: Any,
    *,
    schema_version: int = 1,
) -> Tuple[str, str]:
    """Create (or reuse) an attempt row.

    Returns: (attempt_id, state)
    """
    wid = str(work_id or "").strip()
    pid = str(project_id or "").strip()
    uid = str(created_by_user_id or "").strip()
    k = str(kind or "").strip()
    if not pid:
        raise ValueError("project_id is required")
    if not uid:
        raise ValueError("created_by_user_id is required")
    if not wid:
        raise ValueError("work_id is required")
    if not k:
        raise ValueError("kind is required")

    if int(schema_version) <= 0:
        raise ValueError("schema_version must be >= 1")

    h = settings_hash_for_attempt(wid, k, settings_json)
    settings_blob = _json_dumps(settings_json or {})

    with connect() as conn:
        with conn.cursor() as cur:
            # Prevent concurrent duplicate inserts for the same logical attempt.
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (_advisory_lock_id(h),))
            cur.execute(
                """
                SELECT attempt_id, state
                  FROM attempts
                 WHERE project_id = %s
                   AND work_id = %s
                    AND kind = %s
                    AND settings_hash = %s
                  ORDER BY created_at DESC
                  LIMIT 1
                """,
                (pid, wid, k, h),
            )
            row = cur.fetchone()
            if row is not None:
                conn.commit()
                return (str(row[0]), str(row[1]))

            attempt_id = str(uuid4())
            cur.execute(
                """
                INSERT INTO attempts (
                  attempt_id,
                  work_id,
                  project_id,
                  created_by_user_id,
                  kind,
                  state,
                  schema_version,
                  settings_hash,
                  settings_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                RETURNING attempt_id, state
                """,
                (
                    attempt_id,
                    wid,
                    pid,
                    uid,
                    k,
                    "queued",
                    int(schema_version),
                    h,
                    settings_blob,
                ),
            )
            created = cur.fetchone()
            if created is None:
                raise RuntimeError("attempt insert did not return a row")
        conn.commit()
    return (str(created[0]), str(created[1]))


def get_attempt(attempt_id: str) -> Optional[Dict[str, Any]]:
    aid = str(attempt_id or "").strip()
    if not aid:
        raise ValueError("attempt_id is required")

    cols = (
        "attempt_id",
        "work_id",
        "project_id",
        "created_by_user_id",
        "kind",
        "state",
        "created_at",
        "started_at",
        "finished_at",
        "schema_version",
        "settings_hash",
        "settings_json",
        "provenance_json",
        "quality_json",
        "failure_reason",
        "failure_detail",
    )
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT attempt_id, work_id, project_id, created_by_user_id,
                       kind, state, created_at, started_at,
                       finished_at, schema_version, settings_hash, settings_json,
                       provenance_json, quality_json, failure_reason, failure_detail
                  FROM attempts
                 WHERE attempt_id = %s
                """,
                (aid,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            out: Dict[str, Any] = {str(k): v for k, v in zip(cols, row)}
            for key in ("settings_json", "provenance_json", "quality_json"):
                out[key] = _json_loads(out.get(key)) or {}
            return out


def get_latest_attempt_for_work(
    *, project_id: str, work_id: str, kind: str
) -> Optional[Dict[str, Any]]:
    pid = str(project_id or "").strip()
    wid = str(work_id or "").strip()
    k = str(kind or "").strip()
    if not pid:
        raise ValueError("project_id is required")
    if not wid:
        raise ValueError("work_id is required")
    if not k:
        raise ValueError("kind is required")

    cols = (
        "attempt_id",
        "work_id",
        "project_id",
        "created_by_user_id",
        "kind",
        "state",
        "created_at",
        "started_at",
        "finished_at",
        "schema_version",
        "settings_hash",
        "settings_json",
        "provenance_json",
        "quality_json",
        "failure_reason",
        "failure_detail",
    )
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT attempt_id, work_id, project_id, created_by_user_id,
                       kind, state, created_at, started_at, finished_at,
                       schema_version, settings_hash, settings_json,
                       provenance_json, quality_json, failure_reason, failure_detail
                  FROM attempts
                 WHERE project_id = %s
                   AND work_id = %s
                   AND kind = %s
                 ORDER BY created_at DESC
                 LIMIT 1
                """,
                (pid, wid, k),
            )
            row = cur.fetchone()
            if row is None:
                return None
            out: Dict[str, Any] = {str(k2): v for k2, v in zip(cols, row)}
            for key in ("settings_json", "provenance_json", "quality_json"):
                out[key] = _json_loads(out.get(key)) or {}
            return out


def transition_attempt_state(
    attempt_id: str,
    to_state: str,
    *,
    expected_from: Optional[Iterable[str]] = None,
    failure_reason: Optional[str] = None,
    failure_detail: Optional[str] = None,
) -> Dict[str, Any]:
    """Transition attempt state with basic guardrails.

    If expected_from is provided, the transition must come from one of those
    states (otherwise ValueError).
    """
    aid = str(attempt_id or "").strip()
    target = str(to_state or "").strip()
    if not aid:
        raise ValueError("attempt_id is required")
    if target not in ATTEMPT_STATES:
        raise ValueError(f"invalid attempt state: {target}")

    expected = {str(s).strip() for s in (expected_from or []) if str(s).strip()}

    now = _utc_now()
    started_at = now if target == "running" else None
    finished_at = now if target in TERMINAL_STATES else None

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT state FROM attempts WHERE attempt_id = %s", (aid,))
            row = cur.fetchone()
            if row is None:
                raise ValueError("attempt not found")
            current = str(row[0])

            if expected and current not in expected:
                exp = ", ".join(sorted(expected))
                msg = (
                    f"invalid attempt transition: {current} -> {target} "
                    f"(expected one of [{exp}])"
                )
                raise ValueError(msg)

            # Idempotent: if already in target state, just return.
            if current == target:
                attempt = get_attempt(aid)
                if attempt is None:
                    raise ValueError("attempt not found")
                return attempt

            cur.execute(
                """
                UPDATE attempts
                   SET state = %s,
                       started_at = COALESCE(started_at, %s),
                       finished_at = COALESCE(finished_at, %s),
                       failure_reason = COALESCE(%s::text, failure_reason),
                       failure_detail = COALESCE(%s::text, failure_detail)
                  WHERE attempt_id = %s
                """,
                (
                    target,
                    started_at,
                    finished_at,
                    failure_reason,
                    failure_detail,
                    aid,
                ),
            )
        conn.commit()

    attempt = get_attempt(aid)
    if attempt is None:
        raise ValueError("attempt not found")
    return attempt


def mark_attempt_queued(attempt_id: str) -> Dict[str, Any]:
    return transition_attempt_state(
        attempt_id, "queued", expected_from={"queued", "running"}
    )


def mark_attempt_running(attempt_id: str) -> Dict[str, Any]:
    return transition_attempt_state(
        attempt_id, "running", expected_from={"queued", "running"}
    )


def mark_attempt_succeeded(attempt_id: str) -> Dict[str, Any]:
    return transition_attempt_state(
        attempt_id, "succeeded", expected_from={"running", "queued"}
    )


def mark_attempt_partial(attempt_id: str) -> Dict[str, Any]:
    return transition_attempt_state(
        attempt_id, "partial", expected_from={"running", "queued"}
    )


def mark_attempt_cancelled(attempt_id: str) -> Dict[str, Any]:
    return transition_attempt_state(
        attempt_id, "cancelled", expected_from={"running", "queued"}
    )


def mark_attempt_failed(
    attempt_id: str,
    *,
    failure_reason: str,
    failure_detail: Optional[str] = None,
) -> Dict[str, Any]:
    reason = str(failure_reason or "").strip() or "unknown"
    detail = str(failure_detail) if failure_detail is not None else None
    return transition_attempt_state(
        attempt_id,
        "failed",
        expected_from={"running", "queued"},
        failure_reason=reason,
        failure_detail=detail,
    )
