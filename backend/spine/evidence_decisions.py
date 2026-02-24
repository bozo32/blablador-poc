"""Append-only evidence decision events + projection (Phase 10-04).

Implements durable reviewer-scoped decision streams for evidence triage:

- DEC-01: append-only events (pin/unpin/accept/reject/clear)
- DEC-02: optimistic concurrency control (OCC) via per-stream version integer
- DEC-03: idempotent writes via idempotency_key + request fingerprint

Projection is keyed by stable decision targets: (attachment_id, span_id).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from backend.db import connect
from backend.settings import settings as app_settings


DECISION_ACTIONS = {"pin", "unpin", "accept", "reject", "clear"}
TRIAGE_VALUES = {"none", "accepted", "rejected"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return value


def _project_id(value: Optional[str] = None) -> str:
    if value is not None and str(value).strip():
        return str(value).strip()
    return str(getattr(app_settings, "DEFAULT_PROJECT_ID", "default") or "default")


def _user_id(value: Optional[str] = None) -> str:
    if value is not None and str(value).strip():
        return str(value).strip()
    return str(getattr(app_settings, "DEFAULT_USER_ID", "local") or "local")


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


def _target_key(attachment_id: str, span_id: str) -> str:
    a = str(attachment_id or "").strip()
    s = str(span_id or "").strip()
    if not a or not s:
        raise ValueError("attachment_id and span_id are required")
    return f"{a}:{s}"


def _fingerprint(
    *,
    action: str,
    target: Optional[Dict[str, str]],
    set_value: Optional[bool],
    payload: Optional[dict],
) -> str:
    body: Dict[str, Any] = {"action": str(action or "").strip()}
    body["target"] = target if target is not None else None
    if set_value is not None:
        body["set"] = bool(set_value)
    if payload is not None:
        body["payload"] = payload
    blob = _json_dumps(body).encode("utf-8")
    return hashlib.sha256(blob).hexdigest().lower()


@dataclass(frozen=True)
class EvidenceDecisionConflict(RuntimeError):
    detail: str
    current_version: Optional[int] = None


def get_or_create_stream_version(
    claim_id: str,
    reviewer_uid: str,
    *,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> int:
    cid = str(claim_id or "").strip()
    ruid = str(reviewer_uid or "").strip() or "default"
    if not cid:
        raise ValueError("claim_id is required")
    pid = _project_id(project_id)
    uid = _user_id(user_id)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO evidence_decision_streams(
                  project_id,
                  claim_id,
                  reviewer_uid,
                  version,
                  created_by_user_id
                )
                VALUES (%s, %s, %s, 0, %s)
                ON CONFLICT(project_id, claim_id, reviewer_uid) DO NOTHING
                """,
                (pid, cid, ruid, uid),
            )
            cur.execute(
                """
                SELECT version
                  FROM evidence_decision_streams
                 WHERE project_id=%s AND claim_id=%s AND reviewer_uid=%s
                 LIMIT 1
                """,
                (pid, cid, ruid),
            )
            row = cur.fetchone()
        conn.commit()
    return int(row[0]) if row else 0


def _get_stream_version(
    claim_id: str,
    reviewer_uid: str,
    *,
    project_id: Optional[str] = None,
) -> int:
    cid = str(claim_id or "").strip()
    ruid = str(reviewer_uid or "").strip() or "default"
    if not cid:
        raise ValueError("claim_id is required")
    pid = _project_id(project_id)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT version
                  FROM evidence_decision_streams
                 WHERE project_id=%s AND claim_id=%s AND reviewer_uid=%s
                 LIMIT 1
                """,
                (pid, cid, ruid),
            )
            row = cur.fetchone()
    return int(row[0]) if row else 0


def list_recent_events(
    claim_id: str,
    reviewer_uid: str,
    *,
    project_id: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    cid = str(claim_id or "").strip()
    ruid = str(reviewer_uid or "").strip() or "default"
    if not cid:
        raise ValueError("claim_id is required")
    pid = _project_id(project_id)

    lim = int(limit)
    if lim <= 0:
        lim = 20
    lim = min(lim, 200)

    cols = (
        "event_id",
        "event_uid",
        "project_id",
        "claim_id",
        "reviewer_uid",
        "created_by_user_id",
        "idempotency_key",
        "action",
        "target_attachment_id",
        "target_span_id",
        "target_key",
        "expected_version",
        "resulting_version",
        "payload_json",
        "created_at",
    )
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT event_id, event_uid, project_id, claim_id, reviewer_uid,
                       created_by_user_id, idempotency_key, action,
                       target_attachment_id, target_span_id, target_key,
                       expected_version, resulting_version,
                       payload_json, created_at
                  FROM evidence_decision_events
                 WHERE project_id=%s AND claim_id=%s AND reviewer_uid=%s
                 ORDER BY event_id DESC
                 LIMIT %s
                """,
                (pid, cid, ruid, lim),
            )
            rows = cur.fetchall() or []

    out: List[Dict[str, Any]] = []
    for row in rows:
        d = {str(k): _to_iso(v) for k, v in zip(cols, row)}
        d["payload_json"] = _json_loads(d.get("payload_json")) or {}
        out.append(d)
    return out


def get_projection(
    claim_id: str,
    reviewer_uid: str,
    *,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    cid = str(claim_id or "").strip()
    ruid = str(reviewer_uid or "").strip() or "default"
    if not cid:
        raise ValueError("claim_id is required")
    pid = _project_id(project_id)

    version = get_or_create_stream_version(cid, ruid, project_id=pid)

    cols = (
        "target_key",
        "attachment_id",
        "span_id",
        "pinned",
        "triage",
        "updated_at",
        "last_event_id",
        "last_event_uid",
        "payload_json",
    )
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT t.target_key,
                       t.attachment_id,
                       t.span_id,
                       t.pinned,
                       t.triage,
                       t.updated_at,
                       t.last_event_id,
                       t.last_event_uid,
                       e.payload_json
                  FROM evidence_decision_targets t
                  LEFT JOIN evidence_decision_events e
                    ON e.project_id=t.project_id
                   AND e.event_id=t.last_event_id
                 WHERE t.project_id=%s AND t.claim_id=%s AND t.reviewer_uid=%s
                 ORDER BY t.target_key ASC
                """,
                (pid, cid, ruid),
            )
            rows = cur.fetchall() or []

    targets_by_key: Dict[str, Dict[str, Any]] = {}
    pinned_targets: List[Dict[str, str]] = []
    pinned_keys: List[str] = []
    for row in rows:
        d = {str(k): _to_iso(v) for k, v in zip(cols, row)}
        payload = _json_loads(d.get("payload_json")) or {}
        snippet = None
        if isinstance(payload, dict):
            raw = payload.get("snippet")
            snippet = str(raw).strip() if raw is not None else None
            if snippet == "":
                snippet = None

        target_key = str(d.get("target_key") or "").strip()
        if not target_key:
            continue
        entry = {
            "attachment_id": str(d.get("attachment_id") or "").strip(),
            "span_id": str(d.get("span_id") or "").strip(),
            "pinned": bool(d.get("pinned")),
            "triage": str(d.get("triage") or "none").strip() or "none",
            "updated_at": d.get("updated_at"),
            "last_event_id": d.get("last_event_id"),
            "last_event_uid": d.get("last_event_uid"),
            "snippet": snippet,
        }
        targets_by_key[target_key] = entry
        if entry["pinned"]:
            pinned_keys.append(target_key)
            pinned_targets.append(
                {
                    "attachment_id": entry["attachment_id"],
                    "span_id": entry["span_id"],
                    "target_key": target_key,
                }
            )

    return {
        "claim_id": cid,
        "reviewer_uid": ruid,
        "version": int(version),
        "targets_by_key": targets_by_key,
        "pinned_keys": pinned_keys,
        "pinned_targets": pinned_targets,
    }


def _current_target_state(
    pid: str,
    cid: str,
    ruid: str,
    *,
    target_key: str,
    conn: Any,
) -> Tuple[bool, str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT pinned, triage
              FROM evidence_decision_targets
             WHERE project_id=%s AND claim_id=%s AND reviewer_uid=%s AND target_key=%s
             LIMIT 1
            """,
            (pid, cid, ruid, target_key),
        )
        row = cur.fetchone()
    if not row:
        return False, "none"
    pinned = bool(row[0])
    triage = str(row[1] or "none").strip() or "none"
    if triage not in TRIAGE_VALUES:
        triage = "none"
    return pinned, triage


def _stream_has_any_targets(pid: str, cid: str, ruid: str, *, conn: Any) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
              FROM evidence_decision_targets
             WHERE project_id=%s AND claim_id=%s AND reviewer_uid=%s
             LIMIT 1
            """,
            (pid, cid, ruid),
        )
        row = cur.fetchone()
    return bool(row)


def append_event(
    claim_id: str,
    reviewer_uid: str,
    *,
    expected_version: int,
    idempotency_key: str,
    action: str,
    target_attachment_id: Optional[str] = None,
    target_span_id: Optional[str] = None,
    set_value: Optional[bool] = None,
    payload: Optional[dict] = None,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    cid = str(claim_id or "").strip()
    ruid = str(reviewer_uid or "").strip() or "default"
    act = str(action or "").strip().lower()
    idem = str(idempotency_key or "").strip()
    exp = int(expected_version)
    if not cid:
        raise ValueError("claim_id is required")
    if not idem:
        raise ValueError("idempotency_key is required")
    if act not in DECISION_ACTIONS:
        raise ValueError(f"invalid action: {act}")

    pid = _project_id(project_id)
    uid = _user_id(user_id)

    target: Optional[Dict[str, str]] = None
    tkey: Optional[str] = None
    attachment_id_clean = None
    span_id_clean = None
    if act != "clear":
        attachment_id_clean = str(target_attachment_id or "").strip()
        span_id_clean = str(target_span_id or "").strip()
        if not attachment_id_clean or not span_id_clean:
            raise ValueError("target_attachment_id and target_span_id are required")
        tkey = _target_key(attachment_id_clean, span_id_clean)
        target = {"attachment_id": attachment_id_clean, "span_id": span_id_clean}

    fp = _fingerprint(action=act, target=target, set_value=set_value, payload=payload)
    now = _utc_now().isoformat().replace("+00:00", "Z")

    with connect() as conn:
        with conn.cursor() as cur:
            # Ensure stream row exists.
            cur.execute(
                """
                INSERT INTO evidence_decision_streams(
                  project_id,
                  claim_id,
                  reviewer_uid,
                  version,
                  created_by_user_id
                )
                VALUES (%s, %s, %s, 0, %s)
                ON CONFLICT(project_id, claim_id, reviewer_uid) DO NOTHING
                """,
                (pid, cid, ruid, uid),
            )

            # Idempotency replay or key-reuse conflict.
            cur.execute(
                """
                SELECT event_id,
                       event_uid,
                       resulting_version,
                       request_fingerprint,
                       created_at
                  FROM evidence_decision_events
                 WHERE project_id=%s
                   AND claim_id=%s
                   AND reviewer_uid=%s
                   AND idempotency_key=%s
                  LIMIT 1
                """,
                (pid, cid, ruid, idem),
            )
            existing = cur.fetchone()
            if existing is not None:
                (
                    event_id,
                    event_uid,
                    resulting_version,
                    existing_fp,
                    created_at,
                ) = existing
                if str(existing_fp or "") != fp:
                    raise EvidenceDecisionConflict(
                        detail="idempotency_key already used", current_version=None
                    )
                pinned, triage = (False, "none")
                if tkey:
                    pinned, triage = _current_target_state(
                        pid, cid, ruid, target_key=tkey, conn=conn
                    )
                return {
                    "ok": True,
                    "event_id": int(event_id),
                    "event_uid": str(event_uid),
                    "version": int(resulting_version),
                    "no_op": False,
                    "created_at": _to_iso(created_at) or now,
                    "state": {"pinned": bool(pinned), "triage": str(triage)},
                }

            # OCC check.
            cur.execute(
                """
                SELECT version
                  FROM evidence_decision_streams
                 WHERE project_id=%s AND claim_id=%s AND reviewer_uid=%s
                 LIMIT 1
                """,
                (pid, cid, ruid),
            )
            row = cur.fetchone()
            current_version = int(row[0]) if row else 0
            if current_version != exp:
                raise EvidenceDecisionConflict(
                    detail="version conflict", current_version=current_version
                )

            # No-op semantics (only evaluated with matching expected_version).
            if act == "clear":
                if not _stream_has_any_targets(pid, cid, ruid, conn=conn):
                    return {
                        "ok": True,
                        "event_id": None,
                        "event_uid": None,
                        "version": current_version,
                        "no_op": True,
                        "created_at": None,
                        "state": {"pinned": False, "triage": "none"},
                    }
            else:
                pinned, triage = _current_target_state(
                    pid, cid, ruid, target_key=str(tkey), conn=conn
                )
                if act == "pin" and pinned:
                    return {
                        "ok": True,
                        "event_id": None,
                        "event_uid": None,
                        "version": current_version,
                        "no_op": True,
                        "created_at": None,
                        "state": {"pinned": True, "triage": triage},
                    }
                if act == "unpin" and not pinned:
                    return {
                        "ok": True,
                        "event_id": None,
                        "event_uid": None,
                        "version": current_version,
                        "no_op": True,
                        "created_at": None,
                        "state": {"pinned": False, "triage": triage},
                    }
                if act == "accept":
                    if set_value is False:
                        if triage != "accepted":
                            return {
                                "ok": True,
                                "event_id": None,
                                "event_uid": None,
                                "version": current_version,
                                "no_op": True,
                                "created_at": None,
                                "state": {"pinned": pinned, "triage": triage},
                            }
                    else:
                        if triage == "accepted":
                            return {
                                "ok": True,
                                "event_id": None,
                                "event_uid": None,
                                "version": current_version,
                                "no_op": True,
                                "created_at": None,
                                "state": {"pinned": pinned, "triage": "accepted"},
                            }
                if act == "reject":
                    if set_value is False:
                        if triage != "rejected":
                            return {
                                "ok": True,
                                "event_id": None,
                                "event_uid": None,
                                "version": current_version,
                                "no_op": True,
                                "created_at": None,
                                "state": {"pinned": pinned, "triage": triage},
                            }
                    else:
                        if triage == "rejected":
                            return {
                                "ok": True,
                                "event_id": None,
                                "event_uid": None,
                                "version": current_version,
                                "no_op": True,
                                "created_at": None,
                                "state": {"pinned": pinned, "triage": "rejected"},
                            }

            # Bump stream version.
            cur.execute(
                """
                UPDATE evidence_decision_streams
                   SET version=version+1,
                       updated_at=now(),
                       created_by_user_id=%s
                 WHERE project_id=%s AND claim_id=%s AND reviewer_uid=%s AND version=%s
                 RETURNING version
                """,
                (uid, pid, cid, ruid, exp),
            )
            bumped = cur.fetchone()
            if bumped is None:
                current = _get_stream_version(cid, ruid, project_id=pid)
                raise EvidenceDecisionConflict(
                    detail="version conflict", current_version=int(current)
                )
            new_version = int(bumped[0])

            event_uid = str(uuid4())
            payload_blob = _json_dumps(payload or {})

            cur.execute(
                """
                INSERT INTO evidence_decision_events(
                  event_uid,
                  project_id,
                  claim_id,
                  reviewer_uid,
                  created_by_user_id,
                  idempotency_key,
                  action,
                  target_attachment_id,
                  target_span_id,
                  target_key,
                  expected_version,
                  resulting_version,
                  request_fingerprint,
                  payload_json
                )
                VALUES (
                  %s,
                  %s, %s, %s,
                  %s,
                  %s,
                  %s,
                  %s, %s, %s,
                  %s, %s,
                  %s,
                  %s::jsonb
                )
                RETURNING event_id, created_at
                """,
                (
                    event_uid,
                    pid,
                    cid,
                    ruid,
                    uid,
                    idem,
                    act,
                    attachment_id_clean,
                    span_id_clean,
                    tkey,
                    exp,
                    new_version,
                    fp,
                    payload_blob,
                ),
            )
            inserted = cur.fetchone()
            if inserted is None:
                raise RuntimeError("evidence_decision_events insert did not return")
            event_id = int(inserted[0])
            created_at = _to_iso(inserted[1]) or now

            # Apply projection update.
            if act == "clear":
                cur.execute(
                    """
                    DELETE FROM evidence_decision_targets
                     WHERE project_id=%s AND claim_id=%s AND reviewer_uid=%s
                    """,
                    (pid, cid, ruid),
                )
                state = {"pinned": False, "triage": "none"}
            else:
                pinned, triage = _current_target_state(
                    pid, cid, ruid, target_key=str(tkey), conn=conn
                )
                next_pinned = pinned
                next_triage = triage
                if act == "pin":
                    next_pinned = True
                elif act == "unpin":
                    next_pinned = False
                elif act == "accept":
                    if set_value is False:
                        next_triage = "none" if triage == "accepted" else triage
                    else:
                        next_triage = "accepted"
                elif act == "reject":
                    if set_value is False:
                        next_triage = "none" if triage == "rejected" else triage
                    else:
                        next_triage = "rejected"

                cur.execute(
                    """
                    INSERT INTO evidence_decision_targets(
                      project_id,
                      claim_id,
                      reviewer_uid,
                      target_key,
                      attachment_id,
                      span_id,
                      pinned,
                      triage,
                      updated_at,
                      last_event_id,
                      last_event_uid
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now(), %s, %s)
                    ON CONFLICT(project_id, claim_id, reviewer_uid, target_key)
                    DO UPDATE SET
                      pinned=EXCLUDED.pinned,
                      triage=EXCLUDED.triage,
                      updated_at=now(),
                      last_event_id=EXCLUDED.last_event_id,
                      last_event_uid=EXCLUDED.last_event_uid
                    """,
                    (
                        pid,
                        cid,
                        ruid,
                        str(tkey),
                        attachment_id_clean,
                        span_id_clean,
                        bool(next_pinned),
                        str(next_triage),
                        int(event_id),
                        str(event_uid),
                    ),
                )
                state = {"pinned": bool(next_pinned), "triage": str(next_triage)}

        conn.commit()

    return {
        "ok": True,
        "event_id": int(event_id),
        "event_uid": str(event_uid),
        "version": int(new_version),
        "no_op": False,
        "created_at": created_at,
        "state": state,
    }


__all__ = [
    "DECISION_ACTIONS",
    "EvidenceDecisionConflict",
    "get_or_create_stream_version",
    "get_projection",
    "list_recent_events",
    "append_event",
]
