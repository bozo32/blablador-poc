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
        "set_value",
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
                       set_value,
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

    set_effective: Optional[bool] = set_value
    if act in {"accept", "reject"}:
        set_effective = True if set_value is None else bool(set_value)

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

    fp = _fingerprint(
        action=act,
        target=target,
        set_value=set_effective,
        payload=payload,
    )
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
                    if set_effective is False:
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
                    if set_effective is False:
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
                  set_value,
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
                    set_effective,
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
                    if set_effective is False:
                        next_triage = "none" if triage == "accepted" else triage
                    else:
                        next_triage = "accepted"
                elif act == "reject":
                    if set_effective is False:
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


def export_events_ndjson(*, project_id: Optional[str] = None) -> bytes:
    pid = _project_id(project_id)
    cols = (
        "event_uid",
        "project_id",
        "claim_id",
        "reviewer_uid",
        "created_by_user_id",
        "idempotency_key",
        "action",
        "target_attachment_id",
        "target_span_id",
        "expected_version",
        "resulting_version",
        "set_value",
        "payload_json",
        "created_at",
        "event_id",
    )
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT event_uid,
                       project_id,
                       claim_id,
                       reviewer_uid,
                       created_by_user_id,
                       idempotency_key,
                       action,
                       target_attachment_id,
                       target_span_id,
                       expected_version,
                       resulting_version,
                       set_value,
                       payload_json,
                       created_at,
                       event_id
                  FROM evidence_decision_events
                 WHERE project_id=%s
                 ORDER BY claim_id ASC, reviewer_uid ASC, event_id ASC
                """,
                (pid,),
            )
            rows = cur.fetchall() or []

    lines: list[str] = []
    for row in rows:
        d = {str(k): _to_iso(v) for k, v in zip(cols, row)}
        payload = _json_loads(d.get("payload_json")) or {}
        if not isinstance(payload, dict):
            payload = {}
        action = str(d.get("action") or "").strip()
        attachment_id = str(d.get("target_attachment_id") or "").strip() or None
        span_id = str(d.get("target_span_id") or "").strip() or None
        target = (
            {"attachment_id": attachment_id, "span_id": span_id}
            if action != "clear" and attachment_id and span_id
            else None
        )
        line_obj: dict[str, Any] = {
            "event_uid": str(d.get("event_uid") or "").strip(),
            "project_id": str(d.get("project_id") or pid),
            "claim_id": str(d.get("claim_id") or "").strip(),
            "reviewer_uid": str(d.get("reviewer_uid") or "default").strip()
            or "default",
            "created_by_user_id": str(d.get("created_by_user_id") or "local"),
            "idempotency_key": str(d.get("idempotency_key") or "").strip(),
            "action": action,
            "target": target,
            "expected_version": int(d.get("expected_version") or 0),
            "resulting_version": int(d.get("resulting_version") or 0),
            "set": d.get("set_value"),
            "payload": payload,
            "created_at": d.get("created_at"),
        }
        lines.append(_json_dumps(line_obj))
    blob = "\n".join(lines) + ("\n" if lines else "")
    return blob.encode("utf-8")


def wipe_project(*, project_id: Optional[str] = None) -> None:
    pid = _project_id(project_id)
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM evidence_decision_targets WHERE project_id=%s",
                (pid,),
            )
            cur.execute(
                "DELETE FROM evidence_decision_events WHERE project_id=%s",
                (pid,),
            )
            cur.execute(
                "DELETE FROM evidence_decision_streams WHERE project_id=%s",
                (pid,),
            )


def rebuild_project_projections(
    *,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, int]:
    pid = _project_id(project_id)
    uid = _user_id(user_id)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT claim_id, reviewer_uid
                  FROM evidence_decision_events
                 WHERE project_id=%s
                 ORDER BY claim_id ASC, reviewer_uid ASC
                """,
                (pid,),
            )
            streams = cur.fetchall() or []

            rebuilt_targets = 0
            for claim_id, reviewer_uid in streams:
                cid = str(claim_id or "").strip()
                ruid = str(reviewer_uid or "").strip() or "default"
                if not cid:
                    continue

                cur.execute(
                    """
                    DELETE FROM evidence_decision_targets
                     WHERE project_id=%s AND claim_id=%s AND reviewer_uid=%s
                    """,
                    (pid, cid, ruid),
                )

                cur.execute(
                    """
                    SELECT event_id,
                           event_uid,
                           action,
                           target_attachment_id,
                           target_span_id,
                           set_value,
                           payload_json,
                           created_at,
                           resulting_version
                      FROM evidence_decision_events
                     WHERE project_id=%s AND claim_id=%s AND reviewer_uid=%s
                     ORDER BY resulting_version ASC, event_id ASC
                    """,
                    (pid, cid, ruid),
                )
                rows = cur.fetchall() or []

                # Replay into in-memory state then write the projection.
                by_key: dict[str, dict[str, Any]] = {}
                max_version = 0
                for (
                    event_id,
                    event_uid,
                    action,
                    att,
                    span,
                    set_value,
                    payload_json,
                    created_at,
                    resulting_version,
                ) in rows:
                    act = str(action or "").strip().lower()
                    max_version = max(max_version, int(resulting_version or 0))
                    if act == "clear":
                        by_key.clear()
                        continue

                    attachment_id = str(att or "").strip()
                    span_id = str(span or "").strip()
                    if not attachment_id or not span_id:
                        continue
                    key = _target_key(attachment_id, span_id)
                    entry = by_key.get(key) or {
                        "attachment_id": attachment_id,
                        "span_id": span_id,
                        "pinned": False,
                        "triage": "none",
                        "updated_at": None,
                        "last_event_id": None,
                        "last_event_uid": None,
                    }

                    pinned = bool(entry.get("pinned"))
                    triage = str(entry.get("triage") or "none")
                    if act == "pin":
                        pinned = True
                    elif act == "unpin":
                        pinned = False
                    elif act == "accept":
                        set_eff = True if set_value is None else bool(set_value)
                        if set_eff is False:
                            if triage == "accepted":
                                triage = "none"
                        else:
                            triage = "accepted"
                    elif act == "reject":
                        set_eff = True if set_value is None else bool(set_value)
                        if set_eff is False:
                            if triage == "rejected":
                                triage = "none"
                        else:
                            triage = "rejected"

                    entry["pinned"] = bool(pinned)
                    entry["triage"] = triage if triage in TRIAGE_VALUES else "none"
                    entry["updated_at"] = _to_iso(created_at)
                    entry["last_event_id"] = int(event_id)
                    entry["last_event_uid"] = str(event_uid)
                    by_key[key] = entry

                cur.execute(
                    """
                    INSERT INTO evidence_decision_streams(
                      project_id,
                      claim_id,
                      reviewer_uid,
                      version,
                      created_by_user_id,
                      updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, now())
                    ON CONFLICT(project_id, claim_id, reviewer_uid)
                    DO UPDATE SET
                      version=EXCLUDED.version,
                      updated_at=now(),
                      created_by_user_id=EXCLUDED.created_by_user_id
                    """,
                    (pid, cid, ruid, int(max_version), uid),
                )

                for key, entry in by_key.items():
                    updated_at = entry.get("updated_at")
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
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::timestamptz,%s,%s)
                        ON CONFLICT(project_id, claim_id, reviewer_uid, target_key)
                        DO UPDATE SET
                          pinned=EXCLUDED.pinned,
                          triage=EXCLUDED.triage,
                          updated_at=EXCLUDED.updated_at,
                          last_event_id=EXCLUDED.last_event_id,
                          last_event_uid=EXCLUDED.last_event_uid
                        """,
                        (
                            pid,
                            cid,
                            ruid,
                            str(key),
                            str(entry.get("attachment_id") or ""),
                            str(entry.get("span_id") or ""),
                            bool(entry.get("pinned")),
                            str(entry.get("triage") or "none"),
                            str(updated_at or _utc_now().isoformat()),
                            entry.get("last_event_id"),
                            entry.get("last_event_uid"),
                        ),
                    )
                    rebuilt_targets += 1

        conn.commit()

    return {"streams": int(len(streams)), "targets": int(rebuilt_targets)}


def import_events_ndjson(
    *,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
    blob: bytes,
    overwrite: bool = False,
) -> Dict[str, int]:
    pid = _project_id(project_id)
    uid = _user_id(user_id)
    if overwrite:
        wipe_project(project_id=pid)

    try:
        text = blob.decode("utf-8", errors="ignore")
    except Exception:
        text = ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    parsed: list[dict[str, Any]] = []
    for ln in lines:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if isinstance(obj, dict):
            parsed.append(obj)

    inserted = 0
    with connect() as conn:
        with conn.cursor() as cur:
            for obj in parsed:
                claim_id = str(obj.get("claim_id") or "").strip()
                reviewer_uid = str(obj.get("reviewer_uid") or "default").strip()
                if not claim_id:
                    continue
                action = str(obj.get("action") or "").strip().lower()
                if action not in DECISION_ACTIONS:
                    continue
                target = obj.get("target")
                attachment_id = None
                span_id = None
                if action != "clear" and isinstance(target, dict):
                    attachment_id = (
                        str(target.get("attachment_id") or "").strip() or None
                    )
                    span_id = str(target.get("span_id") or "").strip() or None
                target_key = (
                    _target_key(str(attachment_id), str(span_id))
                    if attachment_id and span_id
                    else None
                )

                set_val = obj.get("set")
                set_value = None
                if action in {"accept", "reject"}:
                    set_value = True if set_val is None else bool(set_val)

                payload = obj.get("payload")
                if not isinstance(payload, dict):
                    payload = {}
                fp = _fingerprint(
                    action=action,
                    target={"attachment_id": attachment_id, "span_id": span_id}
                    if attachment_id and span_id
                    else None,
                    set_value=set_value,
                    payload=payload,
                )

                created_at = obj.get("created_at") or _utc_now().isoformat().replace(
                    "+00:00", "Z"
                )
                created_by = str(obj.get("created_by_user_id") or uid)
                event_uid = str(obj.get("event_uid") or "").strip() or str(uuid4())
                idempotency_key = (
                    str(obj.get("idempotency_key") or "").strip() or uuid4().hex
                )
                expected_version = int(obj.get("expected_version") or 0)
                resulting_version = int(obj.get("resulting_version") or 0)

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
                      set_value,
                      payload_json,
                      created_at
                    )
                    VALUES (
                      %s,
                      %s,%s,%s,
                      %s,
                      %s,
                      %s,
                      %s,%s,%s,
                      %s,%s,
                      %s,
                      %s,
                      %s::jsonb,
                      %s::timestamptz
                    )
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        event_uid,
                        pid,
                        claim_id,
                        reviewer_uid,
                        created_by,
                        idempotency_key,
                        action,
                        attachment_id,
                        span_id,
                        target_key,
                        expected_version,
                        resulting_version,
                        fp,
                        set_value,
                        _json_dumps(payload),
                        str(created_at),
                    ),
                )
                inserted += int(cur.rowcount or 0)

        conn.commit()

    rebuilt = rebuild_project_projections(project_id=pid, user_id=uid)
    return {"inserted": int(inserted), **rebuilt}


__all__ = [
    "DECISION_ACTIONS",
    "EvidenceDecisionConflict",
    "get_or_create_stream_version",
    "get_projection",
    "list_recent_events",
    "append_event",
    "export_events_ndjson",
    "import_events_ndjson",
    "rebuild_project_projections",
    "wipe_project",
]
