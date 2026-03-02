# backend/spine/opinion_events.py
"""Append-only opinion events for reader follow/ignore/complete state.

This module provides:
- Append-only event storage for reviewer opinions (follow status)
- Projection helpers to get current state from events
- Export/import as NDJSON for project portability

Opinion events are project-scoped via project_id and reviewer-scoped via owner_uid.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from backend.db.pg import connect


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def normalize_visibility(visibility: Optional[str]) -> str:
    raw = str(visibility or "private").strip().lower()
    if raw == "shared":
        raw = "selectable"
    if raw not in {"private", "selectable", "public"}:
        raise ValueError("visibility must be one of: private, selectable, public, shared")
    return raw


def _acl_clause_for_owner_stream(*, include_selectable: bool) -> str:
    if include_selectable:
        return "(owner_uid = %s OR visibility IN ('public', 'selectable'))"
    return "(owner_uid = %s OR visibility = 'public')"


def _can_view_event(
    *,
    owner_uid: str,
    visibility: str,
    group_id: Optional[str],
    mode: Optional[int],
    viewer_uid: str,
    viewer_is_project_member: bool,
    selectable_group_ids: Optional[Sequence[str]],
) -> bool:
    normalized_visibility = normalize_visibility(visibility)
    if str(viewer_uid or "").strip() == str(owner_uid or "").strip():
        return True
    if normalized_visibility == "public":
        return True
    if normalized_visibility != "selectable":
        return False

    # Fail-closed policy: selectable visibility is group-scoped only when we have
    # explicit group ACL context. If ACL context is absent, do not leak events.
    if not viewer_is_project_member:
        return False
    groups = {str(g or "").strip() for g in (selectable_group_ids or []) if str(g or "").strip()}
    required_group = str(group_id or "").strip()
    if not groups or not required_group or required_group not in groups:
        return False

    # Group readability via POSIX-style mode bit: 0o040 => group-read.
    try:
        parsed_mode = int(mode if mode is not None else 0)
    except Exception:
        parsed_mode = 0
    return bool(parsed_mode & 0o040)


def append_event(
    project_id: str,
    user_id: str,
    owner_uid: str,
    kind: str,
    target_key: str,
    visibility: str = "private",
    group_id: Optional[str] = None,
    mode: int = 0o600,
    payload: Optional[Dict[str, Any]] = None,
    idempotency_key: Optional[str] = None,
    doc_id: Optional[str] = None,
    citation_index: Optional[int] = None,
    target_id: Optional[str] = None,
    span_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Append an opinion event.
    
    Returns the created event with generated event_id.
    """
    payload_json = _json_dumps(payload or {})
    normalized_visibility = normalize_visibility(visibility)
    
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO opinion_events (
                    project_id, actor_uid, owner_uid, kind, target_key,
                    visibility, group_id, mode, payload_json, idempotency_key,
                    doc_id, citation_index, target_id, span_id
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                RETURNING event_id, created_at
                """,
                (
                    project_id, user_id, owner_uid, kind, target_key,
                    normalized_visibility, group_id, mode, payload_json, idempotency_key,
                    doc_id, citation_index, target_id, span_id,
                )
            )
            row = cur.fetchone()
    
    return {
        "project_id": project_id,
        "event_id": row[0],
        "created_at": row[1],
        "actor_uid": user_id,
        "owner_uid": owner_uid,
        "kind": kind,
        "target_key": target_key,
        "visibility": normalized_visibility,
        "group_id": group_id,
        "mode": mode,
        "payload": payload or {},
        "idempotency_key": idempotency_key,
        "doc_id": doc_id,
        "citation_index": citation_index,
        "target_id": target_id,
        "span_id": span_id,
    }


def list_recent_events(
    project_id: str,
    owner_uid: str,
    viewer_uid: str,
    viewer_is_project_member: bool,
    selectable_group_ids: Optional[Sequence[str]] = None,
    target_key: Optional[str] = None,
    kind: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """List recent opinion events for a reviewer."""
    include_selectable = bool(selectable_group_ids)
    with connect() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT event_id, created_at, actor_uid, owner_uid, kind, 
                       target_key, visibility, group_id, mode, payload_json,
                       doc_id, citation_index, target_id, span_id
                FROM opinion_events
                WHERE project_id = %s
                  AND owner_uid = %s
                  AND
            """
            query += _acl_clause_for_owner_stream(include_selectable=include_selectable)
            params: list[object] = [project_id, owner_uid, viewer_uid]
            
            if target_key:
                query += " AND target_key = %s"
                params.append(target_key)
            
            if kind:
                query += " AND kind = %s"
                params.append(kind)
            
            query += " ORDER BY event_id DESC LIMIT %s"
            params.append(limit)
            
            cur.execute(query, params)
            rows = cur.fetchall()
    
    events = [
        {
            "event_id": row[0],
            "created_at": row[1].isoformat().replace("+00:00", "Z") if row[1] else None,
            "actor_uid": row[2],
            "owner_uid": row[3],
            "kind": row[4],
            "target_key": row[5],
            "visibility": normalize_visibility(row[6]),
            "group_id": row[7],
            "mode": row[8],
            "payload": json.loads(row[9]) if row[9] else {},
            "doc_id": row[10],
            "citation_index": row[11],
            "target_id": row[12],
            "span_id": row[13],
        }
        for row in rows
    ]
    return [
        ev
        for ev in events
        if _can_view_event(
            owner_uid=str(ev.get("owner_uid") or ""),
            visibility=str(ev.get("visibility") or "private"),
            group_id=ev.get("group_id"),
            mode=ev.get("mode") if isinstance(ev.get("mode"), int) else None,
            viewer_uid=viewer_uid,
            viewer_is_project_member=viewer_is_project_member,
            selectable_group_ids=selectable_group_ids,
        )
    ]


def get_follow_status(
    project_id: str,
    owner_uid: str,
    viewer_uid: str,
    viewer_is_project_member: bool,
    target_key: str,
    selectable_group_ids: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Get the latest follow status for a target key.
    
    Returns the most recent follow/ignore/complete event for the given target.
    """
    include_selectable = bool(selectable_group_ids)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT event_id, created_at, actor_uid, owner_uid, kind,
                       target_key, visibility, group_id, mode, payload_json
                FROM opinion_events
                WHERE project_id = %s 
                  AND owner_uid = %s 
                  AND
                """
                + _acl_clause_for_owner_stream(include_selectable=include_selectable)
                +
                """
                  AND target_key = %s
                  AND kind = 'follow'
                ORDER BY event_id DESC
                LIMIT 1
                """,
                (project_id, owner_uid, viewer_uid, target_key)
            )
            row = cur.fetchone()
    
    if not row:
        return None
    
    if not _can_view_event(
        owner_uid=str(row[3] or ""),
        visibility=str(row[6] or "private"),
        group_id=row[7],
        mode=row[8],
        viewer_uid=viewer_uid,
        viewer_is_project_member=viewer_is_project_member,
        selectable_group_ids=selectable_group_ids,
    ):
        return None

    payload = json.loads(row[9]) if row[9] else {}

    return {
        "event_id": row[0],
        "created_at": row[1].isoformat().replace("+00:00", "Z") if row[1] else None,
        "actor_uid": row[2],
        "owner_uid": row[3],
        "target_key": row[5],
        "visibility": normalize_visibility(row[6]),
        "payload": payload,
        "status": payload.get("status", "follow"),
    }


def list_follow_by_doc(
    project_id: str,
    owner_uid: str,
    viewer_uid: str,
    viewer_is_project_member: bool,
    doc_id: str,
    limit: int = 500,
    selectable_group_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """List projected follow entries for a document.
    
    Returns one row per span_id with current_status and sort_key (min citation_index).
    """
    include_selectable = bool(selectable_group_ids)
    with connect() as conn:
        with conn.cursor() as cur:
            # Get latest follow status per span_id
            cur.execute(
                """
                SELECT DISTINCT ON (span_id)
                    span_id,
                    target_key,
                    payload_json,
                    citation_index,
                    target_id,
                    visibility,
                    group_id,
                    mode,
                    owner_uid
                FROM opinion_events
                WHERE project_id = %s
                  AND owner_uid = %s
                  AND
                """
                + _acl_clause_for_owner_stream(include_selectable=include_selectable)
                +
                """
                  AND doc_id = %s
                  AND kind = 'follow'
                  AND span_id IS NOT NULL
                ORDER BY span_id, event_id DESC
                LIMIT %s
                """,
                (project_id, owner_uid, viewer_uid, doc_id, limit)
            )
            rows = cur.fetchall()
    
    result = []
    for row in rows:
        raw_payload = row[2]
        if isinstance(raw_payload, dict):
            payload = raw_payload
        elif raw_payload:
            payload = json.loads(raw_payload)
        else:
            payload = {}
        if not _can_view_event(
            owner_uid=str(row[8] or ""),
            visibility=str(row[5] or "private"),
            group_id=row[6],
            mode=row[7],
            viewer_uid=viewer_uid,
            viewer_is_project_member=viewer_is_project_member,
            selectable_group_ids=selectable_group_ids,
        ):
            continue
        result.append({
            "span_id": row[0],
            "target_key": row[1],
            "current_status": payload.get("status", "follow"),
            "sort_key": row[3] if row[3] is not None else 0,
            "citation_index": row[3] if row[3] is not None else 0,
            "target_id": row[4],
        })
    
    # Sort by citation_index (sort_key) for rail ordering
    result.sort(key=lambda x: x["sort_key"])
    return result


def export_events_ndjson(
    project_id: str,
    owner_uid: Optional[str] = None,
    include_private: bool = True,
) -> str:
    """Export opinion events as NDJSON.
    
    If owner_uid is provided, exports only that reviewer's events.
    If include_private is False, excludes private events.
    """
    with connect() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT event_id, created_at, actor_uid, owner_uid, kind,
                       target_key, visibility, group_id, mode, payload_json,
                       doc_id, citation_index, target_id, span_id, idempotency_key
                FROM opinion_events
                WHERE project_id = %s
            """
            params = [project_id]
            
            if owner_uid:
                query += " AND owner_uid = %s"
                params.append(owner_uid)
            
            if not include_private:
                query += " AND visibility != 'private'"
            
            query += " ORDER BY event_id ASC"
            
            cur.execute(query, params)
            rows = cur.fetchall()
    
    lines = []
    for row in rows:
        event = {
            "event_id": row[0],
            "created_at": row[1].isoformat().replace("+00:00", "Z") if row[1] else None,
            "actor_uid": row[2],
            "owner_uid": row[3],
            "kind": row[4],
            "target_key": row[5],
            "visibility": row[6],
            "visibility": normalize_visibility(row[6]),
            "group_id": row[7],
            "mode": row[8],
            "payload": json.loads(row[9]) if row[9] else {},
            "doc_id": row[10],
            "citation_index": row[11],
            "target_id": row[12],
            "span_id": row[13],
            "idempotency_key": row[14],
        }
        lines.append(_json_dumps(event))
    
    return "\n".join(lines)


def import_events_ndjson(
    project_id: str,
    user_id: str,
    blob: bytes,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Import opinion events from NDJSON.
    
    If overwrite=False (default), skips events that already exist.
    Returns import statistics.
    """
    lines = blob.decode("utf-8").strip().split("\n")
    
    imported = 0
    skipped = 0
    errors = 0
    
    with connect(autocommit=True) as conn:
        for line in lines:
            if not line.strip():
                continue
            
            try:
                event = json.loads(line)
                
                # Check for existing event
                if not overwrite and event.get("idempotency_key"):
                    cur = conn.cursor()
                    cur.execute(
                        """
                        SELECT 1 FROM opinion_events
                        WHERE project_id = %s 
                          AND owner_uid = %s 
                          AND idempotency_key = %s
                        """,
                        (project_id, event.get("owner_uid"), event.get("idempotency_key"))
                    )
                    if cur.fetchone():
                        skipped += 1
                        continue
                
                # Insert the event
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO opinion_events (
                        project_id, actor_uid, owner_uid, kind, target_key,
                        visibility, group_id, mode, payload_json, idempotency_key,
                        doc_id, citation_index, target_id, span_id
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s
                    )
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        project_id,
                        event.get("actor_uid", user_id),
                        event.get("owner_uid", user_id),
                        event.get("kind", "follow"),
                        event.get("target_key", ""),
                        normalize_visibility(event.get("visibility", "private")),
                        event.get("group_id"),
                        event.get("mode", 0o600),
                        _json_dumps(event.get("payload", {})),
                        event.get("idempotency_key"),
                        event.get("doc_id"),
                        event.get("citation_index"),
                        event.get("target_id"),
                        event.get("span_id"),
                    )
                )
                imported += 1
                
            except Exception:
                errors += 1
    
    return {
        "imported": imported,
        "skipped": skipped,
        "errors": errors,
        "total": len(lines),
    }
