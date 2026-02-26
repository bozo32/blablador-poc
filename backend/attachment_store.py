"""Attachment persistence utilities.

Phase 09.3: attachments are spine-backed (Postgres metadata + object-store blobs).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from uuid import uuid4

from backend.db.pg import connect
from backend.object_store import s3 as object_store_s3

from lxml import etree

from backend import extraction


STATUS_PENDING = "pending"
STATUS_CONVERTING = "converting"
STATUS_PARSING = "parsing"
STATUS_MATCHED = "matched"
STATUS_READY = STATUS_MATCHED  # Backward-compatible alias
STATUS_ERROR = "error"

MAX_TIMELINE_EVENTS = 5
MAX_EVENTS_STORED = 25
DEFAULT_MAX_ATTEMPTS = 2


_SENTENCE_CACHE: dict[str, Tuple[str, List[dict]]] = {}


TEI_NS = extraction.NS


class AttachmentNotFound(RuntimeError):
    """Raised when an attachment id cannot be resolved."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sanitize_filename(filename: str) -> str:
    name = str(filename or "").strip() or "attachment.pdf"
    return name.replace("/", "_").replace("\\", "_")


def find_attachment_by_content_sha256(
    *, project_id: str, content_sha256: str, archived: bool = False
) -> Optional[dict]:
    pid = str(project_id or "default").strip() or "default"
    digest = str(content_sha256 or "").strip().lower()
    if not digest:
        return None
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT attachment_id
                  FROM attachments
                 WHERE project_id=%s
                   AND content_sha256=%s
                   AND archived=%s
                 ORDER BY (claim_id IS NULL) DESC, created_at ASC
                 LIMIT 1
                """,
                (pid, digest, bool(archived)),
            )
            row = cur.fetchone()
    if not row:
        return None
    return get_attachment(str(row[0]))


def create_attachment_from_bytes(
    *,
    claim_id: Optional[str],
    doc_id: Optional[str],
    file_bytes: bytes,
    filename: str,
    size_bytes: Optional[int] = None,
    reference_hint: Optional[dict] = None,
    claim_text: Optional[str] = None,
    citation_index: Optional[int] = None,
    target_id: Optional[str] = None,
    source_ingest_id: Optional[str] = None,
    project_id: str = "default",
    user_id: str = "local",
) -> dict:
    safe_name = _sanitize_filename(filename)
    data = bytes(file_bytes or b"")
    digest = hashlib.sha256(data).hexdigest().lower()

    attachment_id = str(uuid4())
    pdf_object_key = f"attachments/{attachment_id}/{digest}/{safe_name}"
    object_store_s3.put_bytes(
        pdf_object_key,
        data,
        content_type="application/pdf",
    )

    size = int(size_bytes if size_bytes is not None else len(data))
    now = _now()
    rec = {
        "id": attachment_id,
        "attachment_id": attachment_id,
        "project_id": str(project_id or "default"),
        "created_by_user_id": str(user_id or "local"),
        "created_at": now,
        "updated_at": now,
        "claim_id": claim_id,
        "doc_id": doc_id,
        "citation_index": citation_index,
        "target_id": target_id,
        "source_ingest_id": source_ingest_id,
        "filename": safe_name,
        "size": size,
        "size_bytes": size,
        "status": STATUS_PENDING,
        "error": None,
        "parsed_at": None,
        "archived": False,
        "archived_at": None,
        "attempts": 0,
        "max_attempts": DEFAULT_MAX_ATTEMPTS,
        "reference_hint": reference_hint or {},
        "claim_text": claim_text,
        "content_sha256": digest,
        "pdf_object_key": pdf_object_key,
        "artifacts": {},
    }

    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO attachments(
                  attachment_id,
                  project_id,
                  created_by_user_id,
                  claim_id,
                  doc_id,
                  citation_index,
                  target_id,
                  source_ingest_id,
                  filename,
                  size_bytes,
                  status,
                  error,
                  parsed_at,
                  archived,
                  archived_at,
                  attempts,
                  max_attempts,
                  reference_hint,
                  claim_text,
                  content_sha256,
                  pdf_object_key,
                  artifacts_json,
                  created_at,
                  updated_at
                )
                VALUES (
                  %s, %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s, NULL,
                  false, NULL,
                  0, %s,
                  %s::jsonb, %s,
                  %s,
                  %s,
                  %s::jsonb,
                  now(), now()
                )
                """,
                (
                    attachment_id,
                    rec["project_id"],
                    rec["created_by_user_id"],
                    claim_id,
                    doc_id,
                    citation_index,
                    target_id,
                    source_ingest_id,
                    safe_name,
                    size,
                    STATUS_PENDING,
                    None,
                    int(rec["max_attempts"]),
                    json.dumps(rec["reference_hint"], ensure_ascii=True),
                    claim_text,
                    digest,
                    pdf_object_key,
                    json.dumps({}, ensure_ascii=True),
                ),
            )

    _log_event(
        attachment_id=attachment_id,
        project_id=rec["project_id"],
        user_id=rec["created_by_user_id"],
        event="queued",
        detail="Attachment received",
    )
    return rec


def _normalize_status(value: Optional[str]) -> str:
    if value == "ready":
        return STATUS_MATCHED
    if value in {
        STATUS_PENDING,
        STATUS_CONVERTING,
        STATUS_PARSING,
        STATUS_MATCHED,
        STATUS_ERROR,
    }:
        return str(value)
    return STATUS_PENDING


def _log_event(
    *,
    attachment_id: str,
    project_id: str,
    user_id: str,
    event: str,
    detail: Optional[str] = None,
) -> None:
    eid = str(uuid4())
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO attachment_events(
                  event_id,
                  attachment_id,
                  project_id,
                  created_by_user_id,
                  event,
                  detail,
                  at
                )
                VALUES (%s, %s, %s, %s, %s, %s, now())
                """,
                (eid, attachment_id, project_id, user_id, event, detail),
            )


def _fetch_events(*, attachment_id: str) -> list[dict]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT event, detail, at
                FROM attachment_events
                WHERE attachment_id=%s
                ORDER BY at DESC
                LIMIT %s
                """,
                (attachment_id, int(MAX_EVENTS_STORED)),
            )
            rows = cur.fetchall() or []
    out: list[dict] = []
    for row in rows:
        out.append(
            {
                "event": str(row[0] or ""),
                "detail": row[1],
                "at": row[2].isoformat().replace("+00:00", "Z") if row[2] else None,
            }
        )
    return out


def _public_view(record: dict) -> dict:
    data = dict(record or {})
    data["status"] = _normalize_status(data.get("status"))
    data.setdefault("claim_id", None)
    data.setdefault("doc_id", None)
    data.setdefault("citation_index", None)
    data.setdefault("target_id", None)
    data.setdefault("source_ingest_id", None)
    data.setdefault("archived", False)
    data.setdefault("archived_at", None)
    data.setdefault("reference_hint", {})
    timeline = data.get("timeline") or []
    data["history"] = timeline[:MAX_TIMELINE_EVENTS]
    data["timeline"] = data["history"]
    data["retry_available"] = (
        int(data.get("attempts", 0) or 0)
        < int(data.get("max_attempts", DEFAULT_MAX_ATTEMPTS) or DEFAULT_MAX_ATTEMPTS)
        and data.get("status") != STATUS_MATCHED
    )
    return data


def is_ready(record: dict) -> bool:
    return _normalize_status(record.get("status")) == STATUS_MATCHED


def create_attachment(
    *,
    claim_id: Optional[str],
    doc_id: Optional[str],
    local_path: str | Path,
    filename: Optional[str] = None,
    size_bytes: Optional[int] = None,
    reference_hint: Optional[dict] = None,
    claim_text: Optional[str] = None,
    citation_index: Optional[int] = None,
    target_id: Optional[str] = None,
    source_ingest_id: Optional[str] = None,
    project_id: str = "default",
    user_id: str = "local",
) -> dict:
    source_path = Path(local_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Attachment source not found: {source_path}")

    attachment_id = str(uuid4())
    safe_name = _sanitize_filename(filename or source_path.name)
    file_bytes = source_path.read_bytes()
    digest = hashlib.sha256(file_bytes).hexdigest().lower()

    pdf_object_key = f"attachments/{attachment_id}/{digest}/{safe_name}"
    object_store_s3.put_bytes(
        pdf_object_key,
        file_bytes,
        content_type="application/pdf",
    )

    size = int(size_bytes if size_bytes is not None else len(file_bytes))
    now = _now()
    rec = {
        "id": attachment_id,
        "attachment_id": attachment_id,
        "project_id": str(project_id or "default"),
        "created_by_user_id": str(user_id or "local"),
        "created_at": now,
        "updated_at": now,
        "claim_id": claim_id,
        "doc_id": doc_id,
        "citation_index": citation_index,
        "target_id": target_id,
        "source_ingest_id": source_ingest_id,
        "filename": safe_name,
        "size": size,
        "size_bytes": size,
        "status": STATUS_PENDING,
        "error": None,
        "parsed_at": None,
        "archived": False,
        "archived_at": None,
        "attempts": 0,
        "max_attempts": DEFAULT_MAX_ATTEMPTS,
        "reference_hint": reference_hint or {},
        "claim_text": claim_text,
        "content_sha256": digest,
        "pdf_object_key": pdf_object_key,
        "artifacts": {},
    }

    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO attachments(
                  attachment_id,
                  project_id,
                  created_by_user_id,
                  claim_id,
                  doc_id,
                  citation_index,
                  target_id,
                  source_ingest_id,
                  filename,
                  size_bytes,
                  status,
                  error,
                  parsed_at,
                  archived,
                  archived_at,
                  attempts,
                  max_attempts,
                   reference_hint,
                   claim_text,
                   content_sha256,
                   pdf_object_key,
                   artifacts_json,
                   created_at,
                   updated_at
                 )
                 VALUES (
                  %s, %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s, NULL,
                  false, NULL,
                   0, %s,
                   %s::jsonb, %s,
                   %s,
                   %s,
                   %s::jsonb,
                   now(), now()
                 )
                """,
                (
                    attachment_id,
                    rec["project_id"],
                    rec["created_by_user_id"],
                    claim_id,
                    doc_id,
                    citation_index,
                    target_id,
                    source_ingest_id,
                    safe_name,
                    size,
                    STATUS_PENDING,
                    None,
                    int(rec["max_attempts"]),
                    json.dumps(rec["reference_hint"], ensure_ascii=True),
                    claim_text,
                    digest,
                    pdf_object_key,
                    json.dumps({}, ensure_ascii=True),
                ),
            )

    _log_event(
        attachment_id=attachment_id,
        project_id=rec["project_id"],
        user_id=rec["created_by_user_id"],
        event="queued",
        detail="Attachment received",
    )
    return rec


def get_attachment(attachment_id: str, public: bool = False) -> Optional[dict]:
    aid = str(attachment_id or "").strip()
    if not aid:
        return None
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  attachment_id,
                  project_id,
                  created_by_user_id,
                  created_at,
                  updated_at,
                  claim_id,
                  doc_id,
                  citation_index,
                  target_id,
                  source_ingest_id,
                  filename,
                  size_bytes,
                  status,
                  error,
                  parsed_at,
                  archived,
                  archived_at,
                  attempts,
                  max_attempts,
                  reference_hint,
                  claim_text,
                  content_sha256,
                  pdf_object_key,
                  artifacts_json
                FROM attachments
                WHERE attachment_id=%s
                """,
                (aid,),
            )
            row = cur.fetchone()
            if not row:
                return None

    artifacts = row[23] if isinstance(row[23], dict) else json.loads(row[23] or "{}")
    ref_hint = row[19] if isinstance(row[19], dict) else json.loads(row[19] or "{}")
    timeline = _fetch_events(attachment_id=aid)

    rec = {
        "id": row[0],
        "attachment_id": row[0],
        "project_id": row[1],
        "created_by_user_id": row[2],
        "created_at": row[3].isoformat().replace("+00:00", "Z") if row[3] else None,
        "updated_at": row[4].isoformat().replace("+00:00", "Z") if row[4] else None,
        "claim_id": row[5],
        "doc_id": row[6],
        "citation_index": row[7],
        "target_id": row[8],
        "source_ingest_id": row[9],
        "filename": row[10],
        "size": int(row[11] or 0),
        "status": row[12],
        "error": row[13],
        "parsed_at": row[14].isoformat().replace("+00:00", "Z") if row[14] else None,
        "archived": bool(row[15]),
        "archived_at": row[16].isoformat().replace("+00:00", "Z") if row[16] else None,
        "attempts": int(row[17] or 0),
        "max_attempts": int(row[18] or DEFAULT_MAX_ATTEMPTS),
        "reference_hint": ref_hint,
        "claim_text": row[20],
        "content_sha256": row[21],
        "pdf_object_key": row[22],
        "artifacts": artifacts,
        "timeline": timeline,
    }
    return _public_view(rec) if public else rec


def list_attachments(
    claim_id: Optional[str] = None,
    *,
    archived: Optional[bool] = False,
    public: bool = False,
) -> List[dict]:
    where = []
    params: list = []
    if claim_id:
        where.append("claim_id=%s")
        params.append(str(claim_id))
    if archived is not None:
        where.append("archived=%s")
        params.append(bool(archived))
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT attachment_id
                FROM attachments
                {clause}
                ORDER BY created_at DESC
                """,
                tuple(params),
            )
            rows = cur.fetchall() or []
    out: list[dict] = []
    for (aid,) in rows:
        rec = get_attachment(str(aid), public=public)
        if rec:
            out.append(rec)
    return out


def set_archived(attachment_id: str, *, archived: bool) -> dict:
    rec = get_attachment(attachment_id)
    if rec is None:
        raise AttachmentNotFound(f"Attachment {attachment_id} not found")
    aid = str(attachment_id)
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE attachments
                   SET archived=%s,
                       archived_at=CASE WHEN %s THEN now() ELSE NULL END,
                       updated_at=now()
                 WHERE attachment_id=%s
                """,
                (bool(archived), bool(archived), aid),
            )

    _log_event(
        attachment_id=aid,
        project_id=str(rec.get("project_id") or "default"),
        user_id=str(rec.get("created_by_user_id") or "local"),
        event="archive" if archived else "unarchive",
        detail=None,
    )
    return get_attachment(aid) or rec


def set_placement(
    attachment_id: str,
    *,
    claim_id: Optional[str],
    doc_id: Optional[str],
    citation_index: Optional[int],
    target_id: Optional[str],
) -> dict:
    rec = get_attachment(attachment_id)
    if rec is None:
        raise AttachmentNotFound(f"Attachment {attachment_id} not found")
    aid = str(attachment_id)
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE attachments
                   SET claim_id=%s,
                       doc_id=%s,
                       citation_index=%s,
                       target_id=%s,
                       updated_at=now()
                 WHERE attachment_id=%s
                """,
                (claim_id, doc_id, citation_index, target_id, aid),
            )

    detail = (
        f"claim_id={claim_id or 'null'} doc_id={doc_id or 'null'} "
        f"citation_index={citation_index if citation_index is not None else 'null'} "
        f"target_id={target_id or 'null'}"
    )
    _log_event(
        attachment_id=aid,
        project_id=str(rec.get("project_id") or "default"),
        user_id=str(rec.get("created_by_user_id") or "local"),
        event="placement",
        detail=detail,
    )
    return get_attachment(aid) or rec


def clone_attachment(
    source_attachment_id: str,
    *,
    claim_id: Optional[str] = None,
    doc_id: Optional[str] = None,
    citation_index: Optional[int] = None,
    target_id: Optional[str] = None,
    reference_hint: Optional[dict] = None,
    claim_text: Optional[str] = None,
) -> dict:
    """Clone an existing attachment row.

    The clone reuses the same stored PDF object and artifacts. If the source is
    already matched, the clone is created as matched immediately.
    """
    src = get_attachment(str(source_attachment_id))
    if src is None:
        raise AttachmentNotFound(f"Attachment {source_attachment_id} not found")

    new_id = str(uuid4())
    now = _now()
    status = _normalize_status(src.get("status"))
    artifacts = src.get("artifacts") or {}
    ref_hint = reference_hint or (src.get("reference_hint") or {})
    parsed_at = src.get("parsed_at")

    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO attachments(
                  attachment_id,
                  project_id,
                  created_by_user_id,
                  claim_id,
                  doc_id,
                  citation_index,
                  target_id,
                  source_ingest_id,
                  filename,
                  size_bytes,
                  status,
                  error,
                  parsed_at,
                  archived,
                  archived_at,
                  attempts,
                  max_attempts,
                   reference_hint,
                   claim_text,
                   content_sha256,
                   pdf_object_key,
                   artifacts_json,
                   created_at,
                   updated_at
                 )
                 VALUES (
                  %s,%s,%s,%s,%s,%s,%s,%s,
                  %s,%s,%s,%s,%s,
                  false,NULL,
                   0,%s,
                   %s::jsonb,%s,
                   %s,
                   %s,
                   %s::jsonb,
                   now(),now()
                 )
                """,
                (
                    new_id,
                    str(src.get("project_id") or "default"),
                    str(src.get("created_by_user_id") or "local"),
                    claim_id,
                    doc_id,
                    citation_index,
                    target_id,
                    src.get("source_ingest_id"),
                    src.get("filename"),
                    int(src.get("size") or 0),
                    STATUS_MATCHED if status == STATUS_MATCHED else STATUS_PENDING,
                    None,
                    parsed_at,
                    int(src.get("max_attempts") or DEFAULT_MAX_ATTEMPTS),
                    json.dumps(ref_hint, ensure_ascii=True),
                    claim_text,
                    src.get("content_sha256"),
                    str(src.get("pdf_object_key") or ""),
                    json.dumps(artifacts, ensure_ascii=True),
                ),
            )

    _log_event(
        attachment_id=new_id,
        project_id=str(src.get("project_id") or "default"),
        user_id=str(src.get("created_by_user_id") or "local"),
        event="cloned",
        detail=f"from:{source_attachment_id}",
    )
    return get_attachment(new_id) or {
        "id": new_id,
        "attachment_id": new_id,
        "project_id": src.get("project_id"),
        "created_by_user_id": src.get("created_by_user_id"),
        "created_at": now,
        "updated_at": now,
        "claim_id": claim_id,
        "doc_id": doc_id,
        "citation_index": citation_index,
        "target_id": target_id,
        "source_ingest_id": src.get("source_ingest_id"),
        "filename": src.get("filename"),
        "size": int(src.get("size") or 0),
        "status": STATUS_MATCHED if status == STATUS_MATCHED else STATUS_PENDING,
        "error": None,
        "parsed_at": parsed_at,
        "archived": False,
        "attempts": 0,
        "max_attempts": int(src.get("max_attempts") or DEFAULT_MAX_ATTEMPTS),
        "reference_hint": ref_hint,
        "claim_text": claim_text,
        "content_sha256": src.get("content_sha256"),
        "pdf_object_key": src.get("pdf_object_key"),
        "artifacts": artifacts,
        "timeline": [],
    }


def update_attachment(
    attachment_id: str,
    *,
    status: Optional[str] = None,
    error: Optional[str] = None,
    parsed_at: Optional[str] = None,
    timeline_event: Optional[str] = None,
    timeline_detail: Optional[str] = None,
    artifacts: Optional[dict] = None,
    **extra_fields,
) -> dict:
    rec = get_attachment(attachment_id)
    if rec is None:
        raise AttachmentNotFound(f"Attachment {attachment_id} not found")

    aid = str(attachment_id)
    updates: dict[str, object] = {}
    if status is not None:
        updates["status"] = str(status)
    if error is not None:
        updates["error"] = error
    if parsed_at is not None:
        updates["parsed_at"] = parsed_at
    if artifacts is not None:
        updates["artifacts_json"] = json.dumps(artifacts, ensure_ascii=True)

    allowed = {
        "source_ingest_id",
        "claim_id",
        "doc_id",
        "citation_index",
        "target_id",
        "attempts",
        "max_attempts",
    }
    for k, v in (extra_fields or {}).items():
        if k in allowed:
            updates[k] = v

    if updates:
        cols = []
        params = []
        for k, v in updates.items():
            if k == "artifacts_json":
                cols.append("artifacts_json=%s::jsonb")
                params.append(v)
            elif k == "reference_hint":
                cols.append("reference_hint=%s::jsonb")
                params.append(json.dumps(v, ensure_ascii=True))
            else:
                cols.append(f"{k}=%s")
                params.append(v)
        cols.append("updated_at=now()")
        params.append(aid)
        with connect(autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE attachments SET {', '.join(cols)} WHERE attachment_id=%s",
                    tuple(params),
                )

    if timeline_event:
        _log_event(
            attachment_id=aid,
            project_id=str(rec.get("project_id") or "default"),
            user_id=str(rec.get("created_by_user_id") or "local"),
            event=str(timeline_event),
            detail=timeline_detail,
        )
    clear_sentence_cache(aid)
    return get_attachment(aid) or rec


def save_artifacts(
    attachment_id: str,
    *,
    tei_xml: str,
    tei_json: dict,
    sentences: Iterable[dict],
) -> dict:
    rec = get_attachment(attachment_id)
    if rec is None:
        raise AttachmentNotFound(f"Attachment {attachment_id} not found")

    aid = str(attachment_id)
    pid = str(rec.get("project_id") or "default")
    uid = str(rec.get("created_by_user_id") or "local")

    tei_xml_key = f"attachments/{aid}/tei.xml"
    tei_json_key = f"attachments/{aid}/tei.json"
    sentences_key = f"attachments/{aid}/sentences.ndjson"

    tei_xml_bytes = (tei_xml or "").encode("utf-8", errors="ignore")
    tei_json_bytes = json.dumps(
        tei_json or {},
        indent=2,
        ensure_ascii=True,
        sort_keys=True,
    ).encode("utf-8")
    lines: list[str] = []
    for row in sentences:
        if not isinstance(row, dict):
            continue
        lines.append(json.dumps(row, ensure_ascii=True, separators=(",", ":")))
    sentences_bytes = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")

    object_store_s3.put_bytes(
        tei_xml_key, tei_xml_bytes, content_type="application/xml"
    )
    object_store_s3.put_bytes(
        tei_json_key,
        tei_json_bytes,
        content_type="application/json",
    )
    object_store_s3.put_bytes(
        sentences_key,
        sentences_bytes,
        content_type="application/x-ndjson",
    )

    arts = {
        "tei_xml": tei_xml_key,
        "tei_json": tei_json_key,
        "sentences": sentences_key,
    }
    with connect(autocommit=True) as conn:
        with conn.cursor() as cur:
            for typ, key, blob, ctype in (
                ("tei.xml", tei_xml_key, tei_xml_bytes, "application/xml"),
                ("tei.json", tei_json_key, tei_json_bytes, "application/json"),
                (
                    "sentences.ndjson",
                    sentences_key,
                    sentences_bytes,
                    "application/x-ndjson",
                ),
            ):
                cur.execute(
                    """
                    INSERT INTO attachment_artifacts(
                      artifact_id,
                      attachment_id,
                      project_id,
                      created_by_user_id,
                      artifact_type,
                      object_key,
                      bytes,
                      content_type
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(uuid4()),
                        aid,
                        pid,
                        uid,
                        typ,
                        key,
                        int(len(blob)),
                        ctype,
                    ),
                )

            cur.execute(
                """
                UPDATE attachments
                   SET artifacts_json=%s::jsonb,
                       parsed_at=now(),
                       updated_at=now()
                 WHERE attachment_id=%s
                """,
                (json.dumps(arts, ensure_ascii=True), aid),
            )

    clear_sentence_cache(aid)
    return arts


def list_resumable(statuses: Optional[Iterable[str]] = None) -> List[dict]:
    wanted = set(statuses or {STATUS_PENDING, STATUS_CONVERTING, STATUS_PARSING})
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT attachment_id, status FROM attachments WHERE archived=false"
            )
            rows = cur.fetchall() or []
    out: list[dict] = []
    for aid, status in rows:
        if _normalize_status(status) not in wanted:
            continue
        rec = get_attachment(str(aid), public=False)
        if rec:
            out.append(rec)
    return out


def mark_converting(attachment_id: str, attempt: int) -> dict:
    return update_attachment(
        attachment_id,
        status=STATUS_CONVERTING,
        timeline_event="converting",
        timeline_detail=f"Attempt {attempt}",
        attempts=int(attempt),
    )


def mark_parsing(attachment_id: str, attempt: int) -> dict:
    return update_attachment(
        attachment_id,
        status=STATUS_PARSING,
        timeline_event="parsing",
        timeline_detail=f"Attempt {attempt}",
        attempts=int(attempt),
    )


def mark_matched(attachment_id: str, artifacts: dict) -> dict:
    return update_attachment(
        attachment_id,
        status=STATUS_MATCHED,
        error=None,
        parsed_at=_now(),
        timeline_event="matched",
        timeline_detail="Artifacts written",
        artifacts=artifacts,
    )


def mark_ready(attachment_id: str, artifacts: dict) -> dict:
    return mark_matched(attachment_id, artifacts=artifacts)


def mark_error(attachment_id: str, message: str) -> dict:
    return update_attachment(
        attachment_id,
        status=STATUS_ERROR,
        error=message,
        timeline_event="error",
        timeline_detail=message,
    )


def reset_for_retry(attachment_id: str) -> dict:
    rec = get_attachment(attachment_id)
    if rec is None:
        raise AttachmentNotFound(f"Attachment {attachment_id} not found")
    return update_attachment(
        attachment_id,
        status=STATUS_PENDING,
        error=None,
        timeline_event="retry",
        timeline_detail="Manual retry scheduled",
    )


def public_status(attachment_id: str) -> Optional[dict]:
    return get_attachment(attachment_id, public=True)


def public_claim_status(claim_id: str) -> List[dict]:
    return list_attachments(claim_id=claim_id, public=True)


def load_sentences_for_attachment(
    attachment_id: str, *, use_cache: bool = True
) -> List[dict]:
    rec = get_attachment(attachment_id)
    if rec is None:
        raise AttachmentNotFound(f"Attachment {attachment_id} not found")
    if not is_ready(rec):
        raise RuntimeError(
            f"Attachment {attachment_id} is not ready (status={rec.get('status')})"
        )
    artifacts = rec.get("artifacts") or {}
    key = str(artifacts.get("sentences") or "").strip()
    if not key:
        # Best-effort fallback: synthesize sentence rows from TEI XML.
        tei_key = str(artifacts.get("tei_xml") or "").strip()
        if not tei_key:
            raise FileNotFoundError(f"Attachment {attachment_id} is missing sentences")
        raw = object_store_s3.get_bytes(tei_key)
        rows = _synthesize_sentences_from_tei(raw)
        if not rows:
            raise FileNotFoundError(f"Attachment {attachment_id} is missing sentences")
        return rows

    if use_cache:
        cached = _SENTENCE_CACHE.get(str(attachment_id))
        if cached and cached[0] == key:
            return [dict(x) for x in cached[1]]

    blob = object_store_s3.get_bytes(key)
    rows: list[dict] = []
    for line in blob.decode("utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            rows.append(json.loads(s))
        except Exception:
            continue
    _SENTENCE_CACHE[str(attachment_id)] = (key, rows)
    return [dict(x) for x in rows]


def _collapse_ws(text: str) -> str:
    return " ".join(str(text or "").split())


def _text_content(elem: etree._Element) -> str:
    return _collapse_ws("".join(elem.itertext()))


def _xml_id(elem: etree._Element) -> Optional[str]:
    return elem.get(f"{{{extraction.XML_NS}}}id") or elem.get("xml:id")


def _page_label(elem: etree._Element) -> str:
    nodes = elem.xpath("ancestor::tei:pb[1]/@n", namespaces=TEI_NS)
    if nodes:
        return str(nodes[0])
    nodes = elem.xpath("preceding::tei:pb[1]/@n", namespaces=TEI_NS)
    if nodes:
        return str(nodes[0])
    return "Unknown"


def _section_path(elem: etree._Element) -> str:
    divs: List[etree._Element] = list(
        elem.xpath("ancestor::tei:div", namespaces=TEI_NS)
    )
    labels: List[str] = []
    for div in divs:
        heads = div.xpath("./tei:head[1]", namespaces=TEI_NS)
        if not heads:
            continue
        label = _text_content(heads[0]).strip()
        if label:
            labels.append(label)
    return " > ".join(labels) if labels else "Body"


def _synthesize_sentences_from_tei(tei_xml: str | bytes) -> List[dict]:
    """Synthesize attachment sentence rows from TEI.

    The evidence pipeline prefers persisted `sentences.ndjson`, but some seed
    datasets (and older artifacts) may have only `tei_xml`. This fallback keeps
    evidence review usable.
    """
    if isinstance(tei_xml, bytes):
        tei_xml = tei_xml.decode("utf-8", errors="ignore")
    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(str(tei_xml).encode("utf-8"), parser=parser)

    paragraph_xpath = (
        "//tei:text//tei:body//*" "[self::tei:p or self::tei:item or self::tei:cell]"
    )
    para_nodes: List[etree._Element] = list(
        root.xpath(paragraph_xpath, namespaces=TEI_NS)
    )

    rows: List[dict] = []
    global_sentence_idx = 0
    for para_idx, para in enumerate(para_nodes):
        sent_nodes: List[etree._Element] = list(
            para.xpath(".//tei:s", namespaces=TEI_NS)
        )
        if sent_nodes:
            for sent_idx, sent in enumerate(sent_nodes):
                text = _text_content(sent).strip()
                if not text:
                    continue
                sentence_id = _xml_id(sent) or f"auto-sent-{global_sentence_idx}"
                rows.append(
                    {
                        "sentence_id": sentence_id,
                        "text": text,
                        "page": _page_label(sent),
                        "section": _section_path(sent),
                        "tei_ids": [sentence_id],
                        "position": global_sentence_idx,
                        "sentence_index": sent_idx,
                        "paragraph_id": _xml_id(para) or f"auto-par-{para_idx}",
                    }
                )
                global_sentence_idx += 1
        else:
            text = _text_content(para).strip()
            if not text:
                continue
            sentence_id = f"auto-sent-{global_sentence_idx}"
            rows.append(
                {
                    "sentence_id": sentence_id,
                    "text": text,
                    "page": _page_label(para),
                    "section": _section_path(para),
                    "tei_ids": [sentence_id],
                    "position": global_sentence_idx,
                    "sentence_index": 0,
                    "paragraph_id": _xml_id(para) or f"auto-par-{para_idx}",
                }
            )
            global_sentence_idx += 1
    return rows


def clear_sentence_cache(attachment_id: Optional[str] = None) -> None:
    if attachment_id is None:
        _SENTENCE_CACHE.clear()
        return
    _SENTENCE_CACHE.pop(str(attachment_id), None)


__all__ = [
    "AttachmentNotFound",
    "STATUS_PENDING",
    "STATUS_CONVERTING",
    "STATUS_PARSING",
    "STATUS_MATCHED",
    "STATUS_READY",
    "STATUS_ERROR",
    "create_attachment_from_bytes",
    "create_attachment",
    "find_attachment_by_content_sha256",
    "get_attachment",
    "list_attachments",
    "set_archived",
    "set_placement",
    "update_attachment",
    "save_artifacts",
    "list_resumable",
    "mark_converting",
    "mark_parsing",
    "mark_matched",
    "mark_ready",
    "mark_error",
    "reset_for_retry",
    "public_status",
    "public_claim_status",
    "load_sentences_for_attachment",
    "clear_sentence_cache",
    "is_ready",
]
