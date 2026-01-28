"""Attachment persistence utilities for claim evidence uploads."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from uuid import uuid4

from backend.settings import settings


STATUS_PENDING = "pending"
STATUS_CONVERTING = "converting"
STATUS_PARSING = "parsing"
STATUS_MATCHED = "matched"
STATUS_READY = STATUS_MATCHED  # Backward-compatible alias
STATUS_ERROR = "error"

MAX_TIMELINE_EVENTS = 5
MAX_EVENTS_STORED = 25
DEFAULT_MAX_ATTEMPTS = 2


_SENTENCE_CACHE: dict[str, Tuple[float, List[dict]]] = {}


class AttachmentNotFound(RuntimeError):
    """Raised when an attachment id cannot be resolved."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _root() -> Path:
    root = Path(settings.ATTACHMENT_DIR)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _attachment_dir(attachment_id: str) -> Path:
    return _root() / attachment_id


def _metadata_path(attachment_id: str) -> Path:
    return _attachment_dir(attachment_id) / "metadata.json"


def _load_record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _log_event(record: dict, event: str, detail: Optional[str] = None) -> None:
    timeline: List[dict] = record.setdefault("timeline", [])
    timeline.insert(0, {"event": event, "detail": detail, "at": _now()})
    record["timeline"] = timeline[:MAX_EVENTS_STORED]


def _write_record(record: dict) -> dict:
    meta_path = _metadata_path(record["id"])
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record


def _sanitize_filename(filename: str) -> str:
    name = filename.strip() or "attachment.pdf"
    return name.replace("/", "_").replace("\\", "_")


def _normalize_status(value: Optional[str]) -> str:
    if value == "ready":  # legacy persisted value
        return STATUS_MATCHED
    if value in {
        STATUS_PENDING,
        STATUS_CONVERTING,
        STATUS_PARSING,
        STATUS_MATCHED,
        STATUS_ERROR,
    }:
        return value
    return STATUS_PENDING


def _public_view(record: dict) -> dict:
    data = dict(record)
    data.pop("file_path", None)
    data["status"] = _normalize_status(data.get("status"))
    data["history"] = data.get("timeline", [])[:MAX_TIMELINE_EVENTS]
    data["timeline"] = data["history"]
    data.setdefault("reference_hint", {})
    data["retry_available"] = (
        data.get("attempts", 0) < data.get("max_attempts", DEFAULT_MAX_ATTEMPTS)
        and data.get("status") != STATUS_MATCHED
    )
    return data


def is_ready(record: dict) -> bool:
    """Return True if the attachment record is ready for evidence matching."""
    return _normalize_status(record.get("status")) == STATUS_MATCHED


def create_attachment(
    *,
    claim_id: str,
    doc_id: Optional[str],
    local_path: str | Path,
    filename: Optional[str] = None,
    size_bytes: Optional[int] = None,
    reference_hint: Optional[dict] = None,
) -> dict:
    source_path = Path(local_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Attachment source not found: {source_path}")

    attachment_id = str(uuid4())
    dest_dir = _attachment_dir(attachment_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_filename(filename or source_path.name)
    dest_path = dest_dir / safe_name
    shutil.copy2(source_path, dest_path)

    size = size_bytes if size_bytes is not None else dest_path.stat().st_size
    now = _now()
    record = {
        "id": attachment_id,
        "claim_id": claim_id,
        "doc_id": doc_id,
        "filename": safe_name,
        "size": size,
        "status": STATUS_PENDING,
        "error": None,
        "uploaded_at": now,
        "updated_at": now,
        "parsed_at": None,
        "timeline": [],
        "file_path": str(dest_path),
        "reference_hint": reference_hint or {},
        "attempts": 0,
        "max_attempts": DEFAULT_MAX_ATTEMPTS,
        "artifacts": {},
    }
    _log_event(record, "queued", "Attachment received")
    return _write_record(record)


def get_attachment(attachment_id: str, public: bool = False) -> Optional[dict]:
    meta_path = _metadata_path(attachment_id)
    if not meta_path.exists():
        return None
    record = _load_record(meta_path)
    return _public_view(record) if public else record


def list_attachments(
    claim_id: Optional[str] = None, public: bool = False
) -> List[dict]:
    records: List[dict] = []
    for meta_path in _root().glob("*/metadata.json"):
        record = _load_record(meta_path)
        if claim_id and record.get("claim_id") != claim_id:
            continue
        records.append(_public_view(record) if public else record)
    records.sort(key=lambda rec: rec.get("uploaded_at", ""), reverse=True)
    return records


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
    record = get_attachment(attachment_id)
    if record is None:
        raise AttachmentNotFound(f"Attachment {attachment_id} not found")
    if status:
        record["status"] = status
    if error is not None:
        record["error"] = error
    if parsed_at is not None:
        record["parsed_at"] = parsed_at
    if artifacts:
        record["artifacts"] = artifacts
    record.update(extra_fields)
    record["updated_at"] = _now()
    if timeline_event:
        _log_event(record, timeline_event, timeline_detail)
    return _write_record(record)


def save_artifacts(
    attachment_id: str,
    *,
    tei_xml: str,
    tei_json: dict,
    sentences: Iterable[dict],
) -> dict:
    record = get_attachment(attachment_id)
    if record is None:
        raise AttachmentNotFound(f"Attachment {attachment_id} not found")
    dest_dir = _attachment_dir(attachment_id)
    tei_xml_path = dest_dir / "tei.xml"
    tei_json_path = dest_dir / "tei.json"
    sentences_path = dest_dir / "sentences.ndjson"

    tei_xml_path.write_text(tei_xml, encoding="utf-8")
    tei_json_path.write_text(json.dumps(tei_json, indent=2), encoding="utf-8")
    with sentences_path.open("w", encoding="utf-8") as handle:
        for sentence in sentences:
            handle.write(json.dumps(sentence) + "\n")

    return {
        "tei_xml": str(tei_xml_path),
        "tei_json": str(tei_json_path),
        "sentences": str(sentences_path),
    }


def list_resumable(statuses: Optional[Iterable[str]] = None) -> List[dict]:
    wanted = set(statuses or {STATUS_PENDING, STATUS_CONVERTING, STATUS_PARSING})
    return [
        record
        for record in list_attachments(public=False)
        if _normalize_status(record.get("status")) in wanted
    ]


def mark_converting(attachment_id: str, attempt: int) -> dict:
    return update_attachment(
        attachment_id,
        status=STATUS_CONVERTING,
        timeline_event="converting",
        timeline_detail=f"Attempt {attempt}",
        attempts=attempt,
    )


def mark_parsing(attachment_id: str, attempt: int) -> dict:
    return update_attachment(
        attachment_id,
        status=STATUS_PARSING,
        timeline_event="parsing",
        timeline_detail=f"Attempt {attempt}",
        attempts=attempt,
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
    record = get_attachment(attachment_id)
    if record is None:
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
    """Load persisted sentence rows for an attachment."""
    record = get_attachment(attachment_id)
    if record is None:
        raise AttachmentNotFound(f"Attachment {attachment_id} not found")
    if not is_ready(record):
        raise RuntimeError(
            f"Attachment {attachment_id} is not ready (status={record.get('status')})"
        )
    artifacts = record.get("artifacts") or {}
    sentences_path = artifacts.get("sentences")
    if not sentences_path:
        raise FileNotFoundError(
            f"Attachment {attachment_id} is missing persisted sentences"
        )
    path = Path(sentences_path)
    if not path.exists():
        raise FileNotFoundError(path)
    mtime = path.stat().st_mtime
    if use_cache:
        cached = _SENTENCE_CACHE.get(attachment_id)
        if cached and cached[0] == mtime:
            return [dict(row) for row in cached[1]]
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    _SENTENCE_CACHE[attachment_id] = (mtime, rows)
    return [dict(row) for row in rows]


def clear_sentence_cache(attachment_id: Optional[str] = None) -> None:
    """Invalidate the in-memory sentence cache."""
    if attachment_id is None:
        _SENTENCE_CACHE.clear()
        return
    _SENTENCE_CACHE.pop(attachment_id, None)
