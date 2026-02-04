"""State helpers for the evidence attachment queue UI."""

from __future__ import annotations

import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, cast

import requests
import streamlit as st

from frontend import claim_queue


QUEUE_KEY = "attachment_queue"
TMP_DIR_KEY = "attachment_tmp_dir"
SHOW_ARCHIVED_KEY = "attachment_queue_show_archived"
SHOW_HISTORY_KEY = "attachment_queue_show_history"
SESSION_ATTACHMENT_IDS_KEY = "attachment_queue_session_attachment_ids"
STATUS_FLOW = ("pending", "converting", "parsing", "matched")
DEFAULT_STATUSES = ("pending", "converting", "parsing", "matched", "error")
API_TIMEOUT = 15


@dataclass
class QueueItem:
    id: str
    filename: str
    size: Optional[int]
    local_path: str
    status: str = "pending"
    claim_id: Optional[str] = None
    doc_id: Optional[str] = None
    citation_index: Optional[int] = None
    target_id: Optional[str] = None
    source: str = "drop"
    reference_hint: Optional[dict] = None
    uploaded_at: str = ""
    uploaded_epoch: float = 0.0
    next_transition: Optional[float] = None
    history: Optional[List[dict]] = None
    ambiguous_matches: Optional[List[str]] = None
    errors: Optional[List[str]] = None
    attachment_id: Optional[str] = None
    archived: bool = False


def _normalize_status(value: Optional[str]) -> str:
    if not value:
        return "pending"
    if value == "ready":
        return "matched"
    return value


def init_attachment_queue_state() -> None:
    """Ensure session state keys exist for the attachment queue."""
    if QUEUE_KEY not in st.session_state:
        st.session_state[QUEUE_KEY] = {
            "items": {},
            "order": [],
            "panel_open": True,
            "active_modal_claim": None,
            "active_drop_target": None,
            "summary": {status: 0 for status in DEFAULT_STATUSES},
        }
    st.session_state.setdefault(SHOW_ARCHIVED_KEY, False)
    st.session_state.setdefault(SHOW_HISTORY_KEY, False)
    st.session_state.setdefault(SESSION_ATTACHMENT_IDS_KEY, [])
    if TMP_DIR_KEY not in st.session_state:
        tmp_dir = tempfile.mkdtemp(prefix="attachments-")
        st.session_state[TMP_DIR_KEY] = tmp_dir


def _queue_state() -> dict:
    init_attachment_queue_state()
    return st.session_state[QUEUE_KEY]


def _tmp_dir() -> Path:
    init_attachment_queue_state()
    return Path(st.session_state[TMP_DIR_KEY])


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _log_item_history(item: dict, event: str, detail: Optional[str] = None) -> None:
    history = item.setdefault("history", [])
    history.insert(0, {"event": event, "detail": detail, "at": _now()})
    del history[15:]


def _persist_file(uploaded_file) -> str:
    tmp_dir = _tmp_dir()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest = tmp_dir / f"{uuid.uuid4()}_{uploaded_file.name}"
    with open(dest, "wb") as fh:
        fh.write(uploaded_file.getbuffer())
    return str(dest)


def _api_url() -> str:
    api_url = st.session_state.get("api_url") or "http://localhost:8000"
    return api_url.rstrip("/")


def _request(method: str, path: str, **kwargs) -> Optional[dict]:
    url = f"{_api_url()}{path}"
    try:
        response = requests.request(method, url, timeout=API_TIMEOUT, **kwargs)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - UI only
        raise RuntimeError(f"Attachment API error: {exc}") from exc
    if not response.text:
        return None
    return response.json()


def get_show_archived() -> bool:
    init_attachment_queue_state()
    return bool(st.session_state.get(SHOW_ARCHIVED_KEY, False))


def set_show_archived(value: bool) -> None:
    init_attachment_queue_state()
    st.session_state[SHOW_ARCHIVED_KEY] = bool(value)


def get_show_history() -> bool:
    """Whether to include persisted attachments from previous sessions."""
    init_attachment_queue_state()
    return bool(st.session_state.get(SHOW_HISTORY_KEY, False))


def set_show_history(value: bool) -> None:
    init_attachment_queue_state()
    st.session_state[SHOW_HISTORY_KEY] = bool(value)


def clear_session_state() -> None:
    """Clear in-memory Source bin items without touching backend data."""
    init_attachment_queue_state()
    queue = st.session_state.get(QUEUE_KEY) or {}
    queue["items"] = {}
    queue["order"] = []
    queue["summary"] = {status: 0 for status in DEFAULT_STATUSES}
    st.session_state[SESSION_ATTACHMENT_IDS_KEY] = []


def _hydrate_from_backend(item: dict, payload: dict) -> None:
    if not payload:
        return
    previous_status = item.get("status")
    history = payload.get("history") or payload.get("timeline") or []
    item["attachment_id"] = payload.get("id")
    item["status"] = _normalize_status(
        payload.get("status", item.get("status", "pending"))
    )
    item["history"] = history[:5]
    item["timeline"] = item["history"]
    item["error"] = payload.get("error")
    item["doc_id"] = payload.get("doc_id") or item.get("doc_id")
    item["citation_index"] = payload.get("citation_index")
    item["target_id"] = payload.get("target_id")
    item["reference_hint"] = payload.get("reference_hint") or item.get("reference_hint")
    item["filename"] = payload.get("filename") or item.get("filename")
    item["size"] = payload.get("size") or item.get("size")
    item["archived"] = bool(payload.get("archived") or False)
    item["backend_details"] = payload
    claim_id = cast(Optional[str], payload.get("claim_id"))
    if claim_id and previous_status != "matched" and item.get("status") == "matched":
        claim_queue.record_timeline_event(
            claim_id,
            "matched",
            {"attachment_id": payload.get("id")},
        )


def _remember_session_attachment_id(attachment_id: Optional[str]) -> None:
    if not attachment_id:
        return
    init_attachment_queue_state()
    ids = st.session_state.setdefault(SESSION_ATTACHMENT_IDS_KEY, [])
    if attachment_id not in ids:
        ids.append(attachment_id)


def _find_by_attachment_id(attachment_id: Optional[str]) -> Optional[dict]:
    if not attachment_id:
        return None
    queue = _queue_state()
    for item in queue["items"].values():
        if item.get("attachment_id") == attachment_id:
            return item
    return None


def _apply_backend_payload(payload: dict) -> dict:
    queue = _queue_state()
    existing = _find_by_attachment_id(payload.get("id"))
    if not existing:
        item_id = payload.get("id") or str(uuid.uuid4())
        queue_item = QueueItem(
            id=item_id,
            filename=payload.get("filename") or payload.get("id") or "attachment.pdf",
            size=payload.get("size"),
            local_path="",
            status=_normalize_status(payload.get("status", "pending")),
            claim_id=payload.get("claim_id"),
            doc_id=payload.get("doc_id"),
            citation_index=payload.get("citation_index"),
            target_id=payload.get("target_id"),
            source="backend",
            reference_hint=payload.get("reference_hint"),
            uploaded_at=payload.get("uploaded_at", _now()),
            uploaded_epoch=time.time(),
            next_transition=None,
            history=[],
            ambiguous_matches=[],
            errors=[],
            attachment_id=payload.get("id"),
            archived=bool(payload.get("archived") or False),
        ).__dict__
        queue["items"][item_id] = queue_item
        queue["order"].insert(0, item_id)
        existing = queue_item
    _hydrate_from_backend(existing, payload)
    return existing


def _upload_to_backend(queue_item_id: str) -> None:
    queue = _queue_state()
    item = queue["items"].get(queue_item_id)
    if not item or not item.get("local_path"):
        return
    payload = {
        "claim_id": item.get("claim_id"),
        "doc_id": item.get("doc_id"),
        "citation_index": item.get("citation_index"),
        "target_id": item.get("target_id"),
        "filename": item.get("filename"),
        "local_path": item.get("local_path"),
        "size_bytes": item.get("size"),
        "reference_hint": item.get("reference_hint"),
    }
    claim_id = item.get("claim_id")
    if claim_id:
        # Include claim text when available so backend has context.
        record = claim_queue.get_claim_record(claim_id) or {}
        claim_text = record.get("claim") or ""
        if claim_text.strip():
            payload["claim_text"] = claim_text.strip()
    try:
        response = _request("post", "/attachments", json=payload)
    except RuntimeError as exc:
        mark_item_error(queue_item_id, str(exc))
        return
    attachment = (response or {}).get("attachment")
    if attachment:
        _hydrate_from_backend(item, attachment)
        item["status"] = _normalize_status(item.get("status"))
        _remember_session_attachment_id(item.get("attachment_id"))
    refresh_summary_counts()


def sync_backend_state() -> None:
    queue = _queue_state()
    archived = get_show_archived()
    include_history = get_show_history()

    if include_history:
        try:
            payload = _request("get", f"/attachments?archived={str(archived).lower()}")
        except RuntimeError:
            return
        attachments = (payload or {}).get("attachments") or []
        for record in attachments:
            _apply_backend_payload(record)
        refresh_summary_counts()
        auto_match_queue_items()
        return

    # Clean start: only sync attachments that this Streamlit session knows about.
    session_ids = set(st.session_state.get(SESSION_ATTACHMENT_IDS_KEY) or [])
    for item in (queue.get("items") or {}).values():
        attachment_id = (item or {}).get("attachment_id")
        if attachment_id:
            session_ids.add(str(attachment_id))

    for attachment_id in sorted(session_ids):
        try:
            payload = _request("get", f"/attachments/{attachment_id}")
        except RuntimeError:
            continue
        attachment = (payload or {}).get("attachment")
        if attachment:
            _apply_backend_payload(attachment)
    refresh_summary_counts()
    auto_match_queue_items()


def archive_all_active() -> int:
    """Archive all non-archived attachments (explicit user action)."""
    try:
        payload = _request("get", "/attachments?archived=false")
    except RuntimeError:
        return 0
    attachments = (payload or {}).get("attachments") or []
    archived = 0
    for record in attachments:
        attachment_id = (record or {}).get("id")
        if not attachment_id:
            continue
        try:
            _ = _request(
                "patch",
                f"/attachments/{attachment_id}",
                json={"archived": True},
            )
        except RuntimeError:
            continue
        archived += 1
    return archived


def has_inflight_jobs() -> bool:
    queue = _queue_state()
    return any(
        item.get("status") in {"pending", "converting", "parsing"}
        for item in queue["items"].values()
    )


def retry_attachment(queue_item_id: str) -> None:
    queue = _queue_state()
    item = queue["items"].get(queue_item_id)
    attachment_id = (item or {}).get("attachment_id")
    if not attachment_id:
        return
    try:
        response = _request("post", f"/attachments/{attachment_id}/retry")
    except RuntimeError as exc:
        mark_item_error(queue_item_id, str(exc))
        return
    attachment = (response or {}).get("attachment")
    if attachment and item:
        _hydrate_from_backend(item, attachment)
    refresh_summary_counts()


def bulk_retry_failed() -> int:
    """Retry all failed attachments currently in session."""
    queue = _queue_state()
    retried = 0
    for item_id in list(queue.get("order") or []):
        item = queue["items"].get(item_id)
        if not item:
            continue
        if bool(item.get("archived")) and not get_show_archived():
            continue
        if item.get("status") != "error":
            continue
        if not item.get("attachment_id"):
            continue
        retry_attachment(item_id)
        retried += 1
    return retried


def enqueue_files(
    files: Iterable,
    *,
    source_claim_id: Optional[str] = None,
    doc_id: Optional[str] = None,
    reference_hint: Optional[dict] = None,
    source: str = "drop",
) -> List[str]:
    """Add uploaded files to the attachment queue."""
    queue = _queue_state()
    created_ids: List[str] = []
    for file_obj in files or []:
        if file_obj is None:
            continue

        filename = getattr(file_obj, "name", None) or "attachment.pdf"
        size = getattr(file_obj, "size", None)
        if _is_duplicate_upload(filename, size):
            continue

        item_id = str(uuid.uuid4())
        local_path = _persist_file(file_obj)
        queue_item = QueueItem(
            id=item_id,
            filename=filename,
            size=size,
            local_path=local_path,
            claim_id=source_claim_id,
            doc_id=doc_id,
            source=source,
            reference_hint=reference_hint or {},
            uploaded_at=_now(),
            uploaded_epoch=time.time(),
            next_transition=time.time() + 2,
            history=[],
            ambiguous_matches=[],
            errors=[],
            archived=False,
        )
        item_dict = queue_item.__dict__
        queue["items"][item_id] = item_dict
        queue["order"].insert(0, item_id)
        _log_item_history(item_dict, "queued", detail=source)
        created_ids.append(item_id)

        # Global upload starts processing immediately.
        _upload_to_backend(item_id)
    refresh_summary_counts()
    auto_match_queue_items()
    return created_ids


def _is_duplicate_upload(filename: str, size: Optional[int]) -> bool:
    if not filename or size is None:
        return False
    queue = _queue_state()
    for item in queue.get("items", {}).values():
        if bool(item.get("archived")):
            continue
        if item.get("filename") == filename and item.get("size") == size:
            return True
    return False


def handle_drop(
    claim_id: Optional[str],
    files: Iterable,
    *,
    doc_id: Optional[str] = None,
    reference_hint: Optional[dict] = None,
    source: str = "drop",
) -> List[str]:
    """Process files dropped onto a claim card."""
    queue = _queue_state()
    queue["active_drop_target"] = claim_id
    created = enqueue_files(
        files,
        source_claim_id=claim_id,
        doc_id=doc_id,
        reference_hint=reference_hint,
        source=source,
    )
    # If the caller provides a claim_id, treat it as an immediate placement.
    for queue_item_id in created:
        if claim_id:
            place_attachment(
                queue_item_id,
                claim_id,
                doc_id=doc_id,
                target_id=(reference_hint or {}).get("reference_id"),
                via=source,
            )
    if claim_id:
        claim_queue.record_timeline_event(
            claim_id,
            "dropped",
            {
                "count": len(created),
                "source": source,
            },
        )
    return created


def advance_inflight_items() -> None:
    """Refresh queue state from backend."""
    sync_backend_state()


def get_queue_snapshot() -> dict:
    queue = _queue_state()
    return {
        "items": queue["items"],
        "order": queue["order"],
        "panel_open": queue["panel_open"],
        "summary": queue.get("summary", {}),
        "active_modal_claim": queue.get("active_modal_claim"),
    }


def refresh_summary_counts() -> None:
    queue = _queue_state()
    summary: Dict[str, int] = {status: 0 for status in DEFAULT_STATUSES}
    for item in queue["items"].values():
        if bool(item.get("archived")) and not get_show_archived():
            continue
        status = _normalize_status(item.get("status"))
        summary.setdefault(status, 0)
        summary[status] += 1
    queue["summary"] = summary


def auto_match_queue_items(min_score: float = 0.90) -> None:
    queue = _queue_state()
    updated = False
    for item_id, item in queue["items"].items():
        if bool(item.get("archived")):
            continue
        if item.get("claim_id") or not item.get("filename"):
            continue
        candidates = claim_queue.auto_match_claim(item)
        if not candidates:
            continue
        top = candidates[0]
        if top.get("score", 0) >= min_score:
            claim_record = claim_queue.get_claim_record(top["id"]) or {}
            place_attachment(
                item_id,
                top["id"],
                doc_id=claim_record.get("doc_id"),
                via="auto",
                score=top.get("score"),
            )
            _log_item_history(item, "auto-placed", detail=top["id"])
            updated = True
            continue
        _flag_ambiguous(item, candidates)
        updated = True
    if updated:
        refresh_summary_counts()


def _flag_ambiguous(item: dict, candidates: List[dict]) -> None:
    shortlist: List[dict] = []
    for candidate in candidates:
        record = candidate.get("record") or {}
        shortlist.append(
            {
                "id": candidate.get("id"),
                "score": candidate.get("score"),
                "callout": record.get("callout"),
                "claim": record.get("claim"),
            }
        )
    item["ambiguous_matches"] = shortlist
    item["status"] = "pending"
    _log_item_history(
        item,
        "ambiguous",
        detail=", ".join(match.get("id") or "?" for match in shortlist),
    )


def place_attachment(
    queue_item_id: str,
    claim_id: str,
    *,
    doc_id: Optional[str] = None,
    citation_index: Optional[int] = None,
    target_id: Optional[str] = None,
    via: str = "manual",
    score: Optional[float] = None,
) -> None:
    """Assign/re-place a source-bin item onto a claim via PATCH /attachments/{id}."""
    queue = _queue_state()
    item = queue["items"].get(queue_item_id)
    if not item:
        return
    if bool(item.get("archived")):
        return

    # Ensure we have a backend record to place.
    if not item.get("attachment_id"):
        _upload_to_backend(queue_item_id)
    attachment_id = item.get("attachment_id")
    if not attachment_id:
        mark_item_error(queue_item_id, "Attachment upload missing backend id")
        return

    item["claim_id"] = claim_id
    item["ambiguous_matches"] = []
    if score is not None:
        item["match_score"] = round(float(score), 3)
    if doc_id is not None:
        item["doc_id"] = doc_id
    if citation_index is not None:
        item["citation_index"] = citation_index
    if target_id is not None:
        item["target_id"] = target_id

    patch_payload = {
        "claim_id": claim_id,
        "doc_id": item.get("doc_id"),
        "citation_index": item.get("citation_index"),
        "target_id": item.get("target_id"),
    }
    try:
        response = _request(
            "patch", f"/attachments/{attachment_id}", json=patch_payload
        )
    except RuntimeError as exc:
        mark_item_error(queue_item_id, str(exc))
        return
    attachment = (response or {}).get("attachment")
    if attachment:
        _hydrate_from_backend(item, attachment)
    _log_item_history(item, "placed", detail=f"{via}:{claim_id}")
    claim_queue.record_timeline_event(
        claim_id,
        "source-placed",
        {"filename": item.get("filename"), "method": via},
    )
    _trigger_evidence_rerun(claim_id, note="auto-placement")
    refresh_summary_counts()


# Backwards-compatible alias.
attach_to_claim = place_attachment


def detach_attachment(claim_id: str, queue_item_id: Optional[str] = None) -> None:
    queue = _queue_state()
    targets = []
    if queue_item_id:
        targets.append(queue_item_id)
    else:
        for candidate_id, candidate in queue["items"].items():
            if candidate.get("claim_id") == claim_id:
                targets.append(candidate_id)
    for target in targets:
        item = queue["items"].get(target)
        if not item:
            continue
        item["claim_id"] = None
        item["status"] = "pending"
        item["next_transition"] = time.time() + 2
        _log_item_history(item, "detached", detail=claim_id)
        claim_queue.record_timeline_event(
            claim_id,
            "detached",
            {"filename": item["filename"]},
        )
    refresh_summary_counts()


def archive_attachment(queue_item_id: str, *, archived: bool) -> None:
    queue = _queue_state()
    item = queue["items"].get(queue_item_id)
    if not item:
        return
    attachment_id = item.get("attachment_id")
    if not attachment_id:
        return
    try:
        response = _request(
            "patch", f"/attachments/{attachment_id}", json={"archived": bool(archived)}
        )
    except RuntimeError as exc:
        mark_item_error(queue_item_id, str(exc))
        return
    attachment = (response or {}).get("attachment")
    if attachment:
        _hydrate_from_backend(item, attachment)
    item["archived"] = bool(archived)
    _log_item_history(item, "archived" if archived else "unarchived")
    refresh_summary_counts()


def get_claim_attachment(claim_id: str) -> Optional[dict]:
    queue = _queue_state()
    for item in queue["items"].values():
        if item.get("claim_id") == claim_id:
            return item
    return None


def mark_item_error(queue_item_id: str, message: str) -> None:
    queue = _queue_state()
    item = queue["items"].get(queue_item_id)
    if not item:
        return
    item["status"] = "error"
    item.setdefault("errors", []).append({"message": message, "at": _now()})
    _log_item_history(item, "error", detail=message)
    refresh_summary_counts()


def remove_queue_item(queue_item_id: str) -> None:
    queue = _queue_state()
    if queue_item_id in queue["items"]:
        del queue["items"][queue_item_id]
    if queue_item_id in queue["order"]:
        queue["order"].remove(queue_item_id)
    refresh_summary_counts()


def open_attachment_modal(claim_id: Optional[str]) -> None:
    queue = _queue_state()
    queue["active_modal_claim"] = claim_id


def close_attachment_modal() -> None:
    queue = _queue_state()
    queue["active_modal_claim"] = None


def get_modal_claim_id() -> Optional[str]:
    queue = _queue_state()
    return queue.get("active_modal_claim")


def toggle_panel(open_panel: bool) -> None:
    queue = _queue_state()
    queue["panel_open"] = open_panel


def set_active_drop_target(claim_id: Optional[str]) -> None:
    queue = _queue_state()
    queue["active_drop_target"] = claim_id


def clear_active_drop_target() -> None:
    set_active_drop_target(None)


def get_active_drop_target() -> Optional[str]:
    queue = _queue_state()
    return queue.get("active_drop_target")


def collapse_when_idle() -> None:
    queue = _queue_state()
    summary = queue.get("summary", {})
    pending = (
        summary.get("pending", 0)
        + summary.get("converting", 0)
        + summary.get("parsing", 0)
    )
    if pending == 0 and queue.get("panel_open"):
        queue["panel_open"] = False


def ensure_open_when_activity() -> None:
    queue = _queue_state()
    summary = queue.get("summary", {})
    if summary.get("pending", 0) or summary.get("converting", 0):
        queue["panel_open"] = True


def summarize_chip_label() -> str:
    queue = _queue_state()
    summary = queue.get("summary", {})
    pending = summary.get("pending", 0)
    converting = summary.get("converting", 0)
    errors = summary.get("error", 0)
    matched = summary.get("matched", 0)
    parts = [f"Queue • {pending} pending"]
    if converting:
        parts.append(f"{converting} converting")
    parts.append(f"{matched} matched")
    if errors:
        parts.append(f"{errors} errors")
    return " • ".join(parts)


def summarize_counts() -> Dict[str, int]:
    return dict(_queue_state().get("summary", {}))


def get_queue_items() -> List[dict]:
    queue = _queue_state()
    items: List[dict] = []
    for item_id in queue["order"]:
        item = queue["items"].get(item_id)
        if item:
            if bool(item.get("archived")) and not get_show_archived():
                continue
            items.append(item)
    return items


def _trigger_evidence_rerun(claim_id: str, *, note: str) -> None:
    if not claim_id or str(claim_id).strip().lower() == "none":
        return
    record = claim_queue.get_claim_record(claim_id) or {}
    claim_text = (record.get("claim") or "").strip()
    if not claim_text:
        return
    profile = (st.session_state.get("execution_profile") or "").strip()
    advanced_settings = {"profile": profile} if profile else {}
    try:
        _ = _request(
            "post",
            f"/claims/{claim_id}/evidence/rerun",
            json={
                "claim_text": claim_text,
                "note": note,
                "advanced_settings": advanced_settings,
            },
        )
    except RuntimeError:
        # Quiet failure: evidence view will show errors inline.
        return
