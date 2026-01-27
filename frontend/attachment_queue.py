"""State helpers for the evidence attachment queue UI."""

from __future__ import annotations

import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import streamlit as st

from frontend import claim_queue


QUEUE_KEY = "attachment_queue"
TMP_DIR_KEY = "attachment_tmp_dir"
STATUS_FLOW = ("pending", "converting", "parsing", "matched")
DEFAULT_STATUSES = ("pending", "converting", "parsing", "matched", "error")


@dataclass
class QueueItem:
    id: str
    filename: str
    size: Optional[int]
    local_path: str
    status: str = "pending"
    claim_id: Optional[str] = None
    doc_id: Optional[str] = None
    source: str = "drop"
    reference_hint: Optional[dict] = None
    uploaded_at: str = ""
    uploaded_epoch: float = 0.0
    next_transition: Optional[float] = None
    history: Optional[List[dict]] = None
    ambiguous_matches: Optional[List[str]] = None
    errors: Optional[List[str]] = None


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
        item_id = str(uuid.uuid4())
        local_path = _persist_file(file_obj)
        queue_item = QueueItem(
            id=item_id,
            filename=file_obj.name,
            size=getattr(file_obj, "size", None),
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
        )
        item_dict = queue_item.__dict__
        queue["items"][item_id] = item_dict
        queue["order"].insert(0, item_id)
        _log_item_history(item_dict, "queued", detail=source)
        created_ids.append(item_id)
    refresh_summary_counts()
    return created_ids


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
    """Simulate async processing by advancing statuses over time."""
    queue = _queue_state()
    now = time.time()
    for item in queue["items"].values():
        if item["status"] in {"matched", "error"}:
            continue
        threshold = item.get("next_transition")
        if threshold is None or now < threshold:
            continue
        current_index = STATUS_FLOW.index(item["status"])
        if current_index < len(STATUS_FLOW) - 1:
            new_status = STATUS_FLOW[current_index + 1]
            item["status"] = new_status
            item["next_transition"] = now + 2
            _log_item_history(item, f"advanced_to_{new_status}")
        else:
            item["next_transition"] = None
    refresh_summary_counts()


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
        summary[item.get("status", "pending")] += 1
    queue["summary"] = summary


def auto_match_queue_items(min_score: float = 0.6) -> None:
    """Attempt to auto-match queued files to claims via claim_queue heuristics."""
    queue = _queue_state()
    for item_id in list(queue["order"]):
        item = queue["items"].get(item_id)
        if not item:
            continue
        if item.get("claim_id") or item.get("status") in {"matched", "error"}:
            continue
        matches = claim_queue.auto_match_claim(item)
        if not matches:
            continue
        confident = [m for m in matches if m["score"] >= min_score]
        if len(confident) == 1:
            attach_to_claim(
                confident[0]["id"], item_id, via="auto", score=confident[0]["score"]
            )
            continue
        item["ambiguous_matches"] = [match["id"] for match in matches]
        _log_item_history(item, "ambiguous", detail="needs_manual_resolution")
    refresh_summary_counts()


def attach_to_claim(
    claim_id: str,
    queue_item_id: str,
    *,
    via: str = "manual",
    score: Optional[float] = None,
) -> None:
    queue = _queue_state()
    item = queue["items"].get(queue_item_id)
    if not item:
        return
    item["claim_id"] = claim_id
    item["status"] = "matched"
    item["next_transition"] = None
    item["ambiguous_matches"] = []
    if score is not None:
        item["match_score"] = round(score, 3)
    _log_item_history(item, "matched", detail=via)
    claim_queue.record_timeline_event(
        claim_id,
        "attached",
        {"filename": item["filename"], "method": via},
    )
    refresh_summary_counts()


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
    errors = summary.get("error", 0)
    matched = summary.get("matched", 0)
    return f"Queue • {pending} pending • {matched} matched" + (
        f" • {errors} errors" if errors else ""
    )


def get_queue_items() -> List[dict]:
    queue = _queue_state()
    items: List[dict] = []
    for item_id in queue["order"]:
        item = queue["items"].get(item_id)
        if item:
            items.append(item)
    return items
