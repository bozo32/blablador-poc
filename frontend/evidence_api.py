"""HTTP helpers for evidence endpoints with client-side concurrency guards."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, MutableMapping, Optional

import requests
import streamlit as st

DEFAULT_TIMEOUT = 45


def _project_headers() -> Dict[str, str]:
    pid = str(st.session_state.get("project_id") or "").strip()
    if not pid:
        pid = "default"
    return {"X-Project-Id": pid}


MAX_LIST_REQUESTS = 2
MAX_TOTAL_CANDIDATES = 50
INFLIGHT_KEY = "_evidence_list_inflight"


class EvidenceApiError(RuntimeError):
    """Raised when evidence API helpers encounter an unrecoverable error."""


class EvidenceDecisionConflict(EvidenceApiError):
    def __init__(
        self,
        message: str,
        *,
        current_version: Optional[int] = None,
        status_code: int = 409,
    ) -> None:
        """Create a conflict error with optional current_version."""
        super().__init__(message)
        self.current_version = current_version
        self.status_code = int(status_code)


def show_api_error(message: str, *, icon: str = "⚠️") -> None:
    """Show API errors via toast when possible, otherwise fall back to st.error."""
    toast = getattr(st, "toast", None)
    if callable(toast):  # pragma: no cover - toast not available in tests
        toast(message, icon=icon)
        return
    st.error(message)


def _session_state() -> MutableMapping[str, Any]:
    return st.session_state


def _api_root() -> str:
    value = _session_state().get("api_url") or "http://localhost:8000"
    return str(value).rstrip("/")


def _auth_headers() -> Dict[str, str]:
    api_key = _session_state().get("api_key")
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _extract_response_detail(response: Optional[requests.Response]) -> Optional[str]:
    if not response:
        return None
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("message")
        if detail:
            return str(detail).strip()
    text = response.text or ""
    return text.strip() or None


def _request(
    method: str,
    path: str,
    *,
    timeout: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Wrap requests.request with shared error handling."""
    url = f"{_api_root()}{path}"
    headers = kwargs.pop("headers", {})
    headers = {**_auth_headers(), **headers}
    if str(path or "").startswith("/attachments"):
        headers = {**_project_headers(), **headers}
    try:
        response = requests.request(
            method,
            url,
            timeout=timeout or DEFAULT_TIMEOUT,
            headers=headers,
            **kwargs,
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network not in tests
        message = str(exc)
        detail = _extract_response_detail(getattr(exc, "response", None))
        if detail:
            message = f"{message} ({detail})"
        show_api_error(f"Evidence service error: {message}")
        raise EvidenceApiError(message) from exc
    if not response.text:
        return {}
    try:
        return response.json()
    except ValueError:  # pragma: no cover - fallback when API returns text bodies
        return {"content": response.text}


def _inflight_state() -> Dict[str, int]:
    return _session_state().setdefault(INFLIGHT_KEY, {})


def _reserve_slot(claim_id: str) -> None:
    inflight = _inflight_state()
    count = inflight.get(claim_id, 0)
    if count >= MAX_LIST_REQUESTS:
        message = "Only two simultaneous evidence fetch requests are allowed per claim."
        show_api_error(message)
        raise EvidenceApiError(message)
    inflight[claim_id] = count + 1


def _release_slot(claim_id: str) -> int:
    inflight = _inflight_state()
    if claim_id not in inflight:
        return MAX_LIST_REQUESTS
    count = max(0, inflight[claim_id] - 1)
    if count:
        inflight[claim_id] = count
    else:
        inflight.pop(claim_id, None)
    return MAX_LIST_REQUESTS - count


def list_evidence(
    claim_id: str,
    *,
    limit: int = 5,
    offset: int = 0,
    label: Optional[str] = None,
    include_neutral: bool = True,
    pinned_only: bool = False,
    claim_text: Optional[str] = None,
    reviewer_uid: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch ranked evidence for a claim with concurrency + metadata helpers."""
    limit = max(1, min(limit, MAX_TOTAL_CANDIDATES))
    _reserve_slot(claim_id)
    remaining_slots = MAX_LIST_REQUESTS
    try:
        params: Dict[str, Any] = {
            "offset": offset,
            "limit": limit,
            "include_neutral": include_neutral,
            "pinned_only": pinned_only,
        }
        if reviewer_uid:
            params["reviewer_uid"] = str(reviewer_uid)
        if label:
            params["label"] = label
        if claim_text:
            params["claim_text"] = claim_text
        payload = _request("get", f"/claims/{claim_id}/evidence", params=params)
    finally:
        remaining_slots = _release_slot(claim_id)

    payload = deepcopy(payload) if payload else {}
    candidates = payload.setdefault("candidates", [])
    total = payload.get("total", len(candidates))
    current_offset = payload.get("offset", offset)
    returned = len(candidates)
    remaining_candidates = max(0, total - (current_offset + returned))
    meta = payload.setdefault("meta", {})
    meta["request_slots"] = {
        "max": MAX_LIST_REQUESTS,
        "remaining": remaining_slots,
    }
    meta["remaining_candidates"] = remaining_candidates
    meta["next_offset"] = current_offset + returned
    return payload


def get_evidence_decisions(
    claim_id: str,
    *,
    reviewer_uid: str,
    events_limit: int = 20,
) -> Dict[str, Any]:
    params = {
        "reviewer_uid": str(reviewer_uid or "default").strip() or "default",
        "events_limit": max(0, min(int(events_limit), 200)),
    }
    return _request("get", f"/claims/{claim_id}/evidence/decisions", params=params)


def append_evidence_decision_event(
    claim_id: str,
    *,
    reviewer_uid: str,
    idempotency_key: str,
    expected_version: int,
    action: str,
    target: Optional[Dict[str, str]] = None,
    set_value: Optional[bool] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    url = f"{_api_root()}/claims/{claim_id}/evidence/decisions/events"
    params = {"reviewer_uid": str(reviewer_uid or "default").strip() or "default"}
    body: Dict[str, Any] = {
        "idempotency_key": str(idempotency_key or "").strip(),
        "expected_version": int(expected_version),
        "action": str(action or "").strip().lower(),
    }
    if target is not None:
        body["target"] = {
            "attachment_id": str((target or {}).get("attachment_id") or "").strip(),
            "span_id": str((target or {}).get("span_id") or "").strip(),
        }
    if set_value is not None:
        body["set"] = bool(set_value)
    if payload is not None:
        body["payload"] = dict(payload)

    headers = {**_auth_headers(), "Content-Type": "application/json"}
    try:
        response = requests.request(
            "post",
            url,
            params=params,
            json=body,
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:  # pragma: no cover
        message = str(exc)
        detail = _extract_response_detail(getattr(exc, "response", None))
        if detail:
            message = f"{message} ({detail})"
        show_api_error(f"Evidence decision error: {message}")
        raise EvidenceApiError(message) from exc

    if response.status_code == 409:
        try:
            payload_obj = response.json()
        except ValueError:
            payload_obj = {}
        current_version = None
        if (
            isinstance(payload_obj, dict)
            and payload_obj.get("current_version") is not None
        ):
            try:
                current_version = int(payload_obj.get("current_version"))
            except Exception:
                current_version = None
        detail = None
        if isinstance(payload_obj, dict):
            detail = payload_obj.get("detail")
        msg = str(detail or "version conflict")
        raise EvidenceDecisionConflict(
            msg,
            current_version=current_version,
            status_code=int(response.status_code),
        )

    try:
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover
        message = str(exc)
        detail = _extract_response_detail(getattr(exc, "response", None))
        if detail:
            message = f"{message} ({detail})"
        show_api_error(f"Evidence decision error: {message}")
        raise EvidenceApiError(message) from exc

    if not response.text:
        return {}
    try:
        return response.json()
    except ValueError:  # pragma: no cover
        return {"content": response.text}


def request_rerun(
    claim_id: str,
    *,
    claim_text: Optional[str] = None,
    note: Optional[str] = None,
    advanced_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "claim_text": claim_text,
        "note": note,
        "advanced_settings": advanced_settings or {},
    }
    return _request(
        "post",
        f"/claims/{claim_id}/evidence/rerun",
        json=payload,
    )


def fetch_history(claim_id: str, *, limit: int = 5) -> Dict[str, Any]:
    params = {"limit": max(1, limit)}
    return _request("get", f"/claims/{claim_id}/evidence/history", params=params)


def export_rank_json(claim_id: str) -> Dict[str, Any]:
    """Download the latest ranking payload for archival/export flows."""
    return _request("get", f"/claims/{claim_id}/evidence/export")


def fetch_span_excerpt(
    attachment_id: str,
    span_id: str,
    *,
    before: int = 2,
    after: int = 1,
) -> Dict[str, Any]:
    params = {"before": max(0, int(before)), "after": max(0, int(after))}
    return _request(
        "get",
        f"/attachments/{attachment_id}/spans/{span_id}/excerpt",
        params=params,
    )


def fetch_span_jump(attachment_id: str, span_id: str) -> Dict[str, Any]:
    return _request("get", f"/attachments/{attachment_id}/spans/{span_id}/jump")


def jump_to_pdf_span(attachment_id: str, span_id: str) -> Dict[str, Any]:
    """Return viewer metadata for jumping to a PDF span."""
    return fetch_span_jump(attachment_id, span_id)


def get_evidence_selection(claim_id: str) -> Dict[str, Any]:
    return _request("get", f"/claims/{claim_id}/evidence/selection")


def put_evidence_selection(claim_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return _request("put", f"/claims/{claim_id}/evidence/selection", json=payload)


__all__ = [
    "EvidenceApiError",
    "EvidenceDecisionConflict",
    "INFLIGHT_KEY",
    "MAX_LIST_REQUESTS",
    "list_evidence",
    "request_rerun",
    "fetch_history",
    "export_rank_json",
    "fetch_span_excerpt",
    "fetch_span_jump",
    "jump_to_pdf_span",
    "get_evidence_selection",
    "put_evidence_selection",
    "get_evidence_decisions",
    "append_evidence_decision_event",
    "show_api_error",
]
