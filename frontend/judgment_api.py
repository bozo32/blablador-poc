"""HTTP helpers for judgment CRUD/list/export endpoints.

Modeled after frontend.evidence_api so the Streamlit UI can keep judgment wiring
isolated and testable.
"""

from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional

import requests
import streamlit as st

DEFAULT_TIMEOUT = 45


class JudgmentApiError(RuntimeError):
    """Raised when judgment API helpers encounter an unrecoverable error."""


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


def _normalize_reviewer_uid(value: Optional[str]) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    return text


def _strict_scope_headers(*, reviewer_uid: Optional[str] = None) -> Dict[str, str]:
    pid = str(_session_state().get("project_id") or "").strip()
    if not pid:
        raise JudgmentApiError("judgment request requires project_id")

    session_uid = str(_session_state().get("active_reviewer_uid") or "").strip()
    reviewer = _normalize_reviewer_uid(reviewer_uid) or session_uid
    user_id = session_uid or reviewer
    if not user_id:
        raise JudgmentApiError("judgment request requires user_id")
    if not reviewer:
        raise JudgmentApiError("judgment request requires reviewer_uid")
    return {
        "X-Project-Id": pid,
        "X-User-Id": user_id,
        "X-Reviewer-Uid": reviewer,
    }


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
        show_api_error(f"Judgment service error: {message}")
        raise JudgmentApiError(message) from exc

    if not response.text:
        return {}
    try:
        return response.json()
    except ValueError:  # pragma: no cover - fallback when API returns text bodies
        return {"content": response.text}


def get_judgment(
    claim_id: str, *, reviewer_uid: Optional[str] = None
) -> Dict[str, Any]:
    reviewer = _normalize_reviewer_uid(reviewer_uid)
    return _request(
        "get",
        f"/claims/{claim_id}/judgment",
        headers=_strict_scope_headers(reviewer_uid=reviewer),
        params={"reviewer_uid": reviewer},
    )


def put_judgment(
    claim_id: str,
    payload: Dict[str, Any],
    *,
    reviewer_uid: Optional[str] = None,
) -> Dict[str, Any]:
    reviewer = _normalize_reviewer_uid(reviewer_uid)
    return _request(
        "put",
        f"/claims/{claim_id}/judgment",
        headers=_strict_scope_headers(reviewer_uid=reviewer),
        json=payload,
        params={"reviewer_uid": reviewer},
    )


def get_all_judgments(claim_id: str) -> Dict[str, Any]:
    return _request(
        "get",
        f"/claims/{claim_id}/judgments",
        headers=_strict_scope_headers(),
    )


def list_judgments(
    *, doc_id: str | None = None, include_drafts: bool = False
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"include_drafts": include_drafts}
    if doc_id:
        params["doc_id"] = doc_id
    return _request("get", "/judgments", params=params, headers=_strict_scope_headers())


def download_export(
    *,
    shape: str,
    format: str = "json",
    include_drafts: bool = False,
    mode: str = "core",
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Download an export payload suitable for st.download_button.

    Returns a dict with keys:
    - content: str|bytes
    - mime: str
    - filename: str
    """
    filename = (
        f"judgments_{shape}_{mode}{'_with_drafts' if include_drafts else ''}.{format}"
    )
    url = f"{_api_root()}/judgments/export"
    params: Dict[str, Any] = {
        "shape": shape,
        "format": format,
        "include_drafts": include_drafts,
        "mode": mode,
    }
    try:
        response = requests.get(
            url,
            params=params,
            headers={**_auth_headers(), **_strict_scope_headers()},
            timeout=timeout or DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network not in tests
        message = str(exc)
        detail = _extract_response_detail(getattr(exc, "response", None))
        if detail:
            message = f"{message} ({detail})"
        show_api_error(f"Judgment export error: {message}")
        raise JudgmentApiError(message) from exc

    if format == "json":
        return {
            "content": response.text,
            "mime": "application/json",
            "filename": filename,
        }
    return {"content": response.content, "mime": "text/csv", "filename": filename}


__all__ = [
    "JudgmentApiError",
    "get_all_judgments",
    "get_judgment",
    "put_judgment",
    "list_judgments",
    "download_export",
    "show_api_error",
]
