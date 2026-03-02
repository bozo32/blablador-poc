from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import requests
import streamlit as st


DEFAULT_TIMEOUT = 90


class ProjectApiError(RuntimeError):
    pass


def _require_project_headers(
    *,
    project_id: Optional[str],
    user_id: Optional[str] = None,
    require_user: bool = False,
) -> Dict[str, str]:
    pid = str(project_id or "").strip()
    if not pid:
        raise ProjectApiError("project API request requires project_id")

    headers = {"X-Project-Id": pid}
    uid = str(user_id or "").strip()
    if require_user and not uid:
        raise ProjectApiError("project API request requires user_id")
    if uid:
        headers["X-User-Id"] = uid
    return headers


def _api_root() -> str:
    value = st.session_state.get("api_url") or "http://localhost:8000"
    return str(value).rstrip("/")


def get_meta(*, project_id: Optional[str], user_id: Optional[str] = None) -> Dict[str, Any]:
    url = f"{_api_root()}/project"
    headers = _require_project_headers(
        project_id=project_id,
        user_id=user_id,
        require_user=True,
    )
    try:
        resp = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json() if resp.text else {}
    except requests.RequestException as exc:
        raise ProjectApiError(str(exc)) from exc


def put_meta(
    payload: Any, *, project_id: Optional[str], user_id: Optional[str] = None
) -> Dict[str, Any]:
    url = f"{_api_root()}/project"
    headers = _require_project_headers(
        project_id=project_id,
        user_id=user_id,
        require_user=True,
    )
    json_payload: Dict[str, Any]

    if isinstance(payload, Mapping):
        json_payload = {str(k): v for k, v in dict(payload).items()}
        # Avoid sending explicit nulls unless a caller truly intends to clear.
        json_payload = {k: v for k, v in json_payload.items() if v is not None}
    else:
        name = str(payload or "").strip() or "default"
        json_payload = {"name": name}

    try:
        resp = requests.put(
            url,
            json=json_payload,
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json() if resp.text else {}
    except requests.RequestException as exc:
        raise ProjectApiError(str(exc)) from exc


def export_zip(*, project_id: Optional[str], user_id: Optional[str] = None) -> bytes:
    url = f"{_api_root()}/project/export"
    headers = _require_project_headers(
        project_id=project_id,
        user_id=user_id,
        require_user=True,
    )
    try:
        resp = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as exc:
        raise ProjectApiError(str(exc)) from exc


def import_zip(
    zip_bytes: bytes,
    *,
    overwrite: bool,
    project_id: Optional[str],
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    url = f"{_api_root()}/project/import"
    headers = _require_project_headers(
        project_id=project_id,
        user_id=user_id,
        require_user=True,
    )
    params = {"overwrite": "true" if overwrite else "false"}
    files = {"file": ("project.zip", zip_bytes, "application/zip")}
    resp: Optional[requests.Response] = None
    try:
        resp = requests.post(
            url,
            params=params,
            files=files,
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json() if resp.text else {}
    except requests.RequestException as exc:
        detail = None
        try:
            detail = resp.text if resp is not None else None
        except Exception:
            detail = None
        msg = str(exc)
        if detail:
            msg = f"{msg} ({detail})"
        raise ProjectApiError(msg) from exc


def wipe_everything(*, confirm: str) -> Dict[str, Any]:
    url = f"{_api_root()}/dev/wipe"
    payload = {"confirm": str(confirm or "")}
    resp: Optional[requests.Response] = None
    try:
        resp = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json() if resp.text else {}
    except requests.RequestException as exc:
        detail = None
        try:
            detail = resp.text if resp is not None else None
        except Exception:
            detail = None
        msg = str(exc)
        if detail:
            msg = f"{msg} ({detail})"
        raise ProjectApiError(msg) from exc
