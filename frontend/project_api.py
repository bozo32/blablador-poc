from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import requests
import streamlit as st


DEFAULT_TIMEOUT = 90


class ProjectApiError(RuntimeError):
    pass


def _api_root() -> str:
    value = st.session_state.get("api_url") or "http://localhost:8000"
    return str(value).rstrip("/")


def get_meta() -> Dict[str, Any]:
    url = f"{_api_root()}/project"
    try:
        resp = requests.get(url, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json() if resp.text else {}
    except requests.RequestException as exc:
        raise ProjectApiError(str(exc)) from exc


def put_meta(payload: Any) -> Dict[str, Any]:
    url = f"{_api_root()}/project"
    json_payload: Dict[str, Any]

    if isinstance(payload, Mapping):
        json_payload = {str(k): v for k, v in dict(payload).items()}
        # Avoid sending explicit nulls unless a caller truly intends to clear.
        json_payload = {k: v for k, v in json_payload.items() if v is not None}
    else:
        name = str(payload or "").strip() or "default"
        json_payload = {"name": name}

    try:
        resp = requests.put(url, json=json_payload, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json() if resp.text else {}
    except requests.RequestException as exc:
        raise ProjectApiError(str(exc)) from exc


def export_zip() -> bytes:
    url = f"{_api_root()}/project/export"
    try:
        resp = requests.get(url, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as exc:
        raise ProjectApiError(str(exc)) from exc


def import_zip(zip_bytes: bytes, *, overwrite: bool) -> Dict[str, Any]:
    url = f"{_api_root()}/project/import"
    params = {"overwrite": "true" if overwrite else "false"}
    files = {"file": ("project.zip", zip_bytes, "application/zip")}
    resp: Optional[requests.Response] = None
    try:
        resp = requests.post(url, params=params, files=files, timeout=DEFAULT_TIMEOUT)
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
