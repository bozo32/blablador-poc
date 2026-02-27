from __future__ import annotations

from typing import List, Optional

import requests


DEFAULT_TIMEOUT = 30


def _headers(project_id: Optional[str]) -> dict:
    pid = str(project_id or "").strip()
    if not pid:
        return {}
    return {"X-Project-Id": pid}


def _parse_json(response: requests.Response) -> dict:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(response.text) from exc
    if not response.text:
        return {}
    return response.json()


def get_ledger(api_url: str, *, project_id: Optional[str] = None) -> dict:
    url = f"{api_url.rstrip('/')}/ledger"
    try:
        resp = requests.get(url, headers=_headers(project_id), timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach backend at {url}") from exc
    return _parse_json(resp) or {}


def set_outgoing(
    api_url: str, doc_num: int, targets: List[int], *, project_id: Optional[str] = None
) -> dict:
    url = f"{api_url.rstrip('/')}/ledger/{int(doc_num)}/outgoing"
    try:
        resp = requests.patch(
            url,
            json={"targets": [int(t) for t in targets]},
            headers=_headers(project_id),
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach backend at {url}") from exc
    return _parse_json(resp) or {}


def set_incoming(
    api_url: str, doc_num: int, targets: List[int], *, project_id: Optional[str] = None
) -> dict:
    url = f"{api_url.rstrip('/')}/ledger/{int(doc_num)}/incoming"
    try:
        resp = requests.patch(
            url,
            json={"targets": [int(t) for t in targets]},
            headers=_headers(project_id),
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach backend at {url}") from exc
    return _parse_json(resp) or {}


def set_assigned(
    api_url: str, doc_num: int, assigned: bool, *, project_id: Optional[str] = None
) -> dict:
    url = f"{api_url.rstrip('/')}/ledger/{int(doc_num)}/assign"
    try:
        resp = requests.patch(
            url,
            json={"assigned": bool(assigned)},
            headers=_headers(project_id),
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach backend at {url}") from exc
    return _parse_json(resp) or {}
