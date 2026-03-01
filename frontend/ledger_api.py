from __future__ import annotations

from typing import List, Optional

import requests


DEFAULT_TIMEOUT = 30


def _headers(project_id: Optional[str]) -> dict:
    pid = str(project_id or "").strip()
    if not pid:
        return {}
    return {"X-Project-Id": pid}


def _require_mutation_headers(
    *, project_id: Optional[str], user_id: Optional[str]
) -> dict:
    pid = str(project_id or "").strip()
    uid = str(user_id or "").strip()
    if not pid:
        raise RuntimeError("ledger mutation requires project_id")
    if not uid:
        raise RuntimeError("ledger mutation requires user_id")
    return {
        "X-Project-Id": pid,
        "X-User-Id": uid,
    }


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
    api_url: str,
    doc_num: int,
    targets: List[int],
    *,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    url = f"{api_url.rstrip('/')}/ledger/{int(doc_num)}/outgoing"
    try:
        resp = requests.patch(
            url,
            json={"targets": [int(t) for t in targets]},
            headers=_require_mutation_headers(project_id=project_id, user_id=user_id),
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach backend at {url}") from exc
    return _parse_json(resp) or {}


def set_incoming(
    api_url: str,
    doc_num: int,
    targets: List[int],
    *,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    url = f"{api_url.rstrip('/')}/ledger/{int(doc_num)}/incoming"
    try:
        resp = requests.patch(
            url,
            json={"targets": [int(t) for t in targets]},
            headers=_require_mutation_headers(project_id=project_id, user_id=user_id),
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach backend at {url}") from exc
    return _parse_json(resp) or {}


def set_assigned(
    api_url: str,
    doc_num: int,
    assigned: bool,
    *,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    url = f"{api_url.rstrip('/')}/ledger/{int(doc_num)}/assign"
    try:
        resp = requests.patch(
            url,
            json={"assigned": bool(assigned)},
            headers=_require_mutation_headers(project_id=project_id, user_id=user_id),
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach backend at {url}") from exc
    return _parse_json(resp) or {}


def place_relation(
    api_url: str,
    *,
    source_num: int,
    target_num: int,
    relation: str,
    canonical: bool = False,
    reviewer_uid: Optional[str] = None,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    url = f"{api_url.rstrip('/')}/ledger/place"
    payload = {
        "source_num": int(source_num),
        "target_num": int(target_num),
        "relation": str(relation or "is cited by"),
        "canonical": bool(canonical),
        "reviewer_uid": str(reviewer_uid or "").strip() or None,
    }
    try:
        resp = requests.post(
            url,
            json=payload,
            headers=_require_mutation_headers(project_id=project_id, user_id=user_id),
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach backend at {url}") from exc
    return _parse_json(resp) or {}


def place_reference(
    api_url: str,
    *,
    citing_doc_id: str,
    reference_id: str,
    cited_ingest_id: str,
    canonical: bool = False,
    reviewer_uid: Optional[str] = None,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    url = f"{api_url.rstrip('/')}/ledger/place-reference"
    payload = {
        "citing_doc_id": str(citing_doc_id or "").strip(),
        "reference_id": str(reference_id or "").strip(),
        "cited_ingest_id": str(cited_ingest_id or "").strip(),
        "canonical": bool(canonical),
        "reviewer_uid": str(reviewer_uid or "").strip() or None,
    }
    try:
        resp = requests.post(
            url,
            json=payload,
            headers=_require_mutation_headers(project_id=project_id, user_id=user_id),
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach backend at {url}") from exc
    return _parse_json(resp) or {}
