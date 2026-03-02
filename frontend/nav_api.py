from __future__ import annotations

from typing import Optional

import requests


DEFAULT_TIMEOUT = 30


def _parse_response(response: requests.Response) -> Optional[dict]:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(response.text) from exc
    if not response.text:
        return None
    return response.json()


def _request_error_message(url: str, exc: requests.RequestException) -> str:
    if isinstance(exc, requests.Timeout):
        return f"Request timed out ({DEFAULT_TIMEOUT}s): {url}"
    return f"Failed to reach API at {url}"


def get_nav_graph(
    api_url: str,
    *,
    project_id: str,
    reviewer_uid: str,
    focus_type: Optional[str],
    focus_id: Optional[str],
    show_claimspans: bool,
) -> dict:
    url = f"{api_url.rstrip('/')}/nav/graph"
    pid = str(project_id or "").strip()
    reviewer = str(reviewer_uid or "").strip()
    if not pid:
        raise RuntimeError("nav graph requires project_id")
    if not reviewer:
        raise RuntimeError("nav graph requires reviewer_uid")
    params = {
        "reviewer_uid": reviewer,
        "focus_type": str(focus_type or "").strip() or None,
        "focus_id": str(focus_id or "").strip() or None,
        "show_claimspans": "true" if bool(show_claimspans) else "false",
    }
    params = {k: v for k, v in params.items() if v is not None}
    try:
        response = requests.get(
            url,
            params=params,
            headers={"X-Project-Id": pid, "X-Reviewer-Uid": reviewer},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    return _parse_response(response) or {}


def get_work_contexts(
    api_url: str,
    *,
    work_id: str,
    project_id: str,
    reviewer_uid: str,
) -> dict:
    wid = str(work_id or "").strip()
    pid = str(project_id or "").strip()
    reviewer = str(reviewer_uid or "").strip()
    if not wid:
        return {}
    if not pid:
        raise RuntimeError("work contexts require project_id")
    if not reviewer:
        raise RuntimeError("work contexts require reviewer_uid")
    url = f"{api_url.rstrip('/')}/nav/works/{wid}/contexts"
    params = {"reviewer_uid": reviewer}
    try:
        response = requests.get(
            url,
            params=params,
            headers={"X-Project-Id": pid, "X-Reviewer-Uid": reviewer},
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    return _parse_response(response) or {}


def get_reference_retrieval(api_url: str, *, doc_id: str, reference_id: str) -> dict:
    did = str(doc_id or "").strip()
    rid = str(reference_id or "").strip()
    if not did or not rid:
        return {}
    url = f"{api_url.rstrip('/')}/references/{did}/{rid}/retrieval"
    try:
        response = requests.get(url, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    return _parse_response(response) or {}
