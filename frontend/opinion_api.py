# frontend/opinion_api.py
"""Client helpers for opinion events API (follow/ignore/complete)."""

from typing import Any, Dict, List, Optional, Union

import requests

# Extraction can take >30s on first run.
DEFAULT_TIMEOUT = 180


def _opinion_headers(project_id: Optional[str]) -> Dict[str, str]:
    """Return headers with X-Project-Id when project_id is set."""
    pid = str(project_id or "").strip()
    if not pid:
        return {}
    return {"X-Project-Id": pid}


def _request(
    method: str,
    url: str,
    *,
    project_id: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Union[int, str]]] = None,
    json: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> requests.Response:
    """Make an HTTP request with optional X-Project-Id header."""
    req_headers = _opinion_headers(project_id)
    if headers:
        req_headers.update({str(k): str(v) for k, v in headers.items()})
    
    try:
        response = requests.request(
            method=method,
            url=url,
            headers=req_headers,
            params=params,
            json=json,
            timeout=timeout,
        )
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        raise RuntimeError(f"Request to {url} failed: {exc}") from exc


def append_follow(
    api_url: str,
    project_id: str,
    reviewer_uid: str,
    doc_id: str,
    citation_index: int,
    target_id: Optional[str],
    span_id: str,
    status: str,
    *,
    idempotency_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Append a follow event for a citation.
    
    Args:
        api_url: Base API URL
        project_id: Project ID for scoping
        reviewer_uid: Reviewer ID
        doc_id: Document ID
        citation_index: Citation index
        target_id: Target ID (optional)
        span_id: Cite-window span ID
        status: One of "follow", "ignore", "complete"
        idempotency_key: Optional idempotency key for retries
    
    Returns:
        The created event
    """
    pid = str(project_id or "").strip()
    rid = str(reviewer_uid or "").strip()
    if not pid:
        raise RuntimeError("opinion mutation requires project_id")
    if not rid:
        raise RuntimeError("opinion mutation requires reviewer_uid")

    url = f"{api_url.rstrip('/')}/opinions/events"
    
    payload = {
        "kind": "follow",
        "target_key": f"citespan:{span_id}",
        "visibility": "private",
        "payload": {"status": status},
        "doc_id": doc_id,
        "citation_index": citation_index,
        "target_id": target_id,
        "span_id": span_id,
    }
    
    if idempotency_key:
        payload["idempotency_key"] = idempotency_key
    
    response = _request(
        "POST",
        url,
        project_id=pid,
        headers={"X-User-Id": rid, "X-Reviewer-Uid": rid},
        params={"reviewer_uid": reviewer_uid},
        json=payload,
    )
    
    return response.json() if response.text else {}


def list_follows_by_doc(
    api_url: str,
    project_id: str,
    reviewer_uid: str,
    doc_id: str,
) -> List[Dict[str, Any]]:
    """List projected follow entries for a document.
    
    Args:
        api_url: Base API URL
        project_id: Project ID for scoping
        reviewer_uid: Reviewer ID
        doc_id: Document ID
    
    Returns:
        List of follow entries with current_status and sort_key
    """
    url = f"{api_url.rstrip('/')}/opinions/follow/by-doc"
    reviewer = str(reviewer_uid or "").strip()
    if not reviewer:
        raise RuntimeError("opinion read requires reviewer_uid")
    
    response = _request(
        "GET",
        url,
        project_id=project_id,
        headers={"X-Reviewer-Uid": reviewer},
        params={
            "reviewer_uid": reviewer,
            "doc_id": doc_id,
        },
    )
    
    data = response.json() if response.text else {}
    return data.get("follows", [])


def get_follow_for_target(
    api_url: str,
    project_id: str,
    reviewer_uid: str,
    target_key: str,
) -> Optional[Dict[str, Any]]:
    """Get follow status for a specific target.
    
    Args:
        api_url: Base API URL
        project_id: Project ID for scoping
        reviewer_uid: Reviewer ID
        target_key: Target key (e.g., "citespan:span:xxx")
    
    Returns:
        Follow status dict or None if not found
    """
    url = f"{api_url.rstrip('/')}/opinions/follow/target"
    reviewer = str(reviewer_uid or "").strip()
    if not reviewer:
        raise RuntimeError("opinion read requires reviewer_uid")
    
    try:
        response = _request(
            "GET",
            url,
            project_id=project_id,
            headers={"X-Reviewer-Uid": reviewer},
            params={
                "reviewer_uid": reviewer,
                "target_key": target_key,
            },
        )
        return response.json() if response.text else None
    except RuntimeError:
        # Not found is OK - return None
        return None


def resolve_span_id(
    api_url: str,
    project_id: Optional[str],
    doc_id: str,
    citation_index: int,
    target_id: Optional[str],
) -> Optional[str]:
    """Resolve cite-window span_id from (doc_id, citation_index, target_id).
    
    Calls the /spans/lookup-citation-window endpoint.
    
    Args:
        api_url: Base API URL
        project_id: Project ID for scoping
        doc_id: Document ID (can also serve as ingest_id)
        citation_index: Citation index
        target_id: Target ID (optional)
    
    Returns:
        span_id string if found, None otherwise
    """
    url = f"{api_url.rstrip('/')}/spans/lookup-citation-window"
    params: Dict[str, Union[int, str]] = {
        "ingest_id": doc_id,  # doc_id serves as ingest_id
        "citation_index": int(citation_index),
    }
    if target_id:
        params["target_id"] = str(target_id).strip()
    
    try:
        response = requests.get(
            url,
            headers=_opinion_headers(project_id),
            params=params,
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json() if response.text else {}
        return data.get("span_id")
    except Exception:
        return None
