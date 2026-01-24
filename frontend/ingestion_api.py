from typing import Dict, Optional, Union

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


def upload_pdf(api_url: str, file) -> dict:
    url = f"{api_url.rstrip('/')}/ingest"
    content_type = getattr(file, "type", None) or "application/pdf"
    files = {"file": (file.name, file.getbuffer(), content_type)}
    try:
        response = requests.post(url, files=files, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach ingestion API at {url}") from exc
    payload = _parse_response(response) or {}
    return payload.get("document") or {}


def list_documents(api_url: str) -> list[dict]:
    url = f"{api_url.rstrip('/')}/ingest"
    try:
        response = requests.get(url, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach ingestion API at {url}") from exc
    payload = _parse_response(response) or {}
    return payload.get("documents") or []


def get_document(api_url: str, doc_id: str) -> dict:
    url = f"{api_url.rstrip('/')}/ingest/{doc_id}"
    try:
        response = requests.get(url, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach ingestion API at {url}") from exc
    payload = _parse_response(response) or {}
    return payload


def trigger_extraction(api_url: str, doc_id: str) -> dict:
    url = f"{api_url.rstrip('/')}/ingest/{doc_id}/extract"
    try:
        response = requests.post(url, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach ingestion API at {url}") from exc
    return _parse_response(response) or {}


def trigger_resolution(api_url: str, doc_id: str) -> dict:
    url = f"{api_url.rstrip('/')}/ingest/{doc_id}/resolve"
    try:
        response = requests.post(url, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach ingestion API at {url}") from exc
    return _parse_response(response) or {}


def submit_resolution_choice(
    api_url: str, doc_id: str, reference_id: str, selected_source: str
) -> dict:
    url = f"{api_url.rstrip('/')}/ingest/{doc_id}/resolution/{reference_id}/select"
    payload = {"selected_source": selected_source}
    try:
        response = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach ingestion API at {url}") from exc
    return _parse_response(response) or {}


def get_citation_context(
    api_url: str,
    doc_id: str,
    citation_index: int,
    target_id: Optional[str] = None,
) -> dict:
    """Fetch citation context (sentence + neighbors) for a callout."""
    url = f"{api_url.rstrip('/')}/ingest/{doc_id}/citation-context"
    params: Dict[str, Union[int, str]] = {"citation_index": int(citation_index)}
    if target_id:
        params["target_id"] = target_id
    try:
        response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach ingestion API at {url}") from exc
    return _parse_response(response) or {}


def get_citation_graph(
    api_url: str,
    doc_id: str,
    target_id: Optional[str],
    depth: int,
    max_nodes: int,
    doi: Optional[str] = None,
) -> dict:
    """Fetch citation graph data for the selected cited work."""
    url = f"{api_url.rstrip('/')}/ingest/{doc_id}/citation-graph"
    params: Dict[str, Union[int, str]] = {
        "depth": int(depth),
        "max_nodes": int(max_nodes),
    }
    if target_id:
        params["target_id"] = target_id
    if doi:
        params["doi"] = doi
    try:
        response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach ingestion API at {url}") from exc
    return _parse_response(response) or {}
