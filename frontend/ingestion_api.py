from typing import Any, Dict, List, Optional, Union

import requests
import urllib.parse

# Extraction (GROBID) can take >30s on first run.
DEFAULT_TIMEOUT = 180


def _ingest_headers(project_id: Optional[str]) -> Dict[str, str]:
    pid = str(project_id or "").strip()
    if not pid:
        return {}
    return {"X-Project-Id": pid}


def _require_mutation_headers(
    *,
    project_id: Optional[str],
    user_id: Optional[str],
    operation: str,
) -> Dict[str, str]:
    pid = str(project_id or "").strip()
    uid = str(user_id or "").strip()
    if not pid:
        raise RuntimeError(f"{operation} requires project_id")
    if not uid:
        raise RuntimeError(f"{operation} requires user_id")
    return {
        "X-Project-Id": pid,
        "X-User-Id": uid,
    }


def _require_scoped_read_headers(
    *,
    project_id: Optional[str],
    user_id: Optional[str],
    operation: str,
) -> Dict[str, str]:
    pid = str(project_id or "").strip()
    uid = str(user_id or "").strip()
    if not pid:
        raise RuntimeError(f"{operation} requires project_id")
    if not uid:
        raise RuntimeError(f"{operation} requires user_id")
    return {
        "X-Project-Id": pid,
        "X-User-Id": uid,
    }


def _parse_response(response: requests.Response) -> Optional[dict]:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(response.text) from exc
    if not response.text:
        return None
    return response.json()


def upload_pdf(
    api_url: str,
    file,
    *,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
    auto_process: bool = True,
) -> dict:
    url = f"{api_url.rstrip('/')}/ingest"
    content_type = getattr(file, "type", None) or "application/pdf"
    files = {"file": (file.name, file.getbuffer(), content_type)}
    try:
        response = requests.post(
            url,
            params={"auto_process": "true" if bool(auto_process) else "false"},
            headers=_require_mutation_headers(
                project_id=project_id,
                user_id=user_id,
                operation="ingest upload",
            ),
            files=files,
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    payload = _parse_response(response) or {}
    return payload.get("document") or {}


def list_documents(api_url: str, *, project_id: Optional[str] = None) -> list[dict]:
    url = f"{api_url.rstrip('/')}/ingest"
    try:
        response = requests.get(
            url,
            headers=_ingest_headers(project_id),
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    payload = _parse_response(response) or {}
    return payload.get("documents") or []


def get_document(
    api_url: str, doc_id: str, *, project_id: Optional[str] = None
) -> dict:
    url = f"{api_url.rstrip('/')}/ingest/{doc_id}"
    try:
        response = requests.get(
            url,
            headers=_ingest_headers(project_id),
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    payload = _parse_response(response) or {}
    return payload


def trigger_extraction(
    api_url: str,
    doc_id: str,
    *,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
    force: bool = False,
) -> dict:
    url = f"{api_url.rstrip('/')}/ingest/{doc_id}/extract"
    if bool(force):
        url = url + "?force=true"
    try:
        response = requests.post(
            url,
            headers=_require_mutation_headers(
                project_id=project_id,
                user_id=user_id,
                operation="ingest extraction",
            ),
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    return _parse_response(response) or {}


def trigger_resolution(
    api_url: str,
    doc_id: str,
    *,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
    force: bool = False,
) -> dict:
    url = f"{api_url.rstrip('/')}/ingest/{doc_id}/resolve"
    if bool(force):
        url = url + "?force=true"
    try:
        response = requests.post(
            url,
            headers=_require_mutation_headers(
                project_id=project_id,
                user_id=user_id,
                operation="ingest resolution",
            ),
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    return _parse_response(response) or {}


def submit_resolution_choice(
    api_url: str,
    doc_id: str,
    reference_id: str,
    selected_source: str,
    *,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    url = f"{api_url.rstrip('/')}/ingest/{doc_id}/resolution/{reference_id}/select"
    payload = {"selected_source": selected_source}
    try:
        response = requests.post(
            url,
            headers=_require_mutation_headers(
                project_id=project_id,
                user_id=user_id,
                operation="ingest resolution select",
            ),
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    return _parse_response(response) or {}


def get_citation_context(
    api_url: str,
    doc_id: str,
    citation_index: int,
    target_id: Optional[str] = None,
    *,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    """Fetch citation context (sentence + neighbors) for a callout."""
    url = f"{api_url.rstrip('/')}/ingest/{doc_id}/citation-context"
    params: Dict[str, Union[int, str]] = {"citation_index": int(citation_index)}
    if target_id:
        params["target_id"] = target_id
    try:
        response = requests.get(
            url,
            headers=_require_scoped_read_headers(
                project_id=project_id,
                user_id=user_id,
                operation="citation context lookup",
            ),
            params=params,
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    return _parse_response(response) or {}


def get_citation_graph(
    api_url: str,
    doc_id: str,
    target_id: Optional[str],
    depth: int,
    max_nodes: int,
    doi: Optional[str] = None,
    *,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
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
        response = requests.get(
            url,
            headers=_require_scoped_read_headers(
                project_id=project_id,
                user_id=user_id,
                operation="citation graph lookup",
            ),
            params=params,
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    return _parse_response(response) or {}


def auto_place_claim_source(
    api_url: str,
    *,
    claim_id: str,
    doc_id: str,
    citation_index: Optional[int],
    target_id: Optional[str],
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    url = f"{api_url.rstrip('/')}/claims/{claim_id}/auto-place"
    payload = {
        "doc_id": str(doc_id),
        "citation_index": int(citation_index) if citation_index is not None else None,
        "target_id": target_id,
    }
    try:
        response = requests.post(
            url,
            headers=_require_mutation_headers(
                project_id=project_id,
                user_id=user_id,
                operation="claim auto-place",
            ),
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    return _parse_response(response) or {}


def confirm_claims(
    api_url: str,
    *,
    document_id: str,
    sentence_id: str,
    sentence_text: str,
    citation_index: int,
    target_id: Optional[str],
    reviewer_uid: str,
    confirmed_claims: List[Dict[str, Any]],
    segmentation_model: Optional[str] = None,
    cited_work_id: Optional[str] = None,
    citation_anchor: Optional[Dict[str, Any]] = None,
    project_id: Optional[str] = None,
) -> dict:
    """Persist confirmed claims for a citing sentence.

    This powers backend claim indexing (graph nodes) and stable re-loads.
    """
    url = f"{api_url.rstrip('/')}/claims/confirm"
    payload: Dict[str, Any] = {
        "document_id": str(document_id or "").strip(),
        "sentence_id": str(sentence_id or "").strip(),
        "sentence_text": str(sentence_text or "").strip(),
        "citation_index": int(citation_index),
        "target_id": str(target_id).strip() if target_id else None,
        "segmentation_model": str(segmentation_model).strip()
        if segmentation_model
        else None,
        "reviewer_uid": str(reviewer_uid or "").strip(),
        "cited_work_id": str(cited_work_id).strip() if cited_work_id else None,
        "citation_anchor": citation_anchor,
        "confirmed_claims": list(confirmed_claims or []),
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    try:
        response = requests.post(
            url,
            headers=_require_mutation_headers(
                project_id=project_id,
                user_id=reviewer_uid,
                operation="claim confirm",
            ),
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    return _parse_response(response) or {}


def get_document_body(
    api_url: str, doc_id: str, *, project_id: Optional[str] = None
) -> dict:
    url = f"{api_url.rstrip('/')}/ingest/{doc_id}/body"
    try:
        response = requests.get(
            url,
            headers=_ingest_headers(project_id),
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    return _parse_response(response) or {}


def get_reference_retrieval(
    api_url: str,
    doc_id: str,
    reference_id: str,
) -> dict:
    """Fetch retrieval dossier for a reference."""
    url = f"{api_url.rstrip('/')}/references/{doc_id}/{reference_id}/retrieval"
    try:
        response = requests.get(url, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    return _parse_response(response) or {}


def get_claim_span_context(
    api_url: str,
    claim_id: str,
    *,
    target_id: Optional[str] = None,
) -> dict:
    url = f"{api_url.rstrip('/')}/claims/{claim_id}/span-context"
    params: Dict[str, Union[int, str]] = {}
    if target_id:
        params["target_id"] = str(target_id)
    try:
        response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    return _parse_response(response) or {}


def get_claim_status(
    api_url: str,
    claim_id: str,
    *,
    target_id: Optional[str] = None,
) -> dict:
    url = f"{api_url.rstrip('/')}/claims/{claim_id}/status"
    params: Dict[str, Union[int, str]] = {}
    if target_id:
        params["target_id"] = str(target_id)
    try:
        response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    return _parse_response(response) or {}


def get_span_status(
    api_url: str,
    span_id: str,
    *,
    reviewer_uid: str = "default",
    project_id: Optional[str] = None,
) -> dict:
    url = f"{api_url.rstrip('/')}/spans/{span_id}/status"
    params: Dict[str, Union[int, str]] = {
        "reviewer_uid": str(reviewer_uid or "default")
    }
    try:
        response = requests.get(
            url,
            headers=_ingest_headers(project_id),
            params=params,
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    return _parse_response(response) or {}


def get_span_bundle(
    api_url: str,
    span_id: str,
    *,
    reviewer_uid: str = "default",
    include_history: bool = False,
    project_id: Optional[str] = None,
) -> dict:
    url = f"{api_url.rstrip('/')}/spans/{span_id}/bundle"
    params: Dict[str, Union[int, str]] = {
        "reviewer_uid": str(reviewer_uid or "default"),
        "include_history": "true" if include_history else "false",
    }
    try:
        response = requests.get(
            url,
            headers=_ingest_headers(project_id),
            params=params,
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    return _parse_response(response) or {}


def lookup_citation_window_span(
    api_url: str,
    *,
    ingest_id: str,
    citation_index: int,
    target_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> dict:
    url = f"{api_url.rstrip('/')}/spans/lookup-citation-window"
    params: Dict[str, Union[int, str]] = {
        "ingest_id": str(ingest_id or "").strip(),
        "citation_index": int(citation_index),
    }
    if target_id:
        params["target_id"] = str(target_id).strip()
    try:
        response = requests.get(
            url,
            headers=_ingest_headers(project_id),
            params=params,
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    return _parse_response(response) or {}


def set_span_cite_role(
    api_url: str,
    *,
    span_id: str,
    cited_work_id: str,
    reviewer_uid: str,
    role: str,
    project_id: Optional[str] = None,
) -> dict:
    span_id = str(span_id or "").strip()
    cited_work_id = str(cited_work_id or "").strip()
    if not span_id or not cited_work_id:
        raise RuntimeError("span_id and cited_work_id are required")
    quoted = urllib.parse.quote(cited_work_id, safe="")
    url = f"{api_url.rstrip('/')}/spans/{span_id}/cites/{quoted}/role"
    payload = {
        "reviewer_uid": str(reviewer_uid or "").strip(),
        "role": role,
    }
    try:
        response = requests.put(
            url,
            headers=_require_mutation_headers(
                project_id=project_id,
                user_id=reviewer_uid,
                operation="set span cite role",
            ),
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    return _parse_response(response) or {}


def compact_span_graph(
    api_url: str,
    *,
    dry_run: bool = False,
    aggressive: bool = False,
) -> dict:
    url = f"{api_url.rstrip('/')}/maintenance/span-graph/compact"
    payload: Dict[str, Any] = {
        "dry_run": bool(dry_run),
        "aggressive": bool(aggressive),
    }
    try:
        response = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    return _parse_response(response) or {}


def neighborhood_search(
    api_url: str,
    *,
    span_id: str,
    reviewer_uid: str,
    query_text: Optional[str] = None,
    max_per_seed: int = 25,
    min_bib_intersection: int = 1,
    min_abstract_score: float = 0.0,
) -> dict:
    url = f"{api_url.rstrip('/')}/neighborhood/search"
    payload: Dict[str, Any] = {
        "span_id": str(span_id),
        "reviewer_uid": str(reviewer_uid or "default") or "default",
        "max_per_seed": int(max_per_seed),
        "min_bib_intersection": int(min_bib_intersection),
        "query_text": str(query_text).strip() if query_text else None,
        "min_abstract_score": float(min_abstract_score),
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    try:
        response = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    return _parse_response(response) or {}


def get_neighborhood_run(
    api_url: str,
    run_id: str,
    *,
    limit: int = 50,
) -> dict:
    url = f"{api_url.rstrip('/')}/neighborhood/{run_id}"
    params: Dict[str, Union[int, str]] = {"limit": int(limit)}
    try:
        response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    except requests.RequestException as exc:
        raise RuntimeError(_request_error_message(url, exc)) from exc
    return _parse_response(response) or {}


def _request_error_message(url: str, exc: requests.RequestException) -> str:
    if isinstance(exc, requests.Timeout):
        return (
            f"Request to ingestion API timed out ({DEFAULT_TIMEOUT}s): {url}. "
            "The backend may still be processing (e.g., GROBID extraction); "
            "try refreshing the document status in a moment."
        )
    return f"Failed to reach ingestion API at {url}"
