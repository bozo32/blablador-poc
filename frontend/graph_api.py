from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import requests
import streamlit as st


DEFAULT_TIMEOUT = 30


class GraphApiError(RuntimeError):
    pass


def _api_root() -> str:
    value = st.session_state.get("api_url") or "http://localhost:8000"
    return str(value).rstrip("/")


def _project_headers(project_id: Optional[str]) -> Dict[str, str]:
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
        raise GraphApiError(f"{operation} requires project_id")
    if not uid:
        raise GraphApiError(f"{operation} requires user_id")
    return {
        "X-Project-Id": pid,
        "X-User-Id": uid,
    }


def _sources_param(sources: Any) -> str:
    if sources is None:
        return "auto,manual,external_search"
    if isinstance(sources, str):
        text = ",".join(part.strip() for part in sources.split(",") if part.strip())
        return text or "auto,manual,external_search"
    if isinstance(sources, Iterable):
        parts = [str(part).strip() for part in sources if str(part).strip()]
        text = ",".join(parts)
        return text or "auto,manual,external_search"
    return "auto,manual,external_search"


def get_claim_subgraph(
    center_claim_id: str,
    hops: int = 1,
    edge_cap: int = 25,
    min_votes: int = 0,
    sources: Any = None,
) -> Dict[str, Any]:
    url = f"{_api_root()}/graph/claim-subgraph"
    params = {
        "center_claim_id": str(center_claim_id or "").strip(),
        "hops": int(hops),
        "edge_cap": int(edge_cap),
        "min_votes": int(min_votes),
        "sources": _sources_param(sources),
    }
    try:
        resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json() if resp.text else {}
    except requests.RequestException as exc:
        raise GraphApiError(str(exc)) from exc


def get_edge_votes(edge_id: int) -> Dict[str, Any]:
    url = f"{_api_root()}/graph/edge/{int(edge_id)}/votes"
    try:
        resp = requests.get(url, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json() if resp.text else {}
    except requests.RequestException as exc:
        raise GraphApiError(str(exc)) from exc


def put_edge_vote(
    edge_id: int,
    reviewer_uid: str,
    verdict: str,
    confidence: Optional[float] = None,
    comment: Optional[str] = None,
    *,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    url = f"{_api_root()}/graph/edge/{int(edge_id)}/vote"
    reviewer = str(reviewer_uid or "").strip()
    if not reviewer:
        raise GraphApiError("put edge vote requires reviewer_uid")
    params = {"reviewer_uid": reviewer}
    payload: Dict[str, Any] = {
        "verdict": str(verdict or "neutral"),
        "confidence": confidence,
        "comment": comment,
    }
    # Avoid explicit nulls unless the user truly supplied them.
    payload = {k: v for k, v in payload.items() if v is not None}
    try:
        resp = requests.put(
            url,
            params=params,
            headers=_require_mutation_headers(
                project_id=project_id,
                user_id=reviewer,
                operation="graph vote mutation",
            ),
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json() if resp.text else {}
    except requests.RequestException as exc:
        raise GraphApiError(str(exc)) from exc


def create_claim_link(
    source_claim_id: str,
    target_claim_id: str,
    reviewer_uid: str,
    *,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    url = f"{_api_root()}/graph/claim-link"
    reviewer = str(reviewer_uid or "").strip()
    if not reviewer:
        raise GraphApiError("create claim link requires reviewer_uid")
    params = {"reviewer_uid": reviewer}
    payload = {
        "source_claim_id": str(source_claim_id or "").strip(),
        "target_claim_id": str(target_claim_id or "").strip(),
    }
    try:
        resp = requests.post(
            url,
            params=params,
            headers=_require_mutation_headers(
                project_id=project_id,
                user_id=reviewer,
                operation="graph claim-link mutation",
            ),
            json=payload,
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json() if resp.text else {}
    except requests.RequestException as exc:
        raise GraphApiError(str(exc)) from exc


def delete_claim_link(
    edge_id: int,
    reviewer_uid: str,
    *,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    url = f"{_api_root()}/graph/claim-link/{int(edge_id)}"
    reviewer = str(reviewer_uid or "").strip()
    if not reviewer:
        raise GraphApiError("delete claim link requires reviewer_uid")
    params = {"reviewer_uid": reviewer}
    try:
        resp = requests.delete(
            url,
            params=params,
            headers=_require_mutation_headers(
                project_id=project_id,
                user_id=reviewer,
                operation="graph claim-link deletion",
            ),
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json() if resp.text else {}
    except requests.RequestException as exc:
        raise GraphApiError(str(exc)) from exc


def reindex_docs(*, project_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    url = f"{_api_root()}/graph/reindex-docs"
    try:
        resp = requests.post(
            url,
            headers=_require_mutation_headers(
                project_id=project_id,
                user_id=user_id,
                operation="graph reindex",
            ),
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json() if resp.text else {}
    except requests.RequestException as exc:
        raise GraphApiError(str(exc)) from exc


def get_claim_candidates(claim_id: str, limit: int = 10) -> Dict[str, Any]:
    url = f"{_api_root()}/graph/claim/{str(claim_id).strip()}/candidates"
    params = {"limit": int(limit)}
    try:
        resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json() if resp.text else {}
    except requests.RequestException as exc:
        raise GraphApiError(str(exc)) from exc


def list_claim_nodes(doc_id: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
    url = f"{_api_root()}/graph/claim-nodes"
    params: Dict[str, Any] = {"limit": int(limit)}
    if doc_id:
        params["doc_id"] = str(doc_id).strip()
    try:
        resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        return resp.json() if resp.text else {}
    except requests.RequestException as exc:
        raise GraphApiError(str(exc)) from exc


def resolve_references(
    *,
    citing_doc_id: str,
    reference_ids: list[str],
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    url = f"{_api_root()}/graph/resolve-references"
    payload = {
        "citing_doc_id": str(citing_doc_id or "").strip(),
        "reference_ids": [
            str(r).strip() for r in (reference_ids or []) if str(r).strip()
        ],
    }
    try:
        resp = requests.post(
            url,
            json=payload,
            headers=_require_mutation_headers(
                project_id=project_id,
                user_id=user_id,
                operation="graph resolve references",
            ),
            timeout=DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json() if resp.text else {}
    except requests.RequestException as exc:
        raise GraphApiError(str(exc)) from exc
