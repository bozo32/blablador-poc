from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from backend.settings import settings

OPENALEX_CACHE: Dict[str, Dict[str, Any]] = {}
CITED_BY_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def _normalize_identifier(identifier: str) -> str:
    value = identifier.strip()
    lower_value = value.lower()
    if lower_value.startswith("https://doi.org/"):
        return value
    if lower_value.startswith("doi:"):
        return f"https://doi.org/{value[4:].strip()}"
    if lower_value.startswith("10."):
        return f"https://doi.org/{value}"
    if "openalex.org/" in lower_value:
        return value.rsplit("/", 1)[-1]
    return value


def _openalex_id(work_id: Optional[str]) -> Optional[str]:
    if not work_id:
        return None
    if "openalex.org/" in work_id:
        return work_id.rsplit("/", 1)[-1]
    return work_id


def _request_params() -> Dict[str, str]:
    params: Dict[str, str] = {}
    if settings.OPENALEX_API_KEY:
        params["api_key"] = settings.OPENALEX_API_KEY
    return params


def _fetch_json(url: str, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    request_params = _request_params()
    if params:
        request_params.update(params)
    response = requests.get(url, params=request_params, timeout=10)
    if response.status_code in (403, 429):
        raise RuntimeError(
            "OpenAlex request blocked. Check OPENALEX_API_KEY or rate limits."
        )
    if not response.ok:
        raise RuntimeError(
            f"OpenAlex request failed ({response.status_code}): {response.text}"
        )
    return response.json()


def fetch_work(identifier: str) -> Dict[str, Any]:
    normalized = _normalize_identifier(identifier)
    if normalized in OPENALEX_CACHE:
        return OPENALEX_CACHE[normalized]

    url = f"{settings.OPENALEX_API_URL}/{normalized}"
    work = _fetch_json(url)
    work_id = _openalex_id(work.get("id")) or normalized
    OPENALEX_CACHE[work_id] = work
    OPENALEX_CACHE[normalized] = work
    return work


def _fetch_cited_by(work: Dict[str, Any], max_nodes: int) -> List[Dict[str, Any]]:
    work_id = _openalex_id(work.get("id")) or work.get("id")
    if work_id and work_id in CITED_BY_CACHE:
        return CITED_BY_CACHE[work_id]

    cited_by_url = work.get("cited_by_api_url")
    if not cited_by_url:
        return []

    response = _fetch_json(cited_by_url, {"per-page": str(max_nodes)})
    results = response.get("results", [])
    if work_id:
        CITED_BY_CACHE[work_id] = results
    return results


def extract_abstract(work: Dict[str, Any]) -> Optional[str]:
    inverted = work.get("abstract_inverted_index")
    if not isinstance(inverted, dict) or not inverted:
        return None
    positions: Dict[int, str] = {}
    max_pos = -1
    for token, pos_list in inverted.items():
        if not isinstance(token, str) or not isinstance(pos_list, list):
            continue
        for pos in pos_list:
            try:
                idx = int(pos)
            except Exception:
                continue
            positions[idx] = token
            if idx > max_pos:
                max_pos = idx
    if max_pos < 0:
        return None
    words = [""] * (max_pos + 1)
    for idx, token in positions.items():
        if 0 <= idx < len(words):
            words[idx] = token
    text = " ".join(w for w in words if w).strip()
    return text or None


def fetch_cited_by(identifier: str, max_nodes: int = 25) -> List[Dict[str, Any]]:
    work = fetch_work(identifier)
    return _fetch_cited_by(work, max_nodes=max_nodes)


def _work_node(work: Dict[str, Any]) -> Dict[str, Any]:
    work_id = _openalex_id(work.get("id")) or work.get("id") or "unknown"
    return {
        "id": work_id,
        "label": work.get("display_name") or work.get("title") or "Untitled work",
        "doi": work.get("doi"),
        "year": work.get("publication_year"),
        "kind": "work",
    }


def _add_stub(
    nodes: Dict[str, Dict[str, Any]],
    edges: List[Dict[str, str]],
    root: str,
    relation: str,
) -> None:
    stub_id = f"{root}:{relation}:stub"
    if stub_id in nodes:
        return
    nodes[stub_id] = {
        "id": stub_id,
        "label": "data unavailable",
        "kind": "stub",
    }
    if relation == "cited_by":
        edges.append({"source": stub_id, "target": root, "relation": relation})
    else:
        edges.append({"source": root, "target": stub_id, "relation": relation})


def build_citation_graph(
    identifier: str,
    depth: int = 1,
    max_nodes: int = 10,
) -> Dict[str, Any]:
    root_work = fetch_work(identifier)
    root_id = _openalex_id(root_work.get("id")) or _normalize_identifier(identifier)

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, str]] = []

    def add_node(work: Dict[str, Any]) -> str:
        node = _work_node(work)
        node_id = node["id"]
        nodes.setdefault(node_id, node)
        return node_id

    def expand(work: Dict[str, Any], current_depth: int) -> None:
        if current_depth >= depth:
            return

        work_id = add_node(work)

        references = work.get("referenced_works") or []
        if references:
            for ref_id in references[:max_nodes]:
                ref_work = fetch_work(ref_id)
                ref_node_id = add_node(ref_work)
                edges.append(
                    {"source": work_id, "target": ref_node_id, "relation": "references"}
                )
                expand(ref_work, current_depth + 1)
        else:
            _add_stub(nodes, edges, work_id, "references")

        cited_by = _fetch_cited_by(work, max_nodes)
        if cited_by:
            for cited_work in cited_by[:max_nodes]:
                cited_node_id = add_node(cited_work)
                edges.append(
                    {"source": cited_node_id, "target": work_id, "relation": "cited_by"}
                )
                expand(cited_work, current_depth + 1)
        else:
            _add_stub(nodes, edges, work_id, "cited_by")

    expand(root_work, 0)
    if root_id not in nodes:
        nodes[root_id] = _work_node(root_work)

    return {
        "root_id": root_id,
        "nodes": list(nodes.values()),
        "edges": edges,
    }


def build_local_citation_graph(
    document: Dict[str, Any],
    target_id: Optional[str],
    depth: int = 1,
    max_nodes: int = 10,
) -> Dict[str, Any]:
    extraction = (document.get("extraction") or {}).get("data") or {}
    references = extraction.get("references") or []
    resolution = (document.get("resolution") or {}).get("data") or []
    metadata = extraction.get("metadata") or {}

    ref_lookup = {ref.get("id"): ref for ref in references if ref.get("id")}
    res_lookup = {
        ref.get("reference_id"): ref for ref in resolution if ref.get("reference_id")
    }

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, str]] = []

    source_id = "source-document"
    source_label = metadata.get("title") or "Current document"
    source_year = metadata.get("year")
    nodes[source_id] = {
        "id": source_id,
        "label": source_label,
        "year": source_year,
        "kind": "source",
    }

    def add_reference(ref_id: str) -> str:
        reference = ref_lookup.get(ref_id) or {}
        resolved = res_lookup.get(ref_id) or {}
        label = (
            resolved.get("title")
            or reference.get("raw_reference")
            or reference.get("url")
            or "Untitled work"
        )
        year = resolved.get("year")
        nodes[ref_id] = {
            "id": ref_id,
            "label": label,
            "year": year,
            "doi": resolved.get("doi") or reference.get("doi"),
            "kind": "work",
        }
        return ref_id

    if target_id:
        root_id = add_reference(target_id)
        edges.append({"source": source_id, "target": root_id, "relation": "references"})
        _add_stub(nodes, edges, root_id, "references")
        _add_stub(nodes, edges, root_id, "cited_by")
    else:
        root_id = source_id
        for ref in references[:max_nodes]:
            ref_id = ref.get("id")
            if not ref_id:
                continue
            add_reference(ref_id)
            edges.append(
                {"source": source_id, "target": ref_id, "relation": "references"}
            )

    return {
        "root_id": root_id,
        "nodes": list(nodes.values()),
        "edges": edges,
    }
