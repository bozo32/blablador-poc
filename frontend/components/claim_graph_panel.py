from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import streamlit as st


try:
    from streamlit_agraph import Config, Edge, Node, agraph
except Exception:  # pragma: no cover
    Config = None  # type: ignore[assignment]
    Edge = None  # type: ignore[assignment]
    Node = None  # type: ignore[assignment]
    agraph = None  # type: ignore[assignment]


def _node_title(node: Dict[str, Any]) -> str:
    props = node.get("properties") or {}
    text = (props.get("parsed_text") or node.get("label") or "").strip()
    meta_bits: List[str] = []
    for key in ("document_id", "sentence_id", "citation_index", "claim_index"):
        val = props.get(key)
        if val is None or val == "":
            continue
        meta_bits.append(f"{key}={val}")
    meta = "\n" + "\n".join(meta_bits) if meta_bits else ""
    return f"{text}{meta}".strip()


def _edge_badge(edge: Dict[str, Any]) -> str:
    return ""


def _edge_color(edge: Dict[str, Any]) -> str:
    aggs = edge.get("aggregates") or {}
    n_support = int(aggs.get("n_support") or 0)
    n_contra = int(aggs.get("n_contradict") or 0)
    n_total = int(aggs.get("n_total") or 0)
    if n_total <= 0:
        return "#d9e2ef"  # ws-border
    if n_support > n_contra:
        return "#16a34a"  # green-600
    if n_contra > n_support:
        return "#dc2626"  # red-600
    return "#4b5563"  # ws-muted


def _edge_width(edge: Dict[str, Any]) -> int:
    aggs = edge.get("aggregates") or {}
    n_total = int(aggs.get("n_total") or 0)
    return int(max(1, min(8, 1 + n_total)))


def _edge_dashes(edge: Dict[str, Any]) -> bool:
    props = edge.get("properties") or {}
    return str(props.get("source") or "").strip() == "external_search"


def _build_elements(
    payload: Dict[str, Any], *, center_id: Optional[str]
) -> Tuple[List[Any], List[Any]]:
    nodes: List[Any] = []
    edges: List[Any] = []
    raw_nodes = payload.get("nodes") or []
    raw_edges = payload.get("edges") or []

    for raw in raw_nodes:
        node_id = str((raw or {}).get("id") or "").strip()
        if not node_id:
            continue
        label = ""
        is_center = bool(center_id and node_id == center_id)
        nodes.append(
            Node(
                id=node_id,
                label=label,
                title=_node_title(raw or {}),
                size=28 if is_center else 14,
                color="#2780e3" if is_center else "#111827",
            )
        )

    for raw in raw_edges:
        src = str((raw or {}).get("source_id") or "").strip()
        tgt = str((raw or {}).get("target_id") or "").strip()
        if not src or not tgt:
            continue
        edge_id = (raw or {}).get("edge_id")
        label = _edge_badge(raw or {})
        edges.append(
            Edge(
                source=src,
                target=tgt,
                color=_edge_color(raw or {}),
                id=str(edge_id) if edge_id is not None else None,
                label=label,
                width=_edge_width(raw or {}),
                dashes=_edge_dashes(raw or {}),
            )
        )

    return nodes, edges


def _parse_selection(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        text = value.strip()
        return {"type": "node", "id": text} if text else {}
    if isinstance(value, dict):
        nodes = value.get("nodes")
        edges = value.get("edges")
        if isinstance(nodes, list) and nodes:
            node_id = str(nodes[0]).strip()
            return {"type": "node", "id": node_id} if node_id else {}
        if isinstance(edges, list) and edges:
            edge_id = str(edges[0]).strip()
            return {"type": "edge", "id": edge_id} if edge_id else {}
        for key in ("node", "node_id", "selected_node"):
            if key in value and value.get(key):
                node_id = str(value.get(key)).strip()
                return {"type": "node", "id": node_id} if node_id else {}
        for key in ("edge", "edge_id", "selected_edge"):
            if key in value and value.get(key):
                edge_id = str(value.get(key)).strip()
                return {"type": "edge", "id": edge_id} if edge_id else {}
    return {}


def render(
    payload: Dict[str, Any],
    *,
    center_claim_id: Optional[str],
    height: int = 560,
    key: str = "claim-graph",
) -> Dict[str, Any]:
    if agraph is None or Node is None or Edge is None or Config is None:
        st.info(
            "Claim graph renderer unavailable (missing streamlit-agraph dependency)."
        )
        return {}

    nodes, edges = _build_elements(payload or {}, center_id=center_claim_id)

    config = Config(
        width=1200,
        height=int(height),
        directed=True,
        physics=True,
        hierarchical=False,
        link_length=180,
        node_highlight_behavior=True,
    )

    selection_raw = agraph(nodes=nodes, edges=edges, config=config)
    selection = _parse_selection(selection_raw)
    if selection:
        st.session_state[f"{key}::selection"] = selection
    return selection
