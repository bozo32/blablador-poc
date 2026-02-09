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


_WS_ACCENT = "#2780e3"
_WS_INK = "#111827"
_WS_MUTED = "#4b5563"
_WS_BORDER = "#d9e2ef"


def _node_title(node: Dict[str, Any]) -> str:
    label = str((node or {}).get("label") or "").strip()
    year = (node or {}).get("year")
    doi = str((node or {}).get("doi") or "").strip()
    kind = str((node or {}).get("kind") or "").strip()
    bits: List[str] = []
    if label:
        bits.append(label)
    if year:
        bits.append(f"year={year}")
    if doi:
        bits.append(f"doi={doi}")
    if kind:
        bits.append(f"kind={kind}")
    return "\n".join(bits).strip() or str((node or {}).get("id") or "").strip()


def _build_elements(
    payload: Dict[str, Any], *, root_id: Optional[str]
) -> Tuple[List[Any], List[Any]]:
    nodes: List[Any] = []
    edges: List[Any] = []
    raw_nodes = payload.get("nodes") or []
    raw_edges = payload.get("edges") or []

    for raw in raw_nodes:
        node_id = str((raw or {}).get("id") or "").strip()
        if not node_id:
            continue
        is_root = bool(root_id and node_id == root_id)
        kind = str((raw or {}).get("kind") or "").strip()
        color = _WS_ACCENT if is_root else _WS_INK
        if kind == "stub":
            color = _WS_MUTED
        nodes.append(
            Node(
                id=node_id,
                label="",  # no text on graph
                title=_node_title(raw or {}),
                size=28 if is_root else 14,
                color=color,
                borderWidth=1,
            )
        )

    for raw in raw_edges:
        src = str((raw or {}).get("source") or "").strip()
        tgt = str((raw or {}).get("target") or "").strip()
        if not src or not tgt:
            continue
        edges.append(
            Edge(
                source=src,
                target=tgt,
                color=_WS_BORDER,
                label="",  # no text on graph
                width=1,
                dashes=str((raw or {}).get("relation") or "") in {"cited_by"},
            )
        )

    return nodes, edges


def render(
    payload: Dict[str, Any],
    *,
    height: int = 560,
    key: str = "work-citation-graph",
) -> Dict[str, Any]:
    if agraph is None or Node is None or Edge is None or Config is None:
        st.info("Work graph renderer unavailable (missing streamlit-agraph).")
        return {}

    root_id = str((payload or {}).get("root_id") or "").strip() or None
    nodes, edges = _build_elements(payload or {}, root_id=root_id)

    config = Config(
        width=1200,
        height=int(height),
        directed=True,
        physics=True,
        hierarchical=False,
        link_length=170,
        node_highlight_behavior=True,
        highlightColor=_WS_ACCENT,
    )

    selection_raw = agraph(nodes=nodes, edges=edges, config=config)
    if selection_raw:
        st.session_state[f"{key}::selection"] = selection_raw
    return selection_raw or {}
