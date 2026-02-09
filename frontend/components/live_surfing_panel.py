from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import streamlit as st

from frontend import graph_api
from frontend.ingestion_api import get_citation_graph, get_document_body
from frontend.components import cytoscape_panel


@dataclass(frozen=True)
class CiteSpanKey:
    doc_id: str
    sentence_id: str
    citation_index: int
    target_id: str


def _citespan_id(key: CiteSpanKey) -> str:
    tgt = str(key.target_id or "").strip() or "_"
    return f"citespan:{key.doc_id}:{key.sentence_id}:{int(key.citation_index)}:{tgt}"


def _parse_citespan_id(csid: str) -> Optional[CiteSpanKey]:
    text = str(csid or "").strip()
    if not text.startswith("citespan:"):
        return None
    parts = text.split(":")
    if len(parts) < 5:
        return None
    _prefix, doc_id, sentence_id, cite_idx, tgt = parts[:5]
    try:
        citation_index = int(cite_idx)
    except Exception:
        return None
    target_id = "" if tgt == "_" else str(tgt)
    return CiteSpanKey(
        doc_id=str(doc_id),
        sentence_id=str(sentence_id),
        citation_index=int(citation_index),
        target_id=str(target_id),
    )


def _seed_state() -> None:
    st.session_state.setdefault("surf_live_seed_doc", "")
    st.session_state.setdefault("surf_live_expanded_works", [])
    st.session_state.setdefault("surf_live_expanded_citespans", [])
    st.session_state.setdefault("surf_live_selection", {})
    st.session_state.setdefault("surf_live_show_claim_links", False)
    st.session_state.setdefault("surf_live_show_labels", False)
    st.session_state.setdefault("surf_live_follow_active_citation", True)
    st.session_state.setdefault("surf_live_last_callout", "")


def _expanded_set(key: str) -> set[str]:
    raw = st.session_state.get(key)
    if isinstance(raw, set):
        return set(str(x) for x in raw if str(x).strip())
    if isinstance(raw, list):
        return set(str(x) for x in raw if str(x).strip())
    return set()


def _store_expanded(key: str, values: set[str]) -> None:
    st.session_state[key] = sorted(values)


def _work_node_id(raw_id: Any) -> str:
    return str(raw_id or "").strip()


def render(*, api_url: str, seed_doc_id: str) -> None:
    """Live Surfing explorer (Work -> CiteSpan -> ClaimSpan).

    This is the single meandering interface. It starts at document level and
    expands on demand.
    """
    _seed_state()
    seed_doc_id = str(seed_doc_id or "").strip()
    if seed_doc_id:
        st.session_state["surf_live_seed_doc"] = seed_doc_id

    seed = str(st.session_state.get("surf_live_seed_doc") or "").strip()
    if not seed:
        st.info("Select a document in Reading to seed Surfing.")
        return

    expanded_works = _expanded_set("surf_live_expanded_works")
    expanded_cites = _expanded_set("surf_live_expanded_citespans")
    selection = st.session_state.get("surf_live_selection")
    if not isinstance(selection, dict):
        selection = {}

    # --- Fetch work-level citation graph (seeded) ---------------------------
    try:
        work_graph = get_citation_graph(
            api_url,
            seed,
            target_id=None,
            depth=1,
            max_nodes=80,
        )
    except Exception as exc:
        st.error(f"Work graph unavailable: {exc}")
        return

    work_nodes = work_graph.get("nodes") or []
    work_edges = work_graph.get("edges") or []

    work_by_id: dict[str, dict] = {}
    for n in work_nodes:
        if not isinstance(n, dict):
            continue
        nid = _work_node_id(n.get("id"))
        if nid:
            work_by_id[nid] = n

    # Auto-expand the seed work.
    expanded_works.add(seed)

    # --- Claim-node index for "work needs work" indicators -----------------
    try:
        claim_payload = graph_api.list_claim_nodes(doc_id=seed, limit=5000)
    except Exception:
        claim_payload = {}
    claim_nodes = claim_payload.get("nodes") or []

    claim_count_by_citespan: dict[tuple[str, str, int, str], int] = {}
    for cn in claim_nodes:
        if not isinstance(cn, dict):
            continue
        props = cn.get("properties") or {}
        doc_id = str(props.get("document_id") or "").strip()
        sentence_id = str(props.get("sentence_id") or "").strip()
        target_id = str(props.get("target_id") or "").strip()
        try:
            citation_index = int(props.get("citation_index"))
        except Exception:
            continue
        if not doc_id or not sentence_id:
            continue
        key = (doc_id, sentence_id, int(citation_index), str(target_id))
        claim_count_by_citespan[key] = int(claim_count_by_citespan.get(key, 0) + 1)

    # --- Build Cytoscape elements ------------------------------------------
    elements: list[dict] = []
    style = [
        {
            "selector": "node",
            "style": {
                "background-color": "#111827",
                "border-color": "#d9e2ef",
                "border-width": 1,
                "width": 18,
                "height": 18,
                "label": "",
                "font-size": 10,
                "text-wrap": "wrap",
                "text-max-width": 220,
                "text-valign": "bottom",
                "text-halign": "center",
                "color": "#111827",
            },
        },
        {
            "selector": "node[type = 'Work']",
            "style": {
                "background-color": "#2780e3",
                "width": 26,
                "height": 26,
                "border-width": 1,
                "border-color": "#1f2937",
            },
        },
        {
            "selector": "node[type = 'CiteSpan']",
            "style": {
                "background-color": "#0f766e",
                "width": 18,
                "height": 18,
                "border-width": 2,
                "border-color": "#d9e2ef",
            },
        },
        {
            "selector": "node[type = 'CiteSpan'][claim_count = 0]",
            "style": {
                "border-width": 4,
                "border-color": "#ef4444",
            },
        },
        {
            "selector": "node[type = 'ClaimSpan']",
            "style": {
                "background-color": "#6b7280",
                "width": 14,
                "height": 14,
                "border-width": 1,
                "border-color": "#d9e2ef",
            },
        },
        {
            "selector": "edge",
            "style": {
                "width": 2,
                "line-color": "#d9e2ef",
                "target-arrow-color": "#d9e2ef",
                "target-arrow-shape": "triangle",
                "curve-style": "bezier",
                "label": "",
            },
        },
        {
            "selector": "edge[arrow = 'none']",
            "style": {
                "target-arrow-shape": "none",
            },
        },
        {
            "selector": ":selected",
            "style": {
                "border-width": 4,
                "border-color": "#111827",
                "line-color": "#111827",
                "target-arrow-color": "#111827",
            },
        },
    ]

    for wid, wn in work_by_id.items():
        label = str((wn or {}).get("label") or wid)
        year = (wn or {}).get("year")
        author = str((wn or {}).get("author") or "").strip()
        hover_bits = [bit for bit in [author, f"({year})" if year else ""] if bit]
        hover = " ".join(hover_bits) or label
        elements.append(
            {
                "data": {
                    "id": wid,
                    "type": "Work",
                    "label": label,
                    "hover": hover,
                    "kind": str((wn or {}).get("kind") or "").strip(),
                }
            }
        )

    for e in work_edges:
        if not isinstance(e, dict):
            continue
        src = _work_node_id(e.get("source"))
        tgt = _work_node_id(e.get("target"))
        if not src or not tgt:
            continue
        # Plain line at doc-level: no arrows.
        elements.append(
            {
                "data": {
                    "id": str(e.get("id") or f"workedge:{src}->{tgt}"),
                    "source": src,
                    "target": tgt,
                    "type": "CITES",
                    "arrow": "none",
                }
            }
        )

    # Expanded works -> citespans
    citespan_records: dict[str, dict] = {}
    for wid in sorted(expanded_works):
        try:
            body = get_document_body(api_url, wid)
        except Exception:
            continue
        paragraphs = body.get("paragraphs") or []
        for para in paragraphs:
            for sent in para.get("sentences") or []:
                sentence_id = str(sent.get("sentence_id") or "").strip()
                for seg in sent.get("segments") or []:
                    if not isinstance(seg, dict):
                        continue
                    if str(seg.get("type") or "").strip() != "citation":
                        continue
                    target_id = str(seg.get("target_id") or "").strip()
                    try:
                        citation_index = int(seg.get("citation_index"))
                    except Exception:
                        continue
                    key = CiteSpanKey(
                        doc_id=str(wid),
                        sentence_id=str(seg.get("sentence_id") or sentence_id or ""),
                        citation_index=int(citation_index),
                        target_id=str(target_id),
                    )
                    if not key.sentence_id:
                        continue
                    csid = _citespan_id(key)
                    label = str(seg.get("label") or seg.get("callout") or "citation")
                    claim_n = int(
                        claim_count_by_citespan.get(
                            (
                                key.doc_id,
                                key.sentence_id,
                                int(key.citation_index),
                                str(key.target_id),
                            ),
                            0,
                        )
                    )
                    # Store for inspector.
                    citespan_records[csid] = {
                        "id": csid,
                        "doc_id": key.doc_id,
                        "sentence_id": key.sentence_id,
                        "citation_index": int(key.citation_index),
                        "target_id": key.target_id,
                        "label": label,
                        "callout": seg.get("callout"),
                        "claim_count": claim_n,
                    }

                    # Citespan node (unlabeled on canvas).
                    elements.append(
                        {
                            "data": {
                                "id": csid,
                                "type": "CiteSpan",
                                "label": label,
                                "hover": label,
                                "doc_id": key.doc_id,
                                "sentence_id": key.sentence_id,
                                "citation_index": int(key.citation_index),
                                "target_id": key.target_id,
                                "claim_count": int(claim_n),
                            }
                        }
                    )
                    # Work -- CiteSpan (no arrows).
                    elements.append(
                        {
                            "data": {
                                "id": f"work2cs:{wid}->{csid}",
                                "source": wid,
                                "target": csid,
                                "type": "HAS_CITESPAN",
                                "arrow": "none",
                            }
                        }
                    )
                    # CiteSpan -- target Work (if present in work graph).
                    if key.target_id and key.target_id in work_by_id:
                        elements.append(
                            {
                                "data": {
                                    "id": f"cs2work:{csid}->{key.target_id}",
                                    "source": csid,
                                    "target": key.target_id,
                                    "type": "CITES_TARGET",
                                    "arrow": "none",
                                }
                            }
                        )

    # Expanded citespans -> claimspans.
    claim_by_id: dict[str, dict] = {}
    for cn in claim_nodes:
        if not isinstance(cn, dict):
            continue
        cid = str(cn.get("id") or "").strip()
        if cid:
            claim_by_id[cid] = cn

    for csid in sorted(expanded_cites):
        key = _parse_citespan_id(csid)
        if not key:
            continue
        for cid, cn in claim_by_id.items():
            props = cn.get("properties") or {}
            if str(props.get("document_id") or "").strip() != key.doc_id:
                continue
            if str(props.get("sentence_id") or "").strip() != key.sentence_id:
                continue
            try:
                ci = int(props.get("citation_index"))
            except Exception:
                continue
            if int(ci) != int(key.citation_index):
                continue
            # ClaimSpan node.
            parsed_text = str(props.get("parsed_text") or cn.get("label") or "").strip()
            elements.append(
                {
                    "data": {
                        "id": cid,
                        "type": "ClaimSpan",
                        "label": parsed_text,
                        "hover": parsed_text,
                        "doc_id": key.doc_id,
                        "sentence_id": key.sentence_id,
                        "citation_index": int(key.citation_index),
                    }
                }
            )
            elements.append(
                {
                    "data": {
                        "id": f"cs2claim:{csid}->{cid}",
                        "source": csid,
                        "target": cid,
                        "type": "HAS_CLAIM",
                        "arrow": "none",
                    }
                }
            )

    # --- Render layout ------------------------------------------------------
    left, center, right = st.columns([1, 3, 1], gap="large")

    with left:
        with st.container(height=680):
            st.markdown("**Surfing**")
            st.caption("Start at documents; expand into citation spans and claims.")
            st.checkbox("Show labels (debug)", key="surf_live_show_labels")
            st.checkbox(
                "Follow active citation",
                key="surf_live_follow_active_citation",
                help="When you click a citation in Reading/Chasing, focus it here.",
            )

            st.divider()
            st.markdown("**Expanded**")
            if st.button(
                "Collapse all", key="surf-live-collapse", use_container_width=True
            ):
                expanded_works = {seed}
                expanded_cites = set()

            st.caption(f"Works: {len(expanded_works)}")
            st.caption(f"CiteSpans: {len(expanded_cites)}")

            st.divider()
            st.markdown("**Work list**")
            for wid in list(work_by_id.keys())[:40]:
                label = str((work_by_id.get(wid) or {}).get("label") or wid)
                is_on = wid in expanded_works
                if st.toggle(label, value=is_on, key=f"surf-live-w:{wid}") != is_on:
                    if is_on:
                        expanded_works.discard(wid)
                    else:
                        expanded_works.add(wid)

    # Focus request from outside (e.g. right-pane click).
    pending_focus = st.session_state.pop("surf_live_pending_focus", None)
    focus = {}
    if isinstance(pending_focus, dict):
        node_id = str(pending_focus.get("node_id") or "").strip()
        if node_id:
            focus = {"nodeIds": [node_id], "padding": 40}

    with center:
        # If a citation is selected in Reading/Chasing, optionally auto-focus it.
        if bool(st.session_state.get("surf_live_follow_active_citation")):
            active = st.session_state.get("selected_callout_tuple")
            if isinstance(active, dict):
                try:
                    cite_idx = int(active.get("citation_index"))
                except Exception:
                    cite_idx = None
                doc_id = str(active.get("doc_id") or "").strip()
                sentence_id = str(active.get("sentence_id") or "").strip()
                target_id = str(active.get("target_id") or "").strip()
                if cite_idx is not None and doc_id and sentence_id:
                    key = CiteSpanKey(
                        doc_id=doc_id,
                        sentence_id=sentence_id,
                        citation_index=int(cite_idx),
                        target_id=target_id,
                    )
                    csid = _citespan_id(key)
                    last = str(st.session_state.get("surf_live_last_callout") or "")
                    if csid and csid != last:
                        st.session_state["surf_live_last_callout"] = csid
                        expanded_works.add(doc_id)
                        st.session_state["surf_live_pending_focus"] = {"node_id": csid}
                        st.session_state["surf_live_selection"] = {
                            "type": "node",
                            "id": csid,
                        }
                        pending_focus = {"node_id": csid}
                        focus = {"nodeIds": [csid], "padding": 40}

        picked = cytoscape_panel.render(
            elements,
            style=style,
            height=720,
            key="surf-live",
            selection=selection,
            focus=focus,
            options={
                "showLabels": bool(st.session_state.get("surf_live_show_labels")),
            },
        )
        if picked:
            st.session_state["surf_live_selection"] = picked
            selection = picked

    with right:
        with st.container(height=680):
            st.markdown("**Inspector**")
            if not selection:
                st.caption("Click a document (work) or cite span.")
            else:
                sel_id = str(selection.get("id") or "").strip()
                sel_type = str(selection.get("type") or "").strip()
                if sel_type == "node":
                    if sel_id in work_by_id:
                        rec = work_by_id.get(sel_id) or {}
                        st.caption("Work")
                        st.write(str(rec.get("label") or sel_id))
                        if st.button(
                            "Expand cite spans",
                            key="surf-live-expand-work",
                            use_container_width=True,
                        ):
                            expanded_works.add(sel_id)
                        if st.button(
                            "Collapse cite spans",
                            key="surf-live-collapse-work",
                            use_container_width=True,
                        ):
                            expanded_works.discard(sel_id)
                    elif sel_id.startswith("citespan:"):
                        rec = citespan_records.get(sel_id) or {}
                        st.caption("CiteSpan")
                        st.write(str(rec.get("label") or sel_id))
                        st.caption(
                            f"claims confirmed: {int(rec.get('claim_count') or 0)}"
                        )
                        if st.button(
                            "Expand claim spans",
                            key="surf-live-expand-cs",
                            use_container_width=True,
                        ):
                            expanded_cites.add(sel_id)
                        if st.button(
                            "Collapse claim spans",
                            key="surf-live-collapse-cs",
                            use_container_width=True,
                        ):
                            expanded_cites.discard(sel_id)

                        # Focus to this citespan from elsewhere.
                        st.session_state["surf_live_pending_focus"] = {
                            "node_id": sel_id
                        }
                    elif sel_id in claim_by_id:
                        cn = claim_by_id.get(sel_id) or {}
                        props = cn.get("properties") or {}
                        st.caption("ClaimSpan")
                        st.write(
                            str(
                                (props.get("parsed_text") or cn.get("label") or "")
                            ).strip()
                        )
                        st.caption(
                            (
                                f"doc={props.get('document_id')} "
                                f"cite={props.get('citation_index')}"
                            )
                        )
                    else:
                        st.caption(sel_id)
                else:
                    st.caption(sel_id)

            st.divider()
            st.markdown("**CiteSpans**")
            # If a citation is selected in Reading/Chasing, allow focusing it.
            active = st.session_state.get("selected_callout_tuple")
            if isinstance(active, dict):
                try:
                    cite_idx = int(active.get("citation_index"))
                except Exception:
                    cite_idx = None
                doc_id = str(active.get("doc_id") or "").strip()
                sentence_id = str(active.get("sentence_id") or "").strip()
                target_id = str(active.get("target_id") or "").strip()
                if cite_idx is not None and doc_id and sentence_id:
                    key = CiteSpanKey(
                        doc_id=doc_id,
                        sentence_id=sentence_id,
                        citation_index=int(cite_idx),
                        target_id=target_id,
                    )
                    csid = _citespan_id(key)
                    if st.button(
                        "Focus active citation",
                        key="surf-live-focus-active",
                        use_container_width=True,
                    ):
                        expanded_works.add(doc_id)
                        st.session_state["surf_live_pending_focus"] = {"node_id": csid}
                        st.session_state["surf_live_selection"] = {
                            "type": "node",
                            "id": csid,
                        }

            # To-do list: citespans in the seed doc with 0 confirmed claims.
            todo: list[dict] = []
            for csid, rec in citespan_records.items():
                if str(rec.get("doc_id") or "").strip() != seed:
                    continue
                if int(rec.get("claim_count") or 0) > 0:
                    continue
                todo.append(rec)
            todo = sorted(todo, key=lambda r: int(r.get("citation_index") or 0))
            if not todo:
                st.caption("No unclaimed citation spans detected in the seed work.")
            else:
                for rec in todo[:30]:
                    label = str(rec.get("label") or "citation")
                    cite_idx = int(rec.get("citation_index") or 0)
                    csid = str(rec.get("id") or "")
                    if st.button(
                        f"{cite_idx}: {label}",
                        key=f"surf-live-todo:{csid}",
                        use_container_width=True,
                    ):
                        expanded_works.add(seed)
                        st.session_state["surf_live_pending_focus"] = {"node_id": csid}
                        st.session_state["surf_live_selection"] = {
                            "type": "node",
                            "id": csid,
                        }

    _store_expanded("surf_live_expanded_works", expanded_works)
    _store_expanded("surf_live_expanded_citespans", expanded_cites)
