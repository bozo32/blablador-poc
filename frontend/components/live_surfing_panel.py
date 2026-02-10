from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import streamlit as st

from frontend import graph_api, ledger_api
from frontend.ingestion_api import get_document_body
from frontend.components import cytoscape_panel


@dataclass(frozen=True)
class CiteSpanKey:
    doc_id: str
    sentence_id: str
    citation_index: int
    reference_id: str


def _citespan_id(key: CiteSpanKey) -> str:
    ref = str(key.reference_id or "").strip() or "_"
    return f"citespan:{key.doc_id}:{key.sentence_id}:{int(key.citation_index)}:{ref}"


def _parse_citespan_id(csid: str) -> Optional[CiteSpanKey]:
    text = str(csid or "").strip()
    if not text.startswith("citespan:"):
        return None
    parts = text.split(":")
    if len(parts) < 5:
        return None
    _prefix, doc_id, sentence_id, cite_idx, ref = parts[:5]
    try:
        citation_index = int(cite_idx)
    except Exception:
        return None
    reference_id = "" if ref == "_" else str(ref)
    return CiteSpanKey(
        doc_id=str(doc_id),
        sentence_id=str(sentence_id),
        citation_index=int(citation_index),
        reference_id=str(reference_id),
    )


def _seed_state() -> None:
    st.session_state.setdefault("surf_live_seed_doc", "")
    st.session_state.setdefault("surf_live_expanded_works", [])
    st.session_state.setdefault("surf_live_expanded_citespans", [])
    st.session_state.setdefault("surf_live_expanded_work_citespans", [])
    st.session_state.setdefault("surf_live_active_citespan_doc", "")
    st.session_state.setdefault("surf_live_selection", {})
    st.session_state.setdefault("surf_live_show_labels", False)
    st.session_state.setdefault("surf_live_follow_active_citation", True)
    st.session_state.setdefault("surf_live_last_callout", "")
    st.session_state.setdefault("surf_live_claim_cache", {})
    st.session_state.setdefault("surf_live_ref_cache", {})


def _expanded_set(key: str) -> set[str]:
    raw = st.session_state.get(key)
    if isinstance(raw, set):
        return set(str(x) for x in raw if str(x).strip())
    if isinstance(raw, list):
        return set(str(x) for x in raw if str(x).strip())
    return set()


def _store_expanded(key: str, values: set[str]) -> None:
    st.session_state[key] = sorted(values)


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _get_claim_nodes(doc_id: str) -> list[dict]:
    cache = _as_dict(st.session_state.get("surf_live_claim_cache"))
    doc_id = str(doc_id or "").strip()
    if not doc_id:
        return []
    if doc_id in cache and isinstance(cache.get(doc_id), list):
        return cache.get(doc_id) or []
    try:
        payload = graph_api.list_claim_nodes(doc_id=doc_id, limit=5000)
    except Exception:
        payload = {}
    nodes = payload.get("nodes") or []
    cache[doc_id] = nodes
    st.session_state["surf_live_claim_cache"] = cache
    return nodes


def _resolve_reference_targets(
    *,
    citing_doc_id: str,
    reference_ids: list[str],
) -> dict[str, Optional[str]]:
    cache = _as_dict(st.session_state.get("surf_live_ref_cache"))
    citing_doc_id = str(citing_doc_id or "").strip()
    if not citing_doc_id:
        return {}

    # cache key = f"{citing_doc_id}::{reference_id}"
    missing: list[str] = []
    out: dict[str, Optional[str]] = {}
    for rid in reference_ids:
        rid = str(rid or "").strip()
        if not rid:
            continue
        key = f"{citing_doc_id}::{rid}"
        if key in cache:
            val = cache.get(key)
            out[rid] = str(val) if val else None
        else:
            missing.append(rid)

    if missing:
        try:
            resp = graph_api.resolve_references(
                citing_doc_id=citing_doc_id,
                reference_ids=missing,
            )
        except Exception:
            resp = {}
        mapping = _as_dict(resp.get("mapping"))
        for rid in missing:
            val = mapping.get(rid)
            cache[f"{citing_doc_id}::{rid}"] = val
            out[rid] = str(val) if val else None
        st.session_state["surf_live_ref_cache"] = cache

    return out


def _ledger_work_graph(api_url: str) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    """Return ingest-id keyed works and directed cite edges between ingested docs."""
    payload = ledger_api.get_ledger(api_url)
    rows = payload.get("rows") or []

    num_to_ingest: dict[int, str] = {}
    work_by_id: dict[str, dict] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        ingest_id = str(r.get("ingest_id") or "").strip()
        num = r.get("num")
        try:
            num_i = int(num)
        except Exception:
            continue
        if not ingest_id:
            continue
        num_to_ingest[num_i] = ingest_id
        work_by_id[ingest_id] = {
            "id": ingest_id,
            "short": r.get("short"),
            "title": r.get("title"),
            "status": r.get("status"),
            "assigned": r.get("assigned"),
            "anchored": r.get("anchored"),
            "extracted": r.get("extracted"),
            "resolved": r.get("resolved"),
        }

    edges: list[tuple[str, str]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            src_num = int(r.get("num"))
        except Exception:
            continue
        src = num_to_ingest.get(src_num)
        if not src:
            continue
        for t in r.get("outgoing_live") or r.get("outgoing") or []:
            try:
                tgt_num = int(t)
            except Exception:
                continue
            tgt = num_to_ingest.get(tgt_num)
            if tgt and tgt != src:
                edges.append((src, tgt))

    edges = sorted(set(edges))
    return work_by_id, edges


def render(*, api_url: str, seed_doc_id: str) -> None:
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
    expanded_work_citespans = _expanded_set("surf_live_expanded_work_citespans")
    selection = st.session_state.get("surf_live_selection")
    if not isinstance(selection, dict):
        selection = {}

    # --- Follow active citation from Reading/Chasing -----------------------
    if bool(st.session_state.get("surf_live_follow_active_citation")):
        active = st.session_state.get("selected_callout_tuple")
        if isinstance(active, dict):
            doc_id = str(active.get("doc_id") or "").strip()
            sentence_id = str(active.get("sentence_id") or "").strip()
            ref_id = str(active.get("target_id") or "").strip()
            try:
                cite_idx = int(active.get("citation_index"))
            except Exception:
                cite_idx = None
            if doc_id and sentence_id and cite_idx is not None:
                csid = _citespan_id(
                    CiteSpanKey(
                        doc_id=doc_id,
                        sentence_id=sentence_id,
                        citation_index=int(cite_idx),
                        reference_id=ref_id,
                    )
                )
                last = str(st.session_state.get("surf_live_last_callout") or "")
                if csid and csid != last:
                    st.session_state["surf_live_last_callout"] = csid
                    expanded_works.add(doc_id)
                    st.session_state["surf_live_pending_focus"] = {"node_id": csid}
                    st.session_state["surf_live_selection"] = {
                        "type": "node",
                        "id": csid,
                    }
                    selection = {"type": "node", "id": csid}

    # --- Work graph (ledger-backed) ----------------------------------------
    try:
        work_by_id, work_edges = _ledger_work_graph(api_url)
    except Exception as exc:
        st.error(f"Work graph unavailable: {exc}")
        return

    if seed not in work_by_id:
        # Fallback: still render a minimal seed node.
        work_by_id[seed] = {
            "id": seed,
            "short": "Selected document",
            "title": seed,
            "status": "orange",
            "assigned": False,
            "anchored": True,
            "extracted": False,
            "resolved": False,
        }

    expanded_works.add(seed)

    # --- Claims for expanded docs (for "needs work" signals) --------------
    claim_count_by_citespan: dict[tuple[str, str, int, str], int] = {}
    claim_by_id: dict[str, dict] = {}
    for wid in sorted(expanded_works):
        for cn in _get_claim_nodes(wid):
            if not isinstance(cn, dict):
                continue
            cid = str(cn.get("id") or "").strip()
            if not cid:
                continue
            claim_by_id[cid] = cn
            props = cn.get("properties") or {}
            doc_id = str(props.get("document_id") or "").strip()
            sentence_id = str(props.get("sentence_id") or "").strip()
            ref_id = str(props.get("target_id") or "").strip()
            try:
                citation_index = int(props.get("citation_index"))
            except Exception:
                continue
            key = (doc_id, sentence_id, int(citation_index), ref_id)
            claim_count_by_citespan[key] = int(claim_count_by_citespan.get(key, 0) + 1)

    # --- Elements ----------------------------------------------------------
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
            "selector": "node[type = 'CiteSpanBucket']",
            "style": {
                "background-color": "#0f766e",
                "width": 26,
                "height": 26,
                "border-width": 4,
                "border-color": "#111827",
                "label": "data(count_label)",
                "text-valign": "center",
                "text-halign": "center",
                "color": "#ffffff",
                "font-size": 12,
                "font-weight": "bold",
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

    for wid, w in work_by_id.items():
        short = str(w.get("short") or "").strip() or "Work"
        title = str(w.get("title") or "").strip()
        hover = (short + ("\n" + title if title else "")).strip()
        elements.append(
            {
                "data": {
                    "id": wid,
                    "type": "Work",
                    "label": short,
                    "hover": hover,
                }
            }
        )

    for src, tgt in work_edges:
        if src not in work_by_id or tgt not in work_by_id:
            continue
        elements.append(
            {
                "data": {
                    "id": f"workedge:{src}->{tgt}",
                    "source": src,
                    "target": tgt,
                    "type": "CITES",
                    "arrow": "none",
                }
            }
        )

    citespan_records: dict[str, dict] = {}
    citespans_by_doc: dict[str, list[dict]] = {}

    for wid in sorted(expanded_works):
        try:
            body = get_document_body(api_url, wid)
        except Exception:
            continue
        paragraphs = body.get("paragraphs") or []
        ref_ids: list[str] = []
        raw_segments: list[dict] = []

        for para in paragraphs:
            for sent in para.get("sentences") or []:
                sentence_id = str(sent.get("sentence_id") or "").strip()
                for seg in sent.get("segments") or []:
                    if not isinstance(seg, dict):
                        continue
                    if str(seg.get("type") or "").strip() != "citation":
                        continue
                    ref_id = str(seg.get("target_id") or "").strip()
                    if ref_id:
                        ref_ids.append(ref_id)
                    raw_segments.append({"sentence_id": sentence_id, **seg})

        ref_ids = sorted(set(ref_ids))
        resolved = _resolve_reference_targets(
            citing_doc_id=wid,
            reference_ids=ref_ids,
        )

        work_cites: list[dict] = []

        for seg in raw_segments:
            sentence_id = str(seg.get("sentence_id") or "").strip()
            ref_id = str(seg.get("target_id") or "").strip()
            try:
                citation_index = int(seg.get("citation_index"))
            except Exception:
                continue
            if not sentence_id:
                continue
            key = CiteSpanKey(
                doc_id=wid,
                sentence_id=sentence_id,
                citation_index=int(citation_index),
                reference_id=ref_id,
            )
            csid = _citespan_id(key)
            label = str(seg.get("label") or seg.get("callout") or "citation")
            claim_n = int(
                claim_count_by_citespan.get(
                    (
                        key.doc_id,
                        key.sentence_id,
                        int(key.citation_index),
                        str(key.reference_id),
                    ),
                    0,
                )
            )
            tgt_ingest = resolved.get(ref_id)
            citespan_records[csid] = {
                "id": csid,
                "doc_id": wid,
                "sentence_id": sentence_id,
                "citation_index": int(citation_index),
                "reference_id": ref_id,
                "target_ingest_id": tgt_ingest,
                "label": label,
                "callout": seg.get("callout"),
                "claim_count": claim_n,
            }

            work_cites.append(citespan_records[csid])

        citespans_by_doc[wid] = sorted(
            work_cites,
            key=lambda r: int(r.get("citation_index") or 0),
        )

        # Collapse or expand cite spans for this work.
        if wid not in expanded_work_citespans:
            unclaimed = sum(
                1
                for r in citespans_by_doc.get(wid) or []
                if int(r.get("claim_count") or 0) == 0
            )
            bucket_id = f"citespanbucket:{wid}"
            elements.append(
                {
                    "data": {
                        "id": bucket_id,
                        "type": "CiteSpanBucket",
                        "count": int(len(citespans_by_doc.get(wid) or [])),
                        "count_label": str(int(len(citespans_by_doc.get(wid) or []))),
                        "hover": (
                            f"CiteSpans: {len(citespans_by_doc.get(wid) or [])}\n"
                            f"Needs claims: {unclaimed}"
                        ),
                        "doc_id": wid,
                    }
                }
            )
            elements.append(
                {
                    "data": {
                        "id": f"work2bucket:{wid}",
                        "source": wid,
                        "target": bucket_id,
                        "type": "HAS_CITESPANS",
                        "arrow": "none",
                    }
                }
            )
        else:
            cap = 40
            visible = (citespans_by_doc.get(wid) or [])[:cap]
            remainder = (citespans_by_doc.get(wid) or [])[cap:]
            for rec in visible:
                csid = str(rec.get("id") or "")
                tgt_ingest = rec.get("target_ingest_id")
                elements.append(
                    {
                        "data": {
                            "id": csid,
                            "type": "CiteSpan",
                            "label": str(rec.get("label") or "citation"),
                            "hover": str(rec.get("label") or "citation"),
                            "doc_id": wid,
                            "sentence_id": str(rec.get("sentence_id") or ""),
                            "citation_index": int(rec.get("citation_index") or 0),
                            "reference_id": str(rec.get("reference_id") or ""),
                            "claim_count": int(rec.get("claim_count") or 0),
                        }
                    }
                )
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
                if tgt_ingest and tgt_ingest in work_by_id:
                    elements.append(
                        {
                            "data": {
                                "id": f"cs2work:{csid}->{tgt_ingest}",
                                "source": csid,
                                "target": tgt_ingest,
                                "type": "CITES_TARGET",
                                "arrow": "none",
                            }
                        }
                    )

            if remainder:
                bucket_id = f"citespanbucket:{wid}:more"
                elements.append(
                    {
                        "data": {
                            "id": bucket_id,
                            "type": "CiteSpanBucket",
                            "count": int(len(remainder)),
                            "count_label": str(int(len(remainder))),
                            "hover": f"More CiteSpans: {len(remainder)}",
                            "doc_id": wid,
                        }
                    }
                )
                elements.append(
                    {
                        "data": {
                            "id": f"work2bucket:{wid}:more",
                            "source": wid,
                            "target": bucket_id,
                            "type": "HAS_MORE_CITESPANS",
                            "arrow": "none",
                        }
                    }
                )

    # Expanded CiteSpans -> ClaimSpans
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
            if str(props.get("target_id") or "").strip() != str(key.reference_id or ""):
                continue

            parsed_text = str(props.get("parsed_text") or cn.get("label") or "").strip()
            elements.append(
                {
                    "data": {
                        "id": cid,
                        "type": "ClaimSpan",
                        "label": parsed_text,
                        "hover": parsed_text,
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

    # --- Layout: top panels (scrollable) + full-width graph ----------------
    top_left, top_right = st.columns([1, 1], gap="large")
    with top_left:
        with st.container(height=240):
            st.markdown("**Expanded**")
            st.checkbox("Show labels (debug)", key="surf_live_show_labels")
            st.checkbox(
                "Follow active citation",
                key="surf_live_follow_active_citation",
            )
            if st.button("Collapse all", key="surf-live-collapse"):
                expanded_works = {seed}
                expanded_cites = set()
                expanded_work_citespans = set()
                st.session_state["surf_live_active_citespan_doc"] = ""

            st.caption(f"Works expanded: {len(expanded_works)}")
            st.caption(f"CiteSpans expanded: {len(expanded_cites)}")
            st.caption(f"CiteSpan lists: {len(expanded_work_citespans)}")

            st.divider()
            st.markdown("**Work list**")
            work_ids = sorted(work_by_id.keys())
            for wid in work_ids[:10]:
                short = str((work_by_id.get(wid) or {}).get("short") or wid)
                is_on = wid in expanded_works
                if st.toggle(short, value=is_on, key=f"surf-live-w:{wid}") != is_on:
                    if is_on:
                        expanded_works.discard(wid)
                    else:
                        expanded_works.add(wid)

    with top_right:
        with st.container(height=240):
            st.markdown("**Inspector**")
            sel_id = str(selection.get("id") or "").strip()
            if sel_id in work_by_id:
                rec = work_by_id.get(sel_id) or {}
                st.caption("Work")
                st.write(str(rec.get("short") or sel_id))
                if st.button("Expand cite spans", key="surf-live-expand-work"):
                    expanded_works.add(sel_id)
                if st.button("Collapse cite spans", key="surf-live-collapse-work"):
                    expanded_works.discard(sel_id)
            elif sel_id.startswith("citespan:"):
                rec = citespan_records.get(sel_id) or {}
                st.caption("CiteSpan")
                st.write(str(rec.get("label") or sel_id))
                st.caption(f"claims confirmed: {int(rec.get('claim_count') or 0)}")
                if st.button("Expand claim spans", key="surf-live-expand-cs"):
                    expanded_cites.add(sel_id)
                if st.button("Collapse claim spans", key="surf-live-collapse-cs"):
                    expanded_cites.discard(sel_id)
            elif sel_id and sel_id in claim_by_id:
                cn = claim_by_id.get(sel_id) or {}
                props = cn.get("properties") or {}
                st.caption("ClaimSpan")
                st.write(
                    str((props.get("parsed_text") or cn.get("label") or "")).strip()
                )
            else:
                st.caption("Click a Work or CiteSpan.")

    st.divider()

    pending_focus = st.session_state.pop("surf_live_pending_focus", None)
    focus = {}
    if isinstance(pending_focus, dict):
        node_id = str(pending_focus.get("node_id") or "").strip()
        if node_id:
            focus = {"nodeIds": [node_id], "padding": 40}

    picked = cytoscape_panel.render(
        elements,
        style=style,
        height=720,
        key="surf-live",
        selection=selection,
        focus=focus,
        options={"showLabels": bool(st.session_state.get("surf_live_show_labels"))},
    )
    if picked:
        st.session_state["surf_live_selection"] = picked

        # Double-click on collapsed citespan bucket expands it.
        try:
            action = str((picked or {}).get("action") or "").strip()
            node_id = str((picked or {}).get("id") or "").strip()
        except Exception:
            action = ""
            node_id = ""
        if action == "dblclick" and node_id.startswith("citespanbucket:"):
            parts = node_id.split(":")
            doc_id = str(parts[1] if len(parts) > 1 else "").strip()
            if doc_id:
                expanded_work_citespans.add(doc_id)
                st.session_state["surf_live_active_citespan_doc"] = doc_id
                st.rerun()
        elif node_id.startswith("citespanbucket:"):
            parts = node_id.split(":")
            doc_id = str(parts[1] if len(parts) > 1 else "").strip()
            if doc_id:
                st.session_state["surf_live_active_citespan_doc"] = doc_id
        elif node_id in work_by_id:
            st.session_state["surf_live_active_citespan_doc"] = node_id

    active_doc = str(
        st.session_state.get("surf_live_active_citespan_doc") or ""
    ).strip()
    if not active_doc:
        active_doc = seed
    with st.expander("CiteSpans", expanded=True):
        rows = citespans_by_doc.get(active_doc) or []
        if not rows:
            st.caption(
                "No cite spans found for this document (or extraction not complete)."
            )
        else:
            with st.container(height=320):
                for rec in rows:
                    csid = str(rec.get("id") or "")
                    cite_idx = int(rec.get("citation_index") or 0)
                    label = str(rec.get("label") or "citation")
                    claim_n = int(rec.get("claim_count") or 0)
                    prefix = "TODO" if claim_n == 0 else "OK"
                    if st.button(
                        f"{prefix} {cite_idx}: {label}",
                        key=f"surf-live-cs:{csid}",
                        use_container_width=True,
                    ):
                        expanded_works.add(active_doc)
                        expanded_work_citespans.add(active_doc)
                        st.session_state["surf_live_pending_focus"] = {"node_id": csid}
                        st.session_state["surf_live_selection"] = {
                            "type": "node",
                            "id": csid,
                        }
                        st.session_state["surf_live_active_citespan_doc"] = active_doc
                        st.rerun()

    _store_expanded("surf_live_expanded_works", expanded_works)
    _store_expanded("surf_live_expanded_citespans", expanded_cites)
    _store_expanded("surf_live_expanded_work_citespans", expanded_work_citespans)
