from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import streamlit as st

from frontend import graph_api, ledger_api
from frontend.ingestion_api import (
    get_document_body,
    get_span_bundle,
    lookup_citation_window_span,
    trigger_resolution,
)
from frontend.components import cytoscape_panel
from frontend.state_keys import (
    WORKSPACE_ACTIVE_TAB,
    WORKSPACE_TAB_DOCUMENT,
    WORKSPACE_TAB_GRAPH,
    WORKSPACE_TAB_REVIEW,
)


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
    st.session_state.setdefault("surf_live_debug", False)
    st.session_state.setdefault("surf_live_follow_active_citation", True)
    st.session_state.setdefault("surf_live_show_work_graph", False)
    st.session_state.setdefault("surf_live_last_callout", "")
    st.session_state.setdefault("surf_live_layout_nonce", 0)
    st.session_state.setdefault("surf_live_component_nonce", 0)


def _active_reviewer_uid() -> str:
    meta = st.session_state.get("project_meta")
    if isinstance(meta, dict):
        value = str(meta.get("active_reviewer_uid") or "").strip()
        if value:
            return value
    value2 = str(st.session_state.get("active_reviewer_uid") or "").strip()
    return value2 or "default"
    st.session_state.setdefault("surf_live_claim_cache", {})
    st.session_state.setdefault("surf_live_ref_cache", {})
    st.session_state.setdefault("surf_live_span_cache", {})
    st.session_state.setdefault("surf_live_bundle_cache", {})
    st.session_state.setdefault("surf_live_last_event_seq", 0)
    st.session_state.setdefault("surf_live_citespan_followup", {})
    st.session_state.setdefault("surf_live_open_todo_by_work", {})
    st.session_state.setdefault("surf_live_todo_cap_by_work", {})


def _get_span_id(
    *, api_url: str, ingest_id: str, citation_index: int, target_id: str
) -> str:
    cache = _as_dict(st.session_state.get("surf_live_span_cache"))
    key = f"{ingest_id}::{int(citation_index)}::{str(target_id or '').strip()}"
    if key in cache:
        return str(cache.get(key) or "")
    try:
        resp = lookup_citation_window_span(
            api_url,
            ingest_id=str(ingest_id),
            citation_index=int(citation_index),
            target_id=str(target_id).strip() or None,
        )
    except Exception:
        resp = {}
    span_id = str(resp.get("span_id") or "").strip()
    cache[key] = span_id
    st.session_state["surf_live_span_cache"] = cache
    return span_id


def _get_span_bundle_cached(
    *,
    api_url: str,
    span_id: str,
    reviewer_uid: str,
) -> dict:
    cache = _as_dict(st.session_state.get("surf_live_bundle_cache"))
    key = f"{span_id}::{reviewer_uid}"
    if key in cache and isinstance(cache.get(key), dict):
        return cache.get(key) or {}
    try:
        bundle = get_span_bundle(
            api_url,
            str(span_id),
            reviewer_uid=str(reviewer_uid or "default"),
            include_history=False,
        )
    except Exception:
        bundle = {}
    cache[key] = bundle
    st.session_state["surf_live_bundle_cache"] = cache
    return bundle


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


def _citespan_triage(*, citespan_id: str, unresolved: bool, claim_count: int) -> str:
    """Return triage marker for a CiteSpan.

    Values:
    - "todo": follow up (stub/needs work)
    - "ignore": do not follow up
    - "": automatic / none
    """
    csid = str(citespan_id or "").strip()
    overrides = _as_dict(st.session_state.get("surf_live_citespan_followup"))
    raw = str(overrides.get(csid) or "").strip().lower()
    if raw in {"todo", "ignore"}:
        return raw
    if bool(unresolved) or int(claim_count) == 0:
        return "todo"
    return ""


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


def _limit_work_graph(
    *,
    seed: str,
    work_by_id: dict[str, dict],
    work_edges: list[tuple[str, str]],
    expanded_works: set[str],
    expanded_work_citespans: set[str],
    max_nodes: int = 80,
) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    """Reduce the ledger graph to a small neighborhood.

    Surfing uses CiteSpans for deep navigation; the work-level graph is meant to be
    a lightweight scaffold. Rendering the full ledger graph makes expansions feel
    like they "explode".
    """
    seed = str(seed or "").strip()
    if not seed:
        return {}, []
    max_nodes = int(max_nodes) if int(max_nodes) > 5 else 80

    out_adj: dict[str, set[str]] = {}
    in_adj: dict[str, set[str]] = {}
    for s, t in work_edges:
        s = str(s)
        t = str(t)
        out_adj.setdefault(s, set()).add(t)
        in_adj.setdefault(t, set()).add(s)

    visible: list[str] = []
    seen: set[str] = set()

    def _add(wid: str) -> None:
        wid = str(wid or "").strip()
        if not wid or wid not in work_by_id:
            return
        if wid in seen:
            return
        seen.add(wid)
        visible.append(wid)

    _add(seed)
    for wid in sorted(expanded_works):
        if len(visible) >= max_nodes:
            break
        _add(wid)

    # 1-hop neighborhood around seed.
    if seed not in expanded_work_citespans:
        for wid in sorted(out_adj.get(seed, set())):
            if len(visible) >= max_nodes:
                break
            _add(wid)
    for wid in sorted(in_adj.get(seed, set())):
        if len(visible) >= max_nodes:
            break
        _add(wid)

    limited_work_by_id = {wid: work_by_id[wid] for wid in visible if wid in work_by_id}
    limited_edges = [
        (s, t)
        for (s, t) in work_edges
        if s in limited_work_by_id and t in limited_work_by_id
    ]
    return limited_work_by_id, limited_edges


def _set_active_callout(
    *,
    doc_id: str,
    sentence_id: Optional[str],
    citation_index: int,
    target_id: Optional[str],
) -> None:
    doc_id = str(doc_id or "").strip()
    if not doc_id:
        return
    st.session_state["selected_doc_id"] = doc_id
    st.session_state["citation_selected_index"] = int(citation_index)
    st.session_state["citation_selected_target"] = (
        str(target_id).strip() if target_id else None
    )
    st.session_state["citation_selected_sentence_id"] = (
        str(sentence_id).strip() if sentence_id else None
    )
    st.session_state["selected_callout_tuple"] = {
        "doc_id": doc_id,
        "citation_index": int(citation_index),
        "target_id": str(target_id).strip() if target_id else None,
        "sentence_id": str(sentence_id).strip() if sentence_id else None,
    }
    # Mirror ui.select_citation cache busting so the destination tab reloads.
    st.session_state["citation_context_key"] = None
    st.session_state["citation_context"] = None
    st.session_state["citation_context_error"] = None
    st.session_state["citation_last_context_request"] = None
    st.session_state["citation_follow_open"] = False
    st.session_state["citation_graph_key"] = None
    st.session_state["citation_graph"] = None
    st.session_state["citation_graph_error"] = None
    st.session_state["citation_last_graph_request"] = None
    st.session_state["workflow_active_citation"] = int(citation_index)


def render(*, api_url: str, seed_doc_id: str) -> None:
    _seed_state()

    reviewer_uid = _active_reviewer_uid()

    component_nonce = int(st.session_state.get("surf_live_component_nonce") or 0)
    component_key = f"surf-live::{component_nonce}"

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

    # Stable selection for display.
    selection = st.session_state.get("surf_live_selection")
    if not isinstance(selection, dict):
        selection = {}
    else:
        selection = {
            "type": str(selection.get("type") or "").strip(),
            "id": str(selection.get("id") or "").strip(),
        }
        selection = {k: v for k, v in selection.items() if v}

    # Latest component payload (may include transient action/seq).
    comp_value = st.session_state.get(component_key)
    if not isinstance(comp_value, dict):
        comp_value = {}

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
                if csid:
                    # Always bust span-bundle cache for the active callout so
                    # Chasing edits are reflected without manual refresh.
                    try:
                        span_id = _get_span_id(
                            api_url=api_url,
                            ingest_id=str(doc_id),
                            citation_index=int(cite_idx),
                            target_id=str(ref_id),
                        )
                    except Exception:
                        span_id = None
                    if span_id:
                        bundle_cache = _as_dict(
                            st.session_state.get("surf_live_bundle_cache")
                        )
                        bundle_cache.pop(f"{span_id}::{reviewer_uid}", None)
                        st.session_state["surf_live_bundle_cache"] = bundle_cache

                if csid and csid != last:
                    st.session_state["surf_live_last_callout"] = csid
                    expanded_works.add(doc_id)
                    expanded_work_citespans.add(doc_id)
                    st.session_state["surf_live_pending_focus"] = {"node_id": csid}
                    st.session_state["surf_live_selection"] = {
                        "type": "node",
                        "id": csid,
                    }
                    selection = {"type": "node", "id": csid}

    # --- Work graph (ledger-backed) ----------------------------------------
    try:
        ledger_work_by_id, ledger_work_edges = _ledger_work_graph(api_url)
    except Exception as exc:
        st.error(f"Work graph unavailable: {exc}")
        return

    if seed not in ledger_work_by_id:
        # Fallback: still render a minimal seed node.
        ledger_work_by_id[seed] = {
            "id": seed,
            "short": "Selected document",
            "title": seed,
            "status": "orange",
            "assigned": False,
            "anchored": True,
            "extracted": False,
            "resolved": False,
        }

    show_work_graph = bool(st.session_state.get("surf_live_show_work_graph"))
    if show_work_graph:
        # Keep the work-level scaffold small so work expansions reveal CiteSpans,
        # not an overwhelming number of work nodes.
        work_by_id, work_edges = _limit_work_graph(
            seed=seed,
            work_by_id=ledger_work_by_id,
            work_edges=ledger_work_edges,
            expanded_works=expanded_works,
            expanded_work_citespans=expanded_work_citespans,
            max_nodes=80,
        )
    else:
        # Default: only show the seed + explicitly expanded works.
        visible_ids = set(expanded_works) | {seed}
        work_by_id = {
            wid: ledger_work_by_id[wid]
            for wid in visible_ids
            if wid in ledger_work_by_id
        }
        work_edges = []

    expanded_works.add(seed)

    # Process Cytoscape events from the component payload.
    try:
        evt_seq = int(comp_value.get("seq") or 0)
    except Exception:
        evt_seq = 0
    last_seq = int(st.session_state.get("surf_live_last_event_seq") or 0)
    is_new_evt = bool(evt_seq and evt_seq > last_seq)
    if is_new_evt:
        st.session_state["surf_live_last_event_seq"] = evt_seq

        evt_type = str(comp_value.get("type") or "").strip()
        evt_id = str(comp_value.get("id") or "").strip()
        evt_action = str(comp_value.get("action") or "click").strip()
        shift = bool(comp_value.get("shift"))

        # Persist selection.
        if evt_type in {"node", "edge"} and evt_id:
            selection = {"type": evt_type, "id": evt_id}
            st.session_state["surf_live_selection"] = dict(selection)
        else:
            selection = {}
            st.session_state["surf_live_selection"] = {}

        def _collapse_work(wid: str) -> None:
            wid = str(wid or "").strip()
            if not wid:
                return
            expanded_work_citespans.discard(wid)
            expanded_works.discard(wid)
            # Cascade collapse: remove citespans expanded under this work.
            keep: set[str] = set()
            for cs in expanded_cites:
                parsed = _parse_citespan_id(cs)
                if parsed and str(parsed.doc_id) == wid:
                    continue
                keep.add(cs)
            expanded_cites.clear()
            expanded_cites.update(keep)

        if evt_type == "node" and evt_id in work_by_id:
            st.session_state["surf_live_active_citespan_doc"] = evt_id
            if evt_action in {"dblclick", "context"}:
                if shift:
                    _collapse_work(evt_id)
                else:
                    expanded_works.add(evt_id)
                    if evt_id in expanded_work_citespans:
                        expanded_work_citespans.discard(evt_id)
                    else:
                        expanded_work_citespans.add(evt_id)
        elif evt_type == "node" and evt_id.startswith("citespanbucket:"):
            # Bucket ids:
            # - citespanbucket:{work_id}                      (collapsed work view)
            # - citespanbucket:{work_id}:todo                 (todo/unfollowed bucket)
            # - citespanbucket:{work_id}:todo:more            (increase cap)
            parts = evt_id.split(":")
            work_id = str(parts[1] if len(parts) > 1 else "").strip()
            bucket_kind = str(parts[2] if len(parts) > 2 else "").strip()
            bucket_more = bool(len(parts) > 3 and str(parts[3]).strip() == "more")
            if work_id:
                st.session_state["surf_live_active_citespan_doc"] = work_id
                if evt_action in {"dblclick", "context"}:
                    if shift:
                        if bucket_kind == "todo":
                            open_map = _as_dict(
                                st.session_state.get("surf_live_open_todo_by_work")
                            )
                            open_map.pop(work_id, None)
                            st.session_state["surf_live_open_todo_by_work"] = open_map
                        else:
                            _collapse_work(work_id)
                    else:
                        expanded_works.add(work_id)
                        expanded_work_citespans.add(work_id)
                        if bucket_kind == "todo":
                            if bucket_more:
                                cap_map = _as_dict(
                                    st.session_state.get("surf_live_todo_cap_by_work")
                                )
                                cur = int(cap_map.get(work_id) or 40)
                                cap_map[work_id] = int(cur + 40)
                                st.session_state["surf_live_todo_cap_by_work"] = cap_map
                            open_map = _as_dict(
                                st.session_state.get("surf_live_open_todo_by_work")
                            )
                            open_map[work_id] = True
                            st.session_state["surf_live_open_todo_by_work"] = open_map
        elif evt_type == "node" and evt_id.startswith("citespan:"):
            parsed = _parse_citespan_id(evt_id)
            if parsed and parsed.doc_id:
                st.session_state["surf_live_active_citespan_doc"] = parsed.doc_id
            if evt_action in {"dblclick", "context"}:
                if shift:
                    expanded_cites.discard(evt_id)
                else:
                    if evt_id in expanded_cites:
                        expanded_cites.discard(evt_id)
                    else:
                        expanded_cites.add(evt_id)
                    if parsed and parsed.doc_id:
                        expanded_works.add(parsed.doc_id)
                        expanded_work_citespans.add(parsed.doc_id)

    # --- Claims for expanded docs (for "needs work" signals) --------------
    claim_count_by_citespan: dict[tuple[str, str, int, str], int] = {}
    claim_by_id: dict[str, dict] = {}
    claim_rows_by_context: dict[tuple[str, int, str], list[tuple[int, str]]] = {}
    claim_text_by_citespan_order: dict[tuple[str, int, str, int], str] = {}
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
            try:
                claim_index = int(props.get("claim_index"))
            except Exception:
                claim_index = -1
            parsed_text = str(props.get("parsed_text") or "").strip()
            key = (doc_id, sentence_id, int(citation_index), ref_id)
            claim_count_by_citespan[key] = int(claim_count_by_citespan.get(key, 0) + 1)

            ctx_key = (doc_id, int(citation_index), ref_id)
            if claim_index >= 0 and parsed_text:
                claim_rows_by_context.setdefault(ctx_key, []).append(
                    (int(claim_index), parsed_text)
                )

    # Best-effort map order_index -> parsed_text for claimspans.
    for (doc_id, cite_idx, ref_id), rows in claim_rows_by_context.items():
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda t: int(t[0]))
        min_idx = min(int(t[0]) for t in rows_sorted)
        shift = 1 if min_idx == 0 else 0
        for claim_index, parsed_text in rows_sorted:
            order_index = int(claim_index) + int(shift)
            claim_text_by_citespan_order[
                (doc_id, int(cite_idx), ref_id, order_index)
            ] = parsed_text

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
            # Stub/Unresolved cited work.
            "selector": "node[type = 'Work'][stub = 1]",
            "style": {
                "shape": "hexagon",
                "background-color": "#93c5fd",
                "border-width": 3,
                "border-color": "#1f2937",
                "width": 22,
                "height": 22,
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
            # Accessibility: do not rely on color alone for triage.
            # - todo: diamond shape
            # - ignore: triangle shape
            "selector": "node[type = 'CiteSpan'][triage = 'todo']",
            "style": {
                "shape": "diamond",
                "width": 20,
                "height": 20,
                "border-width": 4,
                "border-color": "#1b5e20",
            },
        },
        {
            "selector": "node[type = 'CiteSpan'][triage = 'ignore']",
            "style": {
                "shape": "triangle",
                "width": 20,
                "height": 20,
                "background-color": "#9ca3af",
                "border-width": 4,
                "border-color": "#7f1d1d",
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
            "selector": "node[type = 'CiteSpanBucket'][bucket_kind = 'ignore']",
            "style": {
                "background-color": "#9ca3af",
                "border-color": "#7f1d1d",
            },
        },
        {
            "selector": "node[type = 'CiteSpanBucket'][bucket_kind = 'todo_more']",
            "style": {
                "background-color": "#0f766e",
                "width": 22,
                "height": 22,
                "border-width": 2,
                "border-color": "#111827",
                "label": "data(count_label)",
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
                "line-color": "#cbd5e1",
                "target-arrow-color": "#cbd5e1",
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
            # Structural edges stay neutral; evaluative edges get redundant cues.
            "selector": "edge[type = 'CITES_TARGET'][state = 'unexplored']",
            "style": {
                "line-style": "dashed",
                "line-color": "#6b7280",
                "opacity": 0.45,
                "target-arrow-shape": "none",
            },
        },
        {
            "selector": "edge[type = 'CITES_TARGET'][state = 'explored']",
            "style": {
                "line-style": "solid",
                "line-color": "#6b7280",
                "opacity": 0.9,
                "target-arrow-shape": "none",
            },
        },
        {
            "selector": "edge[type = 'ASSERTS'][verdict = 'support']",
            "style": {
                "line-style": "solid",
                "line-color": "#1b5e20",
                "target-arrow-shape": "triangle",
                "target-arrow-color": "#1b5e20",
                "width": 3,
            },
        },
        {
            "selector": "edge[type = 'ASSERTS'][verdict = 'contradict']",
            "style": {
                "line-style": "solid",
                "line-color": "#b91c1c",
                "target-arrow-shape": "tee",
                "target-arrow-color": "#b91c1c",
                "width": 3,
            },
        },
        {
            "selector": "edge[type = 'ASSERTS'][verdict = 'neutral']",
            "style": {
                "line-style": "dotted",
                "line-color": "#4b5563",
                "target-arrow-shape": "none",
                "target-arrow-color": "#4b5563",
                "width": 3,
            },
        },
        {
            "selector": "edge[type = 'ASSERTS'][verdict = 'uncertain']",
            "style": {
                "line-style": "dashed",
                "line-color": "#b45309",
                "target-arrow-shape": "diamond",
                "target-arrow-color": "#b45309",
                "width": 3,
            },
        },
        {
            "selector": "edge[type = 'ASSERTS'][verdict = 'unknown']",
            "style": {
                "line-style": "dashed",
                "line-color": "#6b7280",
                "target-arrow-shape": "diamond",
                "target-arrow-color": "#6b7280",
                "width": 3,
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

    rendered_work_ids: set[str] = set(work_by_id.keys())

    for src, tgt in work_edges:
        # When a work's citations are expanded, prefer CiteSpan/ClaimSpan level
        # edges to targets. Avoid duplicating with a coarse work->work edge.
        if src in expanded_work_citespans:
            continue
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
                    "hover": "Work cites Work",
                }
            }
        )

    citespan_records: dict[str, dict] = {}
    citespans_by_doc: dict[str, list[dict]] = {}

    for wid in sorted(expanded_works):
        if wid not in work_by_id:
            continue
        try:
            body = get_document_body(api_url, wid)
        except Exception:
            continue
        paragraphs = body.get("paragraphs") or []
        ref_ids: list[str] = []
        raw_segments: list[dict] = []
        sentence_text: dict[str, str] = {}

        for para in paragraphs:
            for sent in para.get("sentences") or []:
                sentence_id = str(sent.get("sentence_id") or "").strip()
                if sentence_id and (sentence_id not in sentence_text):
                    parts: list[str] = []
                    for seg in sent.get("segments") or []:
                        if not isinstance(seg, dict):
                            continue
                        t = str(seg.get("type") or "").strip()
                        if t == "text":
                            parts.append(str(seg.get("text") or "").strip())
                        elif t == "citation":
                            parts.append(str(seg.get("callout") or "").strip())
                    joined = " ".join(p for p in parts if p).strip()
                    if joined:
                        sentence_text[sentence_id] = joined
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
        unindexed_ci = 100000

        for seg in raw_segments:
            sentence_id = str(seg.get("sentence_id") or "").strip()
            ref_id = str(seg.get("target_id") or "").strip()
            indexed = True
            try:
                citation_index = int(seg.get("citation_index"))
            except Exception:
                citation_index = unindexed_ci
                unindexed_ci += 1
                indexed = False
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
            preview = sentence_text.get(sentence_id) or ""
            if preview and len(preview) > 180:
                preview = preview[:179].rstrip() + "…"
            citespan_records[csid] = {
                "id": csid,
                "doc_id": wid,
                "sentence_id": sentence_id,
                "citation_index": int(citation_index),
                "indexed": bool(indexed),
                "reference_id": ref_id,
                "target_ingest_id": tgt_ingest,
                "label": label,
                "callout": seg.get("callout"),
                "claim_count": claim_n,
                "preview": preview or None,
            }

            work_cites.append(citespan_records[csid])

        citespans_by_doc[wid] = sorted(
            work_cites,
            key=lambda r: (
                0 if bool(r.get("indexed")) else 1,
                int(r.get("citation_index") or 0),
            ),
        )

    # Collapse or expand cite spans for this work.
    if wid not in expanded_work_citespans:
        # Bucket count should reflect "not followed" (no confirmed claims), not
        # resolution state.
        todo_n = 0
        for r in citespans_by_doc.get(wid) or []:
            csid = str(r.get("id") or "").strip()
            claim_n = int(r.get("claim_count") or 0)
            triage = _citespan_triage(
                citespan_id=csid,
                unresolved=False,
                claim_count=int(claim_n),
            )
            if claim_n == 0 and triage != "ignore":
                todo_n += 1

        bucket_id = f"citespanbucket:{wid}"
        elements.append(
            {
                "data": {
                    "id": bucket_id,
                    "type": "CiteSpanBucket",
                    "bucket_kind": "todo",
                    "count": int(todo_n),
                    "count_label": str(int(todo_n)),
                    "hover": (
                        f"Follow-up TODO: {todo_n}\n"
                        f"CiteSpans total: {len(citespans_by_doc.get(wid) or [])}"
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
        # Show only "followed" CiteSpans as individual nodes.
        # Anything triaged as TODO stays in a bucket until explicitly opened.
        all_rows = citespans_by_doc.get(wid) or []

        followed: list[dict] = []
        todo_rows: list[dict] = []
        ignore_rows: list[dict] = []

        for rec in all_rows:
            csid = str(rec.get("id") or "").strip()
            claim_n = int(rec.get("claim_count") or 0)
            unresolved = not bool(str(rec.get("target_ingest_id") or "").strip())
            triage = _citespan_triage(
                citespan_id=csid,
                unresolved=bool(unresolved),
                claim_count=int(claim_n),
            )
            if claim_n > 0:
                followed.append(rec)
            elif triage == "ignore":
                ignore_rows.append(rec)
            else:
                todo_rows.append(rec)

        for rec in followed:
            csid = str(rec.get("id") or "")
            tgt_ingest = rec.get("target_ingest_id")
            claim_n = int(rec.get("claim_count") or 0)
            unresolved = not bool(str(tgt_ingest or "").strip())
            triage = _citespan_triage(
                citespan_id=csid,
                unresolved=bool(unresolved),
                claim_count=int(claim_n),
            )

            # Prefer edited/anchored citation-window preview if we have a span.
            span_id = _get_span_id(
                api_url=api_url,
                ingest_id=wid,
                citation_index=int(rec.get("citation_index") or 0),
                target_id=str(rec.get("reference_id") or ""),
            )
            if span_id:
                bundle = _get_span_bundle_cached(
                    api_url=api_url,
                    span_id=str(span_id),
                    reviewer_uid=str(reviewer_uid),
                )
                span = bundle.get("span") or {}
                selector = (
                    (span.get("selector") or {}) if isinstance(span, dict) else {}
                )
                exact = str(selector.get("exact") or "").strip()
                prefix = str(selector.get("prefix") or "").strip()
                suffix = str(selector.get("suffix") or "").strip()
                edited_preview = " ".join(
                    bit for bit in [prefix, exact, suffix] if bit
                ).strip()
                if edited_preview:
                    rec["preview"] = edited_preview
            preview = str(rec.get("preview") or "").strip()
            hover_bits = [str(rec.get("label") or "citation").strip()]
            if preview:
                hover_bits.append(preview)
            if triage == "todo":
                hover_bits.append("follow-up: TODO")
            elif triage == "ignore":
                hover_bits.append("follow-up: ignore")
            if unresolved:
                hover_bits.append("target: unresolved")
            hover = "\n".join(bit for bit in hover_bits if bit)
            elements.append(
                {
                    "data": {
                        "id": csid,
                        "type": "CiteSpan",
                        "label": str(rec.get("label") or "citation"),
                        "hover": hover,
                        "doc_id": wid,
                        "sentence_id": str(rec.get("sentence_id") or ""),
                        "citation_index": int(rec.get("citation_index") or 0),
                        "reference_id": str(rec.get("reference_id") or ""),
                        "claim_count": int(claim_n),
                        "triage": triage,
                        "resolved": 0 if unresolved else 1,
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

            # Create a stable cited-work target node id (resolved or stub).
            if tgt_ingest and str(tgt_ingest).strip():
                target_node_id = str(tgt_ingest).strip()
            else:
                target_node_id = f"stubwork:{csid}"

            rec["target_node_id"] = target_node_id
            if csid in citespan_records:
                citespan_records[csid]["target_node_id"] = target_node_id

            # If we have a real ingested target, show it in the default view
            # ("work only") even when the full work graph is hidden.
            if not target_node_id.startswith("stubwork:"):
                if target_node_id not in rendered_work_ids:
                    meta = (
                        ledger_work_by_id.get(target_node_id)
                        if isinstance(ledger_work_by_id, dict)
                        else None
                    )
                    short = None
                    title = None
                    if isinstance(meta, dict):
                        short = str(meta.get("short") or "").strip() or None
                        title = str(meta.get("title") or "").strip() or None
                    label = short or str(target_node_id)
                    hover = (label + ("\n" + title if title else "")).strip()
                    elements.append(
                        {
                            "data": {
                                "id": str(target_node_id),
                                "type": "Work",
                                "label": label,
                                "hover": hover,
                            }
                        }
                    )
                    rendered_work_ids.add(str(target_node_id))
                if str(target_node_id) not in work_by_id:
                    work_by_id[str(target_node_id)] = {
                        "id": str(target_node_id),
                        "short": str(target_node_id),
                        "title": str(target_node_id),
                    }

            # Only render stub target nodes when CiteSpan is expanded.
            if target_node_id.startswith("stubwork:"):
                if csid in expanded_cites and target_node_id not in rendered_work_ids:
                    stub_label = str(rec.get("label") or "Cited work").strip()
                    elements.append(
                        {
                            "data": {
                                "id": target_node_id,
                                "type": "Work",
                                "stub": 1,
                                "label": stub_label,
                                "hover": f"Cited work (not ingested yet)\n{stub_label}",
                            }
                        }
                    )
                    rendered_work_ids.add(target_node_id)

            # Show citespan->target edge only when citespan is not expanded,
            # and only for resolved, ingested targets.
            if (
                target_node_id
                and (not target_node_id.startswith("stubwork:"))
                and csid not in expanded_cites
            ):
                state = (
                    "explored"
                    if str(target_node_id) in expanded_works
                    else "unexplored"
                )
                elements.append(
                    {
                        "data": {
                            "id": f"cs2work:{csid}->{target_node_id}",
                            "source": csid,
                            "target": target_node_id,
                            "type": "CITES_TARGET",
                            "arrow": "none",
                            "state": state,
                            "hover": f"CiteSpan targets Work\nstate={state}",
                        }
                    }
                )

        # TODO bucket (unfollowed).
        todo_n = int(len(todo_rows))
        if todo_n:
            bucket_id = f"citespanbucket:{wid}:todo"
            elements.append(
                {
                    "data": {
                        "id": bucket_id,
                        "type": "CiteSpanBucket",
                        "bucket_kind": "todo",
                        "count": int(todo_n),
                        "count_label": str(int(todo_n)),
                        "hover": (
                            f"Unfollowed CiteSpans: {todo_n}\n" "Double-click to open."
                        ),
                        "doc_id": wid,
                    }
                }
            )
            elements.append(
                {
                    "data": {
                        "id": f"work2bucket:{wid}:todo",
                        "source": wid,
                        "target": bucket_id,
                        "type": "HAS_TODO_CITESPANS",
                        "arrow": "none",
                    }
                }
            )

            open_map = _as_dict(st.session_state.get("surf_live_open_todo_by_work"))
            is_open = bool(open_map.get(wid))
            if is_open:
                cap_map = _as_dict(st.session_state.get("surf_live_todo_cap_by_work"))
                cap = int(cap_map.get(wid) or 40)
                visible_todo = todo_rows[:cap]
                remainder = todo_rows[cap:]
                for rec in visible_todo:
                    csid = str(rec.get("id") or "")
                    tgt_ingest = rec.get("target_ingest_id")
                    claim_n = int(rec.get("claim_count") or 0)
                    unresolved = not bool(str(tgt_ingest or "").strip())
                    triage = _citespan_triage(
                        citespan_id=csid,
                        unresolved=bool(unresolved),
                        claim_count=int(claim_n),
                    )
                    preview = str(rec.get("preview") or "").strip()
                    hover_bits = [str(rec.get("label") or "citation").strip()]
                    if preview:
                        hover_bits.append(preview)
                    hover_bits.append("follow-up: TODO")
                    if unresolved:
                        hover_bits.append("target: unresolved")
                    hover = "\n".join(bit for bit in hover_bits if bit)
                    elements.append(
                        {
                            "data": {
                                "id": csid,
                                "type": "CiteSpan",
                                "label": str(rec.get("label") or "citation"),
                                "hover": hover,
                                "doc_id": wid,
                                "sentence_id": str(rec.get("sentence_id") or ""),
                                "citation_index": int(rec.get("citation_index") or 0),
                                "reference_id": str(rec.get("reference_id") or ""),
                                "claim_count": int(claim_n),
                                "triage": triage,
                                "resolved": 0 if unresolved else 1,
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

                    # Stable target node id (resolved or stub)
                    if tgt_ingest and str(tgt_ingest).strip():
                        target_node_id = str(tgt_ingest).strip()
                    else:
                        target_node_id = f"stubwork:{csid}"
                    rec["target_node_id"] = target_node_id
                    if csid in citespan_records:
                        citespan_records[csid]["target_node_id"] = target_node_id
                    if target_node_id.startswith("stubwork:"):
                        if target_node_id not in rendered_work_ids:
                            stub_label = str(rec.get("label") or "Cited work").strip()
                            elements.append(
                                {
                                    "data": {
                                        "id": target_node_id,
                                        "type": "Work",
                                        "stub": 1,
                                        "label": stub_label,
                                        "hover": f"Unresolved cited work\n{stub_label}",
                                    }
                                }
                            )
                            rendered_work_ids.add(target_node_id)

                    if target_node_id and csid not in expanded_cites:
                        if target_node_id.startswith("stubwork:"):
                            state = "unresolved"
                        else:
                            state = (
                                "explored"
                                if str(target_node_id) in expanded_works
                                else "unexplored"
                            )
                        elements.append(
                            {
                                "data": {
                                    "id": f"cs2work:{csid}->{target_node_id}",
                                    "source": csid,
                                    "target": target_node_id,
                                    "type": "CITES_TARGET",
                                    "arrow": "none",
                                    "state": state,
                                    "hover": f"CiteSpan targets Work\nstate={state}",
                                }
                            }
                        )

                if remainder:
                    more_id = f"citespanbucket:{wid}:todo:more"
                    elements.append(
                        {
                            "data": {
                                "id": more_id,
                                "type": "CiteSpanBucket",
                                "bucket_kind": "todo_more",
                                "count": int(len(remainder)),
                                "count_label": "+",
                                "hover": f"Show {len(remainder)} more (increase cap)",
                                "doc_id": wid,
                            }
                        }
                    )
                    elements.append(
                        {
                            "data": {
                                "id": f"work2bucket:{wid}:todo:more",
                                "source": wid,
                                "target": more_id,
                                "type": "HAS_MORE_TODO_CITESPANS",
                                "arrow": "none",
                            }
                        }
                    )

        # Ignore bucket (kept collapsed for now).
        if ignore_rows:
            ignore_id = f"citespanbucket:{wid}:ignore"
            elements.append(
                {
                    "data": {
                        "id": ignore_id,
                        "type": "CiteSpanBucket",
                        "bucket_kind": "ignore",
                        "count": int(len(ignore_rows)),
                        "count_label": str(int(len(ignore_rows))),
                        "hover": f"Do not follow up: {len(ignore_rows)}",
                        "doc_id": wid,
                    }
                }
            )
            elements.append(
                {
                    "data": {
                        "id": f"work2bucket:{wid}:ignore",
                        "source": wid,
                        "target": ignore_id,
                        "type": "HAS_IGNORED_CITESPANS",
                        "arrow": "none",
                    }
                }
            )

    # Expanded CiteSpans -> ClaimSpans
    for csid in sorted(expanded_cites):
        key = _parse_citespan_id(csid)
        if not key:
            continue

        # Prefer the citespan-level target node id (resolved or stub) so the
        # cite->work edge can transfer down to claimspans.
        target_node_id = None
        rec = citespan_records.get(csid) or {}
        raw = str(rec.get("target_node_id") or "").strip()
        if raw:
            target_node_id = raw
        else:
            if str(key.reference_id or "").strip():
                mapping = _resolve_reference_targets(
                    citing_doc_id=key.doc_id,
                    reference_ids=[str(key.reference_id)],
                )
                target_doc = mapping.get(str(key.reference_id))
                if target_doc:
                    target_node_id = str(target_doc)
            if not target_node_id:
                target_node_id = f"stubwork:{csid}"

        if target_node_id.startswith("stubwork:"):
            if target_node_id not in rendered_work_ids:
                stub_label = str(
                    (rec.get("label") or "Cited work")
                    if isinstance(rec, dict)
                    else "Cited work"
                ).strip()
                elements.append(
                    {
                        "data": {
                            "id": target_node_id,
                            "type": "Work",
                            "stub": 1,
                            "label": stub_label,
                            "hover": f"Unresolved cited work\n{stub_label}",
                        }
                    }
                )
                rendered_work_ids.add(target_node_id)
        else:
            if target_node_id and target_node_id not in rendered_work_ids:
                meta = (
                    ledger_work_by_id.get(target_node_id)
                    if isinstance(ledger_work_by_id, dict)
                    else None
                )
                short = None
                title = None
                if isinstance(meta, dict):
                    short = str(meta.get("short") or "").strip() or None
                    title = str(meta.get("title") or "").strip() or None
                label = short or str(target_node_id)
                hover = (label + ("\n" + title if title else "")).strip()
                elements.append(
                    {
                        "data": {
                            "id": str(target_node_id),
                            "type": "Work",
                            "label": label,
                            "hover": hover,
                        }
                    }
                )
                rendered_work_ids.add(str(target_node_id))
                # Make the work expandable (event handler uses work_by_id).
                if str(target_node_id) not in work_by_id:
                    work_by_id[str(target_node_id)] = {
                        "id": str(target_node_id),
                        "short": label,
                        "title": title or label,
                    }

        span_id = _get_span_id(
            api_url=api_url,
            ingest_id=key.doc_id,
            citation_index=int(key.citation_index),
            target_id=str(key.reference_id or ""),
        )
        if not span_id:
            continue

        bundle = _get_span_bundle_cached(
            api_url=api_url,
            span_id=span_id,
            reviewer_uid=str(reviewer_uid),
        )
        claim_spans = bundle.get("claim_spans") or []
        span = bundle.get("span") or {}
        selector = (span.get("selector") or {}) if isinstance(span, dict) else {}
        exact = str(selector.get("exact") or "").strip()
        prefix = str(selector.get("prefix") or "").strip()
        suffix = str(selector.get("suffix") or "").strip()
        preview = " ".join(bit for bit in [prefix, exact, suffix] if bit).strip()
        if preview and csid in citespan_records:
            citespan_records[csid]["preview"] = preview

        # ClaimSpan nodes + cite->claim edges.
        for cs in claim_spans:
            if not isinstance(cs, dict):
                continue
            claim_span_id = str(cs.get("claim_span_id") or "").strip()
            if not claim_span_id:
                continue
            order_index = cs.get("order_index")
            try:
                oi = int(order_index)
            except Exception:
                oi = 0

            status = str(cs.get("status") or "").strip() or "unknown"
            label = f"#{oi}" if oi else "claim"
            parsed_text = ""
            if key.doc_id and key.reference_id:
                parsed_text = claim_text_by_citespan_order.get(
                    (
                        key.doc_id,
                        int(key.citation_index),
                        str(key.reference_id),
                        int(oi),
                    ),
                    "",
                )
            hover = f"ClaimSpan {label}\nstatus={status}"
            if parsed_text:
                hover = hover + "\n\n" + parsed_text

            elements.append(
                {
                    "data": {
                        "id": claim_span_id,
                        "type": "ClaimSpan",
                        "label": label,
                        "hover": hover,
                        "status": status,
                    }
                }
            )
            elements.append(
                {
                    "data": {
                        "id": f"cs2claim:{csid}->{claim_span_id}",
                        "source": csid,
                        "target": claim_span_id,
                        "type": "HAS_CLAIM",
                        "arrow": "none",
                    }
                }
            )

            current = cs.get("current") or {}
            verdict = str((current or {}).get("verdict") or "").strip() or None
            if not verdict:
                if status == "supported":
                    verdict = "support"
                elif status == "contradicted":
                    verdict = "contradict"
                elif status in {"contested", "unknown"}:
                    verdict = "uncertain"
                else:
                    verdict = "neutral"

            # When claimspans are visible, transfer the cite->target edge down to
            # work->claimspan edges (evidence direction: cited supports/contradicts).
            if target_node_id and target_node_id in rendered_work_ids:
                edge_id = f"work2claim:{target_node_id}->{claim_span_id}"
                elements.append(
                    {
                        "data": {
                            "id": edge_id,
                            "source": target_node_id,
                            "target": claim_span_id,
                            "type": "ASSERTS",
                            "arrow": "triangle",
                            "verdict": verdict or "unknown",
                            "hover": f"ASSERTS\nverdict={verdict or 'unknown'}",
                        }
                    }
                )

    # --- Layout: top panels (scrollable) + full-width graph ----------------
    top_left, top_right = st.columns([1, 1], gap="large")
    with top_left:
        with st.container(height=240):
            st.markdown("**Expanded**")
            st.checkbox("Show labels (debug)", key="surf_live_show_labels")
            st.checkbox("Debug Surfing", key="surf_live_debug")
            st.checkbox("Show work graph", key="surf_live_show_work_graph")
            st.checkbox(
                "Follow active citation",
                key="surf_live_follow_active_citation",
            )
            if st.button("Collapse all", key="surf-live-collapse"):
                expanded_works = {seed}
                expanded_cites = set()
                expanded_work_citespans = set()
                st.session_state["surf_live_active_citespan_doc"] = ""

            if st.button("Re-layout", key="surf-live-relayout"):
                st.session_state["surf_live_layout_nonce"] = (
                    int(st.session_state.get("surf_live_layout_nonce") or 0) + 1
                )

            st.caption(f"Works expanded: {len(expanded_works)}")
            st.caption(f"CiteSpans expanded: {len(expanded_cites)}")
            st.caption(f"CiteSpan lists: {len(expanded_work_citespans)}")

            with st.expander("Legend", expanded=False):
                st.caption("CiteSpan nodes")
                st.write("diamond = follow up (TODO)")
                st.write("triangle = do not follow up")
                st.caption("Edges")
                st.write("CiteSpan -> Work dashed grey = target not yet explored")
                st.write("CiteSpan -> Work solid grey = target explored")
                st.write("ClaimSpan -> Work (ASSERTS) triangle arrow = supports")
                st.write("ClaimSpan -> Work (ASSERTS) tee arrow = contradicts")
                st.write("ClaimSpan -> Work (ASSERTS) dotted = neutral")
                st.write("ClaimSpan -> Work (ASSERTS) dashed + diamond = uncertain")

            if bool(st.session_state.get("surf_live_debug")):
                st.divider()
                st.caption("State (debug)")
                st.json(
                    {
                        "seed": seed,
                        "active_doc": st.session_state.get(
                            "surf_live_active_citespan_doc"
                        ),
                        "expanded_works": sorted(list(expanded_works))[:50],
                        "expanded_work_citespans": sorted(
                            list(expanded_work_citespans)
                        )[:50],
                        "expanded_cites": sorted(list(expanded_cites))[:50],
                        "selection": selection,
                        "component": comp_value,
                        "last_event_seq": st.session_state.get(
                            "surf_live_last_event_seq"
                        ),
                        "selected_callout_tuple": st.session_state.get(
                            "selected_callout_tuple"
                        ),
                    },
                    expanded=False,
                )

                if st.button(
                    "Reindex documents (graph)",
                    key="surf-live-reindex-docs",
                    help="Rebuild DOI/bib aliases from ingestion store.",
                ):
                    try:
                        graph_api.reindex_docs()
                    except Exception:
                        pass
                    st.session_state["surf_live_ref_cache"] = {}
                    st.rerun()

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

                is_exp = sel_id in expanded_work_citespans
                next_exp = st.toggle(
                    "Show citations",
                    value=bool(is_exp),
                    key=f"surf-live-show-cites:{sel_id}",
                )
                if bool(next_exp) != bool(is_exp):
                    if next_exp:
                        expanded_works.add(sel_id)
                        expanded_work_citespans.add(sel_id)
                    else:
                        expanded_work_citespans.discard(sel_id)
                    st.session_state["surf_live_active_citespan_doc"] = sel_id

                if st.button("Expand citations", key="surf-live-expand-work"):
                    expanded_works.add(sel_id)
                    expanded_work_citespans.add(sel_id)
                    st.session_state["surf_live_active_citespan_doc"] = sel_id
                if st.button("Collapse citations", key="surf-live-collapse-work"):
                    expanded_work_citespans.discard(sel_id)
            elif sel_id.startswith("citespan:"):
                rec = citespan_records.get(sel_id) or {}
                doc_id = str(rec.get("doc_id") or "").strip()
                sentence_id = str(rec.get("sentence_id") or "").strip() or None
                try:
                    citation_index = int(rec.get("citation_index") or 0)
                except Exception:
                    citation_index = 0
                reference_id = str(rec.get("reference_id") or "").strip() or None
                target_ingest_id = (
                    str(rec.get("target_ingest_id") or "").strip() or None
                )

                st.caption("CiteSpan")
                st.write(str(rec.get("label") or sel_id))
                st.caption(f"claims confirmed: {int(rec.get('claim_count') or 0)}")
                if doc_id:
                    st.caption(f"citing: `{doc_id}`  cite_index: `{citation_index}`")
                if reference_id:
                    st.caption(f"reference_id: `{reference_id}`")
                if target_ingest_id:
                    st.caption(f"resolved target: `{target_ingest_id}`")
                else:
                    st.caption("resolved target: (unresolved)")

                follow_map = _as_dict(
                    st.session_state.get("surf_live_citespan_followup")
                )
                raw_choice = str(follow_map.get(sel_id) or "").strip().lower()
                choice_labels = ["Auto", "Follow up", "Do not follow up"]
                if raw_choice == "todo":
                    idx = 1
                elif raw_choice == "ignore":
                    idx = 2
                else:
                    idx = 0
                picked = st.selectbox(
                    "Follow-up",
                    choice_labels,
                    index=idx,
                    key=f"surf-live-followup:{sel_id}",
                    help=(
                        "Affects Surfing styling only; resolution/claims happen in "
                        "Chasing."
                    ),
                )
                next_raw = ""
                if picked == "Follow up":
                    next_raw = "todo"
                elif picked == "Do not follow up":
                    next_raw = "ignore"
                if next_raw != raw_choice:
                    if next_raw:
                        follow_map[sel_id] = next_raw
                    else:
                        follow_map.pop(sel_id, None)
                    st.session_state["surf_live_citespan_followup"] = follow_map
                    st.rerun()

                row = st.columns([1, 1, 1], gap="small")
                with row[0]:
                    if st.button(
                        "Open in Reading",
                        key=f"surf-live-open-reading:{sel_id}",
                        disabled=not bool(doc_id),
                    ):
                        if doc_id:
                            _set_active_callout(
                                doc_id=doc_id,
                                sentence_id=sentence_id,
                                citation_index=int(citation_index),
                                target_id=reference_id,
                            )
                            st.session_state[
                                WORKSPACE_ACTIVE_TAB
                            ] = WORKSPACE_TAB_DOCUMENT
                            st.rerun()
                with row[1]:
                    if st.button(
                        "Open in Chasing",
                        key=f"surf-live-open-chasing:{sel_id}",
                        disabled=not bool(doc_id),
                    ):
                        if doc_id:
                            _set_active_callout(
                                doc_id=doc_id,
                                sentence_id=sentence_id,
                                citation_index=int(citation_index),
                                target_id=reference_id,
                            )
                            st.session_state[
                                WORKSPACE_ACTIVE_TAB
                            ] = WORKSPACE_TAB_REVIEW
                            st.rerun()
                with row[2]:
                    if st.button(
                        "Jump to Target",
                        key=f"surf-live-jump-target:{sel_id}",
                        disabled=not bool(target_ingest_id),
                        help=(
                            "Requires resolved target ingest id."
                            if not target_ingest_id
                            else None
                        ),
                    ):
                        if target_ingest_id:
                            st.session_state["selected_doc_id"] = target_ingest_id
                            st.session_state[WORKSPACE_ACTIVE_TAB] = WORKSPACE_TAB_GRAPH
                            st.session_state["surf_live_seed_doc"] = target_ingest_id
                            st.session_state["surf_live_pending_focus"] = {
                                "node_id": target_ingest_id
                            }
                            st.rerun()

                if doc_id and reference_id and not target_ingest_id:
                    if st.button(
                        "Re-run reference resolution",
                        key=f"surf-live-reresolve:{sel_id}",
                        help=(
                            "Runs /ingest/{doc_id}/resolve and clears Surfing's "
                            "ref cache."
                        ),
                    ):
                        try:
                            trigger_resolution(api_url, doc_id)
                        except Exception:
                            pass
                        cache = _as_dict(st.session_state.get("surf_live_ref_cache"))
                        prefix = f"{doc_id}::"
                        cache = {
                            k: v
                            for k, v in cache.items()
                            if not str(k).startswith(prefix)
                        }
                        st.session_state["surf_live_ref_cache"] = cache
                        st.rerun()

                if doc_id:
                    # ClaimSpan generation lives in Chasing (confirm_claims).
                    span_id = _get_span_id(
                        api_url=api_url,
                        ingest_id=doc_id,
                        citation_index=int(citation_index),
                        target_id=str(reference_id or ""),
                    )
                    if not span_id:
                        st.caption(
                            "No citation-window span yet. Create it in Chasing by "
                            "confirming claims."
                        )
                    else:
                        if st.button(
                            "Refresh claim spans",
                            key=f"surf-live-refresh-claims:{sel_id}",
                            help=(
                                "Clears Surfing bundle cache for this span and "
                                "expands the CiteSpan."
                            ),
                        ):
                            bundle_cache = _as_dict(
                                st.session_state.get("surf_live_bundle_cache")
                            )
                            reviewer_uid = (
                                str(
                                    st.session_state.get("active_reviewer_uid")
                                    or "default"
                                ).strip()
                                or "default"
                            )
                            bundle_cache.pop(f"{span_id}::{reviewer_uid}", None)
                            st.session_state["surf_live_bundle_cache"] = bundle_cache
                            expanded_cites.add(sel_id)
                            st.rerun()

                is_exp = sel_id in expanded_cites
                next_exp = st.toggle(
                    "Show claim spans",
                    value=bool(is_exp),
                    key=f"surf-live-show-claims:{sel_id}",
                )
                if bool(next_exp) != bool(is_exp):
                    if next_exp:
                        expanded_cites.add(sel_id)
                    else:
                        expanded_cites.discard(sel_id)
            elif sel_id.startswith("stubwork:"):
                st.caption("Cited work (stub)")
                st.write(str(selection.get("label") or sel_id))
                # Offer a recovery path: jump back to the originating citespan.
                parent_csid = sel_id.split(":", 1)[-1].strip()
                rec = citespan_records.get(parent_csid) or {}
                doc_id = str(rec.get("doc_id") or "").strip()
                sentence_id = str(rec.get("sentence_id") or "").strip() or None
                try:
                    citation_index = int(rec.get("citation_index") or 0)
                except Exception:
                    citation_index = 0
                reference_id = str(rec.get("reference_id") or "").strip() or None
                if st.button(
                    "Open in Chasing",
                    key=f"surf-live-open-chasing-stub:{sel_id}",
                    disabled=not bool(doc_id),
                    help="Resolve/ingest the cited work in Chasing.",
                ):
                    if doc_id:
                        _set_active_callout(
                            doc_id=doc_id,
                            sentence_id=sentence_id,
                            citation_index=int(citation_index),
                            target_id=reference_id,
                        )
                        st.session_state[WORKSPACE_ACTIVE_TAB] = WORKSPACE_TAB_REVIEW
                        st.rerun()

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

    cytoscape_panel.render(
        elements,
        style=style,
        layout={"name": "dagre", "rankDir": "LR", "fit": True, "padding": 30},
        height=720,
        key=component_key,
        selection=selection,
        focus=focus,
        options={
            "showLabels": bool(st.session_state.get("surf_live_show_labels")),
            "stableLayout": True,
            "layoutNonce": int(st.session_state.get("surf_live_layout_nonce") or 0),
        },
    )

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
                    preview = str(rec.get("preview") or "").strip()
                    line = f"{prefix} {cite_idx}: {label}"
                    if preview:
                        line = f"{line} — {preview}"
                    if st.button(
                        line,
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

    _store_expanded("surf_live_expanded_works", expanded_works)
    _store_expanded("surf_live_expanded_citespans", expanded_cites)
    _store_expanded("surf_live_expanded_work_citespans", expanded_work_citespans)
