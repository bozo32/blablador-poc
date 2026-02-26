# Phase 10-03: Graph As Navigation (Cytoscape Routing) - Research

**Researched:** 2026-02-22
**Domain:** Streamlit Cytoscape navigation + deterministic session routing + recursive retrieval queue (FastAPI + Postgres spine)
**Confidence:** MEDIUM

## Summary

This repo already has the essential primitives for "graph as navigation": (1) a vendored Cytoscape Streamlit component that emits deterministic click events with a monotonically increasing `seq`, supports stable layouts across reruns, and allows programmatic selection/focus (`frontend/components/cytoscape_component/index.html`, `frontend/components/cytoscape_panel.py`); (2) a workspace routing surface implemented purely via Streamlit session state (tab key + citation selection keys) that the Surfing panel already uses (`frontend/components/live_surfing_panel.py`, `frontend/state_keys.py`); and (3) durable graph + span-graph stores in Postgres with existing endpoints for citation windows, claimspans, and evidence-to-assertion mirroring (`backend/graph_store.py`, `backend/span_graph_store.py`, `backend/main.py`, `tests/test_span_graph_rebuild.py`).

To plan 10-03 well, treat the Cytoscape component as a pure event emitter + renderer and keep all "navigation semantics" in a thin Surfing controller: decode selection ids, show a compact Streamlit popover with explicit actions, and only route (switch tabs + set citation/claim context) on action clicks. For recursive retrieval, reuse existing retrieval dossier generation (`GET /references/{doc_id}/{reference_id}/retrieval`) and persist user-driven expand/request decisions per project (recommended: `project_meta.graph_settings`) while deriving live state from existing spine tables (ingestion/extraction/resolution, attachments, workflow run status).

**Primary recommendation:** Use the component's `seq`-gated event contract to update a single session-backed `graph_selection` deterministically, and implement routing via the existing `_set_active_callout()` state mutation pattern; persist expansion/request intent in `project_meta.graph_settings` and fetch live availability/processing state from existing endpoints rather than inventing a new store.

## Standard Stack

### Core
| Library/Tool | Version (repo) | Purpose | Why Standard (in this repo) |
|---|---:|---|---|
| Streamlit | 1.54.0 | UI + session-state routing | Existing workspace shell + tabs + polling (`frontend/ui.py`) |
| Streamlit custom component (no-build) | (vendored) | Cytoscape renderer + event bridge | Already implemented in-repo (`frontend/components/cytoscape_component/index.html`) |
| Cytoscape.js + dagre layout | (vendored) | Graph layout + interaction | Vendored JS avoids build/proxy/network failures |
| FastAPI | 0.129.0 | Navigation/graph API endpoints | Existing routing + tests (`backend/main.py`) |
| Postgres spine | (compose) | Durable graph/spans/workflow state | GraphStore + SpanGraphStore are Postgres-backed (09.3+) |

### Supporting
| Library/Tool | Version (repo) | Purpose | When to Use |
|---|---:|---|---|
| streamlit-autorefresh | 1.0.1 | Polling-based "live graph" refresh | When Surfing panel is open and live state should update |
| pytest | (repo lock) | Contract + routing tests | Add endpoint + selection mapping tests |

## Architecture Patterns

### Recommended Project Structure (10-03)
Keep navigation logic local to Surfing components and keep backend additions small and query-oriented.

```
frontend/
|-- components/
|   |-- cytoscape_panel.py           # component wrapper (already)
|   |-- live_surfing_panel.py        # Surfing controller (extend/replace)
|   `-- ...
|-- graph_api.py                     # existing graph endpoints client
|-- project_api.py                   # project_meta (graph_settings persistence)
`-- workflow_api.py                  # run status polling (requested works patterns)

backend/
|-- graph_store.py                   # document graph nodes/edges + votes
|-- span_graph_store.py              # span-first graph + assertions/status
`-- main.py                          # thin /graph/*, /spans/*, /claims/* endpoints
```

### Pattern 1: Seq-Gated Component Event Processing (Deterministic)
**What:** Only process Cytoscape payloads once by comparing `seq` against a stored `last_seq` per component key.
**When to use:** Any Streamlit rerun-driven interaction where the component can resend the previous value.
**Example:**

```python
# Source: frontend/components/live_surfing_panel.py
comp_value = st.session_state.get(component_key)
evt_seq = int(comp_value.get("seq") or 0)
last_seq = int(st.session_state.get("surf_live_last_event_seq") or 0)
is_new_evt = bool(evt_seq and evt_seq > last_seq)
if is_new_evt:
    st.session_state["surf_live_last_event_seq"] = evt_seq
    # ... mutate selection/expansion state exactly once ...
```

**Implementation note:** keep `last_event_seq` scoped to the component instance key (`key=` passed into `cytoscape_panel.render`) so changing the component key intentionally resets event history.

### Pattern 2: Stable Layout + One-Hop Expand Without Forced Relayout
**What:** Render with `options.stableLayout=True` and only trigger full layout when the user explicitly requests it (via `layoutNonce`).
**When to use:** Expand-in-place ("append nodes") while preserving user panning/zoom/dragged positions.
**Example:**

```python
# Source: frontend/components/live_surfing_panel.py
cytoscape_panel.render(
    elements,
    layout={"name": "dagre", "rankDir": "LR", "fit": True, "padding": 30},
    options={
        "stableLayout": True,
        "layoutNonce": int(st.session_state.get("surf_live_layout_nonce") or 0),
    },
)
```

```js
// Source: frontend/components/cytoscape_component/index.html
// stableLayout: seed new nodes near neighbors; restore viewport/positions;
// run global layout only when needed (posCache empty or layoutNonce changes).
```

### Pattern 3: "Soft Focus" Selection + Explicit Route Actions
**What:** A click updates a small `selection` object and renders a compact Streamlit popover; routing only occurs via explicit buttons like `Go Read`, `Go Chase`, `Open PDF`.
**When to use:** Always in 10-03 (locked decision).
**How:**
- Store selection as a single session-state dict (e.g. `graph_nav_selection = {"type": "work|citespan|claimspan|edge", "id": "..."}`) and pass it into `cytoscape_panel.render(selection=...)` so the component reflects it.
- On action click, mutate the workspace routing keys and call `st.rerun()`.

### Pattern 4: Workspace Routing via Session State (No Router)
**What:** Route by mutating shared session keys; Reading and Chasing already respond to these.
**When to use:** For all graph->workspace navigation.
**Example:**

```python
# Source: frontend/components/live_surfing_panel.py
def _set_active_callout(*, doc_id: str, sentence_id: str | None,
                        citation_index: int, target_id: str | None) -> None:
    st.session_state["selected_doc_id"] = doc_id
    st.session_state["citation_selected_index"] = int(citation_index)
    st.session_state["citation_selected_target"] = (str(target_id).strip() if target_id else None)
    st.session_state["citation_selected_sentence_id"] = (str(sentence_id).strip() if sentence_id else None)
    st.session_state["selected_callout_tuple"] = {
        "doc_id": doc_id,
        "citation_index": int(citation_index),
        "target_id": str(target_id).strip() if target_id else None,
        "sentence_id": str(sentence_id).strip() if sentence_id else None,
    }
    # Cache-bust reading/chasing derived state.
    st.session_state["citation_context_key"] = None
    st.session_state["citation_context"] = None
    st.session_state["citation_graph_key"] = None
    st.session_state["citation_graph"] = None
```

Use `frontend/state_keys.py` constants to switch tabs:
- `WORKSPACE_ACTIVE_TAB = WORKSPACE_TAB_DOCUMENT` for `Go Read`
- `WORKSPACE_ACTIVE_TAB = WORKSPACE_TAB_REVIEW` for `Go Chase`

### Backend Contracts Needed For 10-03 (Prescriptive)

The following are the minimal backend additions that materially simplify planning/implementation while keeping scope inside 10-03:

1) **Work contexts for chooser (ambiguity handling)**
- `GET /nav/works/{work_id}/contexts?reviewer_uid=...`
- Returns: ordered list of citing contexts `{citing_doc_id, citation_index, reference_id, sentence_id?, snippet?}` for that work.
- Source of truth: `span_graph_*` cite rows (work cited by many spans) + confirmed claim text for snippets.

2) **Graph view scope payload (optional but simplifies UI correctness)**
- `GET /nav/graph?reviewer_uid=...&focus=...&show_claimspans=...`
- Returns Cytoscape `elements[]` with stable ids and all UI-required metadata (labels, state badges, edge kinds).
- If you skip this endpoint, you must replicate DB joins in Streamlit; planning should assume one place owns the "graph model".

3) **Request/cancel retrieval for missing works (no auto-download)**
- Prefer persistence in `project_meta.graph_settings` (no new tables).
- Provide endpoints only if you need server-side validation or multi-client concurrency:
  - `POST /nav/requests` (add)
  - `POST /nav/requests/{request_id}/cancel`

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---|---|---|---|
| Cytoscape event bridge | new Streamlit component or JS build toolchain | existing vendored component (`frontend/components/cytoscape_component/index.html`) | Already solves rerun + stable layout + event sequencing |
| "Streamlit router" | custom URL-router/state machine | session-state keys + `WORKSPACE_ACTIVE_TAB` (`frontend/state_keys.py`) | Repo already routes this way; keeps behavior consistent |
| Citation-window span ids | new id scheme for citespans/claimspans | `GET /spans/lookup-citation-window` + `/claims/{claim_id}/span-context` | Existing backend mapping is already stable and tested |
| Retrieval instructions | scraping links client-side | `GET /references/{doc_id}/{reference_id}/retrieval` | Centralizes metadata + fallback instructions |

**Key insight:** in this repo, deterministic navigation comes from (a) stable ids, (b) `seq`-gated event handling, and (c) routing only by mutating a small set of session keys - avoid adding extra layers.

## Common Pitfalls

### Pitfall 1: Streamlit reruns re-processing the same Cytoscape click
**What goes wrong:** selection toggles/expands multiple times because the component value remains in `st.session_state` across reruns.
**Why it happens:** Streamlit persists the last component value; reruns re-execute your handler.
**How to avoid:** gate on `seq` and update `last_event_seq` only after successful handling (Pattern 1).
**Warning signs:** expand/collapse triggers twice; selection clears unexpectedly.

### Pitfall 2: Double-click produces both click and dblclick behaviors
**What goes wrong:** single-click action fires, then dblclick action fires for the same gesture.
**Why it happens:** you schedule click with a timer and must cancel it on dblclick.
**How to avoid:** keep the component's timer-based behavior and only attach expand semantics to `action in {dblclick,context}` (already implemented in the component).
**Source:** click is delayed 420ms and cancelled on dblclick (`frontend/components/cytoscape_component/index.html`).

### Pitfall 3: Selection "stickiness" breaks when node ids change
**What goes wrong:** selection highlight disappears when a work transitions from placeholder->ingested or when expansion rewrites ids.
**Why it happens:** Cytoscape selection uses element id; changing ids makes it a different element.
**How to avoid:** choose ids that remain stable across state transitions (e.g., keep a single canonical `work_id` string; store ingest id as metadata, not as node id).

### Pitfall 4: Persisting graph expansions clobbers other project_meta.graph_settings
**What goes wrong:** saving expansions overwrites reviewer identities or other graph settings.
**Why it happens:** `/project` update replaces `graph_settings` as a whole (no deep merge on the backend).
**How to avoid:** read-modify-write `graph_settings` client-side and PUT the full merged dict.

### Pitfall 5: Polling live state causes layout churn and UI flicker
**What goes wrong:** graph re-layouts every poll or jumps viewport.
**Why it happens:** you bump `key=` or `layoutNonce` or you disable `stableLayout`.
**How to avoid:** keep a stable `key=` for the component; only bump `layoutNonce` on user "Reset layout"; keep `stableLayout=True` during polling.

## Code Examples

### Cytoscape component payload shape (what Streamlit receives)
```js
// Source: frontend/components/cytoscape_component/index.html
sendValue({ type: "node", id, action: "dblclick", seq: ++eventSeq, shift, alt, meta, ctrl });
scheduleClick({ type: "node", id, shift, alt, meta, ctrl }); // sends {action:"click"} later
sendValue({ type: "edge", id, action: "click", seq: ++eventSeq, ...mod });
```

### Routing a graph selection into Reading/Chasing
```python
# Source: frontend/components/live_surfing_panel.py + frontend/state_keys.py
_set_active_callout(doc_id=doc_id, sentence_id=sentence_id,
                    citation_index=citation_index, target_id=target_id)
st.session_state[WORKSPACE_ACTIVE_TAB] = WORKSPACE_TAB_DOCUMENT  # Go Read
st.rerun()
```

### Getting a claimspan's resolved span context (backend contract you can lean on)
```python
# Source: backend/main.py
@app.get("/claims/{claim_id}/span-context")
def get_claim_span_context(claim_id: str, target_id: str | None = None):
    # returns span_id + claim_span_id + cited_work_id for routing
```

## State of the Art (Repo-Relevant)

| Older Approach | Current Approach | Where in Repo | Impact |
|---|---|---|---|
| Graph is "view only" and partially working | Graph emits deterministic events (`seq`) and supports stable expand-in-place | `frontend/components/cytoscape_component/index.html` | Enables reliable navigation without JS build complexity |
| Local sqlite graph stores | Postgres-backed GraphStore + SpanGraphStore | `backend/graph_store.py`, `backend/span_graph_store.py` | Makes expansions/requests persistable per project |
| "Streaming" updates in Streamlit | Polling-first via `streamlit_autorefresh` | `frontend/components/chase_queue.py` | Avoids SSE-in-Streamlit complexity |

## Open Questions

1) **Canonical work id for graph selection**
   - What we know: GraphStore uses `doc:{doc_key}` node ids + aliases (`ingest:...`, `doi:...`, `ref:{ingest}:{ref}`).
   - What's unclear: whether the Surfing graph should use GraphStore node ids as Cytoscape ids (stable) or use ingest ids (simple routing).
   - Recommendation: use GraphStore document `node_id` (`doc:...`) as the Cytoscape id and keep `ingest_id` as metadata for routing.

2) **Where to persist expansion state ("one hop per click; per project")**
   - What we know: durable per-project JSON exists (`project_meta.graph_settings`).
   - Risk: `graph_settings` is replaced as a whole on update.
   - Recommendation: store a small, capped set of ids (e.g. `expanded_work_ids`, `expanded_context_ids`) in `graph_settings` and always read-modify-write.

3) **How to represent requested works (no auto-download) without new tables**
   - What we know: workflow runs already expose a requested-target queue, but they are claimspan-scoped; attachments require an uploaded PDF.
   - Recommendation: persist request intent in `project_meta.graph_settings` keyed by `{parent_work_id, reference_id or work_id}` and derive live availability from ledger/ingest/resolution/attachments.

## Sources (Repo)

### Primary (HIGH confidence)
- `frontend/components/cytoscape_component/index.html` (event payloads, stable layout, `seq`, click vs dblclick)
- `frontend/components/cytoscape_panel.py` (component wrapper contract)
- `frontend/components/live_surfing_panel.py` (seq-gated event handling; routing state mutation pattern)
- `frontend/state_keys.py` (tab routing keys/labels)
- `backend/main.py` (`/spans/lookup-citation-window`, `/claims/{claim_id}/span-context`, `/references/*/retrieval`, workflow status)
- `backend/graph_store.py` (document nodes/edges, citation edges, ledger)
- `backend/span_graph_store.py` (span cite indexing, claimspans, assertions/status)
- `tests/test_span_graph_rebuild.py` (span endpoints + claimspan context roundtrip)

## Metadata

**Confidence breakdown:**
- Cytoscape event wiring + Streamlit determinism: HIGH (verified in component + live panel code)
- Routing semantics (graph -> Reading/Chasing): HIGH (existing `_set_active_callout` + tab keys)
- Recursive retrieval queue persistence strategy: MEDIUM (existing primitives exist; exact persistence/endpoint shape must be chosen)

**Research date:** 2026-02-22
**Valid until:** 2026-03-21
