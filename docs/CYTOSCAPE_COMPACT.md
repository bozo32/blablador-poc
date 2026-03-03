# Cytoscape Compact (OS-ERIN)

Repo-specific compact manual for Surfing graph behavior.

## Governance Metadata

- Doc role: implementation compact guide
- Authority tier: protocol/support (non-contract)
- Status: active
- Owner: repo maintainers
- Last reviewed: 2026-03-03
- Canonical for: Cytoscape event/render contract in this repo
- Must not override: `docs/IDENTITY_SCOPE_VISIBILITY_CONTRACT.md` or `docs/WORKFLOW_PROTOCOL.md`

## Purpose

This document captures what matters most when debugging or extending the
Surfing graph. It is intentionally short and tied to the current codebase.

## 1) Component Contract (JS <-> Streamlit)

Primary files:

- `frontend/components/cytoscape_component/index.html`
- `frontend/components/cytoscape_panel.py`

Input contract (`cytoscape_panel.render(...)`):

- `elements`: Cytoscape-compatible elements list
- `style`: Cytoscape style rules
- `layout`: layout config (default dagre LR)
- `selection`: external selection to apply (`{type,id}`)
- `focus`: optional list of node ids to fit viewport
- `options`: includes `stableLayout`, `layoutNonce`, `showLabels`

Output event payload (from component to Python):

- `{type, id, action, seq, shift, alt, meta, ctrl}`
- `type`: `node` or `edge`
- `action`: `click`, `dblclick`, `context`
- `seq`: monotonically increasing event id

Key runtime behavior:

- Click is delayed ~420ms so a double-click can cancel it.
- Double-click threshold is ~600ms on same node id.
- Empty canvas tap sends `{}` and clears selection.

## 2) Determinism Rules (Critical)

- Always process events only when `seq > last_seq` for the same component key.
  - Current state key: `graph_nav_last_event_seq_by_key`.
- Keep a stable component key for normal reruns.
  - Current key: `surf-nav::cytoscape`.
- Keep `stableLayout=True` in normal operation.
- Bump `layoutNonce` only on explicit "Reset layout" actions.

If you violate these, you get duplicate actions, selection races, or viewport
churn/flicker.

## 3) Selection and Focus Model

Primary file:

- `frontend/components/live_surfing_panel.py`

Core state:

- `graph_nav_selection`: current selected element (`type`,`id`)
- `graph_nav_focus`: graph fetch focus (`focus_type`,`focus_id`)
- `graph_nav_selected_context`: chosen citing context for ambiguous work nodes
- `graph_nav_selection_nonce`: invalidates stale context selectors
- `graph_nav_show_claimspans`: density toggle

Double-click drill-down semantics:

- work node dblclick -> set focus `{focus_type:"work", focus_id:work_id}`
- citespan node dblclick -> set focus `{focus_type:"citespan", focus_id:span_id}`

## 4) Backend Nav API Contract

Primary endpoints:

- `GET /nav/graph`
- `GET /nav/works/{work_id}/contexts`

Required headers:

- `X-Project-Id`
- `X-Reviewer-Uid`

Behavior:

- `focus_type` + `focus_id` drive graph content.
- `show_claimspans` toggles claimspan nodes.
- elements are best-effort, but each element must include:
  - `data.id`
  - `data.selectable_type`
  - `data.state`

Node id conventions:

- `work:<work_id>`
- `citespan:<span_id>`
- `claimspan:<claim_span_id>`
- edge ids: `edge:<kind>:<src>-><tgt>`

State vocab (backend):

- `missing`, `requested`, `blocked`, `available`, `processing`, `done`,
  `error`, `cancelled`, `mixed`

## 5) Styling and Meaning

Current surf styles in `live_surfing_panel.py`:

- work nodes: blue rounded rectangles
- citespan nodes: green ellipses
- claimspan nodes: gray rounded rectangles
- state `missing`: dashed/red border
- state `requested`: orange border/emphasis

Use style changes for readability only; do not encode semantics solely in color.
Semantics must remain in `data.state` and routing fields.

## 6) Known Failure Classes

1. Duplicate action processing
   - Cause: missing seq gate or key churn
2. Layout/viewport churn on rerun
   - Cause: non-stable key, nonce misuse, stableLayout disabled
3. Wrong routing target from graph selection
   - Cause: stale `graph_nav_selected_context` or missing routing fields
4. Scope/membership failures (4xx)
   - Cause: missing/invalid `X-Project-Id` or reviewer mismatch
5. Ambiguous work context handling regressions
   - Cause: context selector not nonce-bounded

## 7) Verification Quick Pack

- `bash scripts/dev/verify_10_03_graph_nav.sh`
- `./.venv/bin/pytest -q tests/test_nav_graph_api.py`
- `./.venv/bin/pytest -q tests/test_span_graph_rebuild.py`

Optional full stage workflow:

- `bash scripts/dev/verify_10_04_5_spine_workflow.sh`

## 8) Extension Guardrails

- Treat Cytoscape as renderer/event bridge, not authority.
- Keep navigation semantics in Python (`live_surfing_panel.py`) and backend nav
  endpoints.
- Preserve action-first routing (selection -> explicit route actions) to avoid
  accidental tab/context jumps.
- Any new graph interaction must be scoped and fail-closed under existing
  identity/scope contracts.
