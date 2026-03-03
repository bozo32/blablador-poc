# OS-ERIN Use Case: Bug Hunt (Surfing Graph)

Use this lane when debugging Cytoscape graph behavior in Surfing.

## Primary Goals

- Keep graph navigation deterministic under Streamlit reruns.
- Preserve explicit routing semantics from graph selection into Reading/Chasing.
- Avoid introducing graph-tail complexity while upstream bug hunting is active.

## Read Order

1. `docs/START_HERE.md`
2. `docs/IDENTITY_SCOPE_VISIBILITY_CONTRACT.md`
3. `docs/WORKFLOW_PROTOCOL.md`
4. `docs/CYTOSCAPE_COMPACT.md`
5. `.planning/phases/10-contracts-core-workflow-simplification/done/10-03-SUMMARY.md`
6. `.planning/debug/*.md` (graph-related notes)

## Guardrails

- Do not remove seq gating for Cytoscape events.
- Do not repurpose graph rendering layer as state authority.
- Keep component key stable during normal reruns.
- Treat graph selection context as ephemeral and nonce-bounded.
- Preserve strict scope/membership behavior on `/nav/*` and retrieval routes.

## Failure Triage Order

1. Contract failure (scope/header/membership) or UI bug?
2. Event determinism bug (seq/key/nonce) or backend payload bug?
3. Routing bug (wrong doc/span/ref) or context ambiguity bug?
4. Styling/readability issue or semantic/state issue?

## Verify First

- `bash scripts/dev/verify_10_03_graph_nav.sh`
- `./.venv/bin/pytest -q tests/test_nav_graph_api.py`
- `./.venv/bin/pytest -q tests/test_span_graph_rebuild.py`

## High-Signal Files

- `frontend/components/live_surfing_panel.py`
- `frontend/components/cytoscape_panel.py`
- `frontend/components/cytoscape_component/index.html`
- `frontend/nav_api.py`
- `backend/main.py` (`/nav/graph`, `/nav/works/{work_id}/contexts`)

## Logging Targets

- Findings: `.planning/debug/`
- Deferred cleanup: `.planning/phases/10-contracts-core-workflow-simplification/10-04.5-PRUNE-CANDIDATES.md`
