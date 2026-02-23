# Summary: 10-03 Graph as navigation

**Phase:** 10 - Contracts + Core Workflow Simplification
**Plan:** 03
**Status:** Complete
**Date:** 2026-02-23

## What was built

Made the Cytoscape Surfing graph a real navigation surface:
- Clicks produce stable selection; explicit actions route into Reading/Chasing
- Double-click on work focuses graph to show its citespans
- Double-click on citespan focuses graph to show its claimspans
- Go Read/Go Chase enabled for work nodes that have resolved_ingest_id
- Missing works show Retrieval instructions with Request/Cancel toggle

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | `15daa6b` | Add /nav API endpoints (GET /nav/graph, GET /nav/works/{work_id}/contexts) + tests + verifier script |
| 2 | `e9a55bf` | Implement Surfing selection contract with seq-gated events |
| 2 | `4b6c28b` | Enrich work-focused /nav/graph with routing fields + claimspans |
| 2 | `e454655` | Fix Open PDF to route with selection context |
| 2 | `c821fb2` | Make TEI abstract citations navigable; add fixtures/citations-2.pdf |
| 2 | `eab544f` | Expand cited work nodes to show more citing spans |
| 3 | `b82a2b1` | Enable Go Read/Go Chase for work nodes; add double-click drill-down |

## Files modified

- `backend/main.py` - /nav/graph and /nav/works/{work_id}/contexts endpoints
- `backend/span_graph_store.py` - list_citing_contexts_for_work helper
- `frontend/nav_api.py` - Client wrappers for /nav endpoints
- `frontend/components/live_surfing_panel.py` - Seq-gated selection, action-first routing, double-click drill-down
- `tests/test_nav_graph_api.py` - Unit tests for /nav endpoints
- `scripts/dev/verify_10_03_graph_nav.sh` - Docker-compose verifier
- `fixtures/citations-2.pdf` - Tiny fixture PDF with real citations

## Verification

- [x] `bash scripts/dev/pytest_docker.sh` passes
- [x] `bash scripts/dev/verify_10_03_graph_nav.sh` prints `OK: verify_10_03_graph_nav`
- [x] Manual checkpoint: Surfing selection/routing deterministic; double-click drill-down works

## Decisions

1. **Seq-gated events**: Cytoscape returns `{type, id, action, seq}`; frontend only processes events where `seq > last_seq` to avoid double-processing under Streamlit reruns.
2. **Focus-based drill-down**: Double-click changes `graph_nav_focus` to refetch graph elements rather than client-side node filtering.
3. **Work routing**: Work nodes with `resolved_ingest_id` can use Go Read/Go Chase to open the document directly (no callout context needed).
4. **Cited work contexts**: Missing/available cited works fetch contexts via `/nav/works/{work_id}/contexts` for ambiguity chooser.

## Open items

- Expand button still persists to `project_meta.graph_settings.expanded_work_ids` but focus-based drill-down is now the primary mechanism.
- Loop issue reported during checkpoint (segmentation push-through) - resolved by fixing routing state management.
