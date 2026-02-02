# Cleanup Execution Protocol

Run this before starting the next GSD phase whenever the session has accumulated multiple incremental UI/backend changes.

## Goals

- Restore a coherent structure (especially `frontend/ui.py`).
- Reduce duplicated logic (same workflow rendered in multiple places).
- Prevent Streamlit key collisions and state drift.
- Keep behavior stable; only refactor when there is a test/verification path.

## Protocol

1) Snapshot and scope
- Capture current behavior with a short human-check list (3-5 bullets).
- Identify "hot" files that have grown (typically `frontend/ui.py`) and list the sub-features inside.

2) Extract components
- Move self-contained UI blocks into `frontend/components/` (or a dedicated module) with explicit inputs/outputs.
- Centralize Streamlit widget key generation into one helper (e.g., `scope + claim_id + citation_index`).

3) Normalize state
- Pick one canonical session-state key per piece of data.
- Any duplicated widgets must sync to the canonical value and use unique widget keys.
- Remove stale/unused session keys (or gate them behind a single feature flag).

4) Remove duplication
- Ensure the same workflow isn't rendered twice ("queue expanded" vs "tab view") unless intentionally mirrored.
- Prefer a single rendering function with a `scope` parameter for minor layout differences.

5) Verify
- Run `pytest -q`.
- Manual smoke test:
  - chase queue open/close works and only one item is expanded
  - switching citations prompts on unsaved edits
  - evidence rerun works; no NLI runs on read-only GET
  - attachments UI shows one source drop zone per source group

6) Document
- Update `.planning/STATE.md` with what was stabilized and what remains.
- Add any structural follow-ups to `.planning/research/FEATURES.md` under v2+.

## Commit guidance

- Prefer one commit for "behavior fixes" and another for "refactor-only" changes.
- Avoid mixing UI reshuffles with backend behavior changes in the same commit.
