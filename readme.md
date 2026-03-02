# Session Re-Orient Guide

Use this prompt when restarting OpenCode after a break:

```text
Re-orient in /data/repos/blablador-poc on branch <branch-name>.

Goal:
- <one sentence goal>

Context from last session:
- Implemented ingest-id graph compaction in backend/graph_compaction.py
- Updated tests in tests/test_graph_compaction_store.py and tests/test_graph_compaction_api.py
- Updated smoke script scripts/dev/smoke_stage3_graph_compaction.py
- Pytest for graph compaction passes
- Smoke passes on :8001, but :8000 was running old root-owned server

Please do this first:
1) Show git status and summarize only relevant changed files
2) Run targeted tests: .venv/bin/pytest tests/test_graph_compaction_store.py tests/test_graph_compaction_api.py
3) Verify API behavior (tell me whether running server is old/new strategy)
4) Propose next exact step to finish cleanly (commit or further fix)
```

## Helpful Startup Details

- Repo root: `/data/repos/blablador-poc`
- Files changed for this task:
  - `backend/graph_compaction.py`
  - `tests/test_graph_compaction_store.py`
  - `tests/test_graph_compaction_api.py`
  - `scripts/dev/smoke_stage3_graph_compaction.py`
- Compaction strategy expected in API responses:
  - `doc-key+ingest-id-dedup-v2`
- New report fields expected:
  - `ingest_id_dedup`
  - `mutation_journal` (apply mode)

## Quick Verification Commands

```bash
git status --short
.venv/bin/pytest tests/test_graph_compaction_store.py tests/test_graph_compaction_api.py
.venv/bin/python scripts/dev/smoke_stage3_graph_compaction.py --api-url http://127.0.0.1:8001 --fixture fixtures/text-1.pdf
```

If checking the root-owned server on `:8000`, compare dry-run strategy quickly:

```bash
curl -sS -X POST http://localhost:8000/maintenance/graph/compact/dry-run \
  -H 'content-type: application/json' \
  -d '{"project_id":"default"}'
```

If the strategy still shows `doc-key-dedup-v1`, that server is stale and needs restart/redeploy.
