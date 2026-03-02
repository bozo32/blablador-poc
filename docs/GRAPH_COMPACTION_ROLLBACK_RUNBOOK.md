# Graph Compaction Rollback Runbook

## Scope

This runbook covers rollback for Stage 3 durable graph compaction (`doc-key-dedup-v1`) backed by:

- `graph_compaction_runs` (journal metadata)
- `snapshot_json` capture of graph rows before each apply run

Compaction touch points are limited to graph tables:

- `graph_nodes`
- `graph_aliases`
- `graph_edges`
- `graph_edge_votes`

## Normal Operation

1. Preview candidates:

```bash
curl -sS -X POST http://localhost:8000/maintenance/graph/compact/dry-run \
  -H 'Content-Type: application/json' \
  -d '{"project_id":"default"}'
```

2. Apply compaction:

```bash
curl -sS -X POST http://localhost:8000/maintenance/graph/compact/apply \
  -H 'Content-Type: application/json' \
  -d '{"project_id":"default"}'
```

Save the `run_id` from the apply response.

## Rollback Procedure

Use the original apply `run_id`:

```bash
curl -sS -X POST http://localhost:8000/maintenance/graph/compact/rollback \
  -H 'Content-Type: application/json' \
  -d '{"project_id":"default","run_id":"gcompact:..."}'
```

Expected response:

- `mode=rollback`
- `status=completed`
- `report.source_run_id` matches the apply run id

## Post-Rollback Validation

1. Re-run dry-run and inspect counts.
2. Spot-check known aliases/edges that were compacted.
3. Run targeted tests:

```bash
.venv/bin/pytest -q tests/test_graph_compaction_rollback.py
```

## Failure Handling

- If rollback returns 404: verify run id exists in `graph_compaction_runs`.
- If rollback returns 422:
  - run id may belong to a different project
  - run id may be a dry-run (not apply)
  - apply run may be missing snapshot data

If an apply run has no usable snapshot, restore from database backup.
