#!/usr/bin/env bash
set -euo pipefail

.venv/bin/pytest -q tests/test_graph_compaction_store.py
.venv/bin/pytest -q tests/test_graph_compaction_api.py
.venv/bin/pytest -q tests/test_graph_compaction_rollback.py

.venv/bin/python scripts/dev/smoke_stage3_graph_compaction.py \
  --api-url "${API_URL:-http://localhost:8000}" \
  --fixture "${FIXTURE:-fixtures/text-1.pdf}"

echo "OK: verify_10_04_5_03_graph_compaction"
