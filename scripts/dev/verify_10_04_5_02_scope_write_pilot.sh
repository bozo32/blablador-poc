#!/usr/bin/env bash
set -euo pipefail

PYTEST_BIN="${PYTEST_BIN:-.venv/bin/pytest}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

"${PYTEST_BIN}" -q tests/test_scope_guard_primitive.py
"${PYTEST_BIN}" -q tests/test_scope_write_pilot_api.py
"${PYTEST_BIN}" -q tests/test_frontend_scope_client_guards.py

"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

main_py = Path("backend/main.py").read_text(encoding="utf-8")

required_fragments = [
    'endpoint="/ledger/{doc_num}/outgoing"',
    'endpoint="/ledger/{doc_num}/incoming"',
    'endpoint="/ledger/{doc_num}/assign"',
    'endpoint="/ledger/place"',
    'endpoint="/ledger/place-reference"',
    'endpoint="/attachments/upload"',
    'endpoint="/attachments/{attachment_id}"',
    'endpoint="/attachments/{attachment_id}/clone"',
    'endpoint="/attachments/{attachment_id}/promote-ingest"',
    'endpoint="/attachments/{attachment_id}/retry"',
    'endpoint="/opinions/events"',
]

for fragment in required_fragments:
    if fragment not in main_py:
        raise SystemExit(f"missing strict scope pilot fragment: {fragment}")

print("OK: strict write-path pilot wiring fragments found")
PY

echo "PASS: verify_10_04_5_02_scope_write_pilot"
