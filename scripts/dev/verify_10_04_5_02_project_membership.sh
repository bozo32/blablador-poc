#!/usr/bin/env bash
set -euo pipefail

echo "[PR-08] project membership gates"

PYTEST_BIN="${PYTEST_BIN:-.venv/bin/pytest}"

"${PYTEST_BIN}" -q tests/test_project_membership_spine.py
"${PYTEST_BIN}" -q tests/test_project_membership_api.py
"${PYTEST_BIN}" -q tests/test_project_scope_api.py tests/test_scope_guard_primitive.py tests/test_frontend_scope_selector_lock.py tests/test_project_api_scope_headers.py

echo "PASS: PR-08 project membership verification"
