#!/usr/bin/env bash
set -euo pipefail

echo "[PR-10] opinion visibility ACL + strict scope gates"

PYTEST_BIN="${PYTEST_BIN:-.venv/bin/pytest}"

"${PYTEST_BIN}" -q tests/test_opinion_visibility_acl_api.py tests/test_scope_write_pilot_api.py tests/test_scope_strict_remaining_routes_api.py
"${PYTEST_BIN}" -q tests/test_frontend_scope_client_guards.py
"${PYTEST_BIN}" -q tests/test_project_scope_api.py tests/test_project_membership_api.py

echo "PASS: PR-10 visibility ACL verification"
