#!/bin/sh
set -eu

# Runs pytest inside the app-api container image.
# This avoids host Python binary-compat issues.

docker compose run --rm \
  -v "${PWD}:/repo" \
  -w /repo \
  app-api \
  sh -lc "python -m pip install -q pytest && python scripts/dev/pytest_runner.py"
