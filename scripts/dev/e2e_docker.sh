#!/bin/sh
set -eu

# Full stack E2E smoke, fully containerized.

bash scripts/dev/up.sh
bash scripts/dev/gen_text_fixture.sh fixtures/text-1.pdf
bash scripts/dev/gen_scanned_fixture.sh fixtures/scanned-1.pdf

docker compose run --rm \
  -v "${PWD}:/repo" \
  -w /repo \
  -e API_URL="http://host.docker.internal:8000" \
  app-api \
  python scripts/dev/e2e_flow.py
