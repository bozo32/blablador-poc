#!/usr/bin/env bash
set -euo pipefail

# 10-04.5 umbrella verifier: Stage A-F protocol runner.
#
# Notes:
# - Each underlying verifier is allowed to be self-contained (may wipe data / restart app-api).
# - This runner prints PASS/FAIL per stage and returns non-zero if any stage fails.

repo_root() {
  git rev-parse --show-toplevel 2>/dev/null || pwd
}

say() {
  printf "%s\n" "$*"
}

run_stage() {
  local stage="$1"
  shift

  say ""
  say "=== ${stage} ==="
  say "+ $*"

  local rc=0
  set +e
  "$@"
  rc=$?
  set -e

  if [ "$rc" -eq 0 ]; then
    say "PASS: ${stage}"
  else
    say "FAIL: ${stage} (rc=${rc})"
  fi
  return "$rc"
}

start_api_from_worktree() {
  if ! command -v docker >/dev/null 2>&1; then
    say "WARN: docker not found; assuming API already running at ${API_URL}"
    return 0
  fi
  if ! docker info >/dev/null 2>&1; then
    say "WARN: docker not available; assuming API already running at ${API_URL}"
    return 0
  fi
  if ! command -v docker compose >/dev/null 2>&1; then
    say "WARN: docker compose not found; assuming API already running at ${API_URL}"
    return 0
  fi

  docker compose up -d postgres minio minio-init grobid fallback-worker

  # Ensure the running API container reflects the current working tree without rebuild.
  docker compose stop app-api >/dev/null 2>&1 || true
  docker compose rm -f app-api >/dev/null 2>&1 || true

  # Clean up any prior `docker compose run -d app-api` containers that may still hold port 8000.
  old_ids=$(docker ps -aq --filter "name=blablador-poc-app-api-run" || true)
  if [ -n "${old_ids}" ]; then
    docker rm -f ${old_ids} >/dev/null 2>&1 || true
  fi

  local root
  root="$(repo_root)"

  docker compose run -d --service-ports \
    -v "${root}/backend:/app/backend" \
    -v "${root}/frontend:/app/frontend" \
    -v "${root}/application.py:/app/application.py" \
    app-api >/dev/null
}

API_URL="${API_URL:-http://127.0.0.1:8000}"
API_URL="${API_URL%/}"

# Default to repo fixtures if corpus/workflow is absent.
if [ -z "${CITING_PDF_PATH:-}" ]; then
  if [ -f corpus/workflow/rudko-chatgpt-is-incredible-at-being-average.pdf ]; then
    CITING_PDF_PATH="corpus/workflow/rudko-chatgpt-is-incredible-at-being-average.pdf"
  else
    CITING_PDF_PATH="fixtures/citations-2.pdf"
  fi
fi

if [ -z "${SOURCE_PDF_PATH:-}" ]; then
  if [ -f corpus/workflow/hicks-chatgpt-is-bullshit.pdf ]; then
    SOURCE_PDF_PATH="corpus/workflow/hicks-chatgpt-is-bullshit.pdf"
  else
    SOURCE_PDF_PATH="fixtures/sample.pdf"
  fi
fi

export API_URL
export CITING_PDF_PATH
export SOURCE_PDF_PATH

say "API_URL=${API_URL}"
say "CITING_PDF_PATH=${CITING_PDF_PATH}"
say "SOURCE_PDF_PATH=${SOURCE_PDF_PATH}"

start_api_from_worktree

failures=0

# A-D: Intake hardening verifier (ingest + attachments + clone + project isolation)
if ! run_stage "Stage A-D (Intake/Ingest/Resolve/Attachments)" bash scripts/dev/verify_10_04_5_01_intake.sh; then
  failures=$((failures+1))
fi

# E: Durable evidence decisions/events
if ! run_stage "Stage E (Evidence Decisions Durability)" bash scripts/dev/verify_10_04_decisions.sh; then
  failures=$((failures+1))
fi

# F: Graph navigation surface
if ! run_stage "Stage F (Graph Navigation)" bash scripts/dev/verify_10_03_graph_nav.sh; then
  failures=$((failures+1))
fi

say ""
if [ "${failures}" -eq 0 ]; then
  say "ALL PASS: 10-04.5 spine workflow"
  exit 0
fi

say "FAILED: ${failures} stage(s)"
exit 1
