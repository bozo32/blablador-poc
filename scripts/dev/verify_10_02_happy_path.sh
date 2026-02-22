#!/usr/bin/env bash
set -euo pipefail

curl_json() {
  curl -fsS --retry 20 --retry-delay 1 --retry-connrefused --max-time 30 "$@"
}

docker compose up -d postgres minio minio-init grobid fallback-worker

# Ensure the running API container reflects the current working tree without
# requiring a full image rebuild (use bind mounts to overlay code).
docker compose stop app-api >/dev/null 2>&1 || true
docker compose rm -f app-api >/dev/null 2>&1 || true
docker compose run -d --service-ports \
  -v "${PWD}/backend:/app/backend" \
  -v "${PWD}/frontend:/app/frontend" \
  -v "${PWD}/application.py:/app/application.py" \
  app-api >/dev/null

API_URL="http://127.0.0.1:8000"

for i in {1..120}; do
  code=$(curl -s --max-time 2 -o /dev/null -w "%{http_code}" "${API_URL}/docs" || true)
  if [ "$code" = "200" ]; then
    break
  fi
  sleep 1
done

if [ "${code:-}" != "200" ]; then
  echo "ERROR: API did not become ready at ${API_URL} (last code=${code:-})" >&2
  exit 1
fi

curl_json -X POST "${API_URL}/dev/wipe" \
  -H "Content-Type: application/json" \
  -d '{"confirm":"WIPE"}' >/dev/null

export API_URL

python - <<'PY'
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from scripts.dev import e2e_flow


API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
e2e_flow.API_URL = API_URL


def req(method: str, path: str, **kwargs) -> dict:
    url = f"{API_URL}{path}"
    resp = requests.request(method, url, timeout=kwargs.pop("timeout", 60), **kwargs)
    if resp.status_code >= 400:
        raise RuntimeError(f"{method} {path} failed: {resp.status_code} {resp.text[:200]}")
    return resp.json() if resp.text else {}


repo = Path(".").resolve()
sample = repo / "fixtures" / "text-1.pdf"
if not sample.exists():
    raise RuntimeError(f"missing fixture: {sample}")

doc_id = e2e_flow.upload_pdf(sample)
e2e_flow.trigger_extract(doc_id)
state, _spine = e2e_flow.poll_attempt(doc_id, timeout_s=240)
if state not in {"succeeded", "partial"}:
    raise RuntimeError(f"unexpected extraction state={state}")

body = e2e_flow.get_body(doc_id)
if len(body.get("paragraphs") or []) == 0:
    e2e_flow.force_fallback(doc_id)
    _ = e2e_flow.poll_until_artifact(doc_id, artifact_type="fallback.body.txt", timeout_s=240)
    body = e2e_flow.get_body(doc_id)
e2e_flow.confirm_one_claim(doc_id=doc_id, body=body)

claim_id = f"cite:{doc_id}:0:default:1a"
run = req(
    "POST",
    f"/workflow/claimspans/{claim_id}/runs",
    json={"reviewer_uid": "default", "citing_doc_id": doc_id},
)
run_id = str(run.get("run_id") or "").strip()
if not run_id:
    raise RuntimeError("start run did not return run_id")

_ = req(
    "POST",
    f"/workflow/claimspans/{claim_id}/assessment/finalize",
    json={
        "reviewer_uid": "default",
        "citing_doc_id": doc_id,
        "assessed_at": "2026-02-22T00:00:00Z",
        "rollup_label": "supports",
        "by_target": {},
        "judgment_snapshot": {
            "status": "final",
            "verdict": "support",
            "provenance": {"doc_id": doc_id},
        },
    },
)

terminal = {"complete", "partial", "error", "cancelled", "blocked"}
t0 = time.time()
last_state = None
while True:
    status = req("GET", f"/workflow/runs/{run_id}/status", timeout=30)
    last_state = str(((status.get("run") or {}).get("state")) or "")
    if last_state in terminal:
        break
    if time.time() - t0 > 90:
        raise RuntimeError(f"timeout waiting for terminal run state (last={last_state})")
    time.sleep(1)


def poll_stage(stage: str, *, timeout_s: int = 90) -> dict:
    t0 = time.time()
    while True:
        resp = requests.get(f"{API_URL}/pipeline/runs/{run_id}/stages/{stage}", timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if not isinstance(data, dict):
                raise RuntimeError("stage payload must be a dict")
            return data
        if resp.status_code not in {404}:
            raise RuntimeError(f"fetch stage {stage} failed: {resp.status_code} {resp.text[:200]}")
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"timeout waiting for stage {stage}")
        time.sleep(1)


extract = poll_stage("extract")
assert extract.get("stage") == "extract"
assert extract.get("schema_version") == 1

citespans = poll_stage("citespans")
assert citespans.get("stage") == "citespans"
assert citespans.get("schema_version") == 1

assessment = poll_stage("assessment")
assert assessment.get("stage") == "assessment"
assert assessment.get("data", {}).get("judgment_snapshot", {}).get("verdict") == "support"

print("OK: verify_10_02_happy_path")
PY
