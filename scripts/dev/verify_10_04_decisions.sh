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

# Clean up any prior `docker compose run -d app-api` containers that may still
# be holding the 8000 port.
old_ids=$(docker ps -aq --filter "name=blablador-poc-app-api-run" || true)
if [ -n "${old_ids}" ]; then
  docker rm -f ${old_ids} >/dev/null 2>&1 || true
fi

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

import os
import time
from pathlib import Path

import requests

from scripts.dev import e2e_flow


API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
e2e_flow.API_URL = API_URL


def req(method: str, path: str, **kwargs):
    url = f"{API_URL}{path}"
    resp = requests.request(method, url, timeout=kwargs.pop("timeout", 60), **kwargs)
    if resp.status_code >= 400:
        return resp
    return resp


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

# 1) Decisions start empty.
resp = req("GET", f"/claims/{claim_id}/evidence/decisions", params={"reviewer_uid": "default"})
assert resp.status_code == 200, resp.text
data = resp.json()
assert data["version"] == 0

# 2) Append a pin event (dummy stable target).
pin_body = {
    "idempotency_key": "idem-1",
    "expected_version": 0,
    "action": "pin",
    "target": {"attachment_id": "att_dummy", "span_id": "span_dummy"},
    "payload": {"snippet": "Pinned dummy target", "ui_source": "verifier"},
}

resp1 = req(
    "POST",
    f"/claims/{claim_id}/evidence/decisions/events",
    params={"reviewer_uid": "default"},
    json=pin_body,
)
assert resp1.status_code == 200, resp1.text
pin1 = resp1.json()
assert pin1["version"] == 1

# 3) Idempotency replay returns identical payload.
resp2 = req(
    "POST",
    f"/claims/{claim_id}/evidence/decisions/events",
    params={"reviewer_uid": "default"},
    json=pin_body,
)
assert resp2.status_code == 200, resp2.text
assert resp2.json() == pin1

# 4) OCC conflict returns current_version.
resp3 = req(
    "POST",
    f"/claims/{claim_id}/evidence/decisions/events",
    params={"reviewer_uid": "default"},
    json={
        **pin_body,
        "idempotency_key": "idem-2",
        "expected_version": 0,
    },
)
assert resp3.status_code == 409, resp3.text
conflict = resp3.json()
assert conflict.get("current_version") == 1

# 5) Evidence list shows pinned placeholder even if current candidates empty.
resp4 = req(
    "GET",
    f"/claims/{claim_id}/evidence",
    params={"reviewer_uid": "default", "pinned_only": True, "limit": 5},
)
assert resp4.status_code == 200, resp4.text
ev = resp4.json()
assert ev.get("pinned"), "expected pinned placeholder list"
assert any(
    (item.get("metadata") or {}).get("not_in_current_run")
    for item in (ev.get("pinned") or [])
), "expected not_in_current_run placeholder"

# 6) Export -> wipe -> import roundtrip retains decisions.
export_resp = req("GET", "/project/export")
assert export_resp.status_code == 200
zip_blob = export_resp.content

wipe = req(
    "POST",
    "/dev/wipe",
    json={"confirm": "WIPE"},
    headers={"Content-Type": "application/json"},
)
assert wipe.status_code == 200

files = {"file": ("project.zip", zip_blob, "application/zip")}
imp = req("POST", "/project/import", params={"overwrite": True}, files=files)
assert imp.status_code == 200, imp.text

post = req("GET", f"/claims/{claim_id}/evidence/decisions", params={"reviewer_uid": "default"})
assert post.status_code == 200, post.text
post_data = post.json()
assert post_data.get("pinned_targets"), "expected pinned targets after import"

print("OK: verify_10_04_decisions")
PY
