#!/usr/bin/env bash
set -euo pipefail

curl_json() {
  curl -fsS --retry 20 --retry-delay 1 --retry-connrefused --max-time 20 "$@"
}

docker compose up -d --build

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

RUN_JSON=$(curl_json -X POST "${API_URL}/pipeline/runs" \
  -H "Content-Type: application/json" \
  -d '{"work_id":"work:verify-10-01"}')

export RUN_JSON

RUN_ID=$(python - <<'PY'
import json
import os

data = json.loads(os.environ["RUN_JSON"])
print(data["run_id"])
PY
)

PUT_BODY=$(python - <<'PY'
import json

payload = {
  "status": "complete",
  "warnings": [],
  "data": {
    "structured_doc": {"hello": "world"},
    "citation_anchors": [],
  },
}
print(json.dumps(payload, ensure_ascii=True))
PY
)

curl_json -X PUT "${API_URL}/pipeline/runs/${RUN_ID}/stages/extract" \
  -H "Content-Type: application/json" \
  -d "${PUT_BODY}" >/dev/null

FETCHED=$(curl_json "${API_URL}/pipeline/runs/${RUN_ID}/stages/extract")

export FETCHED

python - <<'PY'
import json
import os

data = json.loads(os.environ["FETCHED"])
assert data["schema_version"] == 1
assert data["stage"] == "extract"
assert data["artifact_type"].startswith("contracts/extract@v1")
assert data["status"] == "complete"
assert data["data"]["structured_doc"]["hello"] == "world"
PY

echo "OK: verify_10_01_contracts"
