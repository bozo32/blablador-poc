#!/usr/bin/env bash
set -euo pipefail

curl_json() {
  curl -fsS --retry 20 --retry-delay 1 --retry-connrefused --max-time 60 "$@"
}

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

pass() {
  echo "PASS: $*"
}

PHASE="pre"
if [ "${1:-}" = "--phase" ]; then
  PHASE="${2:-}"
  shift 2 || true
fi

STATE_FILE="${STATE_FILE:-/tmp/verify_10_04_5_01_intake_state.json}"

API_URL="${API_URL:-http://127.0.0.1:8000}"
API_URL="${API_URL%/}"

CITING_PDF_PATH="${CITING_PDF_PATH:-corpus/workflow/rudko-chatgpt-is-incredible-at-being-average.pdf}"
SOURCE_PDF_PATH="${SOURCE_PDF_PATH:-corpus/workflow/hicks-chatgpt-is-bullshit.pdf}"

P1="${P1:-proj-a}"
P2="${P2:-proj-b}"

wait_api() {
  local url="$1"
  local code=""
  local tmp="/tmp/verify_10_04_5_01_openapi.json"

  for _ in $(seq 1 120); do
    code=$(curl -s --max-time 2 -o "${tmp}" -w "%{http_code}" "${url}/openapi.json" || true)
    if [ "${code}" = "200" ]; then
      if python - "${tmp}" >/dev/null 2>&1 <<'PYCHECK'; then
import json,sys
raw=open(sys.argv[1],'r',encoding='utf-8',errors='replace').read() or ''
p=json.loads(raw)
assert isinstance(p, dict)
assert p.get('openapi')
paths=p.get('paths') or {}
assert isinstance(paths, dict)
assert '/ingest' in paths
print('ok')
PYCHECK
        return 0
      fi
    fi
    sleep 1
  done

  echo "ERROR: API not ready at ${url} (expected FastAPI openapi.json with /ingest; last_code=${code})" >&2
  echo "Hint: ensure the API is running on ${url} (not Streamlit/UI)." >&2
  return 1
}

ingest_upload() {
  local project_id="$1"
  local pdf_path="$2"
  local out
  out=$(curl_json \
    -H "X-Project-Id: ${project_id}" \
    -F "file=@${pdf_path};type=application/pdf" \
    "${API_URL}/ingest?auto_process=true")

  echo "${out}" | python - <<'PY'
import json,sys
raw=sys.stdin.read()
try:
    p=json.loads(raw)
except Exception as exc:
    sys.stderr.write('ERROR: /ingest upload did not return JSON\n')
    sys.stderr.write(f'error={exc}\n')
    sys.stderr.write('body_start=')
    sys.stderr.write((raw[:400].replace('\n',' ') if raw else '<empty>'))
    sys.stderr.write('\n')
    raise SystemExit(2)
d=(p.get('document') or {}) if isinstance(p,dict) else {}
print(d.get('id') or '')

PY
}

ingest_get() {
  local project_id="$1"
  local doc_id="$2"
  curl_json -H "X-Project-Id: ${project_id}" "${API_URL}/ingest/${doc_id}"
}

attachments_upload() {
  local project_id="$1"
  local pdf_path="$2"
  local out
  out=$(curl_json \
    -H "X-Project-Id: ${project_id}" \
    -F "file=@${pdf_path};type=application/pdf" \
    "${API_URL}/attachments/upload")

  echo "${out}" | python - <<'PY'
import json,sys
raw=sys.stdin.read()
try:
    p=json.loads(raw)
except Exception as exc:
    sys.stderr.write('ERROR: /attachments/upload did not return JSON\n')
    sys.stderr.write(f'error={exc}\n')
    sys.stderr.write('body_start=')
    sys.stderr.write((raw[:400].replace('\n',' ') if raw else '<empty>'))
    sys.stderr.write('\n')
    raise SystemExit(2)
a=(p.get('attachment') or {}) if isinstance(p,dict) else {}
print(a.get('id') or '')

PY
}

attachments_get() {
  local project_id="$1"
  local att_id="$2"
  curl_json -H "X-Project-Id: ${project_id}" "${API_URL}/attachments/${att_id}"
}

poll_attachment_terminal() {
  local project_id="$1"
  local att_id="$2"
  local timeout_s="${3:-180}"

  python - <<'PY' "${API_URL}" "${project_id}" "${att_id}" "${timeout_s}"
import sys,time
import requests
api_url, pid, att_id, timeout_s = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
start=time.time()
while time.time()-start < timeout_s:
    r=requests.get(f"{api_url}/attachments/{att_id}", headers={"X-Project-Id": pid}, timeout=15)
    if r.status_code >= 400:
        time.sleep(1)
        continue
    a=(r.json().get('attachment') or {})
    st=str(a.get('status') or '').strip().lower()
    if st in {'matched','error'}:
        print(st)
        sys.exit(0 if st=='matched' else 2)
    time.sleep(1)
print('timeout')
sys.exit(3)
PY
}

if [ "${PHASE}" = "post" ]; then
  test -f "${STATE_FILE}" || fail "state file missing: ${STATE_FILE} (run without --phase post first)"

  API_URL_FROM_STATE=$(python - <<'PY' "${STATE_FILE}"
import json,sys
p=json.loads(open(sys.argv[1],'r',encoding='utf-8').read())
print(p.get('api_url') or '')
PY
)
  if [ -n "${API_URL_FROM_STATE}" ]; then
    API_URL="${API_URL_FROM_STATE%/}"
  fi

  wait_api "${API_URL}" || fail "api not ready"

  DOC1=$(python - <<'PY' "${STATE_FILE}"
import json,sys
p=json.loads(open(sys.argv[1],'r',encoding='utf-8').read())
print(p['doc1'])
PY
)
  ATT1=$(python - <<'PY' "${STATE_FILE}"
import json,sys
p=json.loads(open(sys.argv[1],'r',encoding='utf-8').read())
print(p['att1'])
PY
)

  curl_json -H "X-Project-Id: ${P1}" "${API_URL}/ingest" >/dev/null || fail "post: ingest list failed"
  curl_json -H "X-Project-Id: ${P1}" "${API_URL}/ingest/${DOC1}" >/dev/null || fail "post: ingest read failed"
  curl_json -H "X-Project-Id: ${P1}" "${API_URL}/attachments/${ATT1}" >/dev/null || fail "post: attachment read failed"

  code=$(curl -s -o /dev/null -w "%{http_code}" -H "X-Project-Id: ${P2}" "${API_URL}/ingest/${DOC1}" || true)
  [ "${code}" = "404" ] || fail "post: expected cross-project ingest read 404, got ${code}"

  pass "restart persistence + isolation (post)"
  exit 0
fi

# --- pre phase ---

wait_api "${API_URL}" || fail "api not ready"

[ -f "${CITING_PDF_PATH}" ] || fail "missing CITING_PDF_PATH=${CITING_PDF_PATH}"
[ -f "${SOURCE_PDF_PATH}" ] || fail "missing SOURCE_PDF_PATH=${SOURCE_PDF_PATH}"

# Start clean.
if curl -fsS "${API_URL}/dev/wipe" >/dev/null 2>&1; then
  curl_json -X POST "${API_URL}/dev/wipe" -H "Content-Type: application/json" -d '{"confirm":"WIPE"}' >/dev/null || true
fi

DOC1=$(ingest_upload "${P1}" "${CITING_PDF_PATH}")
[ -n "${DOC1}" ] || fail "citing upload did not return doc id"
pass "citing upload returned doc_id=${DOC1}"

ingest_get "${P1}" "${DOC1}" | python - <<'PY'
import json,sys
p=json.load(sys.stdin)
text=json.dumps(p, sort_keys=True)
assert 'local_path' not in text
print('ok')
PY
pass "citing ingest contract does not include local_path"

curl_json -H "X-Project-Id: ${P1}" "${API_URL}/ingest" | python - <<'PY' "${DOC1}"
import json,sys
p=json.load(sys.stdin)
docs=p.get('documents') or []
ids={d.get('id') for d in docs if isinstance(d,dict)}
assert sys.argv[1] in ids
print('ok')
PY
pass "/ingest list contains doc in project"

code=$(curl -s -o /dev/null -w "%{http_code}" -H "X-Project-Id: ${P2}" "${API_URL}/ingest/${DOC1}" || true)
[ "${code}" = "404" ] || fail "expected cross-project ingest read 404, got ${code}"
pass "cross-project ingest read denied"

ATT1=$(attachments_upload "${P1}" "${SOURCE_PDF_PATH}")
[ -n "${ATT1}" ] || fail "source upload did not return attachment id"
pass "source upload returned attachment_id=${ATT1}"

set +e
st=$(poll_attachment_terminal "${P1}" "${ATT1}" 240)
rc=$?
set -e
if [ "${rc}" -eq 0 ]; then
  pass "source attachment reached matched"
else
  echo "WARN: source attachment terminal status=${st} (rc=${rc})" >&2
fi

ATT2=$(attachments_upload "${P1}" "${SOURCE_PDF_PATH}")
[ "${ATT1}" = "${ATT2}" ] || fail "expected dedupe id equality in project, got ${ATT1} vs ${ATT2}"
pass "dedupe within project (same bytes -> same attachment id)"

ATT_B=$(attachments_upload "${P2}" "${SOURCE_PDF_PATH}")
[ "${ATT_B}" != "${ATT1}" ] || fail "expected cross-project uploads to have different ids"
pass "cross-project uploads do not dedupe across projects"

CL1=$(curl_json -H "X-Project-Id: ${P1}" -X POST "${API_URL}/attachments/${ATT1}/clone" \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"doc-a"}' \
  | python - <<'PY'
import json,sys
p=json.load(sys.stdin)
print((p.get('attachment') or {}).get('id') or '')
PY
)
CL2=$(curl_json -H "X-Project-Id: ${P1}" -X POST "${API_URL}/attachments/${ATT1}/clone" \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"doc-b"}' \
  | python - <<'PY'
import json,sys
p=json.load(sys.stdin)
print((p.get('attachment') or {}).get('id') or '')
PY
)
[ -n "${CL1}" ] && [ -n "${CL2}" ] || fail "clone ids missing"
[ "${CL1}" != "${CL2}" ] || fail "expected distinct clone ids"

attachments_get "${P1}" "${ATT1}" | python - <<'PY'
import json,sys
p=json.load(sys.stdin)
a=p.get('attachment') or {}
ok=(not a.get('doc_id') and not a.get('claim_id') and not a.get('target_id') and a.get('citation_index') is None)
assert ok, a
print('ok')
PY
pass "clone is non-mutating for global source"

code=$(curl -s -o /dev/null -w "%{http_code}" -H "X-Project-Id: ${P2}" "${API_URL}/attachments/${ATT1}" || true)
[ "${code}" = "404" ] || fail "expected cross-project attachment read 404, got ${code}"
pass "cross-project attachment read denied"

code=$(curl -s -o /dev/null -w "%{http_code}" "${API_URL}/attachments?archived=false" || true)
[ "${code}" != "200" ] || fail "expected missing header to fail for attachments list"
pass "missing X-Project-Id header fails for attachments list"

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 && command -v docker compose >/dev/null 2>&1; then
  echo "INFO: attempting docker compose restart-style check" >&2

  docker compose up -d postgres minio minio-init grobid fallback-worker >/dev/null 2>&1 || true

  docker compose stop app-api >/dev/null 2>&1 || true
  docker compose rm -f app-api >/dev/null 2>&1 || true

  old_ids=$(docker ps -aq --filter "name=blablador-poc-app-api-run" || true)
  if [ -n "${old_ids}" ]; then
    docker rm -f ${old_ids} >/dev/null 2>&1 || true
  fi

  docker compose run -d --service-ports \
    -v "${PWD}/backend:/app/backend" \
    -v "${PWD}/frontend:/app/frontend" \
    -v "${PWD}/application.py:/app/application.py" \
    app-api >/dev/null

  wait_api "${API_URL}" || fail "api not ready after restart"

  curl_json -H "X-Project-Id: ${P1}" "${API_URL}/ingest/${DOC1}" >/dev/null || fail "post-restart ingest read failed"
  curl_json -H "X-Project-Id: ${P1}" "${API_URL}/attachments/${ATT1}" >/dev/null || fail "post-restart attachment read failed"

  pass "restart persistence + isolation (docker compose)"
else
  python - <<'PY' "${STATE_FILE}" "${API_URL}" "${DOC1}" "${ATT1}"
import json,sys
state_path, api_url, doc1, att1 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(state_path, 'w', encoding='utf-8') as f:
    json.dump({'api_url': api_url, 'doc1': doc1, 'att1': att1}, f, indent=2, sort_keys=True)
print(state_path)
PY

  echo "" >&2
  echo "INFO: docker compose not available; manual restart required" >&2
  echo "1) Restart your API process/container" >&2
  echo "2) Re-run: bash scripts/dev/verify_10_04_5_01_intake.sh --phase post" >&2
  echo "State saved to: ${STATE_FILE}" >&2
  exit 2
fi

pass "verify_10_04_5_01_intake (pre)"
