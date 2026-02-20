#!/usr/bin/env bash
set -euo pipefail

api_base="${API_BASE:-http://127.0.0.1:8000}"
wait_seconds="${WAIT_SECONDS:-120}"
grobid_base="${GROBID_BASE:-http://127.0.0.1:8070}"

curl_json() {
  # curl: fail on non-2xx, retry transient startup issues
  curl -fsS --retry 20 --retry-delay 1 --retry-connrefused --max-time 20 "$@"
}

if [ "${VERIFY_RM_DATA:-0}" = "1" ]; then
  rm -rf data
fi

docker compose up -d

# Ensure GROBID is actually ready so the "normal" corpus smoke exercises the
# primary path (not fallback due to startup delay).
for i in $(seq 1 "$wait_seconds"); do
  gcode=$(curl -s --max-time 2 -o /dev/null -w "%{http_code}" "$grobid_base/api/isalive" || true)
  if [ "$gcode" = "200" ]; then
    break
  fi
  sleep 1
done

if [ "${gcode:-}" != "200" ]; then
  echo "ERROR: GROBID did not become ready at $grobid_base (last code=$gcode)" >&2
  exit 1
fi

for i in $(seq 1 "$wait_seconds"); do
  code=$(curl -s --max-time 2 -o /dev/null -w "%{http_code}" "$api_base/docs" || true)
  if [ "$code" = "200" ]; then
    break
  fi
  sleep 1
done

if [ "${code:-}" != "200" ]; then
  echo "ERROR: API did not become ready at $api_base (last code=$code)" >&2
  exit 1
fi

curl_json -X POST "$api_base/dev/wipe" \
  -H 'Content-Type: application/json' \
  -d '{"confirm":"WIPE"}' \
  >/dev/null

# Corpus regression smoke (normal + deterministic fallback with GROBID down)
bash scripts/dev/smoke_corpus.sh fixtures/corpus-09.2.txt

# Ensure the GROBID-down run cannot reuse prior successful extraction results.
curl_json -X POST "$api_base/dev/wipe" \
  -H 'Content-Type: application/json' \
  -d '{"confirm":"WIPE"}' \
  >/dev/null

bash scripts/dev/with_grobid_down.sh -- bash scripts/dev/smoke_corpus.sh fixtures/corpus-09.2.txt

# Minimal persistence smoke: selection + judgment persist across API restart.
claim_id="claim:verify-09.3"

curl_json -X PUT "$api_base/claims/$claim_id/evidence/selection" \
  -H 'Content-Type: application/json' \
  -d '{"verdict":"none","primary":null,"secondary":[],"note":null}' \
  >/dev/null

curl_json -X PUT "$api_base/claims/$claim_id/judgment" \
  -H 'Content-Type: application/json' \
  -d '{"reviewer_uid":"default","status":"final","verdict":"support","notes":{"rationale":"verify 09.3"},"claim_text":"verify"}' \
  >/dev/null

docker compose restart app-api >/dev/null

for i in $(seq 1 "$wait_seconds"); do
  code=$(curl -s --max-time 2 -o /dev/null -w "%{http_code}" "$api_base/docs" || true)
  if [ "$code" = "200" ]; then
    break
  fi
  sleep 1
done

if [ "${code:-}" != "200" ]; then
  echo "ERROR: API did not return after restart (last code=$code)" >&2
  exit 1
fi

sel_json=$(curl_json "$api_base/claims/$claim_id/evidence/selection")
python -c "import json,sys; d=json.loads(sys.argv[1]); assert d.get('verdict')=='none'; print('selection_ok')" "$sel_json"

jud_json=$(curl_json "$api_base/claims/$claim_id/judgment?reviewer_uid=default")
python -c "import json,sys; d=json.loads(sys.argv[1]); assert (d.get('status') or '')=='final'; print('judgment_ok')" "$jud_json"

echo "OK: verify_09_3_spine_cutover"
