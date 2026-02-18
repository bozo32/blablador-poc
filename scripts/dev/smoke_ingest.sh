#!/bin/sh
set -eu

if [ "${1:-}" = "" ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Usage: scripts/dev/smoke_ingest.sh path/to/file.pdf

Uploads a PDF to the API, triggers extraction, and fetches the final ingest record.

Env:
  API_URL  (default: http://127.0.0.1:8000)
EOF
  exit 0
fi

pdf_path="$1"
if [ ! -f "$pdf_path" ]; then
  echo "File not found: $pdf_path" 1>&2
  exit 2
fi

api_url="${API_URL:-http://127.0.0.1:8000}"

ok=0
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
  if curl -4 -sf "${api_url}/ingest" >/dev/null 2>&1; then
    ok=1
    break
  fi
  sleep 1
done

if [ "$ok" -ne 1 ]; then
  echo "API not reachable at ${api_url}" 1>&2
  exit 1
fi

set +e
upload_json=$(curl -4 -sf -F "file=@${pdf_path}" "${api_url}/ingest?auto_process=false")
rc=$?
set -e

if [ "$rc" -ne 0 ] || [ "$upload_json" = "" ]; then
  echo "Upload failed (rc=${rc})" 1>&2
  exit 1
fi

doc_id=$(printf '%s' "$upload_json" | python -c "import json,sys; print((json.load(sys.stdin).get('document') or {}).get('id') or '')")
if [ "$doc_id" = "" ]; then
  echo "Upload succeeded but response missing document.id" 1>&2
  printf '%s\n' "$upload_json" 1>&2
  exit 1
fi

curl -4 -sf -X POST "${api_url}/ingest/${doc_id}/extract" >/dev/null

# Poll spine attempt state (spine mode runs extract async).
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
  state=$(curl -4 -sf "${api_url}/ingest/${doc_id}/spine" | python -c "import json,sys; d=json.load(sys.stdin); a=d.get('attempt') or {}; print(a.get('state') or '')")
  if [ "$state" = "succeeded" ] || [ "$state" = "partial" ] || [ "$state" = "failed" ] || [ "$state" = "cancelled" ]; then
    break
  fi
  sleep 1
done

echo "doc_id=${doc_id}"
curl -4 -sf "${api_url}/ingest/${doc_id}" | python -m json.tool
