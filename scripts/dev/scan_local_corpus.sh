#!/bin/sh
set -eu

if [ "${1:-}" = "" ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Usage: scripts/dev/scan_local_corpus.sh path/to/pdf_dir

Scans a local-only directory of PDFs by uploading each to the API and
triggering extraction, writing a per-file record under data/corpus_scans/.

Notes:
  - Designed for discovery: keep going even if some PDFs fail.
  - The PDFs directory should be local-only (gitignored).

Env:
  API_URL  (default: http://127.0.0.1:8000)
  START    (default: 1 = first file)
  LIMIT    (default: 0 = no limit)
  EXTRACT  (default: 1 = trigger extraction + fetch ingest record)
EOF
  exit 0
fi

pdf_dir="$1"
if [ ! -d "$pdf_dir" ]; then
  echo "Directory not found: $pdf_dir" 1>&2
  exit 2
fi

api_url="${API_URL:-http://127.0.0.1:8000}"
start="${START:-1}"
limit="${LIMIT:-0}"
extract="${EXTRACT:-1}"

ok=0
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
  if curl -4 -sf "${api_url}/docs" >/dev/null 2>&1; then
    ok=1
    break
  fi
  sleep 1
done

if [ "$ok" -ne 1 ]; then
  echo "API not reachable at ${api_url}" 1>&2
  exit 1
fi

scan_id="$(date -u +%Y%m%dT%H%M%SZ)__$$"
out_dir="data/corpus_scans/${scan_id}"
mkdir -p "$out_dir/records"

echo "scan_id=${scan_id}" 1>&2
echo "api_url=${api_url}" 1>&2
echo "pdf_dir=${pdf_dir}" 1>&2
echo "start=${start}" 1>&2
echo "limit=${limit}" 1>&2
echo "extract=${extract}" 1>&2
echo "out_dir=${out_dir}" 1>&2

# List PDFs (recursive) via Python for portability.
python - "$pdf_dir" "$start" "$limit" >"$out_dir/files.txt" <<'PY'
import os, sys

root = sys.argv[1]
start = int(sys.argv[2])
limit = int(sys.argv[3])

paths = []
for d, _, files in os.walk(root):
    for fn in files:
        if fn.lower().endswith('.pdf'):
            paths.append(os.path.join(d, fn))

paths.sort()
if start > 1:
    paths = paths[start - 1 :]
if limit > 0:
    paths = paths[:limit]

for p in paths:
    print(p)
PY

total=$(python -c "print(sum(1 for _ in open('$out_dir/files.txt','r',encoding='utf-8')))" )
echo "pdf_count=${total}" 1>&2

ok_upload=0
ok_extract=0
ok_get=0
fail_upload=0
fail_extract=0
fail_get=0

i=0
while IFS= read -r pdf_path || [ -n "$pdf_path" ]; do
  i=$((i + 1))
  base=$(basename "$pdf_path")
  safe="${i}__${base}"
  rec="$out_dir/records/${safe}.json"

  echo "[$i/$total] $pdf_path" 1>&2

  upload_json=""
  upload_err=""
  upload_rc=0
  upload_ms=0
  set +e
  t0=$(date +%s)
  upload_json=$(python - "$api_url" "$pdf_path" 2>"$out_dir/records/${safe}.upload.stderr" <<'PY'
import os
import sys

import requests

api_url = sys.argv[1].rstrip('/')
pdf_path = sys.argv[2]

with open(pdf_path, 'rb') as f:
    resp = requests.post(
        f"{api_url}/ingest?auto_process=false",
        files={"file": (os.path.basename(pdf_path), f, "application/pdf")},
        timeout=120,
    )
resp.raise_for_status()
sys.stdout.write(resp.text)
PY
)
  upload_rc=$?
  t1=$(date +%s)
  upload_ms=$(((t1 - t0) * 1000))
  set -e
  if [ -f "$out_dir/records/${safe}.upload.stderr" ]; then
    upload_err=$(cat "$out_dir/records/${safe}.upload.stderr" || true)
  fi

  doc_id=""
  if [ "$upload_rc" -eq 0 ] && [ "$upload_json" != "" ]; then
    doc_id=$(printf '%s' "$upload_json" | python -c "import json,sys; print(((json.load(sys.stdin).get('document') or {}).get('id')) or '')")
  fi

  extract_rc=0
  extract_ms=0
  extract_http=""
  extract_err=""
  if [ "$doc_id" != "" ] && [ "$extract" != "0" ]; then
    set +e
    t0=$(date +%s)
    extract_http=$(curl -4 -sS --max-time 240 -o "$out_dir/records/${safe}.extract.json" -w '%{http_code}' -X POST "${api_url}/ingest/${doc_id}/extract" 2>"$out_dir/records/${safe}.extract.stderr")
    extract_rc=$?
    t1=$(date +%s)
    extract_ms=$(((t1 - t0) * 1000))
    set -e
    if [ -f "$out_dir/records/${safe}.extract.stderr" ]; then
      extract_err=$(cat "$out_dir/records/${safe}.extract.stderr" || true)
    fi
  else
    extract_rc=2
  fi

  ingest_json=""
  get_rc=0
  get_ms=0
  if [ "$doc_id" != "" ] && [ "$extract" != "0" ]; then
    set +e
    t0=$(date +%s)
    ingest_json=$(curl -4 -sf --max-time 60 "${api_url}/ingest/${doc_id}")
    get_rc=$?
    t1=$(date +%s)
    get_ms=$(((t1 - t0) * 1000))
    set -e
  else
    get_rc=2
  fi

  python - "$pdf_path" "$doc_id" "$upload_rc" "$extract_rc" "$get_rc" "$upload_err" "$extract_err" "$extract_http" "$upload_ms" "$extract_ms" "$get_ms" "$rec" <<'PY'
import json, sys

pdf_path, doc_id, upload_rc, extract_rc, get_rc, upload_err, extract_err, extract_http, upload_ms, extract_ms, get_ms, rec_path = sys.argv[1:]
upload_rc = int(upload_rc)
extract_rc = int(extract_rc)
get_rc = int(get_rc)
upload_ms = int(upload_ms)
extract_ms = int(extract_ms)
get_ms = int(get_ms)

rec = {
    "pdf_path": pdf_path,
    "doc_id": doc_id,
    "upload_rc": upload_rc,
    "upload_err": upload_err,
    "upload_ms": upload_ms,
    "extract_rc": extract_rc,
    "extract_http": extract_http,
    "extract_err": extract_err,
    "extract_ms": extract_ms,
    "get_rc": get_rc,
    "get_ms": get_ms,
}

with open(rec_path, 'w', encoding='utf-8') as f:
    json.dump(rec, f, indent=2, sort_keys=True)
    f.write("\n")
PY

  if [ "$upload_rc" -eq 0 ] && [ "$doc_id" != "" ]; then
    ok_upload=$((ok_upload + 1))
  else
    fail_upload=$((fail_upload + 1))
  fi

  if [ "$extract_rc" -eq 0 ]; then
    ok_extract=$((ok_extract + 1))
  else
    fail_extract=$((fail_extract + 1))
  fi

  if [ "$get_rc" -eq 0 ]; then
    ok_get=$((ok_get + 1))
    printf '%s' "$ingest_json" >"$out_dir/records/${safe}.ingest.json"
  else
    fail_get=$((fail_get + 1))
  fi
done <"$out_dir/files.txt"

cat >"$out_dir/summary.txt" <<EOF
scan_id=${scan_id}
pdf_dir=${pdf_dir}
api_url=${api_url}
start=${start}
limit=${limit}
extract=${extract}
pdf_count=${total}

ok_upload=${ok_upload}
ok_extract=${ok_extract}
ok_get=${ok_get}

fail_upload=${fail_upload}
fail_extract=${fail_extract}
fail_get=${fail_get}
EOF

echo "Done: $out_dir" 1>&2
