#!/bin/sh
set -eu

if [ "${1:-}" = "" ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Usage: scripts/dev/scan_grobid_local_corpus.sh path/to/pdf_dir

Scans a local-only directory of PDFs by calling GROBID endpoints directly
(no ingest persistence) and writing per-file JSON records under
data/grobid_scans/<timestamp>/.

Env:
  GROBID_URL   (default: http://127.0.0.1:8070)
  ENDPOINTS    (default: header,refs,fulltext)
  ORDER        (default: size_desc)  One of: size_desc, size_asc, alpha
  START        (default: 1)
  LIMIT        (default: 0 = no limit)
  TIMEOUT_S    (default: 90)
EOF
  exit 0
fi

pdf_dir="$1"
if [ ! -d "$pdf_dir" ]; then
  echo "Directory not found: $pdf_dir" 1>&2
  exit 2
fi

grobid_url="${GROBID_URL:-http://127.0.0.1:8070}"
endpoints_sel="${ENDPOINTS:-header,refs,fulltext}"
order="${ORDER:-size_desc}"
start="${START:-1}"
limit="${LIMIT:-0}"
timeout_s="${TIMEOUT_S:-90}"

scan_id="$(date -u +%Y%m%dT%H%M%SZ)__$$"
out_dir="data/grobid_scans/${scan_id}"
mkdir -p "$out_dir/records"

echo "scan_id=${scan_id}" 1>&2
echo "grobid_url=${grobid_url}" 1>&2
echo "endpoints=${endpoints_sel}" 1>&2
echo "pdf_dir=${pdf_dir}" 1>&2
echo "order=${order}" 1>&2
echo "start=${start}" 1>&2
echo "limit=${limit}" 1>&2
echo "timeout_s=${timeout_s}" 1>&2
echo "out_dir=${out_dir}" 1>&2

python - "$pdf_dir" "$grobid_url" "$endpoints_sel" "$order" "$start" "$limit" "$timeout_s" "$out_dir" <<'PY'
import json
import os
import sys
import time

import requests

pdf_dir, grobid_url, endpoints_sel, order, start_s, limit_s, timeout_s, out_dir = sys.argv[1:]
start = int(start_s)
limit = int(limit_s)
timeout = float(timeout_s)

want = {s.strip().lower() for s in (endpoints_sel or '').split(',') if s.strip()}
if not want:
    want = {'header', 'refs', 'fulltext'}

paths = []
for d, _, files in os.walk(pdf_dir):
    for fn in files:
        if fn.lower().endswith('.pdf'):
            p = os.path.join(d, fn)
            try:
                size = os.path.getsize(p)
            except OSError:
                size = None
            paths.append((p, size))

if order == 'alpha':
    paths.sort(key=lambda t: t[0].lower())
elif order == 'size_asc':
    paths.sort(key=lambda t: (t[1] is None, t[1] or 0, t[0].lower()))
else:
    paths.sort(key=lambda t: (t[1] is None, -(t[1] or 0), t[0].lower()))

if start > 1:
    paths = paths[start - 1 :]
if limit > 0:
    paths = paths[:limit]

all_endpoints = {
    'header': '/api/processHeaderDocument',
    'refs': '/api/processReferences',
    'fulltext': '/api/processFulltextDocument',
}
endpoints = [(k, all_endpoints[k]) for k in ('header', 'refs', 'fulltext') if k in want]

sess = requests.Session()
total = len(paths)

def call(endpoint_url: str, pdf_path: str):
    t0 = time.time()
    try:
        with open(pdf_path, 'rb') as f:
            resp = sess.post(
                endpoint_url,
                files={'input': (os.path.basename(pdf_path), f, 'application/pdf')},
                timeout=timeout,
            )
        dt_ms = int((time.time() - t0) * 1000)
        return {
            'ok': resp.status_code == 200,
            'status_code': resp.status_code,
            'elapsed_ms': dt_ms,
            'text_head': (resp.text or '')[:1000] if resp.status_code != 200 else '',
            'error': '',
        }
    except Exception as e:
        dt_ms = int((time.time() - t0) * 1000)
        return {
            'ok': False,
            'status_code': None,
            'elapsed_ms': dt_ms,
            'text_head': '',
            'error': f'{type(e).__name__}: {e}',
        }

for idx, (pdf_path, size) in enumerate(paths, 1):
    base = os.path.basename(pdf_path)
    safe = f'{idx}__{base}'
    rec_path = os.path.join(out_dir, 'records', safe + '.json')

    rec = {
        'pdf_path': pdf_path,
        'size_bytes': size,
        'grobid_url': grobid_url.rstrip('/'),
        'results': {},
    }

    for label, ep in endpoints:
        rec['results'][label] = call(grobid_url.rstrip('/') + ep, pdf_path)

    with open(rec_path, 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=2, sort_keys=True)
        f.write('\n')

    # progress
    if idx == 1 or idx == total or idx % 10 == 0:
        print(f'[{idx}/{total}] {base}', file=sys.stderr)

summary = {
    'scan_id': os.path.basename(out_dir),
    'pdf_dir': pdf_dir,
    'grobid_url': grobid_url,
    'order': order,
    'start': start,
    'limit': limit,
    'timeout_s': timeout,
    'count': total,
}

with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, sort_keys=True)
    f.write('\n')

# Derive candidate lists to help pruning/focus.
slow_ms = 20000
ok_paths = []
candidate_paths = []
stats = {
    'total': total,
    'ok_all_endpoints': 0,
    'candidate_any_failure': 0,
    'candidate_slow_only': 0,
    'candidate_error_only': 0,
}

for idx, (pdf_path, size) in enumerate(paths, 1):
    base = os.path.basename(pdf_path)
    rec_path = os.path.join(out_dir, 'records', f'{idx}__{base}.json')
    try:
        rec = json.load(open(rec_path, 'r', encoding='utf-8'))
    except Exception:
        candidate_paths.append(pdf_path)
        stats['candidate_any_failure'] += 1
        continue

    results = rec.get('results') or {}
    any_fail = any(not (v or {}).get('ok') for v in results.values())
    any_slow = any(((v or {}).get('elapsed_ms') or 0) >= slow_ms for v in results.values())
    if not any_fail and not any_slow:
        ok_paths.append(pdf_path)
        stats['ok_all_endpoints'] += 1
    else:
        candidate_paths.append(pdf_path)
        stats['candidate_any_failure'] += 1
        if any_fail and not any_slow:
            stats['candidate_error_only'] += 1
        if any_slow and not any_fail:
            stats['candidate_slow_only'] += 1

with open(os.path.join(out_dir, 'candidates.txt'), 'w', encoding='utf-8') as f:
    for p in candidate_paths:
        f.write(p + '\n')

with open(os.path.join(out_dir, 'ok.txt'), 'w', encoding='utf-8') as f:
    for p in ok_paths:
        f.write(p + '\n')

with open(os.path.join(out_dir, 'triage.json'), 'w', encoding='utf-8') as f:
    json.dump({'slow_ms': slow_ms, 'stats': stats}, f, indent=2, sort_keys=True)
    f.write('\n')

print(f'Done: {out_dir}', file=sys.stderr)
PY
