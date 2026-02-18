#!/bin/sh
set -eu

if [ "${1:-}" = "" ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Usage: scripts/dev/scan_grobid_filelist.sh path/to/files.txt

Scans a fixed list of PDF paths against GROBID endpoints directly.

Env:
  GROBID_URL   (default: http://127.0.0.1:8070)
  ENDPOINTS    (default: header,refs)
  START        (default: 1)
  LIMIT        (default: 0 = no limit)
  TIMEOUT_S    (default: 60)
  RETRIES      (default: 2)
  SLEEP_S      (default: 2)

Outputs:
  data/grobid_scans/<scan_id>/records/*.json
  data/grobid_scans/<scan_id>/ok.txt
  data/grobid_scans/<scan_id>/candidates.txt
EOF
  exit 0
fi

files_txt="$1"
if [ ! -f "$files_txt" ]; then
  echo "File list not found: $files_txt" 1>&2
  exit 2
fi

grobid_url="${GROBID_URL:-http://127.0.0.1:8070}"
endpoints_sel="${ENDPOINTS:-header,refs}"
start="${START:-1}"
limit="${LIMIT:-0}"
timeout_s="${TIMEOUT_S:-60}"
retries="${RETRIES:-2}"
sleep_s="${SLEEP_S:-2}"

scan_id="$(date -u +%Y%m%dT%H%M%SZ)__$$"
out_dir="data/grobid_scans/${scan_id}"
mkdir -p "$out_dir/records"

echo "scan_id=${scan_id}" 1>&2
echo "grobid_url=${grobid_url}" 1>&2
echo "endpoints=${endpoints_sel}" 1>&2
echo "files_txt=${files_txt}" 1>&2
echo "start=${start}" 1>&2
echo "limit=${limit}" 1>&2
echo "timeout_s=${timeout_s}" 1>&2
echo "retries=${retries}" 1>&2
echo "sleep_s=${sleep_s}" 1>&2
echo "out_dir=${out_dir}" 1>&2

python - "$files_txt" "$grobid_url" "$endpoints_sel" "$start" "$limit" "$timeout_s" "$retries" "$sleep_s" "$out_dir" <<'PY'
import json
import os
import sys
import time

import requests

files_txt, grobid_url, endpoints_sel, start_s, limit_s, timeout_s, retries_s, sleep_s, out_dir = sys.argv[1:]
start = int(start_s)
limit = int(limit_s)
timeout = float(timeout_s)
retries = int(retries_s)
sleep_s = float(sleep_s)

want = {s.strip().lower() for s in (endpoints_sel or '').split(',') if s.strip()}
if not want:
    want = {'header', 'refs'}

all_endpoints = {
    'header': '/api/processHeaderDocument',
    'refs': '/api/processReferences',
    'fulltext': '/api/processFulltextDocument',
}
endpoints = [(k, all_endpoints[k]) for k in ('header', 'refs', 'fulltext') if k in want]

with open(files_txt, 'r', encoding='utf-8') as f:
    paths = [ln.strip() for ln in f if ln.strip()]

if start > 1:
    paths = paths[start - 1 :]
if limit > 0:
    paths = paths[:limit]

sess = requests.Session()

def is_transient_error(msg: str) -> bool:
    m = (msg or '').lower()
    return any(
        s in m
        for s in (
            'connection reset',
            'remote end closed',
            'connection refused',
            'failed to establish a new connection',
            'remote disconnected',
        )
    )

def call(endpoint_url: str, pdf_path: str):
    attempt = 0
    last = None
    while True:
        attempt += 1
        t0 = time.time()
        try:
            with open(pdf_path, 'rb') as f:
                resp = sess.post(
                    endpoint_url,
                    files={'input': (os.path.basename(pdf_path), f, 'application/pdf')},
                    timeout=timeout,
                )
            dt_ms = int((time.time() - t0) * 1000)
            last = {
                'ok': resp.status_code == 200,
                'status_code': resp.status_code,
                'elapsed_ms': dt_ms,
                'attempts': attempt,
                'text_head': (resp.text or '')[:1000] if resp.status_code != 200 else '',
                'error': '',
            }
            return last
        except Exception as e:
            dt_ms = int((time.time() - t0) * 1000)
            msg = f'{type(e).__name__}: {e}'
            last = {
                'ok': False,
                'status_code': None,
                'elapsed_ms': dt_ms,
                'attempts': attempt,
                'text_head': '',
                'error': msg,
            }
            if attempt <= retries and is_transient_error(msg):
                time.sleep(sleep_s)
                continue
            return last

slow_ms = 20000
ok_paths = []
candidate_paths = []

total = len(paths)
for idx, pdf_path in enumerate(paths, 1):
    base = os.path.basename(pdf_path)
    safe = f'{idx}__{base}'
    rec_path = os.path.join(out_dir, 'records', safe + '.json')

    try:
        size = os.path.getsize(pdf_path)
    except OSError:
        size = None

    rec = {
        'pdf_path': pdf_path,
        'size_bytes': size,
        'grobid_url': grobid_url.rstrip('/'),
        'results': {},
    }

    any_fail = False
    any_slow = False
    for label, ep in endpoints:
        res = call(grobid_url.rstrip('/') + ep, pdf_path)
        rec['results'][label] = res
        if not res.get('ok'):
            any_fail = True
        if (res.get('elapsed_ms') or 0) >= slow_ms:
            any_slow = True

    with open(rec_path, 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=2, sort_keys=True)
        f.write('\n')

    if not any_fail and not any_slow:
        ok_paths.append(pdf_path)
    else:
        candidate_paths.append(pdf_path)

    if idx == 1 or idx == total or idx % 10 == 0:
        print(f'[{idx}/{total}] {base}', file=sys.stderr)

with open(os.path.join(out_dir, 'ok.txt'), 'w', encoding='utf-8') as f:
    for p in ok_paths:
        f.write(p + '\n')

with open(os.path.join(out_dir, 'candidates.txt'), 'w', encoding='utf-8') as f:
    for p in candidate_paths:
        f.write(p + '\n')

with open(os.path.join(out_dir, 'triage.json'), 'w', encoding='utf-8') as f:
    json.dump(
        {
            'slow_ms': slow_ms,
            'stats': {
                'total': total,
                'ok': len(ok_paths),
                'candidates': len(candidate_paths),
            },
        },
        f,
        indent=2,
        sort_keys=True,
    )
    f.write('\n')

print(f'Done: {out_dir}', file=sys.stderr)
PY

echo "$out_dir"
