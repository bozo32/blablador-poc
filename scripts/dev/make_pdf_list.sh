#!/bin/sh
set -eu

if [ "${1:-}" = "" ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Usage: scripts/dev/make_pdf_list.sh path/to/pdf_dir

Writes a stable, alphabetically-sorted list of PDFs under the directory to
stdout (one path per line).

Notes:
  - Intended for long unattended scans.
EOF
  exit 0
fi

pdf_dir="$1"
if [ ! -d "$pdf_dir" ]; then
  echo "Directory not found: $pdf_dir" 1>&2
  exit 2
fi

python - "$pdf_dir" <<'PY'
import os
import sys

root = sys.argv[1]
paths = []
for d, _, files in os.walk(root):
    for fn in files:
        if fn.lower().endswith('.pdf'):
            paths.append(os.path.join(d, fn))
paths.sort(key=lambda p: p.lower())
for p in paths:
    print(p)
PY
