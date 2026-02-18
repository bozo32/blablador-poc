#!/bin/sh
set -eu

if [ "${1:-}" = "" ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Usage: scripts/dev/build_focus_corpus.sh path/to/candidates.txt

Copies the PDFs listed in candidates.txt into corpus/focus/ for faster
iteration, without deleting the originals.

Notes:
  - candidates.txt should contain one PDF path per line.
  - The destination directory is gitignored.
EOF
  exit 0
fi

manifest="$1"
if [ ! -f "$manifest" ]; then
  echo "Manifest not found: $manifest" 1>&2
  exit 2
fi

dest_dir="corpus/focus"
mkdir -p "$dest_dir"

python - "$manifest" "$dest_dir" <<'PY'
import os
import shutil
import sys

manifest, dest_dir = sys.argv[1:]

copied = 0
missing = 0

with open(manifest, 'r', encoding='utf-8') as f:
    for line in f:
        p = line.strip()
        if not p:
            continue
        if not os.path.isfile(p):
            missing += 1
            continue
        dst = os.path.join(dest_dir, os.path.basename(p))
        shutil.copy2(p, dst)
        copied += 1

print(f"copied={copied} missing={missing}")
PY
