#!/bin/sh
set -eu

if [ "${1:-}" = "" ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Usage: scripts/dev/tidy_local_corpus.sh path/to/grobid_scan_dir

Moves PDFs out of corpus/local into:
  - corpus/ok/    (paths listed in ok.txt)
  - corpus/focus/ (paths listed in candidates.txt)

This keeps corpus/local as an "inbox".

Notes:
  - Preserves relative paths under corpus/local.
  - Destination directories are gitignored.
EOF
  exit 0
fi

scan_dir="$1"
ok_txt="$scan_dir/ok.txt"
cand_txt="$scan_dir/candidates.txt"

if [ ! -f "$ok_txt" ] || [ ! -f "$cand_txt" ]; then
  echo "Expected ok.txt and candidates.txt in: $scan_dir" 1>&2
  exit 2
fi

mkdir -p corpus/ok corpus/focus

python - "$ok_txt" "$cand_txt" <<'PY'
import os
import shutil
import sys

ok_txt, cand_txt = sys.argv[1:]

def move_list(txt_path: str, dest_root: str):
    moved = 0
    skipped = 0
    missing = 0
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            src = line.strip()
            if not src:
                continue
            # only manage corpus/local
            norm = os.path.normpath(src)
            prefix = os.path.normpath('corpus/local') + os.sep
            if not norm.startswith(prefix):
                skipped += 1
                continue
            rel = norm[len(prefix) :]
            dst = os.path.join(dest_root, rel)
            if not os.path.exists(norm):
                missing += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                # already moved
                try:
                    os.remove(norm)
                except OSError:
                    pass
                skipped += 1
                continue
            shutil.move(norm, dst)
            moved += 1
    return moved, skipped, missing

ok_moved, ok_skipped, ok_missing = move_list(ok_txt, 'corpus/ok')
c_moved, c_skipped, c_missing = move_list(cand_txt, 'corpus/focus')

print(
    f"ok_moved={ok_moved} ok_skipped={ok_skipped} ok_missing={ok_missing} "
    f"candidates_moved={c_moved} candidates_skipped={c_skipped} candidates_missing={c_missing}"
)
PY
