#!/bin/sh
set -eu

if [ "${1:-}" = "" ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Usage: scripts/dev/smoke_corpus.sh path/to/corpus-manifest.txt

Runs scripts/dev/smoke_ingest.sh for each PDF listed in a manifest.

Manifest format:
  <path> <class-tags...>

Notes:
  - Blank lines and lines starting with '#' are ignored.
  - Exits non-zero if any fixture is missing or any ingest fails.

Env:
  API_URL  (default: http://127.0.0.1:8000)
EOF
  exit 0
fi

manifest="$1"
if [ ! -f "$manifest" ]; then
  echo "Manifest not found: $manifest" 1>&2
  exit 2
fi

missing=0
failed=0

while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in
    ""|"#"*)
      continue
      ;;
  esac

  # Split on whitespace: first token is path, rest is tags (for printing only).
  set -- $line
  pdf_path="${1:-}"
  shift || true
  tags="$*"

  if [ "$pdf_path" = "" ]; then
    continue
  fi

  if [ ! -f "$pdf_path" ]; then
    echo "Missing fixture: $pdf_path${tags:+ ($tags)}" 1>&2
    missing=$((missing + 1))
    continue
  fi

  echo "=== ingest: $pdf_path${tags:+ ($tags)} ===" 1>&2
  if ! bash scripts/dev/smoke_ingest.sh "$pdf_path"; then
    failed=$((failed + 1))
  fi
done < "$manifest"

if [ "$missing" -ne 0 ] || [ "$failed" -ne 0 ]; then
  echo "Corpus smoke failures: missing=${missing} failed=${failed}" 1>&2
  exit 1
fi

echo "Corpus smoke OK" 1>&2
