#!/bin/sh
set -eu

detach=1

for arg in "$@"; do
  case "$arg" in
    --foreground)
      detach=0
      ;;
    -d|--detach)
      detach=1
      ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/dev/up.sh [--detach|-d] [--foreground]

Boot the local dev stack with Docker Compose.

Default: detached.
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" 1>&2
      echo "Run: scripts/dev/up.sh --help" 1>&2
      exit 2
      ;;
  esac
done

if [ "$detach" -eq 1 ]; then
  docker compose up --build -d
else
  docker compose up --build
fi

cat <<'EOF'

Services:
- UI:           http://localhost:8501
- API:          http://localhost:8000
- API docs:     http://localhost:8000/docs
- GROBID:       http://localhost:8070
- MinIO API:    http://localhost:9000
- MinIO console http://localhost:9001

Tip: If curl hangs/resets on localhost, try IPv4: http://127.0.0.1:8000
EOF
