#!/bin/sh
set -eu

if [ "${1:-}" = "" ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Usage: scripts/dev/with_grobid_down.sh -- <command> [args...]

Stops the grobid container, runs a command, then starts grobid again.

Notes:
  - Intended for deterministic fallback verification.
  - Uses docker compose service name 'grobid'.
EOF
  exit 0
fi

if [ "${1:-}" != "--" ]; then
  echo "Expected '--' before command" 1>&2
  exit 2
fi
shift

if [ "${1:-}" = "" ]; then
  echo "Command is required" 1>&2
  exit 2
fi

docker compose stop grobid >/dev/null

cleanup() {
  docker compose start grobid >/dev/null || true
}
trap cleanup EXIT INT TERM

"$@"
