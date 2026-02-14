#!/bin/sh
set -eu

cat <<'EOF'
Bringing the dev stack down.
WARNING: This removes volumes (-v) and deletes Postgres/MinIO persisted data.
EOF

docker compose down -v
