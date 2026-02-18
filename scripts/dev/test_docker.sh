#!/bin/sh
set -eu

# Runs unit tests and then the E2E smoke flow.
# Intended for CI-like local validation on any host OS.

bash scripts/dev/pytest_docker.sh
bash scripts/dev/e2e_docker.sh
