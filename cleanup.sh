#!/usr/bin/env bash
#
# cleanup_unused_imports.sh
# Automatically strip out unused imports/variables in all .py files.

set -euxo pipefail

# 1) Remove unused imports & variables
autoflake \
  --in-place \
  --remove-unused-variables \
  --remove-all-unused-imports \
  --recursive \
  backend frontend application.py tests

# 2) Re-format with Black to keep things tidy
black backend frontend application.py tests

echo "✔ Step 1 complete: unused imports removed and files formatted."