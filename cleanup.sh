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

# 3) Sort and group imports consistently
isort --profile=black backend frontend application.py tests

# 4) Convert bare except to explicit Exception
find backend frontend -type f -name "*.py" -exec sed -i 's/except:/except Exception:/g' {} +

# 5) Remove f-prefix from string literals with no placeholders
find . -type f -name "*.py" -exec sed -i 's/f"\([^{}]*\)"/"\1"/g' {} +

echo "✔ Step 1 complete: unused imports removed and files formatted."