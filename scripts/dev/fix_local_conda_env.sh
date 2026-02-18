#!/bin/sh
set -eu

if [ "${1:-}" = "" ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Usage: CONFIRM=1 scripts/dev/fix_local_conda_env.sh

Repairs the active conda env by downgrading numpy (<2) and reinstalling
compiled deps so `pytest` can import pandas/scipy/sklearn.

This is intended to fix errors like:
  "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x"

Notes:
  - Modifies the currently-active conda environment in place.
  - If you prefer not to touch your conda env, use:
      scripts/dev/pytest_docker.sh
EOF
  exit 0
fi

if [ "${CONFIRM:-}" != "1" ]; then
  echo "Refusing to modify conda env without CONFIRM=1" 1>&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found" 1>&2
  exit 1
fi

python_bin=$(which python)
echo "python=${python_bin}" 1>&2

echo "Fixing numpy/pandas/scipy/sklearn compatibility in active conda env..." 1>&2

conda install -y -c conda-forge \
  "numpy<2" \
  "pandas" \
  "scipy" \
  "scikit-learn" \
  "pyarrow" >/dev/null

python -m pip install -q -U pytest

python - <<'PY'
import numpy
print('numpy', numpy.__version__)
import pandas
print('pandas', pandas.__version__)
import scipy
print('scipy', scipy.__version__)
import sklearn
print('sklearn', sklearn.__version__)
PY

echo "OK. You can run: pytest -q" 1>&2
