#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PYTHON="${EXTREME_AVOIDANCE_BASE_PYTHON:-/usr/bin/python3}"

if [[ ! -x "$BASE_PYTHON" ]]; then
  echo "Base Python not found at $BASE_PYTHON" >&2
  exit 2
fi

# Ubuntu 22.04 supplies the validated NumPy/Matplotlib versions as apt packages.
# The local venv isolates the interpreter entry point while reusing those audited
# binary packages; requirements-lock.txt verifies their exact versions offline.
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$ROOT/.venv"
fi

"$ROOT/.venv/bin/python" "$ROOT/scripts/check_environment.py"
bash "$ROOT/scripts/build.sh"
"$ROOT/.venv/bin/python" -m unittest discover -s "$ROOT/tests" -v
