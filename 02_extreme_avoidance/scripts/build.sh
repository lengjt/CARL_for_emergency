#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${EXTREME_AVOIDANCE_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Project Python not found at $PYTHON; run bash scripts/bootstrap.sh first." >&2
  exit 2
fi
cmake -S "$ROOT/native" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PYTHON"
cmake --build "$ROOT/build" --parallel "${OCD_BUILD_JOBS:-4}"
