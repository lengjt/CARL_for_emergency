#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${HERE}/../01_flow_planner/.venv/bin/python"

cmake -S "${HERE}/native" -B "${HERE}/build_py39" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="${PYTHON}"
cmake --build "${HERE}/build_py39" --parallel 4
"${PYTHON}" -c "import sys; sys.path.insert(0, '${HERE}/../02_extreme_avoidance'); from extreme_avoidance._ocd_native import UPSTREAM_COMMIT; print(UPSTREAM_COMMIT)"
