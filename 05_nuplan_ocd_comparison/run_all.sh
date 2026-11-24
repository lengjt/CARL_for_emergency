#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${HERE}/../01_flow_planner/.venv/bin/python"
export MPLCONFIGDIR="${HERE}/.cache/matplotlib"

"${HERE}/build_ocd_for_flow.sh"
"${PYTHON}" "${HERE}/check_ocd_interface.py"
"${PYTHON}" "${HERE}/run_comparison.py" --mode both "$@"
"${PYTHON}" "${HERE}/analyze_comparison.py"
"${PYTHON}" "${HERE}/render_comparison.py"
"${PYTHON}" "${HERE}/../04_nuplan_static_obstacle/render_official_interplan.py" \
  kinematic --results-root "${HERE}/results"
"${PYTHON}" "${HERE}/../04_nuplan_static_obstacle/render_official_interplan.py" \
  ocd --results-root "${HERE}/results"

echo "Results: ${HERE}/results"
