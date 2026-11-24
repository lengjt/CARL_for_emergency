#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${HERE}/../01_flow_planner/.venv/bin/python"
export MPLCONFIGDIR="${HERE}/.cache/matplotlib"

"${HERE}/build_ocd_for_flow.sh"
"${PYTHON}" "${HERE}/check_ocd_interface.py"
"${PYTHON}" "${HERE}/run_peak_sweep_15mps.py" "$@"
"${PYTHON}" "${HERE}/analyze_peak_sweep_15mps.py"
for level in low_0p5 medium_0p8 high_1p0
do
  "${PYTHON}" "${HERE}/../04_nuplan_static_obstacle/render_official_interplan.py" \
    "${level}" --results-root "${HERE}/results_ocd_peak_sweep_15mps"
done

echo "Results: ${HERE}/results_ocd_peak_sweep_15mps"
