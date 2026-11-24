#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${HERE}/../01_flow_planner/.venv/bin/python"
export MPLCONFIGDIR="${HERE}/.cache/matplotlib"

"${PYTHON}" "${HERE}/run_nudge_position_sweep.py" "$@"
"${PYTHON}" "${HERE}/analyze_nudge_position_sweep.py"
for variant in baseline closer_8m farther_8m left_1m right_1m
do
  "${PYTHON}" "${HERE}/render_official_interplan.py" "${variant}" \
    --results-root "${HERE}/results_nudge_position_sweep"
done
