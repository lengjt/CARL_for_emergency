#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOW_ROOT="${HERE}/../01_flow_planner"
PYTHON="${FLOW_ROOT}/.venv/bin/python"

export MPLCONFIGDIR="${HERE}/.cache/matplotlib"

"${PYTHON}" "${HERE}/check_official_interplan_data.py"
"${PYTHON}" "${HERE}/run_official_interplan.py" \
  --categories nudge_around_parked_vehicle overtake_parked_vehicle avoid_crashed_cars "$@"
"${PYTHON}" "${HERE}/analyze_official_interplan.py"
"${PYTHON}" "${HERE}/plot_official_interplan_summary.py"

for token in \
  81f400b414215325-s0 \
  168c4e592f1b5984-s1 \
  948201ecfe2c560a-s1 \
  9e7581ef72155c9f-s0
do
  "${PYTHON}" "${HERE}/render_official_interplan.py" "${token}"
done

echo "Results: ${HERE}/results_official_interplan"
