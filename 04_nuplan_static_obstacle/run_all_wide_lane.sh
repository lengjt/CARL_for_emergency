#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/ljt23/research_ca/01_flow_planner/.venv/bin/python

"$PYTHON" "$HERE/audit_candidate.py" --config "$HERE/config_wide_lane.yaml" --target-distance 40
"$PYTHON" "$HERE/run_experiment.py" baseline --config "$HERE/config_wide_lane.yaml"
"$PYTHON" "$HERE/run_experiment.py" obstacle --config "$HERE/config_wide_lane.yaml"
"$PYTHON" "$HERE/verify_results.py" --config "$HERE/config_wide_lane.yaml"
"$PYTHON" "$HERE/render_results.py" --config "$HERE/config_wide_lane.yaml"
