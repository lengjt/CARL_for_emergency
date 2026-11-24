#!/usr/bin/env bash
set -euo pipefail

cd /home/ljt23/research_ca/04_nuplan_static_obstacle
PY=/home/ljt23/research_ca/01_flow_planner/.venv/bin/python
CONFIG=config_interplan_nudge_final.yaml

"$PY" run_experiment.py both --config "$CONFIG"
"$PY" verify_results.py --config "$CONFIG"
"$PY" analyze_interplan_nudge.py --config "$CONFIG"
"$PY" render_results.py --config "$CONFIG"
