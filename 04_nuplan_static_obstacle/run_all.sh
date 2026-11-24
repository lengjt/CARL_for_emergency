#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ljt23/research_ca/04_nuplan_static_obstacle
PYTHON=/home/ljt23/research_ca/01_flow_planner/.venv/bin/python

cd "$ROOT"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 "$PYTHON" run_experiment.py baseline
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 "$PYTHON" run_experiment.py obstacle
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PYTHON" verify_results.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PYTHON" render_results.py

