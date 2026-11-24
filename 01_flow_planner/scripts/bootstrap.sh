#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
mkdir -p .tools .cache upstream checkpoints artifacts
export UV_CACHE_DIR="$TASK_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$TASK_ROOT/.tools/python-managed"
UV="$TASK_ROOT/.tools/python/bin/uv"
if [ ! -x "$UV" ]; then
    python3 -m pip install --target .tools/python --cache-dir .cache/pip uv==0.13.0
fi
if [ ! -x .venv/bin/python ]; then
    "$UV" venv --python 3.9.25 .venv
fi
if [ ! -d upstream/Flow-Planner/.git ]; then
    git clone https://github.com/DiffusionAD/Flow-Planner.git upstream/Flow-Planner
    git -C upstream/Flow-Planner checkout --detach 366df7654d7b7fa77f9d2577279da63127c6d2f8
fi
if [ ! -d upstream/nuplan-devkit/.git ]; then
    git clone https://github.com/motional/nuplan-devkit.git upstream/nuplan-devkit
    git -C upstream/nuplan-devkit checkout --detach e9241677997dd86bfc0bcd44817ab04fe631405b
fi
test "$(git -C upstream/Flow-Planner rev-parse HEAD)" = 366df7654d7b7fa77f9d2577279da63127c6d2f8
test "$(git -C upstream/nuplan-devkit rev-parse HEAD)" = e9241677997dd86bfc0bcd44817ab04fe631405b
for FILE in model.pth model_config.yaml; do
    if [ ! -f "checkpoints/$FILE" ]; then
        curl -fL --retry 3 "https://huggingface.co/ttwhy/flow-planner/resolve/b627934a8db3021892afd13f5278ba0149532efc/$FILE" -o "checkpoints/$FILE.part"
        mv "checkpoints/$FILE.part" "checkpoints/$FILE"
    fi
done
if [ -f checkpoints/SHA256SUMS ]; then
    (cd checkpoints && sha256sum -c SHA256SUMS)
fi
if [ -f requirements-lock.txt ]; then
    "$UV" pip sync --python .venv/bin/python requirements-lock.txt
else
    "$UV" pip install --python .venv/bin/python -r requirements-simulation.txt
fi
"$UV" pip check --python .venv/bin/python
printf '%s\n' 'Environment ready. No training was launched.'
