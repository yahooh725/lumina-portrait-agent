#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${PORTRAIT_PYTHON:-}" ]]; then
  PY="${PORTRAIT_PYTHON}"
elif [[ -x "${ROOT}/tool_acceptance/.conda-env/bin/python" ]]; then
  PY="${ROOT}/tool_acceptance/.conda-env/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi
SCRIPT="${ROOT}/agent_baseline/demo_showcase_v1/web_demo.py"

GPU_ID="${PORTRAIT_GPU_ID:-0}"
PORT="${PORTRAIT_DEMO_PORT:-8787}"
HOST="${PORTRAIT_DEMO_HOST:-127.0.0.1}"
PLANNER="${PORTRAIT_DEMO_PLANNER:-six1800}"

cd "${ROOT}"
export PYTHONDONTWRITEBYTECODE=1
if [[ -d "${ROOT}/tool_acceptance/models" ]]; then
  export LUMINA_VISION_MODEL_DIR="${LUMINA_VISION_MODEL_DIR:-${ROOT}/tool_acceptance/models}"
fi
if [[ -d "${ROOT}/agent_baseline/models/Qwen3-VL-2B-Instruct" ]]; then
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi

if [[ "${PLANNER}" == "six1800" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  exec "${PY}" -u "${SCRIPT}" --host "${HOST}" --port "${PORT}" --planner six1800 --preload
else
  exec "${PY}" -u "${SCRIPT}" --host "${HOST}" --port "${PORT}" --planner "${PLANNER}"
fi
