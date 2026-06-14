#!/usr/bin/env bash
# Launch the synchronous websocket server for QwenPIKM on Astribot S1.
# Pair with examples/Astribot/eval_files/sync_policy_client.py.
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/../../..")"
# Activate your env if needed:
# source /path/to/miniconda3/etc/profile.d/conda.sh && conda activate openclap

CKPT="${CKPT:?Set CKPT to a QwenPIKM checkpoint dir}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_ID="${GPU_ID:-0}"
USE_BF16="${USE_BF16:-1}"
ROBOT_TYPE="${ROBOT_TYPE:-S1-stationary}"
STATS_PATH="${STATS_PATH:-./clap/assets/dataset_statistics_32.json}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-240}"
IMAGE_WIDTH="${IMAGE_WIDTH:-320}"
TRAIN_FREQ="${TRAIN_FREQ:-30}"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CMD=(
  python -m examples.Astribot.eval_files.sync_policy_server
  --ckpt_path "${CKPT}"
  --host "${HOST}"
  --port "${PORT}"
  --robot_type "${ROBOT_TYPE}"
  --stats_path "${STATS_PATH}"
  --image_height "${IMAGE_HEIGHT}"
  --image_width "${IMAGE_WIDTH}"
  --train_freq "${TRAIN_FREQ}"
)

if [[ "${USE_BF16}" == "1" ]]; then
  CMD+=(--use_bf16)
fi
if [[ -n "${DEFAULT_PROMPT}" ]]; then
  CMD+=(--default_prompt "${DEFAULT_PROMPT}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
