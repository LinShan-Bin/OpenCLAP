#!/usr/bin/env bash
# Launcher for QwenAR (CLAP-VLA stage-3, autoregressive) on the four-dataset
# pretraining mixture (Astribot S1-stationary + AgiBot + ego4d + DROID).
#
# All static training knobs live in the YAML next to this script
# (``starvla_astribot_qwenar.yaml``). Only the values that genuinely
# vary per-launch are passed as CLI overrides.
#
# Common knobs:
#   SMOKE_TEST=1            → tiny step budget, frequent logging
#   PET_NPROC_PER_NODE      → override GPU count
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/../../..")"
# Activate your env if needed:
# source /path/to/miniconda3/etc/profile.d/conda.sh && conda activate openclap

###########################################################################################
config_yaml=./examples/Astribot/train_files/starvla_astribot_qwenar.yaml
run_root_dir=./ckpts/Checkpoints
run_id=clap_s3_l32_qwen3vl4b_pretrain

# Smoke-test mode: tiny step budget + frequent logging.
SMOKE_TEST=${SMOKE_TEST:-0}
if [[ "${SMOKE_TEST}" == "1" ]]; then
  max_train_steps=3
  save_interval=999999
  logging_frequency=1
  eval_interval=999999
  run_id=${run_id}_smoke
fi
max_train_steps=${max_train_steps:-800000}
save_interval=${save_interval:-5000}
logging_frequency=${logging_frequency:-100}
eval_interval=${eval_interval:-100}
###########################################################################################

PET_NNODES=${PET_NNODES:-1}
PET_NPROC_PER_NODE=${PET_NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
PET_NODE_RANK=${PET_NODE_RANK:-0}
PET_MASTER_ADDR=${PET_MASTER_ADDR:-127.0.0.1}
PET_MASTER_PORT=${PET_MASTER_PORT:-29501}

TOTAL_GPUS=$((PET_NNODES * PET_NPROC_PER_NODE))

echo "WORLD_SIZE:   $PET_NNODES"
echo "NPROC/NODE:   $PET_NPROC_PER_NODE"
echo "NODE_RANK:    $PET_NODE_RANK"
echo "MASTER_ADDR:  $PET_MASTER_ADDR"
echo "MASTER_PORT:  $PET_MASTER_PORT"
echo "TOTAL_GPUS:   $TOTAL_GPUS"
echo "SMOKE_TEST:   ${SMOKE_TEST}  (max_train_steps=${max_train_steps})"

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_2,mlx5_3}
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
# OpenCLAP makes ``clap`` a first-class top-level package, so we only need
# the repo root on PYTHONPATH; no ``latent_action_model`` shim required.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

output_dir=${run_root_dir}/${run_id}
log_dir=${output_dir}/logs
mkdir -p "${output_dir}" "${log_dir}"
cp "$0" "${output_dir}/" 2>/dev/null || true
cp "${config_yaml}" "${output_dir}/" 2>/dev/null || true

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_ip ${PET_MASTER_ADDR} \
  --main_process_port ${PET_MASTER_PORT} \
  --machine_rank ${PET_NODE_RANK} \
  --num_machines ${PET_NNODES} \
  --num_processes ${TOTAL_GPUS} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --trainer.max_train_steps ${max_train_steps} \
  --trainer.save_interval ${save_interval} \
  --trainer.logging_frequency ${logging_frequency} \
  --trainer.eval_interval ${eval_interval} \
  --run_id ${run_id} \
  2>&1 | tee "${log_dir}/log-${PET_NODE_RANK}.${PET_NNODES}.log"
