#!/usr/bin/env bash
# Trainer-only launcher for the 16-DiT-layer + Knowledge-Matching recipe.
# Model/dataset/KM knobs (vl_layer_indices, data_mix, balance_dataset_weights,
# kl_loss_weight, ...) live in the YAML.
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/../../..")"
# Activate your env if needed:
# source /path/to/miniconda3/etc/profile.d/conda.sh && conda activate openclap

###########################################################################################
# === Modify these for your environment ===
config_yaml=./examples/LIBERO/train_files/starvla_libero_clap_km_l16.yaml
run_root_dir=./ckpts/Checkpoints
run_id=libero_clap_s3_l32_qwen3vl4b_km_l16
###########################################################################################

# --- Trainer overrides (omit a line to fall back to the YAML default) ---
freeze_module_list=''

# Throughput knobs (effective batch = per_device_batch_size * grad_acc * #GPUs)
per_device_batch_size=8
gradient_accumulation_steps=1

# Smoke-test mode: tiny step budget + frequent logging. Override via env:
#   SMOKE_TEST=1 bash run_libero_clap_km_l16.sh
SMOKE_TEST=${SMOKE_TEST:-0}
if [[ "${SMOKE_TEST}" == "1" ]]; then
  max_train_steps=5
  save_interval=999999
  logging_frequency=1
  eval_interval=999999
  run_id=${run_id}_smoke
fi
max_train_steps=${max_train_steps:-30000}
save_interval=${save_interval:-5000}
logging_frequency=${logging_frequency:-100}
eval_interval=${eval_interval:-100}

# --- PyTorch Elastic / torchrun-style env vars (with single-node fallbacks) ---
PET_NNODES=${PET_NNODES:-1}
PET_NPROC_PER_NODE=${PET_NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
PET_NODE_RANK=${PET_NODE_RANK:-0}
PET_MASTER_ADDR=${PET_MASTER_ADDR:-127.0.0.1}
PET_MASTER_PORT=${PET_MASTER_PORT:-29500}

TOTAL_GPUS=$((PET_NNODES * PET_NPROC_PER_NODE))

echo "WORLD_SIZE:   $PET_NNODES"
echo "NPROC/NODE:   $PET_NPROC_PER_NODE"
echo "NODE_RANK:    $PET_NODE_RANK"
echo "MASTER_ADDR:  $PET_MASTER_ADDR"
echo "MASTER_PORT:  $PET_MASTER_PORT"
echo "TOTAL_GPUS:   $TOTAL_GPUS"
echo "SMOKE_TEST:   ${SMOKE_TEST}  (max_train_steps=${max_train_steps})"
echo "config_yaml:  ${config_yaml}"
echo "run_id:       ${run_id}"
echo "batch:        per_device=${per_device_batch_size}  grad_acc=${gradient_accumulation_steps}"

# --- NCCL tuning ---
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_2,mlx5_3}
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
# Keep clap importable for QwenPIKM's CLAP solutionizer
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
log_dir=${output_dir}/logs
mkdir -p "${output_dir}" "${log_dir}"
cp "$0" "${output_dir}/"
cp "${config_yaml}" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_ip ${PET_MASTER_ADDR} \
  --main_process_port ${PET_MASTER_PORT} \
  --machine_rank ${PET_NODE_RANK} \
  --num_machines ${PET_NNODES} \
  --num_processes ${TOTAL_GPUS} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --datasets.vla_data.per_device_batch_size ${per_device_batch_size} \
  --trainer.gradient_accumulation_steps ${gradient_accumulation_steps} \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps ${max_train_steps} \
  --trainer.save_interval ${save_interval} \
  --trainer.logging_frequency ${logging_frequency} \
  --trainer.eval_interval ${eval_interval} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  2>&1 | tee "${log_dir}/log-${PET_NODE_RANK}.${PET_NNODES}.log"
