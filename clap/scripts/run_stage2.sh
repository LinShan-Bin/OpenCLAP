#!/usr/bin/env bash
# Stage-2 (VD-VAE / Vision VAE) training launcher.
#
# Examples:
#   bash clap/scripts/run_stage2.sh                       # full run
#   SMOKE_TEST=1 bash clap/scripts/run_stage2.sh          # 2-step single-GPU smoke
#
# Pre-requisites for SMOKE_TEST=1:
#   • A finished Stage-1 checkpoint at ./clap/ckpts/clap-s1-l32/last.ckpt
#     (or override via --model.stage_one_ckpt=...)
#   • ./data_smoke populated with at least one LeRobot dataset (see run_stage1.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

CONFIG=${CONFIG:-clap/configs/clap-s2-l32.yaml}
SMOKE_TEST=${SMOKE_TEST:-0}

EXTRA=()
if [[ "${SMOKE_TEST}" == "1" ]]; then
  if [[ ! -d ./data_smoke ]] || [[ -z "$(ls -A ./data_smoke 2>/dev/null)" ]]; then
    echo "[SMOKE_TEST] ./data_smoke not populated. Run run_stage1.sh smoke first or populate it." >&2
    exit 1
  fi
  EXTRA+=(--data.data_root='[./data_smoke]'
          --data.data_mix=all
          --data.batch_size=2
          --data.num_workers=0
          --trainer.max_steps=2
          --trainer.limit_train_batches=2
          --trainer.limit_val_batches=0
          --trainer.log_every_n_steps=1
          --trainer.devices=1
          --trainer.strategy=auto
          --trainer.precision=bf16-mixed
          --trainer.logger=null
          --trainer.callbacks=null)
  echo "[SMOKE_TEST=1] 2-step single-GPU smoke against ./data_smoke"
fi

python -u -m clap.main_clap fit --config "${CONFIG}" "${EXTRA[@]}" "$@"
