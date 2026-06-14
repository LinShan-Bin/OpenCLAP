#!/usr/bin/env bash
# Stage-1 (Act-VAE) training launcher.
#
# Examples:
#   bash clap/scripts/run_stage1.sh                       # full run
#   bash clap/scripts/run_stage1.sh --trainer.max_epochs=2
#   SMOKE_TEST=1 bash clap/scripts/run_stage1.sh          # 2-step single-GPU smoke
#
# SMOKE_TEST=1 expects ./data_smoke to contain at least one LeRobot dataset
# (the smoke test points there to avoid scanning the full data root). For
# convenience we auto-symlink one Astribot subdataset if you give us a
# SMOKE_DATASET path.
set -euo pipefail
cd "$(dirname "$0")/../.."        # -> repo root
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

CONFIG=${CONFIG:-clap/configs/clap-s1-l32.yaml}
SMOKE_TEST=${SMOKE_TEST:-0}

EXTRA=()
if [[ "${SMOKE_TEST}" == "1" ]]; then
  if [[ -n "${SMOKE_DATASET:-}" ]] && [[ ! -e ./data_smoke ]]; then
    mkdir -p ./data_smoke
    ln -s "${SMOKE_DATASET}" "./data_smoke/$(basename "${SMOKE_DATASET}")"
  fi
  if [[ ! -d ./data_smoke ]] || [[ -z "$(ls -A ./data_smoke 2>/dev/null)" ]]; then
    echo "[SMOKE_TEST] ./data_smoke not populated. Either:" >&2
    echo "  • run SMOKE_DATASET=/abs/path/to/lerobot_dataset SMOKE_TEST=1 bash clap/scripts/run_stage1.sh" >&2
    echo "  • or populate ./data_smoke/ with one LeRobot v2.0/v2.1 dataset (symlinks fine)" >&2
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
