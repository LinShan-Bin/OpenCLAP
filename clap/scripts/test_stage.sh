#!/usr/bin/env bash
# Open-loop reconstruction test for a stage-1 or stage-2 checkpoint.
#   bash clap/scripts/test_stage.sh clap/configs/clap-s1-l32.yaml clap/ckpts/clap-s1-l32/last.ckpt
#   bash clap/scripts/test_stage.sh clap/configs/clap-s2-l32.yaml clap/ckpts/clap-s2-l32/last.ckpt
#
# Logs per-DOF MSE (left/right arms, VAE and visual branches) to wandb and
# stdout. Internally invokes ``LightningCLI test`` against ``DINO_CLAP``.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CONFIG=${1:-clap/configs/clap-s1-l32.yaml}
CKPT=${2:?"Pass a checkpoint as the second argument."}

python -m clap.main_clap test --config "${CONFIG}" --ckpt_path "${CKPT}" "${@:3}"
