#!/usr/bin/env bash
# Sequential LIBERO eval for a single checkpoint on a single GPU.
# Loops object/goal/spatial/10 (50 trials each), one suite at a time.
set -uo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/../../../..")}"
LIBERO_HOME="${LIBERO_HOME:?Set LIBERO_HOME to your LIBERO checkout (https://github.com/Lifelong-Robot-Learning/LIBERO).}"
STARVLA_PY="${STARVLA_PY:-python}"   # OpenCLAP env (server side)
LIBERO_PY="${LIBERO_PY:-python}"     # LIBERO sim env (client side)
CKPT="${CKPT:-$STARVLA_DIR/ckpts/Checkpoints/libero_clap_s3_l32_qwen3vl4b_km_l16/checkpoints/steps_40000_pytorch_model.pt}"

GPU_ID="${GPU_ID:-0}"
NUM_TRIALS="${NUM_TRIALS:-50}"
# SUITES may be passed either as a pre-set bash array, or as a space-separated
# string in the environment (the more portable form for nested launches).
if [[ -z "${SUITES+x}" ]]; then
  SUITES=(libero_object libero_goal libero_spatial libero_10)
elif [[ "$(declare -p SUITES 2>/dev/null)" != declare\ -a* ]]; then
  read -r -a SUITES <<< "$SUITES"
fi
BASE_PORT="${BASE_PORT:-6700}"

cd "$STARVLA_DIR"
# OpenCLAP makes ``clap`` a top-level package, so only the repo root needs to
# be on PYTHONPATH; the LIBERO sim env additionally needs the LIBERO checkout.
export PYTHONPATH="$STARVLA_DIR:$LIBERO_HOME:${PYTHONPATH:-}"
# EGL renders ~5-10x faster than osmesa for MuJoCo on NVIDIA hosts.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

MODEL_ROOT="$(dirname "$(dirname "$CKPT")")"
FOLDER_NAME="$(echo "$CKPT" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')"
# When running several drivers against the SAME ckpt in parallel (e.g. one per
# seed), set LOG_TAG to disambiguate per-suite log/video files and the summary
# file so they don't race or overwrite each other.
if [[ -n "${LOG_TAG:-}" ]]; then
  FOLDER_NAME="${FOLDER_NAME}__${LOG_TAG}"
fi
SUMMARY="$MODEL_ROOT/logs/eval_summary_${FOLDER_NAME}.txt"
mkdir -p "$MODEL_ROOT/logs" "$MODEL_ROOT/videos"
# By default the summary is reset for a fresh run.  Set APPEND_SUMMARY=1 to keep
# previously-recorded suite results (useful when rerunning a subset of suites).
if [[ "${APPEND_SUMMARY:-0}" != "1" ]]; then
  : > "$SUMMARY"
fi

echo "[driver] CKPT=$CKPT"
echo "[driver] GPU=$GPU_ID trials/task=$NUM_TRIALS suites=(${SUITES[*]})"

i=0
for SUITE in "${SUITES[@]}"; do
  PORT=$((BASE_PORT + i)); i=$((i+1))
  LOG_DIR="$MODEL_ROOT/logs/$SUITE"
  VID_DIR="$MODEL_ROOT/videos/$SUITE/$FOLDER_NAME"
  mkdir -p "$LOG_DIR" "$VID_DIR"
  SERVER_LOG="$LOG_DIR/server_${FOLDER_NAME}.log"
  EVAL_LOG="$LOG_DIR/${FOLDER_NAME}.log"

  echo "[driver] === $SUITE on port $PORT ==="
  echo "[driver] server log: $SERVER_LOG"
  echo "[driver] eval log:   $EVAL_LOG"

  # 1) start policy server
  SEED_ARGS=()
  if [[ -n "${INFERENCE_SEED:-}" ]]; then
    SEED_ARGS=(--seed "${INFERENCE_SEED}")
  fi
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$STARVLA_PY" deployment/model_server/server_policy.py \
      --ckpt_path "$CKPT" --port "$PORT" --use_bf16 "${SEED_ARGS[@]}" \
      >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  echo "[driver] server pid=$SERVER_PID"

  # 2) wait for the server to be listening (or up to 30 min — 3-way concurrent
  # loading of a 12 GB ckpt + frozen reference VLM can push close to that)
  ready=0
  for _ in $(seq 1 1800); do
    if grep -q "server running" "$SERVER_LOG" 2>/dev/null; then ready=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
    sleep 1
  done
  if [[ $ready -ne 1 ]]; then
    echo "[driver] !!! server failed to start for $SUITE; tail of server log:"
    tail -40 "$SERVER_LOG"
    echo "$SUITE: SERVER_FAILED" >> "$SUMMARY"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    continue
  fi
  echo "[driver] server listening; launching eval"

  # 3) run eval client
  "$LIBERO_PY" ./examples/LIBERO/eval_files/eval_libero.py \
      --args.pretrained-path "$CKPT" \
      --args.host 127.0.0.1 \
      --args.port "$PORT" \
      --args.task-suite-name "$SUITE" \
      --args.num-trials-per-task "$NUM_TRIALS" \
      --args.video-out-path "$VID_DIR" \
      >"$EVAL_LOG" 2>&1
  EVAL_RC=$?

  # 4) parse final success rate from log if present
  SR=$(grep -E "Total success rate:" "$EVAL_LOG" | tail -1 | sed -E 's/.*Total success rate:\s*//' | awk '{print $1}')
  printf "%-18s rc=%s  SR=%s\n" "$SUITE" "$EVAL_RC" "${SR:-NA}" | tee -a "$SUMMARY"

  # 5) shut down server
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  sleep 2
done

echo "[driver] done. summary at $SUMMARY"
cat "$SUMMARY"
