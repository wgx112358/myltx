#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash myltx/ode/qz_ode_regression_single_smoke.sh <config_path>" >&2
  exit 1
fi

CONFIG_PATH="$1"
ROOT="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx"
PYTHON_BIN="$ROOT/.venv/bin/python"
TRAIN_SCRIPT="$ROOT/packages/ltx-trainer/scripts/train.py"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing python executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Missing train script: $TRAIN_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file does not exist: $CONFIG_PATH" >&2
  exit 1
fi

cd "$ROOT"
export PYTHONPATH="packages/ltx-trainer/src:packages/ltx-core/src:packages/ltx-pipelines/src:${PYTHONPATH:-}"

OUTPUT_DIR="$("$PYTHON_BIN" - <<'PY' "$CONFIG_PATH"
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
print(config["output_dir"])
PY
)"

mkdir -p "$OUTPUT_DIR"
ATTEMPT_TAG="$(date +%Y%m%dT%H%M%S)"
RUN_LOG="$OUTPUT_DIR/smoke_attempt_${ATTEMPT_TAG}.log"
MEM_LOG="$OUTPUT_DIR/nvidia_smi_${ATTEMPT_TAG}.log"

exec > >(tee -a "$RUN_LOG") 2>&1

echo "attempt_tag: $ATTEMPT_TAG"
echo "attempt_log: $RUN_LOG"

monitor_pid=""
cleanup() {
  if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" >/dev/null 2>&1; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if command -v nvidia-smi >/dev/null 2>&1; then
  (
    while true; do
      date -Iseconds
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
      sleep 5
    done
  ) >"$MEM_LOG" 2>&1 &
  monitor_pid="$!"
  echo "nvidia-smi monitor log: $MEM_LOG"
fi

echo "Starting single-GPU ODE regression smoke test"
echo "  config_path: $CONFIG_PATH"
echo "  output_dir: $OUTPUT_DIR"

set +e
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -u "$TRAIN_SCRIPT" "$CONFIG_PATH" --disable-progress-bars
exit_code=$?
set -e

echo "train_exit_code: $exit_code"
exit "$exit_code"
