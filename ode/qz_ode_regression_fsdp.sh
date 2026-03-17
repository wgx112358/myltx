#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash myltx/ode/qz_ode_regression_fsdp.sh <gpu_count> <config_path>" >&2
  exit 1
fi

GPU_COUNT="$1"
CONFIG_PATH="$2"

ROOT="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx"
ACCELERATE_BIN="$ROOT/.venv/bin/accelerate"
TRAIN_SCRIPT="$ROOT/packages/ltx-trainer/scripts/train.py"
ACCELERATE_CONFIG="$ROOT/packages/ltx-trainer/configs/accelerate/fsdp.yaml"

NNODES="${NNODES:-${PET_NNODES:-1}}"
NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29500}}"

cd "$ROOT"

if ! [[ "$GPU_COUNT" =~ ^[0-9]+$ ]] || [[ "$GPU_COUNT" -le 0 ]]; then
  echo "Invalid gpu_count: $GPU_COUNT" >&2
  exit 1
fi

if [[ ! -x "$ACCELERATE_BIN" ]]; then
  echo "Missing accelerate executable: $ACCELERATE_BIN" >&2
  exit 1
fi

if [[ ! -f "$ACCELERATE_CONFIG" ]]; then
  echo "Missing accelerate config: $ACCELERATE_CONFIG" >&2
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

export PYTHONPATH="packages/ltx-trainer/src:packages/ltx-core/src:packages/ltx-pipelines/src:${PYTHONPATH:-}"

echo "Starting myltx ODE regression training with FSDP"
echo "  gpu_count: $GPU_COUNT"
echo "  config_path: $CONFIG_PATH"
echo "  nnodes: $NNODES"
echo "  node_rank: $NODE_RANK"
echo "  master: $MASTER_ADDR:$MASTER_PORT"
echo "  accelerate_config: $ACCELERATE_CONFIG"

"$ACCELERATE_BIN" launch \
  --config_file "$ACCELERATE_CONFIG" \
  --num_processes "$GPU_COUNT" \
  --num_machines "$NNODES" \
  --machine_rank "$NODE_RANK" \
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port "$MASTER_PORT" \
  "$TRAIN_SCRIPT" \
  "$CONFIG_PATH"
