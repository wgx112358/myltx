#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx"
MYLTX_DIR="${ROOT}/myltx"
VENV_ACTIVATE="${MYLTX_DIR}/.venv/bin/activate"

INPUT_DIR="${INPUT_DIR:-/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/ode_data/ode_data_raw_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-/inspire/qb-ilm/project/agileapplication/zhangkaipeng-24043/wgx/ode_data/ode_data_raw_v1_decoded_samples30_$(date +%Y%m%d-%H%M%S)}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${MYLTX_DIR}/model/ltx-2.3-22b-distilled.safetensors}"
NUM_SAMPLES="${NUM_SAMPLES:-30}"
SEED="${SEED:-42}"
FPS="${FPS:-24}"
DEVICE="${DEVICE:-cuda}"

echo "[decode-raw-v1] start: $(date -Iseconds)"
echo "[decode-raw-v1] hostname: $(hostname)"
echo "[decode-raw-v1] input_dir: ${INPUT_DIR}"
echo "[decode-raw-v1] output_dir: ${OUTPUT_DIR}"
echo "[decode-raw-v1] checkpoint: ${CHECKPOINT_PATH}"
echo "[decode-raw-v1] num_samples: ${NUM_SAMPLES}"
echo "[decode-raw-v1] seed: ${SEED}"
echo "[decode-raw-v1] fps: ${FPS}"
echo "[decode-raw-v1] device: ${DEVICE}"

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "ERROR: venv activate script not found: ${VENV_ACTIVATE}" >&2
  exit 1
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "ERROR: input dir not found: ${INPUT_DIR}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

source "${VENV_ACTIVATE}"
cd "${MYLTX_DIR}"

python ode/decode_distilled_samples.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --num-samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  --fps "${FPS}" \
  --device "${DEVICE}"

echo "[decode-raw-v1] done: $(date -Iseconds)"
