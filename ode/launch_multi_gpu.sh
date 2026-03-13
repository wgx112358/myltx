#!/bin/bash
# 多 GPU 并行生成 ODE 数据
# 每个 GPU 独立运行一个进程，处理 CSV 的不同分片，写入同一输出目录。
#
# 用法:
#   bash ode/launch_multi_gpu.sh [GPU数量] [config路径]
#
# 示例（4 GPU）:
#   bash ode/launch_multi_gpu.sh 4 ode/configs/gen_ode_data.yaml
#
# 示例（8 GPU）:
#   bash ode/launch_multi_gpu.sh 8 ode/configs/gen_ode_data.yaml

set -e

NUM_GPUS=${1:-4}
CONFIG=${2:-ode/configs/gen_ode_data.yaml}
LOG_DIR="ode/logs"

mkdir -p "${LOG_DIR}"

echo "启动 ${NUM_GPUS} 个进程，config: ${CONFIG}"
echo "日志目录: ${LOG_DIR}"
echo ""

for CHUNK_ID in $(seq 0 $((NUM_GPUS - 1))); do
    LOG_FILE="${LOG_DIR}/gpu${CHUNK_ID}.log"
    echo "GPU ${CHUNK_ID}: chunk_id=${CHUNK_ID}/${NUM_GPUS}  log -> ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=${CHUNK_ID} python ode/gen_ode_data.py \
        --config "${CONFIG}" \
        --chunk_id "${CHUNK_ID}" \
        --num_chunks "${NUM_GPUS}" \
        > "${LOG_FILE}" 2>&1 &
done

echo ""
echo "所有进程已在后台启动，PID 列表:"
jobs -p

echo ""
echo "实时查看某个 GPU 的日志（以 GPU 0 为例）:"
echo "  tail -f ${LOG_DIR}/gpu0.log"
echo ""
echo "等待所有进程完成:"
echo "  wait"
