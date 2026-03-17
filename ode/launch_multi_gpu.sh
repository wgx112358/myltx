#!/bin/bash
# 多 GPU 并行生成 ODE 数据（非蒸馏版本）
# 支持两种模式：
# 1. 节点分片模式（推荐）
#    bash ode/launch_multi_gpu.sh <start_group> <node_gpu_count> <total_groups> [config_path]
#    - start_group: 当前节点从第几组开始处理，按 0 开始计数
#    - node_gpu_count: 当前节点可用 GPU 数量
#    - total_groups: 全局总分组数（通常等于总 GPU 数或你想切分的总 chunk 数）
#
#    示例：
#      bash ode/launch_multi_gpu.sh 3 6 100
#    含义：
#      - 整个 CSV 按 100 组切分
#      - 当前节点有 6 张 GPU
#      - 当前节点处理第 3~8 组
#      - 实际传给 Python 的 chunk_id 为 3~8，num_chunks 为 100
#
# 2. 兼容旧模式
#    bash ode/launch_multi_gpu.sh [gpu_count] [config_path]
#    - 等价于从第 0 组开始，总分组数 = 当前 GPU 数

set -euo pipefail

CONFIG_DEFAULT="ode/configs/gen_ode_data.yaml"
LOG_DIR="ode/logs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_ACTIVATE="${PROJECT_DIR}/.venv/bin/activate"

usage() {
    echo "用法（节点分片模式）:"
    echo "  bash ode/launch_multi_gpu.sh <start_group> <node_gpu_count> <total_groups> [config_path]"
    echo ""
    echo "示例:"
    echo "  bash ode/launch_multi_gpu.sh 3 6 100"
    echo "  bash ode/launch_multi_gpu.sh 3 6 100 ode/configs/gen_ode_data.yaml"
    echo ""
    echo "兼容旧模式:"
    echo "  bash ode/launch_multi_gpu.sh [gpu_count] [config_path]"
}

is_non_negative_int() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

if [[ $# -ge 3 ]]; then
    START_GROUP=$1
    NUM_GPUS=$2
    TOTAL_GROUPS=$3
    CONFIG=${4:-${CONFIG_DEFAULT}}
    MODE="node-aware"
else
    NUM_GPUS=${1:-4}
    CONFIG=${2:-${CONFIG_DEFAULT}}
    START_GROUP=0
    TOTAL_GROUPS=${NUM_GPUS}
    MODE="legacy"
fi

if ! is_non_negative_int "${START_GROUP}"; then
    echo "错误: start_group 必须是 >= 0 的整数，当前值: ${START_GROUP}" >&2
    usage
    exit 1
fi

if ! is_non_negative_int "${NUM_GPUS}" || (( NUM_GPUS < 1 )); then
    echo "错误: node_gpu_count / gpu_count 必须是 >= 1 的整数，当前值: ${NUM_GPUS}" >&2
    usage
    exit 1
fi

if ! is_non_negative_int "${TOTAL_GROUPS}" || (( TOTAL_GROUPS < 1 )); then
    echo "错误: total_groups 必须是 >= 1 的整数，当前值: ${TOTAL_GROUPS}" >&2
    usage
    exit 1
fi

if (( START_GROUP >= TOTAL_GROUPS )); then
    echo "错误: start_group=${START_GROUP} 超出范围 [0, $((TOTAL_GROUPS - 1))]" >&2
    exit 1
fi

START_CHUNK_ID=${START_GROUP}
REMAINING_GROUPS=$((TOTAL_GROUPS - START_CHUNK_ID))
LAUNCH_COUNT=${NUM_GPUS}

if (( REMAINING_GROUPS < LAUNCH_COUNT )); then
    LAUNCH_COUNT=${REMAINING_GROUPS}
fi

if (( LAUNCH_COUNT <= 0 )); then
    echo "错误: 没有可启动的任务。start_group=${START_GROUP}, total_groups=${TOTAL_GROUPS}" >&2
    exit 1
fi

cd "${PROJECT_DIR}"

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
    echo "错误: 未找到虚拟环境激活脚本: ${VENV_ACTIVATE}" >&2
    exit 1
fi

source "${VENV_ACTIVATE}"

mkdir -p "${LOG_DIR}"

echo "启动模式: ${MODE}"
echo "config: ${CONFIG}"
echo "日志目录: ${LOG_DIR}"
echo "项目目录: ${PROJECT_DIR}"
echo "虚拟环境: ${VENV_ACTIVATE}"
echo "全局总分组数: ${TOTAL_GROUPS}"
echo "当前节点 GPU 数: ${NUM_GPUS}"
echo "当前节点起始组号(0-indexed): ${START_GROUP}"
echo "本节点实际启动任务数: ${LAUNCH_COUNT}"
echo ""

if (( LAUNCH_COUNT < NUM_GPUS )); then
    echo "注意: 从第 ${START_GROUP} 组开始只剩 ${LAUNCH_COUNT} 组可处理，因此不会占满全部 ${NUM_GPUS} 张 GPU。"
    echo ""
fi

for LOCAL_GPU_ID in $(seq 0 $((LAUNCH_COUNT - 1))); do
    CHUNK_ID=$((START_CHUNK_ID + LOCAL_GPU_ID))
    GROUP_NO=${CHUNK_ID}
    LOG_FILE="${LOG_DIR}/group$(printf '%04d' "${GROUP_NO}")_gpu${LOCAL_GPU_ID}.log"

    echo "本地 GPU ${LOCAL_GPU_ID}: 全局组 ${GROUP_NO}/${TOTAL_GROUPS} (chunk_id=${CHUNK_ID}) -> ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=${LOCAL_GPU_ID} python ode/gen_ode_data.py \
        --config "${CONFIG}" \
        --chunk_id "${CHUNK_ID}" \
        --num_chunks "${TOTAL_GROUPS}" \
        > "${LOG_FILE}" 2>&1 &
done

echo ""
echo "所有进程已在后台启动，PID 列表:"
jobs -p

echo ""
echo "实时查看日志（以当前节点第一个任务为例）:"
echo "  tail -f ${LOG_DIR}/group$(printf '%04d' "${START_GROUP}")_gpu0.log"
echo ""
echo "等待所有进程完成:"
echo "  wait"
wait
