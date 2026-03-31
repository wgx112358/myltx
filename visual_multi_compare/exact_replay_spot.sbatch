#!/usr/bin/env bash
#SBATCH -J vmc-exact-replay
#SBATCH -p Omnilab
#SBATCH --quotatype=spot
#SBATCH -N 1 -n 1 --cpus-per-task=8 --gres=gpu:1
#SBATCH -o /mnt/petrelfs/wanggongxuan/workspace/myltx/visual_multi_compare/log/exact_replay-%j.out
#SBATCH -e /mnt/petrelfs/wanggongxuan/workspace/myltx/visual_multi_compare/log/exact_replay-%j.err
set -euo pipefail

CONTAINER_IMAGE="/mnt/petrelfs/wanggongxuan/node-22.sif"
CONTAINER_BIND="/mnt/petrelfs/wanggongxuan/workspace:/workspace"
CONTAINER_ROOT="/workspace/myltx"
CONTAINER_PYTHONPATH="/workspace/myltx:/workspace/myltx/packages/ltx-trainer/src:/workspace/myltx/packages/ltx-core/src:/workspace/myltx/packages/ltx-pipelines/src"
OUTPUT_ROOT="/workspace/myltx/visual_multi_compare/outputs/exact_replay"

srun -N1 -n1 --gres=gpu:1 --cpus-per-task=8 \
  /usr/bin/env PYTHONPATH="$CONTAINER_PYTHONPATH" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /usr/bin/apptainer exec --nv --bind "$CONTAINER_BIND" "$CONTAINER_IMAGE" \
  /bin/bash -lc "cd $CONTAINER_ROOT && exec .venv/bin/python visual_multi_compare/replay_precompute_exact.py \
    --samples-json visual_multi_compare/sampled_8.json \
    --precompute-root /workspace/ode_precompute_stage2_8gpu_20260324T125612Z \
    --config ode/configs/gen_ode_data_distilled.yaml \
    --output-root $OUTPUT_ROOT \
    --device cuda"
