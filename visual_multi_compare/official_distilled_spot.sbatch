#!/usr/bin/env bash
#SBATCH -J vmc-official-dist
#SBATCH -p Omnilab
#SBATCH --quotatype=spot
#SBATCH -N 1 -n 1 --cpus-per-task=8 --gres=gpu:1
#SBATCH -o /mnt/petrelfs/wanggongxuan/workspace/myltx/visual_multi_compare/log/official_distilled-%j.out
#SBATCH -e /mnt/petrelfs/wanggongxuan/workspace/myltx/visual_multi_compare/log/official_distilled-%j.err
set -euo pipefail

CONTAINER_IMAGE="/mnt/petrelfs/wanggongxuan/node-22.sif"
CONTAINER_BIND="/mnt/petrelfs/wanggongxuan/workspace:/workspace"
CONTAINER_ROOT="/workspace/myltx"
CONTAINER_PYTHONPATH="/workspace/myltx:/workspace/myltx/packages/ltx-trainer/src:/workspace/myltx/packages/ltx-core/src:/workspace/myltx/packages/ltx-pipelines/src"
OUTPUT_ROOT="/workspace/myltx/visual_multi_compare/outputs"

srun -N1 -n1 --gres=gpu:1 --cpus-per-task=8 \
  /usr/bin/env PYTHONPATH="$CONTAINER_PYTHONPATH" PYTHONUNBUFFERED=1 \
  /usr/bin/apptainer exec --nv --bind "$CONTAINER_BIND" "$CONTAINER_IMAGE" \
  /bin/bash -lc "cd $CONTAINER_ROOT && exec .venv/bin/python visual_multi_compare/infer_5prompt_distilled.py \
    --samples-json visual_multi_compare/sampled_8.json \
    --distilled-checkpoint-path model/ltx-2.3-22b-distilled.safetensors \
    --spatial-upsampler-path model/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
    --gemma-root model/gemma \
    --tag official_distilled \
    --output-root $OUTPUT_ROOT"
