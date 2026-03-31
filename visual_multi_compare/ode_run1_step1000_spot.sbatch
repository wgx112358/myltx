#!/usr/bin/env bash
#SBATCH -J vmc-ode-run1
#SBATCH -p Omnilab
#SBATCH --quotatype=spot
#SBATCH -N 1 -n 1 --cpus-per-task=8 --gres=gpu:1
#SBATCH -o /mnt/petrelfs/wanggongxuan/workspace/myltx/visual_multi_compare/log/run1_step1000-%j.out
#SBATCH -e /mnt/petrelfs/wanggongxuan/workspace/myltx/visual_multi_compare/log/run1_step1000-%j.err
set -euo pipefail

CONTAINER_IMAGE="/mnt/petrelfs/wanggongxuan/node-22.sif"
CONTAINER_BIND="/mnt/petrelfs/wanggongxuan/workspace:/workspace"
CONTAINER_ROOT="/workspace/myltx"
CONTAINER_PYTHONPATH="/workspace/myltx:/workspace/myltx/packages/ltx-trainer/src:/workspace/myltx/packages/ltx-core/src:/workspace/myltx/packages/ltx-pipelines/src"
SOURCE_CKPT="/workspace/ode_ckpt/ltx2_av_ode_regression_full_run1/checkpoints/model_weights_step_01000.safetensors"
CONVERTED_CKPT="/workspace/ode_ckpt/ltx2_av_ode_regression_full_run1/checkpoints/model_weights_step_01000.converted.safetensors"
OUTPUT_ROOT="/workspace/myltx/visual_multi_compare/outputs"

srun -N1 -n1 --gres=gpu:1 --cpus-per-task=8 \
  /usr/bin/env PYTHONPATH="$CONTAINER_PYTHONPATH" PYTHONUNBUFFERED=1 \
  /usr/bin/apptainer exec --nv --bind "$CONTAINER_BIND" "$CONTAINER_IMAGE" \
  /bin/bash -lc "cd $CONTAINER_ROOT && \
    if [ ! -f '$CONVERTED_CKPT' ]; then \
      .venv/bin/python ode/utils/convert_torch_zip_ckpt_to_safetensors.py --input '$SOURCE_CKPT' --output '$CONVERTED_CKPT'; \
    fi && \
    exec .venv/bin/python visual_multi_compare/infer_5prompt_ode.py \
    --samples-json visual_multi_compare/sampled_8.json \
    --ode-checkpoint-path '$CONVERTED_CKPT' \
    --distilled-checkpoint-path model/ltx-2.3-22b-distilled.safetensors \
    --spatial-upsampler-path model/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
    --gemma-root model/gemma \
    --tag run1_step1000 \
    --output-root $OUTPUT_ROOT"
