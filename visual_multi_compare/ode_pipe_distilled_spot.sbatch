#!/usr/bin/env bash
#SBATCH -J vmc-ode-pipe-dist
#SBATCH -p Omnilab
#SBATCH --quotatype=spot
#SBATCH -N 1 -n 1 --cpus-per-task=8 --gres=gpu:1
#SBATCH -o /mnt/petrelfs/wanggongxuan/workspace/myltx/visual_multi_compare/log/ode_pipe_distilled-%j.out
#SBATCH -e /mnt/petrelfs/wanggongxuan/workspace/myltx/visual_multi_compare/log/ode_pipe_distilled-%j.err
set -euo pipefail

CONTAINER_IMAGE="/mnt/petrelfs/wanggongxuan/node-22.sif"
CONTAINER_BIND="/mnt/petrelfs/wanggongxuan/workspace:/workspace"
CONTAINER_ROOT="/workspace/myltx"
CONTAINER_PYTHONPATH="/workspace/myltx:/workspace/myltx/packages/ltx-trainer/src:/workspace/myltx/packages/ltx-core/src:/workspace/myltx/packages/ltx-pipelines/src"
EXTRACTED_CKPT="/workspace/ode_ckpt/_distilled_transformer_only_v3.safetensors"
OUTPUT_ROOT="/workspace/myltx/visual_multi_compare/outputs"

srun -N1 -n1 --gres=gpu:1 --cpus-per-task=8 \
  /usr/bin/env PYTHONPATH="$CONTAINER_PYTHONPATH" PYTHONUNBUFFERED=1 \
  /usr/bin/apptainer exec --nv --bind "$CONTAINER_BIND" "$CONTAINER_IMAGE" \
  /bin/bash -lc "cd $CONTAINER_ROOT && \
    if [ ! -f '$EXTRACTED_CKPT' ]; then \
      .venv/bin/python -c \"
from safetensors.torch import load_file, save_file
sd = load_file('model/ltx-2.3-22b-distilled.safetensors', device='cpu')
prefix = 'model.diffusion_model.'
t = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
skip = ('audio_embeddings_connector.', 'video_embeddings_connector.', 'caption_projection.', 'audio_caption_projection.', 'feature_extractor.')
t = {k: v for k, v in t.items() if not k.startswith(skip)}
print(f'Extracted {len(t)} keys')
save_file(t, '$EXTRACTED_CKPT')
\"; \
    fi && \
    exec .venv/bin/python visual_multi_compare/infer_5prompt_ode.py \
    --samples-json visual_multi_compare/sampled_8.json \
    --ode-checkpoint-path '$EXTRACTED_CKPT' \
    --distilled-checkpoint-path model/ltx-2.3-22b-distilled.safetensors \
    --spatial-upsampler-path model/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
    --gemma-root model/gemma \
    --tag ode_pipe_distilled \
    --output-root $OUTPUT_ROOT"
