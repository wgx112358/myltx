#!/bin/bash

# Configuration for LTX-2 distilled 2-stage inference
# Please run this script on your GPU machine.

PROJECT_DIR="/mnt/shared-storage-user/worldmodel-shared/wgx/LTX-2"

# Model paths on shared storage
DISTILLED_CKPT="/mnt/shared-storage-user/worldmodel-shared/wgx/ltx-2.0_old_1/models/ltx2/ltx-2.3-22b-distilled.safetensors"
SPATIAL_UPSAMPLER="/mnt/shared-storage-user/worldmodel-shared/wgx/ltx-2.0_old_1/models/ltx2/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
GEMMA_ROOT="/mnt/shared-storage-user/worldmodel-shared/wgx/ltx-2.0_old_1/models/gemma3"

# Inference parameters
PROMPT="thunderstorm,heavy rain"
OUTPUT_PATH="distilled_output.mp4"

echo "Starting Distilled Inference..."
echo "Project Dir: $PROJECT_DIR"
echo "Prompt: $PROMPT"
echo "Output: $OUTPUT_PATH"

cd "$PROJECT_DIR"

python packages/ltx-pipelines/src/ltx_pipelines/distilled.py \
    --distilled-checkpoint-path "$DISTILLED_CKPT" \
    --spatial-upsampler-path "$SPATIAL_UPSAMPLER" \
    --gemma-root "$GEMMA_ROOT" \
    --prompt "$PROMPT" \
    --output-path "$OUTPUT_PATH"

echo "Inference complete! Output saved to $PROJECT_DIR/$OUTPUT_PATH"
