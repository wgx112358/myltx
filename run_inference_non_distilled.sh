#!/bin/bash

# Configuration for LTX-2 non-distilled inference (One Stage)
# Please run this script on your GPU machine.

PROJECT_DIR="/mnt/shared-storage-user/worldmodel-shared/wgx/LTX-2"

# Model paths on shared storage
CKPT="/mnt/shared-storage-user/worldmodel-shared/wgx/ltx-2.0_old_1/models/ltx-2.3-22b-dev.safetensors"
GEMMA_ROOT="/mnt/shared-storage-user/worldmodel-shared/wgx/ltx-2.0_old_1/models/gemma3"

# Inference parameters
PROMPT="A cinematic thunderstorm with heavy rain"
OUTPUT_PATH="nondistilled_output.mp4"

echo "Starting Non-Distilled Inference (ti2vid_one_stage)..."
echo "Project Dir: $PROJECT_DIR"
echo "Prompt: $PROMPT"
echo "Output: $OUTPUT_PATH"

cd "$PROJECT_DIR"

python packages/ltx-pipelines/src/ltx_pipelines/ti2vid_one_stage.py \
    --checkpoint-path "$CKPT" \
    --gemma-root "$GEMMA_ROOT" \
    --prompt "$PROMPT" \
    --output-path "$OUTPUT_PATH"

echo "Inference complete! Output saved to $PROJECT_DIR/$OUTPUT_PATH"
