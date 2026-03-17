#!/bin/bash

# =============================================================================
# LTX 2.3 Non-Distilled (Full Model) Two-Stage Inference Script
#
# Stage 1: 半分辨率生成 (512x768) + CFG引导, 使用完整非蒸馏模型
# Stage 2: 2x空间上采样 (1024x1536) + 蒸馏LoRA精炼
#
# 这是官方推荐的生产级推理管线 (TI2VidTwoStagesPipeline)
# =============================================================================

set -e

# ---- Paths ----
PROJECT_DIR="/inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx"
CKPT="${PROJECT_DIR}/model/ltx-2.3-22b-dev.safetensors"
GEMMA_ROOT="${PROJECT_DIR}/model/gemma"
DISTILLED_LORA="${PROJECT_DIR}/model/ltx-2.3-22b-distilled-lora-384.safetensors"
SPATIAL_UPSAMPLER="${PROJECT_DIR}/model/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

# ---- Generation Parameters ----
PROMPT="A cinematic thunderstorm with heavy rain pouring down on a dark city street at night, neon lights reflecting off wet pavement"
NEGATIVE_PROMPT="worst quality, inconsistent motion, blurry, jittery, distorted"
OUTPUT_PATH="${PROJECT_DIR}/outputs/nondistilled_two_stage_output.mp4"

# 最终输出分辨率 (Stage 1 自动为此值的一半)
HEIGHT=1024           # Stage 1 -> 512, Stage 2 -> 1024. 必须能被64整除
WIDTH=1536            # Stage 1 -> 768, Stage 2 -> 1536. 必须能被64整除
NUM_FRAMES=121        # 必须满足 8k+1 (e.g. 9,17,...,97,121,129). 121帧 ~ 5s @24fps
FRAME_RATE=24.0
NUM_STEPS=30          # LTX 2.3 默认30步去噪 (Stage 1)
SEED=42

# ---- Distilled LoRA Strength (Stage 2) ----
DISTILLED_LORA_STRENGTH=1.0

# ---- Guidance Scales (Stage 1) ----
VIDEO_CFG=3.0         # Video CFG: 越高越贴合prompt, 典型 2.0-5.0
VIDEO_STG=1.0         # 时空引导: 提高时序一致性, 典型 0.5-1.5
VIDEO_RESCALE=0.7     # 防止过饱和, 典型 0.5-0.7
A2V_GUIDANCE=3.0      # 音视频同步引导

AUDIO_CFG=7.0         # Audio CFG
AUDIO_STG=1.0
AUDIO_RESCALE=0.7
V2A_GUIDANCE=3.0      # 视频到音频同步引导

# ---- Optional: FP8 Quantization (uncomment to reduce VRAM) ----
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# QUANT_ARGS="--quantization fp8-cast"
QUANT_ARGS=""

# ---- Optional: Image Conditioning (uncomment and set path) ----
# IMAGE_ARGS="--image /path/to/image.jpg 0 1.0"
IMAGE_ARGS=""

# ---- Optional: Extra LoRA (uncomment and set path) ----
# LORA_ARGS="--lora /path/to/lora.safetensors 1.0"
LORA_ARGS=""

# =============================================================================

# Create output directory
mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "============================================="
echo "LTX 2.3 Non-Distilled Two-Stage Inference"
echo "============================================="
echo "Checkpoint        : ${CKPT}"
echo "Distilled LoRA    : ${DISTILLED_LORA}"
echo "Spatial Upsampler : ${SPATIAL_UPSAMPLER}"
echo "Gemma Root        : ${GEMMA_ROOT}"
echo "Output Resolution : ${WIDTH}x${HEIGHT} (Stage1: $((WIDTH/2))x$((HEIGHT/2)))"
echo "Frames            : ${NUM_FRAMES} @ ${FRAME_RATE}fps (~$(echo "scale=1; ${NUM_FRAMES}/${FRAME_RATE}" | bc)s)"
echo "Steps (Stage 1)   : ${NUM_STEPS}"
echo "Seed              : ${SEED}"
echo "Prompt            : ${PROMPT}"
echo "Output            : ${OUTPUT_PATH}"
echo "============================================="

cd "${PROJECT_DIR}"

python packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py \
    --checkpoint-path "${CKPT}" \
    --distilled-lora "${DISTILLED_LORA}" ${DISTILLED_LORA_STRENGTH} \
    --spatial-upsampler-path "${SPATIAL_UPSAMPLER}" \
    --gemma-root "${GEMMA_ROOT}" \
    --prompt "${PROMPT}" \
    --negative-prompt "${NEGATIVE_PROMPT}" \
    --output-path "${OUTPUT_PATH}" \
    --height ${HEIGHT} \
    --width ${WIDTH} \
    --num-frames ${NUM_FRAMES} \
    --frame-rate ${FRAME_RATE} \
    --num-inference-steps ${NUM_STEPS} \
    --seed ${SEED} \
    --video-cfg-guidance-scale ${VIDEO_CFG} \
    --video-stg-guidance-scale ${VIDEO_STG} \
    --video-rescale-scale ${VIDEO_RESCALE} \
    --a2v-guidance-scale ${A2V_GUIDANCE} \
    --audio-cfg-guidance-scale ${AUDIO_CFG} \
    --audio-stg-guidance-scale ${AUDIO_STG} \
    --audio-rescale-scale ${AUDIO_RESCALE} \
    --v2a-guidance-scale ${V2A_GUIDANCE} \
    ${QUANT_ARGS} \
    ${IMAGE_ARGS} \
    ${LORA_ARGS}

echo ""
echo "Done! Output saved to: ${OUTPUT_PATH}"
