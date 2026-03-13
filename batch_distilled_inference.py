"""
Batch inference script for LTX-2 distilled model.
Loads the model once, then iterates through prompts from a CSV file.
"""

import csv
import logging
import time
from pathlib import Path

import torch

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import resolve_path
from ltx_pipelines.utils.constants import detect_params
from ltx_pipelines.utils.media_io import encode_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────
CSV_PATH = "/mnt/shared-storage-user/worldmodel-shared/wgx/LTX-2/datagen/ltx_prompts_100.csv"
OUTPUT_DIR = "/mnt/shared-storage-user/worldmodel-shared/wgx/LTX-2/batch_outputs"

DISTILLED_CKPT = "/mnt/shared-storage-user/worldmodel-shared/wgx/ltx-2.0_old_1/models/ltx2/ltx-2.3-22b-distilled.safetensors"
SPATIAL_UPSAMPLER = "/mnt/shared-storage-user/worldmodel-shared/wgx/ltx-2.0_old_1/models/ltx2/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
GEMMA_ROOT = "/mnt/shared-storage-user/worldmodel-shared/wgx/ltx-2.0_old_1/models/gemma3"

NUM_PROMPTS = 30
SEED = 42
# Height/width are the final output resolution (2-stage pipeline does half res -> full res)
# Defaults from LTX_2_3_PARAMS: stage_2_height=768, stage_2_width=1152 (or similar)
# We use detect_params to get the right defaults.
# ─────────────────────────────────────────────────────────────────────────────


def load_prompts(csv_path: str, num_prompts: int) -> list[str]:
    """Load first N prompts from the CSV file."""
    prompts = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num_prompts:
                break
            prompts.append(row["text_prompt"])
    logger.info(f"Loaded {len(prompts)} prompts from {csv_path}")
    return prompts


@torch.inference_mode()
def main() -> None:
    # Detect model params
    checkpoint_path = resolve_path(DISTILLED_CKPT)
    params = detect_params(checkpoint_path)

    height = params.stage_2_height
    width = params.stage_2_width
    num_frames = params.num_frames
    frame_rate = params.frame_rate

    logger.info(f"Resolution: {width}x{height}, frames: {num_frames}, fps: {frame_rate}")

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts = load_prompts(CSV_PATH, NUM_PROMPTS)

    # Build pipeline (loads model once)
    logger.info("Loading distilled pipeline...")
    t0 = time.time()
    pipeline = DistilledPipeline(
        distilled_checkpoint_path=checkpoint_path,
        spatial_upsampler_path=resolve_path(SPATIAL_UPSAMPLER),
        gemma_root=resolve_path(GEMMA_ROOT),
        loras=(),
        quantization=None,
    )
    logger.info(f"Pipeline loaded in {time.time() - t0:.1f}s")

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

    # Iterate through prompts
    for idx, prompt in enumerate(prompts):
        output_path = str(output_dir / f"prompt_{idx:03d}.mp4")
        logger.info(f"[{idx + 1}/{len(prompts)}] Generating: {prompt[:80]}...")

        t1 = time.time()
        try:
            video, audio = pipeline(
                prompt=prompt,
                seed=SEED,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=[],
                tiling_config=tiling_config,
                enhance_prompt=False,
            )

            encode_video(
                video=video,
                fps=frame_rate,
                audio=audio,
                output_path=output_path,
                video_chunks_number=video_chunks_number,
            )
            elapsed = time.time() - t1
            logger.info(f"[{idx + 1}/{len(prompts)}] Saved to {output_path} ({elapsed:.1f}s)")

        except Exception as e:
            logger.error(f"[{idx + 1}/{len(prompts)}] Failed: {e}", exc_info=True)
            continue

    logger.info("Batch inference complete!")


if __name__ == "__main__":
    main()
