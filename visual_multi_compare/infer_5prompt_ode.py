#!/usr/bin/env python3
# ruff: noqa: T201
"""Batch ODE inference using CausalDistilledODEPipeline with per-sample seed alignment.

Reads ``sampled_8.json`` and runs inference for each sample using its own
``noise_seeds.stage2.video`` seed.

Usage::

    python visual_multi_compare/infer_5prompt_ode.py \
        --samples-json visual_multi_compare/sampled_8.json \
        --ode-checkpoint-path /path/to/ode_checkpoint.safetensors \
        --distilled-checkpoint-path model/ltx-2.3-22b-distilled.safetensors \
        --spatial-upsampler-path model/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
        --gemma-root model/gemma \
        --tag ode_eval \
        --output-root visual_multi_compare/outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.utils.media_io import encode_video
from ltx_trainer.causal_distilled_pipeline import CausalDistilledODEPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed_utils import resolve_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch ODE inference with per-sample seed")
    p.add_argument("--samples-json", type=Path, required=True)
    p.add_argument("--ode-checkpoint-path", type=str, required=True)
    p.add_argument("--distilled-checkpoint-path", type=str, required=True)
    p.add_argument("--spatial-upsampler-path", type=str, required=True)
    p.add_argument("--gemma-root", type=str, required=True)
    p.add_argument("--tag", type=str, default="ode")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1536)
    p.add_argument("--num-frames", type=int, default=121)
    p.add_argument("--frame-rate", type=float, default=24.0)
    p.add_argument("--block-size", type=int, default=6)
    p.add_argument("--audio-boundary-mode", type=str, default="left")
    p.add_argument("--local-attn-size", type=int, default=-1)
    p.add_argument("--audio-sink-token-count", type=int, default=2)
    p.add_argument("--audio-sink-identity-rope", action="store_true", default=True)
    return p


@torch.inference_mode()
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    with args.samples_json.open(encoding="utf-8") as fh:
        samples = json.load(fh)

    output_dir = args.output_root / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = CausalDistilledODEPipeline(
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        ode_checkpoint_path=args.ode_checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        block_size=args.block_size,
        audio_boundary_mode=args.audio_boundary_mode,
        local_attn_size=args.local_attn_size,
        audio_sink_token_count=args.audio_sink_token_count,
        audio_sink_identity_rope=args.audio_sink_identity_rope,
    )

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

    for i, sample in enumerate(samples):
        idx = int(sample["index"])
        prompt = str(sample["prompt"])
        seed = resolve_seed(sample, args.seed)

        print(f"[{args.tag}] [{i + 1}/{len(samples)}] idx={idx:05d} seed={seed} prompt={prompt[:70]}...")

        video, audio = pipeline(
            prompt=prompt,
            seed=seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            tiling_config=tiling_config,
            enhance_prompt=False,
        )

        out_path = output_dir / f"{args.tag}_{idx:05d}.mp4"
        encode_video(
            video=video,
            fps=args.frame_rate,
            audio=audio,
            output_path=out_path,
            video_chunks_number=video_chunks_number,
        )
        print(f"  Saved: {out_path}")

    print(f"[{args.tag}] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
