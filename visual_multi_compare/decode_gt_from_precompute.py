#!/usr/bin/env python3
# ruff: noqa: T201
"""Decode ground-truth video+audio from precomputed trajectories.

Reads ``sampled_8.json``, loads the corresponding ``.pt`` files from the
precompute root, decodes with VAE, and writes ``gt_*.mp4`` with muxed audio.

Usage::

    python visual_multi_compare/decode_gt_from_precompute.py \
        --samples-json visual_multi_compare/sampled_8.json \
        --precompute-root /path/to/ode_precompute_stage2_... \
        --checkpoint-path model/ltx-2.3-22b-distilled.safetensors \
        --output-dir visual_multi_compare/outputs/gt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.media_io import encode_video


def extract_video_latent(payload: dict[str, Any]) -> torch.Tensor:
    latent = payload.get("ode_target_latents")
    if latent is None:
        latent = payload["ode_video_trajectory"][-1]
    if latent.ndim == 4:
        latent = latent.unsqueeze(0)
    return latent


def extract_audio_latent(payload: dict[str, Any]) -> torch.Tensor:
    latent = payload.get("ode_target_latents")
    if latent is None:
        latent = payload["ode_audio_trajectory"][-1]
    if latent.ndim == 3:
        latent = latent.unsqueeze(0)
    return latent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Decode GT from precomputed trajectories")
    p.add_argument("--samples-json", type=Path, required=True)
    p.add_argument("--precompute-root", type=Path, required=True)
    p.add_argument("--checkpoint-path", type=str, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--frame-rate", type=float, default=24.0)
    p.add_argument("--spatial-upsampler-path", type=str, default=None)
    p.add_argument("--gemma-root", type=str, default=None)
    return p


@torch.inference_mode()
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    with args.samples_json.open(encoding="utf-8") as fh:
        samples = json.load(fh)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tiling_config = TilingConfig.default()

    # Build a minimal ModelLedger — only need video_decoder, audio_decoder, vocoder
    gemma_root = args.gemma_root
    if gemma_root is None:
        gemma_root = str(Path(args.checkpoint_path).resolve().parent / "gemma")
    spatial_upsampler = args.spatial_upsampler_path
    if spatial_upsampler is None:
        ckpt_dir = Path(args.checkpoint_path).resolve().parent
        candidate = ckpt_dir / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        spatial_upsampler = str(candidate) if candidate.exists() else ""

    ledger = ModelLedger(
        dtype=dtype,
        device=device,
        checkpoint_path=args.checkpoint_path,
        spatial_upsampler_path=spatial_upsampler,
        gemma_root_path=gemma_root,
        loras=(),
    )

    video_traj_dir = args.precompute_root / ".precomputed" / "ode_video_trajectories"
    audio_traj_dir = args.precompute_root / ".precomputed" / "ode_audio_trajectories"

    for i, sample in enumerate(samples):
        idx = int(sample["index"])
        prompt = str(sample["prompt"])
        print(f"[gt] [{i + 1}/{len(samples)}] idx={idx:05d} prompt={prompt[:60]}...")

        video_payload = torch.load(video_traj_dir / f"{idx:05d}.pt", map_location="cpu", weights_only=False)
        audio_payload = torch.load(audio_traj_dir / f"{idx:05d}.pt", map_location="cpu", weights_only=False)

        video_latent = extract_video_latent(video_payload).to(device=device, dtype=dtype)
        audio_latent = extract_audio_latent(audio_payload).to(device=device, dtype=dtype)

        generator = torch.Generator(device=device).manual_seed(42)
        decoded_video = vae_decode_video(video_latent, ledger.video_decoder(), tiling_config, generator)
        decoded_audio = vae_decode_audio(audio_latent, ledger.audio_decoder(), ledger.vocoder())

        num_frames_pixel = (video_latent.shape[2] - 1) * 8 + 1
        video_chunks_number = get_video_chunks_number(num_frames_pixel, tiling_config)

        out_path = args.output_dir / f"gt_{idx:05d}.mp4"
        encode_video(
            video=decoded_video,
            fps=args.frame_rate,
            audio=decoded_audio,
            output_path=out_path,
            video_chunks_number=video_chunks_number,
        )
        print(f"  Saved: {out_path}")

        # Free memory between samples
        del video_payload, audio_payload, video_latent, audio_latent, decoded_video, decoded_audio
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[gt] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
