#!/usr/bin/env python3
# ruff: noqa: T201
"""Decode precomputed GT and exact distilled replay outputs for the same samples."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ode.gen_ode_data_distilled import cleanup_memory, encode_prompt, generate_sample


def resolve_config_path(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def load_generation_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    for key in (
        "checkpoint_path",
        "gemma_root",
        "spatial_upsampler_path",
        "caption_path",
        "output_dir",
        "prompt_ctx_cache_dir",
    ):
        if key in cfg:
            cfg[key] = resolve_config_path(cfg[key])
    return cfg


def extract_video_latent(video_payload: dict[str, Any]) -> torch.Tensor:
    latent = video_payload.get("ode_target_latents")
    if latent is None:
        latent = video_payload["ode_video_trajectory"][-1]
    if latent.ndim == 4:
        latent = latent.unsqueeze(0)
    return latent


def extract_audio_latent(audio_payload: dict[str, Any]) -> torch.Tensor:
    latent = audio_payload.get("ode_target_latents")
    if latent is None:
        latent = audio_payload["ode_audio_trajectory"][-1]
    if latent.ndim == 3:
        latent = latent.unsqueeze(0)
    return latent


def decode_and_save(
    *,
    video_latent: torch.Tensor,
    audio_latent: torch.Tensor,
    output_path: Path,
    frame_rate: float,
    ledger: ModelLedger,
    device: torch.device,
    dtype: torch.dtype,
    tiling_config: TilingConfig,
    decode_seed: int,
) -> None:
    generator = torch.Generator(device=device).manual_seed(int(decode_seed))
    video_latent = video_latent.to(device=device, dtype=dtype)
    audio_latent = audio_latent.to(device=device, dtype=dtype)
    decoded_video = vae_decode_video(video_latent, ledger.video_decoder(), tiling_config, generator)
    decoded_audio = vae_decode_audio(audio_latent, ledger.audio_decoder(), ledger.vocoder())
    num_frames_pixel = (video_latent.shape[2] - 1) * 8 + 1
    video_chunks_number = get_video_chunks_number(num_frames_pixel, tiling_config)
    encode_video(
        video=decoded_video,
        fps=frame_rate,
        audio=decoded_audio,
        output_path=output_path,
        video_chunks_number=video_chunks_number,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-json", type=Path, required=True)
    parser.add_argument("--precompute-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "ode" / "configs" / "gen_ode_data_distilled.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


@torch.inference_mode()
def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    with args.samples_json.open(encoding="utf-8") as handle:
        samples = json.load(handle)

    cfg = load_generation_config(args.config)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tiling_config = TilingConfig.default()

    ledger = ModelLedger(
        dtype=dtype,
        device=device,
        checkpoint_path=cfg["checkpoint_path"],
        spatial_upsampler_path=cfg["spatial_upsampler_path"],
        gemma_root_path=cfg["gemma_root"],
        loras=(),
    )
    components = PipelineComponents(dtype=dtype, device=device)
    transformer = ledger.transformer()

    pt_output_dir = args.output_root / "pt_decode"
    replay_output_dir = args.output_root / "replay"
    pt_output_dir.mkdir(parents=True, exist_ok=True)
    replay_output_dir.mkdir(parents=True, exist_ok=True)

    video_traj_dir = args.precompute_root / ".precomputed" / "ode_video_trajectories"
    audio_traj_dir = args.precompute_root / ".precomputed" / "ode_audio_trajectories"

    for sample in samples:
        idx = int(sample["index"])
        prompt = str(sample["prompt"])
        print(f"[exact-replay] idx={idx:05d} prompt={prompt[:60]}...")

        video_payload = torch.load(video_traj_dir / f"{idx:05d}.pt", map_location="cpu", weights_only=False)
        audio_payload = torch.load(audio_traj_dir / f"{idx:05d}.pt", map_location="cpu", weights_only=False)
        decode_and_save(
            video_latent=extract_video_latent(video_payload),
            audio_latent=extract_audio_latent(audio_payload),
            output_path=pt_output_dir / f"pt_decode_{idx:05d}.mp4",
            frame_rate=float(video_payload.get("fps", cfg["frame_rate"])),
            ledger=ledger,
            device=device,
            dtype=dtype,
            tiling_config=tiling_config,
            decode_seed=int(cfg["seed"]),
        )

        ctx_pos = encode_prompt(prompt, ledger, device)
        replay_data = generate_sample(
            prompt=prompt,
            global_idx=idx,
            ctx_pos=ctx_pos,
            transformer=transformer,
            model_ledger=ledger,
            components=components,
            cfg=cfg,
            device=device,
            dtype=dtype,
        )
        decode_and_save(
            video_latent=replay_data["stage2_video_traj"][-1:].contiguous(),
            audio_latent=replay_data["stage2_audio_traj"][-1:].contiguous(),
            output_path=replay_output_dir / f"replay_{idx:05d}.mp4",
            frame_rate=float(cfg["frame_rate"]),
            ledger=ledger,
            device=device,
            dtype=dtype,
            tiling_config=tiling_config,
            decode_seed=int(cfg["seed"]),
        )
        cleanup_memory()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
