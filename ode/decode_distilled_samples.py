import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, "packages/ltx-core/src")
sys.path.insert(0, "packages/ltx-pipelines/src")

from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.video_vae import TilingConfig, decode_video as vae_decode_video, get_video_chunks_number
from ltx_core.types import VideoLatentShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.media_io import encode_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly decode distilled ODE latent samples into videos.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("ode/data_distilled"),
        help="Directory containing distilled ODE .pt samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ode/distilled_example_videos"),
        help="Directory to save decoded mp4 files.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("model/ltx-2.3-22b-distilled.safetensors"),
        help="Distilled checkpoint used to build the VAE decoder.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="How many samples to decode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Output frame rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for decoding.",
    )
    return parser.parse_args()


def list_samples(input_dir: Path) -> list[Path]:
    samples = sorted(input_dir.glob("*.pt"))
    if not samples:
        raise FileNotFoundError(f"No .pt samples found under {input_dir}")
    return samples


def choose_samples(samples: list[Path], num_samples: int, seed: int) -> list[Path]:
    if num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if num_samples > len(samples):
        raise ValueError(f"Requested {num_samples} samples, but only found {len(samples)}")
    rng = random.Random(seed)
    return sorted(rng.sample(samples, num_samples), key=lambda path: path.name)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint_path = args.checkpoint_path.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    samples = choose_samples(list_samples(input_dir), args.num_samples, args.seed)
    print(f"Selected {len(samples)} samples from {input_dir}")
    selected_list_path = output_dir / "selected_latents.txt"
    selected_list_path.write_text("".join(f"{sample_path.name}\n" for sample_path in samples), encoding="utf-8")
    for sample_path in samples:
        print(sample_path.name)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tiling_config = TilingConfig.default()

    ledger = ModelLedger(
        dtype=dtype,
        device=device,
        checkpoint_path=str(checkpoint_path),
    )
    decoder = ledger.video_decoder()
    audio_decoder = ledger.audio_decoder()
    vocoder = ledger.vocoder()

    try:
        with torch.inference_mode():
            for index, sample_path in enumerate(samples, start=1):
                payload = torch.load(sample_path, map_location="cpu")
                video_latent = payload["stage2_video_traj"][-1].unsqueeze(0).to(device=device, dtype=dtype)
                audio_latent = payload["stage2_audio_traj"][-1].unsqueeze(0).to(device=device, dtype=dtype)
                output_path = output_dir / f"{sample_path.stem}.mp4"
                video = vae_decode_video(video_latent, decoder, tiling_config=tiling_config)
                audio = vae_decode_audio(audio_latent, audio_decoder, vocoder)
                num_frames = VideoLatentShape.from_torch_shape(video_latent.shape).upscale().frames
                video_chunks = get_video_chunks_number(num_frames=num_frames, tiling_config=tiling_config)
                encode_video(
                    video=video,
                    fps=args.fps,
                    audio=audio,
                    output_path=str(output_path),
                    video_chunks_number=video_chunks,
                )
                print(f"[{index}/{len(samples)}] saved {output_path.name}")
    finally:
        decoder.to("cpu")
        audio_decoder.to("cpu")
        vocoder.to("cpu")
        del decoder
        del audio_decoder
        del vocoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
