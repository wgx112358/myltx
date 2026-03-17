#!/usr/bin/env python3

"""
将 myltx/ode/data_distilled 下的 ODE 轨迹样本转换为 ltx-trainer 可直接读取的
.precomputed 数据目录。

默认行为：
1. 读取 ode/configs/gen_ode_data_distilled.yaml，自动拿到 input/model/gemma/fps 默认值
2. 从每个样本中选择 stage2 的最后一个轨迹状态（clean latent）
3. 写出:
   - <output_root>/.precomputed/latents/*.pt
   - <output_root>/.precomputed/audio_latents/*.pt
   - <output_root>/.precomputed/conditions/*.pt

输出格式会对齐 myltx/packages/ltx-trainer 的 PrecomputedDataset。
当使用 `--export-mode ode_regression` 时，还会把当前 step 的 clean target 和 sigma
一并写入 payload，供 `ltx-trainer` 的 ODE regression strategy 直接训练。

示例：
    cd /inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx

    # 只转换 latent，不生成文本条件
    python ode/convert_ode_pt_to_precomputed.py \
        --limit 10 \
        --no-write-conditions

    # 转换 stage1 第 0 步 noisy latent，并写出 conditions
    python ode/convert_ode_pt_to_precomputed.py \
        --stage stage1 \
        --trajectory-step 0 \
        --output-dir ode/data_distilled_stage1_step0

    # 导出 ODE regression 训练数据：把每条轨迹展开为多个 noisy->clean 训练样本
    python ode/convert_ode_pt_to_precomputed.py \
        --export-mode ode_regression \
        --trajectory-step all_non_last \
        --output-dir ode/data_distilled_stage2_ode
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, "packages/ltx-core/src")
sys.path.insert(0, "packages/ltx-pipelines/src")
sys.path.insert(0, "packages/ltx-trainer/src")

from ltx_trainer.model_loader import load_embeddings_processor, load_text_encoder

LOGGER = logging.getLogger("convert_ode_pt_to_precomputed")
DEFAULT_CONFIG_PATH = Path("ode/configs/gen_ode_data_distilled.yaml")
PRECOMPUTED_DIR_NAME = ".precomputed"
SUPPORTED_STAGES = ("stage1", "stage2")
EXPORT_MODES = ("standard", "ode_regression")
VIDEO_TEMPORAL_COMPRESSION = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert distilled ODE .pt samples into myltx .precomputed training data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to distilled ODE yaml config used to provide project-specific defaults.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing distilled ODE .pt samples. Defaults to config.output_dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output dataset root. The script writes into <output-dir>/.precomputed unless the path "
            "already ends with .precomputed."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=SUPPORTED_STAGES,
        default="stage2",
        help="Which ODE stage to export as training latents.",
    )
    parser.add_argument(
        "--trajectory-step",
        type=str,
        default="last",
        help=(
            'Which trajectory state(s) to export: "first", "last", "all", "all_non_last", '
            'or an integer index.'
        ),
    )
    parser.add_argument(
        "--export-mode",
        choices=EXPORT_MODES,
        default="standard",
        help=(
            '"standard" keeps the selected latent as training input; '
            '"ode_regression" additionally stores the clean target latent and sigma.'
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS metadata written into exported video latent files. Defaults to config.frame_rate.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="LTX checkpoint path, used to build the embeddings processor. Defaults to config.checkpoint_path.",
    )
    parser.add_argument(
        "--text-encoder-path",
        type=Path,
        default=None,
        help="Gemma directory path, used to encode prompts. Defaults to config.gemma_root.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for prompt encoding.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Compute dtype used when loading the text encoder / embeddings processor.",
    )
    parser.add_argument(
        "--load-text-encoder-in-8bit",
        action="store_true",
        help="Load Gemma in 8-bit to reduce VRAM during condition generation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Only process the first N .pt files after sorting. -1 means all files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files. Default behavior is resume-friendly skip-existing.",
    )
    parser.add_argument(
        "--no-write-audio",
        action="store_true",
        help="Skip exporting audio_latents.",
    )
    parser.add_argument(
        "--no-write-conditions",
        action="store_true",
        help="Skip exporting prompt embedding conditions.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input-dir for .pt files. Default only scans the top level.",
    )
    parser.add_argument(
        "--manifest-name",
        type=str,
        default="conversion_manifest.json",
        help="Filename used for the summary manifest written under output root.",
    )
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        LOGGER.warning("Config file does not exist, continuing without project defaults: %s", config_path)
        return {}

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must deserialize into a dict, got {type(data)} from {config_path}")

    return data


def resolve_output_root(
    output_dir: Path | None,
    input_dir: Path,
    stage: str,
    trajectory_step: str,
    export_mode: str,
) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve()

    step_label = trajectory_step.strip().lower()
    default_name = f"{input_dir.name}_{stage}_{step_label}"
    if export_mode != "standard":
        default_name = f"{default_name}_{export_mode}"
    return (input_dir.parent / default_name).resolve()


def resolve_precomputed_root(output_root: Path) -> Path:
    if output_root.name == PRECOMPUTED_DIR_NAME:
        return output_root
    return output_root / PRECOMPUTED_DIR_NAME


def parse_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping[dtype_name]


def discover_input_files(input_dir: Path, recursive: bool, limit: int) -> list[Path]:
    if recursive:
        files = sorted(input_dir.glob("**/*.pt"))
    else:
        files = sorted(input_dir.glob("*.pt"))

    if limit >= 0:
        files = files[:limit]

    return files


def resolve_step_index(step_spec: str, trajectory_length: int) -> int:
    normalized = step_spec.strip().lower()
    if normalized == "first":
        return 0
    if normalized == "last":
        return trajectory_length - 1

    try:
        raw_index = int(normalized)
    except ValueError as exc:
        raise ValueError(f'Invalid --trajectory-step "{step_spec}". Use first/last or an integer.') from exc

    if raw_index < 0:
        raw_index += trajectory_length

    if raw_index < 0 or raw_index >= trajectory_length:
        raise IndexError(f"Trajectory step {step_spec} is out of range for trajectory length {trajectory_length}.")

    return raw_index


def resolve_step_indices(step_spec: str, trajectory_length: int) -> list[int]:
    normalized = step_spec.strip().lower()
    if normalized == "all":
        return list(range(trajectory_length))
    if normalized == "all_non_last":
        return list(range(max(trajectory_length - 1, 0)))
    return [resolve_step_index(step_spec, trajectory_length)]


def validate_sample(sample: dict[str, Any], stage: str, require_audio: bool) -> None:
    required_keys = {
        "prompt",
        f"{stage}_video_traj",
        f"{stage}_sigmas",
    }
    if require_audio:
        required_keys.add(f"{stage}_audio_traj")

    missing = sorted(key for key in required_keys if key not in sample)
    if missing:
        raise KeyError(f"Missing keys: {missing}")

    if not isinstance(sample["prompt"], str):
        raise TypeError(f'Expected "prompt" to be str, got {type(sample["prompt"])}')

    video_traj = sample[f"{stage}_video_traj"]
    if not isinstance(video_traj, torch.Tensor) or video_traj.ndim != 5:
        raise TypeError(
            f"Expected {stage}_video_traj to be a 5D tensor, got {type(video_traj)} "
            f"/ ndim={getattr(video_traj, 'ndim', None)}"
        )

    sigmas = sample[f"{stage}_sigmas"]
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1:
        raise TypeError(
            f"Expected {stage}_sigmas to be a 1D tensor, got {type(sigmas)} "
            f"/ ndim={getattr(sigmas, 'ndim', None)}"
        )

    if video_traj.shape[0] != sigmas.shape[0]:
        raise ValueError(
            f"{stage} video trajectory length ({video_traj.shape[0]}) does not match sigma length ({sigmas.shape[0]})."
        )

    if require_audio:
        audio_traj = sample[f"{stage}_audio_traj"]
        if not isinstance(audio_traj, torch.Tensor) or audio_traj.ndim != 4:
            raise TypeError(
                f"Expected {stage}_audio_traj to be a 4D tensor, got {type(audio_traj)} "
                f"/ ndim={getattr(audio_traj, 'ndim', None)}"
            )
        if audio_traj.shape[0] != sigmas.shape[0]:
            raise ValueError(
                f"{stage} audio trajectory length ({audio_traj.shape[0]}) "
                f"does not match sigma length ({sigmas.shape[0]})."
            )


def should_write_file(path: Path | None, overwrite: bool) -> bool:
    if path is None:
        return False
    if overwrite:
        return True
    return not path.exists()


def build_relative_output_path(relative_path: Path, step_index: int, multiple_steps: bool) -> Path:
    if not multiple_steps:
        return relative_path
    return relative_path.with_name(f"{relative_path.stem}__step_{step_index:03d}{relative_path.suffix}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_latent_payload(
    video_latent: torch.Tensor,
    fps: float,
    *,
    ode_target_latent: torch.Tensor | None = None,
    ode_sigma: float | None = None,
    ode_step_index: int | None = None,
    ode_clean_step_index: int | None = None,
    ode_noise_seeds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if video_latent.ndim != 4:
        raise ValueError(f"Expected selected video latent to be 4D [C, F, H, W], got {tuple(video_latent.shape)}")

    payload = {
        "latents": video_latent.cpu().contiguous(),
        "num_frames": int(video_latent.shape[1]),
        "height": int(video_latent.shape[2]),
        "width": int(video_latent.shape[3]),
        "fps": float(fps),
    }
    if ode_target_latent is not None:
        if ode_target_latent.ndim != 4:
            raise ValueError(
                f"Expected ODE target video latent to be 4D [C, F, H, W], got {tuple(ode_target_latent.shape)}"
            )
        payload["ode_target_latents"] = ode_target_latent.cpu().contiguous()
    if ode_sigma is not None:
        payload["ode_sigma"] = float(ode_sigma)
    if ode_step_index is not None:
        payload["ode_step_index"] = int(ode_step_index)
    if ode_clean_step_index is not None:
        payload["ode_clean_step_index"] = int(ode_clean_step_index)
    if ode_noise_seeds is not None:
        payload["ode_noise_seeds"] = ode_noise_seeds
    return payload


def build_audio_payload(
    audio_latent: torch.Tensor,
    duration_seconds: float,
    *,
    ode_target_latent: torch.Tensor | None = None,
    ode_sigma: float | None = None,
    ode_step_index: int | None = None,
    ode_clean_step_index: int | None = None,
    ode_noise_seeds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if audio_latent.ndim != 3:
        raise ValueError(f"Expected selected audio latent to be 3D [C, T, F], got {tuple(audio_latent.shape)}")

    payload = {
        "latents": audio_latent.cpu().contiguous(),
        "num_time_steps": int(audio_latent.shape[1]),
        "frequency_bins": int(audio_latent.shape[2]),
        "duration": float(duration_seconds),
    }
    if ode_target_latent is not None:
        if ode_target_latent.ndim != 3:
            raise ValueError(
                f"Expected ODE target audio latent to be 3D [C, T, F], got {tuple(ode_target_latent.shape)}"
            )
        payload["ode_target_latents"] = ode_target_latent.cpu().contiguous()
    if ode_sigma is not None:
        payload["ode_sigma"] = float(ode_sigma)
    if ode_step_index is not None:
        payload["ode_step_index"] = int(ode_step_index)
    if ode_clean_step_index is not None:
        payload["ode_clean_step_index"] = int(ode_clean_step_index)
    if ode_noise_seeds is not None:
        payload["ode_noise_seeds"] = ode_noise_seeds
    return payload


def estimate_duration_seconds(latent_num_frames: int, fps: float) -> float:
    original_num_frames = (latent_num_frames - 1) * VIDEO_TEMPORAL_COMPRESSION + 1
    return original_num_frames / fps


def encode_prompt_embeddings(
    prompt: str,
    text_encoder: Any,
    embeddings_processor: Any,
) -> dict[str, torch.Tensor]:
    with torch.inference_mode():
        hidden_states, prompt_attention_mask = text_encoder.encode(prompt, padding_side="left")
        video_prompt_embeds, audio_prompt_embeds = embeddings_processor.feature_extractor(
            hidden_states,
            prompt_attention_mask,
            "left",
        )

    embedding_data: dict[str, torch.Tensor] = {
        "video_prompt_embeds": video_prompt_embeds[0].cpu().contiguous(),
        "prompt_attention_mask": prompt_attention_mask[0].cpu().contiguous(),
    }
    if audio_prompt_embeds is not None:
        embedding_data["audio_prompt_embeds"] = audio_prompt_embeds[0].cpu().contiguous()

    return embedding_data


def write_manifest(
    manifest_path: Path,
    *,
    config_path: Path,
    input_dir: Path,
    output_root: Path,
    precomputed_root: Path,
    stage: str,
    trajectory_step: str,
    export_mode: str,
    fps: float,
    total_input_files: int,
    processed_items: int,
    skipped_items: int,
    wrote_audio: bool,
    wrote_conditions: bool,
    model_path: Path | None,
    text_encoder_path: Path | None,
) -> None:
    manifest = {
        "config_path": str(config_path),
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "precomputed_root": str(precomputed_root),
        "stage": stage,
        "trajectory_step": trajectory_step,
        "export_mode": export_mode,
        "fps": fps,
        "total_input_files": total_input_files,
        "processed_items": processed_items,
        "skipped_items": skipped_items,
        "processed_files": processed_items,
        "skipped_files": skipped_items,
        "wrote_audio": wrote_audio,
        "wrote_conditions": wrote_conditions,
        "model_path": str(model_path) if model_path is not None else None,
        "text_encoder_path": str(text_encoder_path) if text_encoder_path is not None else None,
        "directories": {
            "latents": str(precomputed_root / "latents"),
            "audio_latents": str(precomputed_root / "audio_latents") if wrote_audio else None,
            "conditions": str(precomputed_root / "conditions") if wrote_conditions else None,
        },
    }

    ensure_parent(manifest_path)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_yaml_config(config_path)

    input_dir = (
        args.input_dir.expanduser().resolve()
        if args.input_dir is not None
        else Path(config.get("output_dir", "ode/data_distilled")).expanduser().resolve()
    )
    model_path = (
        args.model_path.expanduser().resolve()
        if args.model_path is not None
        else Path(config["checkpoint_path"]).expanduser().resolve() if "checkpoint_path" in config else None
    )
    text_encoder_path = (
        args.text_encoder_path.expanduser().resolve()
        if args.text_encoder_path is not None
        else Path(config["gemma_root"]).expanduser().resolve() if "gemma_root" in config else None
    )
    fps = float(args.fps if args.fps is not None else config.get("frame_rate", 24.0))

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    write_audio = not args.no_write_audio
    write_conditions = not args.no_write_conditions

    if write_conditions and model_path is None:
        raise ValueError("--model-path is required when writing conditions.")
    if write_conditions and text_encoder_path is None:
        raise ValueError("--text-encoder-path is required when writing conditions.")

    output_root = resolve_output_root(
        args.output_dir,
        input_dir,
        args.stage,
        args.trajectory_step,
        args.export_mode,
    )
    precomputed_root = resolve_precomputed_root(output_root)
    latents_dir = precomputed_root / "latents"
    audio_dir = precomputed_root / "audio_latents" if write_audio else None
    conditions_dir = precomputed_root / "conditions" if write_conditions else None

    input_files = discover_input_files(input_dir, recursive=args.recursive, limit=args.limit)
    if not input_files:
        raise ValueError(f"No .pt files found under {input_dir}")

    LOGGER.info(
        "Discovered %d files under %s. Export mode=%s, stage=%s, trajectory-step=%s.",
        len(input_files),
        input_dir,
        args.export_mode,
        args.stage,
        args.trajectory_step,
    )

    text_encoder = None
    embeddings_processor = None
    if write_conditions:
        compute_dtype = parse_dtype(args.dtype)
        LOGGER.info("Loading text encoder from %s", text_encoder_path)
        text_encoder = load_text_encoder(
            text_encoder_path,
            device=args.device,
            dtype=compute_dtype,
            load_in_8bit=args.load_text_encoder_in_8bit,
        )
        LOGGER.info("Loading embeddings processor from %s", model_path)
        embeddings_processor = load_embeddings_processor(
            model_path,
            device=args.device,
            dtype=compute_dtype,
        )

    processed_items = 0
    skipped_items = 0
    multiple_steps = args.trajectory_step.strip().lower() in {"all", "all_non_last"}
    warned_zero_sigma = False

    try:
        for input_path in tqdm(input_files, desc="Converting ODE samples"):
            sample = torch.load(input_path, map_location="cpu")
            if not isinstance(sample, dict):
                raise TypeError(f"Expected sample dict from {input_path}, got {type(sample)}")

            validate_sample(sample, stage=args.stage, require_audio=write_audio)

            relative_path = input_path.relative_to(input_dir)
            video_traj = sample[f"{args.stage}_video_traj"]
            sigmas = sample[f"{args.stage}_sigmas"]
            step_indices = resolve_step_indices(args.trajectory_step, video_traj.shape[0])
            clean_video_latent = video_traj[-1]
            noise_seeds = sample.get("noise_seeds")

            audio_traj = sample[f"{args.stage}_audio_traj"] if write_audio else None
            clean_audio_latent = audio_traj[-1] if audio_traj is not None else None

            cached_embedding_data = None
            for step_index in step_indices:
                output_rel_path = build_relative_output_path(relative_path, step_index, multiple_steps)
                latent_output_path = latents_dir / output_rel_path
                audio_output_path = audio_dir / output_rel_path if audio_dir is not None else None
                condition_output_path = conditions_dir / output_rel_path if conditions_dir is not None else None

                write_latent = should_write_file(latent_output_path, args.overwrite)
                write_audio_item = should_write_file(audio_output_path, args.overwrite)
                write_condition = should_write_file(condition_output_path, args.overwrite)

                if not (write_latent or write_audio_item or write_condition):
                    skipped_items += 1
                    continue

                selected_video_latent = video_traj[step_index]
                selected_sigma = float(sigmas[step_index].item())
                if args.export_mode == "ode_regression" and selected_sigma <= 0 and not warned_zero_sigma:
                    LOGGER.warning(
                        "ODE regression export includes sigma<=0 samples. They remain in the dataset but trainer loss "
                        "will ignore them. Prefer --trajectory-step all_non_last for pure noisy->clean supervision."
                    )
                    warned_zero_sigma = True

                latent_payload_kwargs: dict[str, Any] = {}
                audio_payload_kwargs: dict[str, Any] = {}
                if isinstance(noise_seeds, dict):
                    latent_payload_kwargs["ode_noise_seeds"] = noise_seeds
                    audio_payload_kwargs["ode_noise_seeds"] = noise_seeds
                if args.export_mode == "ode_regression":
                    latent_payload_kwargs = {
                        **latent_payload_kwargs,
                        "ode_target_latent": clean_video_latent,
                        "ode_sigma": selected_sigma,
                        "ode_step_index": step_index,
                        "ode_clean_step_index": video_traj.shape[0] - 1,
                    }
                    if clean_audio_latent is not None:
                        audio_payload_kwargs = {
                            **audio_payload_kwargs,
                            "ode_target_latent": clean_audio_latent,
                            "ode_sigma": selected_sigma,
                            "ode_step_index": step_index,
                            "ode_clean_step_index": video_traj.shape[0] - 1,
                        }

                if write_latent:
                    ensure_parent(latent_output_path)
                    torch.save(
                        build_latent_payload(selected_video_latent, fps=fps, **latent_payload_kwargs),
                        latent_output_path,
                    )

                if write_audio_item:
                    if audio_traj is None:
                        raise RuntimeError("Audio export requested but audio trajectory is missing.")
                    selected_audio_latent = audio_traj[step_index]
                    duration_seconds = estimate_duration_seconds(int(selected_video_latent.shape[1]), fps=fps)
                    ensure_parent(audio_output_path)
                    torch.save(
                        build_audio_payload(
                            selected_audio_latent,
                            duration_seconds,
                            **audio_payload_kwargs,
                        ),
                        audio_output_path,
                    )

                if write_condition:
                    if text_encoder is None or embeddings_processor is None:
                        raise RuntimeError("Condition generation requested but text models are not loaded.")
                    if cached_embedding_data is None:
                        cached_embedding_data = encode_prompt_embeddings(
                            prompt=sample["prompt"],
                            text_encoder=text_encoder,
                            embeddings_processor=embeddings_processor,
                        )
                    ensure_parent(condition_output_path)
                    torch.save(cached_embedding_data, condition_output_path)

                processed_items += 1
    finally:
        if text_encoder is not None:
            del text_encoder
        if embeddings_processor is not None:
            del embeddings_processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest_path = output_root / args.manifest_name
    write_manifest(
        manifest_path=manifest_path,
        config_path=config_path,
        input_dir=input_dir,
        output_root=output_root,
        precomputed_root=precomputed_root,
        stage=args.stage,
        trajectory_step=args.trajectory_step,
        export_mode=args.export_mode,
        fps=fps,
        total_input_files=len(input_files),
        processed_items=processed_items,
        skipped_items=skipped_items,
        wrote_audio=write_audio,
        wrote_conditions=write_conditions,
        model_path=model_path,
        text_encoder_path=text_encoder_path,
    )

    LOGGER.info(
        "Done. Converted %d sample items into %s (skipped %d existing items).",
        processed_items,
        precomputed_root,
        skipped_items,
    )
    LOGGER.info("Manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
