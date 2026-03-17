"""
生成 ODE 轨迹数据（non-distilled two-stage 官方结构版本）。

该脚本遵循官方 two-stage pipeline 的模型结构：
  - Stage 1 使用 base checkpoint + 官方 MultiModalGuiderFactory guidance
  - Stage 2 使用 base checkpoint + distilled LoRA 的 refinement transformer

同时保留 ODE 数据生成所需的自定义采样方式：
  - Stage 1 在 teacher sigma 调度上运行，并保存对齐蒸馏 sigma 的若干中间状态
  - Stage 2 从 Stage 1 clean latent 上采样后出发，在自定义子调度上继续运行并保存轨迹

每个 sample 保存为 {index:05d}.pt，内容:
  {
    "prompt": str,
    "stage1_video_traj": Tensor [9, 128, 16, 16, 24],
    "stage1_audio_traj": Tensor [9, 8, 126, 16],
    "stage1_sigmas":     Tensor [9],
    "stage2_video_traj": Tensor [4, 128, 16, 32, 48],
    "stage2_audio_traj": Tensor [4, 8, 126, 16],
    "stage2_sigmas":     Tensor [4],
  }

用法:
    cd /inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx
    python ode/gen_ode_data.py --config ode/configs/gen_ode_data.yaml
"""

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, "packages/ltx-core/src")
sys.path.insert(0, "packages/ltx-pipelines/src")

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import MultiModalGuiderParams, create_multimodal_guider_factory
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.upsampler import upsample_video
from ltx_core.text_encoders.gemma import EmbeddingsProcessorOutput
from ltx_core.types import VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, detect_params
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    multi_modal_guider_factory_denoising_func,
    noise_audio_state,
    noise_video_state,
    post_process_latent,
    simple_denoising_func,
)
from ltx_pipelines.utils.types import PipelineComponents

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 对齐蒸馏 sigma 的保存索引，基于 teacher_steps=40 的 LTX2Scheduler 最近邻映射
STAGE1_SAVE_INDICES = [0, 2, 3, 5, 6, 17, 29, 36, 40]
STAGE2_SAVE_INDICES_GLOBAL = [17, 29, 36, 40]
STAGE2_START_IDX = 17
PROMPT_CTX_PREENCODE_MODES = {"staged", "fast", "sequential"}
CACHE_FORMAT_VERSION = 1
MAX_BATCH_ATTEMPTS = 2
MAX_TORCH_SEED = (1 << 63) - 1
NOISE_SEED_SCHEME = "per-stage-per-modality-v1"


def cleanup_memory() -> None:
    if torch.cuda.is_available():
        from ltx_pipelines.utils import cleanup_memory as pipeline_cleanup_memory

        pipeline_cleanup_memory()


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype)


def load_prompts(csv_path: str, prompt_column: str, max_samples: int) -> list[str]:
    prompts = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            if max_samples > 0 and i >= max_samples:
                break
            prompts.append(row[prompt_column])
    logger.info("Loaded %d prompts from %s", len(prompts), csv_path)
    return prompts


def get_prompt_ctx_batch_size(cfg: dict) -> int:
    return int(cfg.get("prompt_ctx_batch_size", 512))


def get_prompt_ctx_preencode_mode(cfg: dict) -> str:
    return str(cfg.get("prompt_ctx_preencode_mode", "staged")).strip().lower()


def should_delete_prompt_ctx_after_use(cfg: dict) -> bool:
    return bool(cfg.get("delete_prompt_ctx_after_use", True))


def should_cleanup_after_each_sample(cfg: dict) -> bool:
    return bool(cfg.get("cleanup_after_each_sample", True))


def get_prompt_ctx_cache_dir(cfg: dict, output_dir: Path, chunk_id: int) -> Path:
    configured = cfg.get("prompt_ctx_cache_dir")
    if configured:
        base_dir = Path(configured)
    else:
        base_dir = output_dir.parent / f"{output_dir.name}_prompt_ctx_cache"
    return base_dir / f"chunk_{chunk_id:04d}"


def iter_batches(items: list[int], batch_size: int):
    if batch_size == -1:
        if items:
            yield items
        return
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(payload: dict) -> str:
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def derive_noise_seed(base_seed: int, global_idx: int, stage: str, modality: str) -> int:
    payload = f"{int(base_seed)}:{int(global_idx)}:{stage}:{modality}:{NOISE_SEED_SCHEME}"
    seed = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % MAX_TORCH_SEED
    return seed if seed != 0 else 1


def build_noise_seed_metadata(base_seed: int, global_idx: int) -> dict[str, Any]:
    return {
        "scheme": NOISE_SEED_SCHEME,
        "base_seed": int(base_seed),
        "global_idx": int(global_idx),
        "stage1": {
            "video": derive_noise_seed(base_seed, global_idx, "stage1", "video"),
            "audio": derive_noise_seed(base_seed, global_idx, "stage1", "audio"),
        },
        "stage2": {
            "video": derive_noise_seed(base_seed, global_idx, "stage2", "video"),
            "audio": derive_noise_seed(base_seed, global_idx, "stage2", "audio"),
        },
    }


def build_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


def guider_params_to_dict(params: MultiModalGuiderParams) -> dict:
    return {
        "cfg_scale": float(params.cfg_scale),
        "stg_scale": float(params.stg_scale),
        "rescale_scale": float(params.rescale_scale),
        "modality_scale": float(params.modality_scale),
        "skip_step": int(params.skip_step),
        "stg_blocks": list(params.stg_blocks or []),
    }


def build_cache_signatures(
    cfg: dict,
    checkpoint_path: str,
    gemma_root_path: str,
    negative_prompt: str,
    video_guider_params: MultiModalGuiderParams,
    audio_guider_params: MultiModalGuiderParams,
    dtype: torch.dtype,
) -> tuple[str, str]:
    prompt_ctx_basis = {
        "format_version": CACHE_FORMAT_VERSION,
        "checkpoint_path": checkpoint_path,
        "gemma_root_path": gemma_root_path,
        "dtype": str(dtype),
    }
    stage1_basis = {
        "format_version": CACHE_FORMAT_VERSION,
        "checkpoint_path": checkpoint_path,
        "gemma_root_path": gemma_root_path,
        "dtype": str(dtype),
        "negative_prompt_hash": hash_text(negative_prompt),
        "teacher_steps": int(cfg["teacher_steps"]),
        "seed": int(cfg["seed"]),
        "num_frames": int(cfg["num_frames"]),
        "frame_rate": float(cfg["frame_rate"]),
        "stage1_height": int(cfg["stage1_height"]),
        "stage1_width": int(cfg["stage1_width"]),
        "stage1_save_indices": list(STAGE1_SAVE_INDICES),
        "noise_seed_scheme": NOISE_SEED_SCHEME,
        "video_guider_params": guider_params_to_dict(video_guider_params),
        "audio_guider_params": guider_params_to_dict(audio_guider_params),
    }
    return stable_hash(prompt_ctx_basis), stable_hash(stage1_basis)


def build_prompt_ctx_metadata(prompt_ctx_signature: str, prompt: str, global_idx: int) -> dict:
    return {
        "kind": "prompt_ctx",
        "format_version": CACHE_FORMAT_VERSION,
        "signature": prompt_ctx_signature,
        "prompt_hash": hash_text(prompt),
        "global_idx": int(global_idx),
    }


def build_stage1_artifact_metadata(stage1_signature: str, prompt: str, global_idx: int) -> dict:
    return {
        "kind": "stage1_artifact",
        "format_version": CACHE_FORMAT_VERSION,
        "signature": stage1_signature,
        "prompt_hash": hash_text(prompt),
        "global_idx": int(global_idx),
    }


def validate_cache_metadata(actual: dict | None, expected: dict, path: Path) -> None:
    if not isinstance(actual, dict):
        raise ValueError(f"{path} 缺少 metadata")
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            raise ValueError(
                f"{path} metadata[{key}] 不匹配: got={actual.get(key)!r}, expected={expected_value!r}"
            )


def remove_file_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def discard_invalid_cache(path: Path, cache_label: str, exc: Exception) -> None:
    remove_file_if_exists(path)
    logger.warning("%s %s 无效，已删除并准备重建: %s", cache_label, path, exc)


def cleanup_sample_cache_files(
    ctx_cache_path: Path,
    stage1_artifact_path: Path,
    delete_prompt_ctx_after_use: bool,
    delete_stage1_artifact_after_use: bool,
) -> None:
    if delete_prompt_ctx_after_use:
        remove_file_if_exists(ctx_cache_path)
    if delete_stage1_artifact_after_use:
        remove_file_if_exists(stage1_artifact_path)


def _raw_to_cpu(raw):
    if isinstance(raw, torch.Tensor):
        return raw.cpu()
    if isinstance(raw, (tuple, list)):
        moved = [_raw_to_cpu(x) for x in raw]
        return type(raw)(moved)
    if hasattr(raw, "keys"):
        return {k: _raw_to_cpu(v) for k, v in raw.items()}
    return raw


def _raw_to_device(raw, device):
    if isinstance(raw, torch.Tensor):
        return raw.to(device)
    if isinstance(raw, (tuple, list)):
        moved = [_raw_to_device(x, device) for x in raw]
        return type(raw)(moved)
    if hasattr(raw, "keys"):
        return {k: _raw_to_device(v, device) for k, v in raw.items()}
    return raw


def _ctx_to_device(ctx, device):
    return ctx._replace(
        video_encoding=ctx.video_encoding.to(device),
        audio_encoding=ctx.audio_encoding.to(device) if ctx.audio_encoding is not None else None,
        attention_mask=ctx.attention_mask.to(device),
    )


def _ctx_to_cpu(ctx):
    return ctx._replace(
        video_encoding=ctx.video_encoding.to("cpu", dtype=torch.bfloat16),
        audio_encoding=ctx.audio_encoding.to("cpu", dtype=torch.bfloat16) if ctx.audio_encoding is not None else None,
        attention_mask=ctx.attention_mask.to("cpu", dtype=torch.uint8),
    )


def encode_prompt(prompt: str, model_ledger: ModelLedger, device: torch.device):
    text_encoder = model_ledger.text_encoder()
    raw = text_encoder.encode(prompt)
    raw_cpu = _raw_to_cpu(raw)
    del raw, text_encoder
    cleanup_memory()

    embeddings_processor = model_ledger.gemma_embeddings_processor()
    raw_gpu = _raw_to_device(raw_cpu, device)
    ctx = embeddings_processor.process_hidden_states(*raw_gpu)
    ctx_cpu = _ctx_to_device(ctx, "cpu")
    del raw_gpu, ctx, raw_cpu, embeddings_processor
    cleanup_memory()
    return ctx_cpu


def get_prompt_ctx_cache_path(cache_dir: Path, global_idx: int) -> Path:
    return cache_dir / f"{global_idx:05d}.pt"


def save_prompt_ctx(ctx, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "metadata": metadata,
            "video_encoding": ctx.video_encoding.contiguous(),
            "audio_encoding": ctx.audio_encoding.contiguous() if ctx.audio_encoding is not None else None,
            "attention_mask": ctx.attention_mask.contiguous(),
        },
        tmp_path,
    )
    tmp_path.replace(path)


def load_prompt_ctx(path: Path, expected_metadata: dict | None = None) -> EmbeddingsProcessorOutput:
    payload = torch.load(path, map_location="cpu")
    if expected_metadata is not None:
        validate_cache_metadata(payload.get("metadata"), expected_metadata, path)
    if "video_encoding" not in payload or "attention_mask" not in payload:
        raise ValueError(f"{path} prompt ctx payload 缺少必要字段")
    return EmbeddingsProcessorOutput(
        video_encoding=payload["video_encoding"],
        audio_encoding=payload["audio_encoding"],
        attention_mask=payload["attention_mask"],
    )


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def get_stage1_artifact_dir(cache_dir: Path) -> Path:
    return cache_dir / "stage1_artifacts"


def get_stage1_artifact_path(artifact_dir: Path, global_idx: int) -> Path:
    return artifact_dir / f"{global_idx:05d}.pt"


def should_delete_stage1_artifact_after_use(cfg: dict) -> bool:
    return bool(cfg.get("delete_stage1_artifact_after_use", True))


def load_stage1_artifact(path: Path, expected_metadata: dict | None = None) -> dict:
    payload = torch.load(path, map_location="cpu")
    if expected_metadata is not None:
        validate_cache_metadata(payload.get("metadata"), expected_metadata, path)
    required_keys = {
        "prompt",
        "stage1_video_traj",
        "stage1_audio_traj",
        "stage1_sigmas",
        "stage1_final_video",
        "stage1_final_audio",
        "noise_seeds",
    }
    missing_keys = sorted(key for key in required_keys if key not in payload)
    if missing_keys:
        raise ValueError(f"{path} stage1 artifact 缺少字段: {missing_keys}")
    return payload


def has_valid_prompt_ctx_cache(path: Path, expected_metadata: dict) -> bool:
    if not path.exists():
        return False
    try:
        ctx = load_prompt_ctx(path, expected_metadata=expected_metadata)
        del ctx
        return True
    except Exception as exc:
        discard_invalid_cache(path, "Prompt ctx cache", exc)
        cleanup_memory()
        return False


def has_valid_stage1_artifact(path: Path, expected_metadata: dict) -> bool:
    if not path.exists():
        return False
    try:
        artifact = load_stage1_artifact(path, expected_metadata=expected_metadata)
        del artifact
        return True
    except Exception as exc:
        discard_invalid_cache(path, "Stage1 artifact", exc)
        cleanup_memory()
        return False


def try_load_prompt_ctx(path: Path, expected_metadata: dict) -> EmbeddingsProcessorOutput | None:
    if not path.exists():
        return None
    try:
        return load_prompt_ctx(path, expected_metadata=expected_metadata)
    except Exception as exc:
        discard_invalid_cache(path, "Prompt ctx cache", exc)
        cleanup_memory()
        return None


def try_load_stage1_artifact(path: Path, expected_metadata: dict) -> dict | None:
    if not path.exists():
        return None
    try:
        return load_stage1_artifact(path, expected_metadata=expected_metadata)
    except Exception as exc:
        discard_invalid_cache(path, "Stage1 artifact", exc)
        cleanup_memory()
        return None


@torch.inference_mode()
def ensure_stage1_artifact(
    prompt: str,
    global_idx: int,
    stage1_artifact_path: Path,
    prompt_ctx_cache_dir: Path,
    prompt_ctx_signature: str,
    stage1_signature: str,
    ctx_neg,
    stage1_transformer,
    sigmas: torch.Tensor,
    components: PipelineComponents,
    cfg: dict,
    video_guider_params: MultiModalGuiderParams,
    audio_guider_params: MultiModalGuiderParams,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    artifact_metadata = build_stage1_artifact_metadata(stage1_signature, prompt, global_idx)
    if has_valid_stage1_artifact(stage1_artifact_path, artifact_metadata):
        return

    ctx_cache_path = get_prompt_ctx_cache_path(prompt_ctx_cache_dir, global_idx)
    prompt_ctx_metadata = build_prompt_ctx_metadata(prompt_ctx_signature, prompt, global_idx)
    ctx_pos = try_load_prompt_ctx(ctx_cache_path, prompt_ctx_metadata)
    if ctx_pos is None:
        raise FileNotFoundError(f"Prompt ctx cache 不存在或无效: {ctx_cache_path}")

    try:
        artifact = generate_stage1_artifact(
            prompt=prompt,
            global_idx=global_idx,
            ctx_pos=ctx_pos,
            ctx_neg=ctx_neg,
            stage1_transformer=stage1_transformer,
            sigmas=sigmas,
            components=components,
            cfg=cfg,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            device=device,
            dtype=dtype,
        )
        atomic_torch_save({"metadata": artifact_metadata, **artifact}, stage1_artifact_path)
        del artifact
    finally:
        del ctx_pos


def preencode_prompt_batch_staged(
    missing: list[tuple[int, str]],
    model_ledger: ModelLedger,
    cache_dir: Path,
    device: torch.device,
    prompt_ctx_signature: str,
) -> None:
    raw_batch: list[tuple[int, object]] = []
    missing_map = dict(missing)
    text_encoder = None
    embeddings_processor = None
    try:
        text_encoder = model_ledger.text_encoder()
        for global_idx, prompt in tqdm(missing, desc="prompt-preencode-text", leave=False):
            raw_batch.append((global_idx, _raw_to_cpu(text_encoder.encode(prompt))))

        del text_encoder
        text_encoder = None
        cleanup_memory()

        embeddings_processor = model_ledger.gemma_embeddings_processor()
        for global_idx, raw_cpu in tqdm(raw_batch, desc="prompt-preencode-embed", leave=False):
            cache_path = get_prompt_ctx_cache_path(cache_dir, global_idx)
            prompt_ctx_metadata = build_prompt_ctx_metadata(prompt_ctx_signature, missing_map[global_idx], global_idx)
            raw_gpu = _raw_to_device(raw_cpu, device)
            ctx = embeddings_processor.process_hidden_states(*raw_gpu)
            save_prompt_ctx(_ctx_to_cpu(ctx), cache_path, prompt_ctx_metadata)
            del raw_gpu, ctx, raw_cpu
    finally:
        raw_batch.clear()
        del text_encoder, embeddings_processor
        cleanup_memory()


@torch.inference_mode()
def preencode_prompt_batch(
    prompt_batch: list[tuple[int, str]],
    model_ledger: ModelLedger,
    cache_dir: Path,
    device: torch.device,
    encode_mode: str,
    prompt_ctx_signature: str,
) -> None:
    missing = [
        (global_idx, prompt)
        for global_idx, prompt in prompt_batch
        if not has_valid_prompt_ctx_cache(
            get_prompt_ctx_cache_path(cache_dir, global_idx),
            build_prompt_ctx_metadata(prompt_ctx_signature, prompt, global_idx),
        )
    ]
    if not missing:
        return

    logger.info("Pre-encoding %d prompts to %s with mode=%s", len(missing), cache_dir, encode_mode)

    if encode_mode == "staged":
        try:
            preencode_prompt_batch_staged(
                missing=missing,
                model_ledger=model_ledger,
                cache_dir=cache_dir,
                device=device,
                prompt_ctx_signature=prompt_ctx_signature,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            logger.warning(
                "Staged prompt pre-encode hit OOM; falling back to sequential prompt encoding for this batch."
            )
            cleanup_memory()
            for global_idx, prompt in tqdm(missing, desc="prompt-preencode-sequential", leave=False):
                cache_path = get_prompt_ctx_cache_path(cache_dir, global_idx)
                if cache_path.exists():
                    continue
                ctx = encode_prompt(prompt, model_ledger, device)
                save_prompt_ctx(ctx, cache_path, build_prompt_ctx_metadata(prompt_ctx_signature, prompt, global_idx))
                del ctx
            cleanup_memory()
        return

    if encode_mode == "sequential":
        for global_idx, prompt in tqdm(missing, desc="prompt-preencode-sequential", leave=False):
            cache_path = get_prompt_ctx_cache_path(cache_dir, global_idx)
            ctx = encode_prompt(prompt, model_ledger, device)
            save_prompt_ctx(ctx, cache_path, build_prompt_ctx_metadata(prompt_ctx_signature, prompt, global_idx))
            del ctx
        cleanup_memory()
        return

    text_encoder = None
    embeddings_processor = None
    try:
        text_encoder = model_ledger.text_encoder()
        embeddings_processor = model_ledger.gemma_embeddings_processor()
        for global_idx, prompt in tqdm(missing, desc="prompt-preencode", leave=False):
            cache_path = get_prompt_ctx_cache_path(cache_dir, global_idx)
            raw = text_encoder.encode(prompt)
            ctx = embeddings_processor.process_hidden_states(*raw)
            save_prompt_ctx(
                _ctx_to_cpu(ctx),
                cache_path,
                build_prompt_ctx_metadata(prompt_ctx_signature, prompt, global_idx),
            )
            del raw, ctx
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        logger.warning(
            "Fast prompt pre-encode hit OOM; falling back to sequential prompt encoding for this batch."
        )
        text_encoder = None
        embeddings_processor = None
        cleanup_memory()
        for global_idx, prompt in tqdm(missing, desc="prompt-preencode-fallback", leave=False):
            cache_path = get_prompt_ctx_cache_path(cache_dir, global_idx)
            if cache_path.exists():
                continue
            ctx = encode_prompt(prompt, model_ledger, device)
            save_prompt_ctx(ctx, cache_path, build_prompt_ctx_metadata(prompt_ctx_signature, prompt, global_idx))
            del ctx
    finally:
        text_encoder = None
        embeddings_processor = None
        cleanup_memory()


def _extract_spatial_latent(state, tools) -> torch.Tensor:
    spatial_state = tools.unpatchify(tools.clear_conditioning(state))
    return spatial_state.latent[:1].cpu()


def run_ode_loop(
    video_state,
    audio_state,
    video_tools,
    audio_tools,
    sigmas: torch.Tensor,
    stepper: EulerDiffusionStep,
    denoise_fn,
    save_indices: list[int],
) -> tuple[object, object, list[torch.Tensor], list[torch.Tensor]]:
    save_set = set(save_indices)
    video_traj: list[torch.Tensor] = []
    audio_traj: list[torch.Tensor] = []

    if 0 in save_set:
        video_traj.append(_extract_spatial_latent(video_state, video_tools))
        audio_traj.append(_extract_spatial_latent(audio_state, audio_tools))

    for step_idx in range(len(sigmas) - 1):
        denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_idx)

        denoised_video = post_process_latent(
            denoised_video,
            video_state.denoise_mask,
            video_state.clean_latent,
        )
        denoised_audio = post_process_latent(
            denoised_audio,
            audio_state.denoise_mask,
            audio_state.clean_latent,
        )

        video_state = replace(
            video_state,
            latent=stepper.step(video_state.latent, denoised_video, sigmas, step_idx),
        )
        audio_state = replace(
            audio_state,
            latent=stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx),
        )

        current_idx = step_idx + 1
        if current_idx in save_set:
            video_traj.append(_extract_spatial_latent(video_state, video_tools))
            audio_traj.append(_extract_spatial_latent(audio_state, audio_tools))

    return video_state, audio_state, video_traj, audio_traj


@torch.inference_mode()
def generate_stage1_artifact(
    prompt: str,
    global_idx: int,
    ctx_pos,
    ctx_neg,
    stage1_transformer,
    sigmas: torch.Tensor,
    components: PipelineComponents,
    cfg: dict,
    video_guider_params: MultiModalGuiderParams,
    audio_guider_params: MultiModalGuiderParams,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    seed = int(cfg["seed"])
    num_frames = int(cfg["num_frames"])
    frame_rate = float(cfg["frame_rate"])
    stage1_h = int(cfg["stage1_height"])
    stage1_w = int(cfg["stage1_width"])

    noise_seeds = build_noise_seed_metadata(seed, global_idx)
    stage1_noise_seeds = noise_seeds["stage1"]
    stepper = EulerDiffusionStep()

    v_ctx_pos = ctx_pos.video_encoding.to(dtype=dtype, device=device)
    a_ctx_pos = ctx_pos.audio_encoding.to(dtype=dtype, device=device)
    v_ctx_neg = ctx_neg.video_encoding.to(dtype=dtype, device=device)
    a_ctx_neg = ctx_neg.audio_encoding.to(dtype=dtype, device=device)

    denoise_fn_s1 = multi_modal_guider_factory_denoising_func(
        video_guider_factory=create_multimodal_guider_factory(
            params=video_guider_params,
            negative_context=v_ctx_neg,
        ),
        audio_guider_factory=create_multimodal_guider_factory(
            params=audio_guider_params,
            negative_context=a_ctx_neg,
        ),
        v_context=v_ctx_pos,
        a_context=a_ctx_pos,
        transformer=stage1_transformer,
    )

    stage1_shape = VideoPixelShape(
        batch=1,
        frames=num_frames,
        height=stage1_h,
        width=stage1_w,
        fps=frame_rate,
    )
    video_state_s1, video_tools_s1 = noise_video_state(
        output_shape=stage1_shape,
        noiser=GaussianNoiser(generator=build_generator(device, stage1_noise_seeds["video"])),
        conditionings=[],
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=1.0,
    )
    audio_state_s1, audio_tools_s1 = noise_audio_state(
        output_shape=stage1_shape,
        noiser=GaussianNoiser(generator=build_generator(device, stage1_noise_seeds["audio"])),
        conditionings=[],
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=1.0,
    )

    with autocast_context(device, dtype):
        video_state_s1, audio_state_s1, s1_video_traj, s1_audio_traj = run_ode_loop(
            video_state=video_state_s1,
            audio_state=audio_state_s1,
            video_tools=video_tools_s1,
            audio_tools=audio_tools_s1,
            sigmas=sigmas,
            stepper=stepper,
            denoise_fn=denoise_fn_s1,
            save_indices=STAGE1_SAVE_INDICES,
        )

    s1_final_video = _extract_spatial_latent(video_state_s1, video_tools_s1)
    s1_final_audio = _extract_spatial_latent(audio_state_s1, audio_tools_s1)
    s1_sigma_vals = torch.tensor([sigmas[i].item() for i in STAGE1_SAVE_INDICES], dtype=torch.float32)

    del denoise_fn_s1, video_state_s1, audio_state_s1, video_tools_s1, audio_tools_s1
    del v_ctx_pos, a_ctx_pos, v_ctx_neg, a_ctx_neg
    cleanup_memory()

    return {
        "prompt": prompt,
        "stage1_video_traj": torch.stack(s1_video_traj, dim=0).squeeze(1),
        "stage1_audio_traj": torch.stack(s1_audio_traj, dim=0).squeeze(1),
        "stage1_sigmas": s1_sigma_vals,
        "stage1_final_video": s1_final_video,
        "stage1_final_audio": s1_final_audio,
        "noise_seeds": noise_seeds,
    }


@torch.inference_mode()
def generate_stage2_sample(
    ctx_pos,
    global_idx: int,
    stage1_artifact: dict,
    stage2_transformer,
    stage1_model_ledger: ModelLedger,
    stage2_model_ledger: ModelLedger,
    sigmas: torch.Tensor,
    components: PipelineComponents,
    cfg: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    seed = int(cfg["seed"])
    num_frames = int(cfg["num_frames"])
    frame_rate = float(cfg["frame_rate"])
    stage1_h = int(cfg["stage1_height"])
    stage1_w = int(cfg["stage1_width"])
    stage2_h = stage1_h * 2
    stage2_w = stage1_w * 2

    noise_seeds = stage1_artifact.get("noise_seeds")
    if not isinstance(noise_seeds, dict):
        noise_seeds = build_noise_seed_metadata(seed, global_idx)
    stage2_noise_seeds = noise_seeds["stage2"]
    stepper = EulerDiffusionStep()

    v_ctx_pos = ctx_pos.video_encoding.to(dtype=dtype, device=device)
    a_ctx_pos = ctx_pos.audio_encoding.to(dtype=dtype, device=device)

    s1_final_video = stage1_artifact["stage1_final_video"]
    s1_final_audio = stage1_artifact["stage1_final_audio"]

    video_encoder = stage1_model_ledger.video_encoder()
    upsampler = stage2_model_ledger.spatial_upsampler()
    upsampled_video = upsample_video(
        latent=s1_final_video.to(dtype=dtype, device=device),
        video_encoder=video_encoder,
        upsampler=upsampler,
    )
    del upsampler, video_encoder
    cleanup_memory()

    stage2_start_sigma = float(sigmas[STAGE2_START_IDX].item())
    stage2_sigmas = sigmas[STAGE2_START_IDX:]
    stage2_save_local = [idx - STAGE2_START_IDX for idx in STAGE2_SAVE_INDICES_GLOBAL]

    stage2_shape = VideoPixelShape(
        batch=1,
        frames=num_frames,
        height=stage2_h,
        width=stage2_w,
        fps=frame_rate,
    )
    video_state_s2, video_tools_s2 = noise_video_state(
        output_shape=stage2_shape,
        noiser=GaussianNoiser(generator=build_generator(device, stage2_noise_seeds["video"])),
        conditionings=[],
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=stage2_start_sigma,
        initial_latent=upsampled_video,
    )
    audio_state_s2, audio_tools_s2 = noise_audio_state(
        output_shape=stage2_shape,
        noiser=GaussianNoiser(generator=build_generator(device, stage2_noise_seeds["audio"])),
        conditionings=[],
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=stage2_start_sigma,
        initial_latent=s1_final_audio.to(dtype=dtype, device=device),
    )
    del upsampled_video
    cleanup_memory()

    denoise_fn_s2 = simple_denoising_func(
        video_context=v_ctx_pos,
        audio_context=a_ctx_pos,
        transformer=stage2_transformer,
    )

    with autocast_context(device, dtype):
        _, _, s2_video_traj, s2_audio_traj = run_ode_loop(
            video_state=video_state_s2,
            audio_state=audio_state_s2,
            video_tools=video_tools_s2,
            audio_tools=audio_tools_s2,
            sigmas=stage2_sigmas,
            stepper=stepper,
            denoise_fn=denoise_fn_s2,
            save_indices=stage2_save_local,
        )

    s2_sigma_vals = torch.tensor([sigmas[i].item() for i in STAGE2_SAVE_INDICES_GLOBAL], dtype=torch.float32)
    del denoise_fn_s2, video_state_s2, audio_state_s2, video_tools_s2, audio_tools_s2
    del v_ctx_pos, a_ctx_pos
    cleanup_memory()

    return {
        "prompt": stage1_artifact["prompt"],
        "stage1_video_traj": stage1_artifact["stage1_video_traj"],
        "stage1_audio_traj": stage1_artifact["stage1_audio_traj"],
        "stage1_sigmas": stage1_artifact["stage1_sigmas"],
        "stage2_video_traj": torch.stack(s2_video_traj, dim=0).squeeze(1),
        "stage2_audio_traj": torch.stack(s2_audio_traj, dim=0).squeeze(1),
        "stage2_sigmas": s2_sigma_vals,
        "noise_seeds": noise_seeds,
    }


def build_guider_params(cfg: dict, detected_defaults) -> tuple[MultiModalGuiderParams, MultiModalGuiderParams]:
    video_defaults = detected_defaults.video_guider_params
    audio_defaults = detected_defaults.audio_guider_params

    video_params = MultiModalGuiderParams(
        cfg_scale=float(cfg.get("video_cfg_scale", video_defaults.cfg_scale)),
        stg_scale=float(cfg.get("video_stg_scale", video_defaults.stg_scale)),
        rescale_scale=float(cfg.get("video_rescale_scale", video_defaults.rescale_scale)),
        modality_scale=float(cfg.get("a2v_guidance_scale", video_defaults.modality_scale)),
        skip_step=int(cfg.get("video_skip_step", video_defaults.skip_step)),
        stg_blocks=list(cfg.get("video_stg_blocks", video_defaults.stg_blocks or [])),
    )
    audio_params = MultiModalGuiderParams(
        cfg_scale=float(cfg.get("audio_cfg_scale", audio_defaults.cfg_scale)),
        stg_scale=float(cfg.get("audio_stg_scale", audio_defaults.stg_scale)),
        rescale_scale=float(cfg.get("audio_rescale_scale", audio_defaults.rescale_scale)),
        modality_scale=float(cfg.get("v2a_guidance_scale", audio_defaults.modality_scale)),
        skip_step=int(cfg.get("audio_skip_step", audio_defaults.skip_step)),
        stg_blocks=list(cfg.get("audio_stg_blocks", audio_defaults.stg_blocks or [])),
    )
    return video_params, audio_params


def validate_config(cfg: dict) -> None:
    checkpoint_path = Path(cfg["checkpoint_path"]).expanduser().resolve()
    gemma_root = Path(cfg["gemma_root"]).expanduser().resolve()
    spatial_upsampler_path = Path(cfg["spatial_upsampler_path"]).expanduser().resolve()
    distilled_lora_path = Path(cfg["distilled_lora_path"]).expanduser().resolve()
    caption_path = Path(cfg["caption_path"]).expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")
    if not gemma_root.is_dir():
        raise FileNotFoundError(f"Gemma 目录不存在: {gemma_root}")
    if not spatial_upsampler_path.is_file():
        raise FileNotFoundError(f"Spatial upsampler 不存在: {spatial_upsampler_path}")
    if not distilled_lora_path.is_file():
        raise FileNotFoundError(f"Stage 2 distilled LoRA 不存在: {distilled_lora_path}")
    if not caption_path.is_file():
        raise FileNotFoundError(f"caption CSV 不存在: {caption_path}")

    stage1_h = int(cfg["stage1_height"])
    stage1_w = int(cfg["stage1_width"])
    if stage1_h <= 0 or stage1_w <= 0:
        raise ValueError("stage1_height 和 stage1_width 必须为正整数")
    assert_resolution(height=stage1_h * 2, width=stage1_w * 2, is_two_stage=True)

    num_frames = int(cfg["num_frames"])
    if num_frames <= 0 or (num_frames - 1) % 8 != 0:
        raise ValueError("num_frames 必须满足 (num_frames - 1) % 8 == 0")

    teacher_steps = int(cfg["teacher_steps"])
    if teacher_steps <= 0:
        raise ValueError("teacher_steps 必须为正整数")
    if teacher_steps < STAGE1_SAVE_INDICES[-1]:
        raise ValueError(
            f"teacher_steps={teacher_steps} 过小，至少需要 >= {STAGE1_SAVE_INDICES[-1]} 以覆盖保存索引"
        )

    prompt_column = cfg["prompt_column"]
    if not isinstance(prompt_column, str) or not prompt_column:
        raise ValueError("prompt_column 必须是非空字符串")
    prompt_ctx_batch_size = get_prompt_ctx_batch_size(cfg)
    if prompt_ctx_batch_size == 0 or prompt_ctx_batch_size < -1:
        raise ValueError("prompt_ctx_batch_size 必须为正整数，或设为 -1 表示整 chunk 预编码")
    prompt_ctx_preencode_mode = get_prompt_ctx_preencode_mode(cfg)
    if prompt_ctx_preencode_mode not in PROMPT_CTX_PREENCODE_MODES:
        raise ValueError(
            "prompt_ctx_preencode_mode 必须是 "
            f"{sorted(PROMPT_CTX_PREENCODE_MODES)} 之一，当前值: {prompt_ctx_preencode_mode}"
        )


def main():
    parser = argparse.ArgumentParser(description="Generate ODE trajectory data with official two-stage structure")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--chunk_id", type=int, default=0, help="当前 GPU 处理的分片编号（0-indexed）")
    parser.add_argument("--num_chunks", type=int, default=1, help="总分片数（等于 GPU 数量）")
    args = parser.parse_args()

    if args.chunk_id < 0 or args.chunk_id >= args.num_chunks:
        raise ValueError(f"chunk_id={args.chunk_id} 超出范围 [0, {args.num_chunks - 1}]")

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    validate_config(cfg)

    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    all_prompts = load_prompts(cfg["caption_path"], cfg["prompt_column"], int(cfg["max_samples"]))
    total = len(all_prompts)
    chunk_size = math.ceil(total / args.num_chunks)
    chunk_start = args.chunk_id * chunk_size
    chunk_end = min(chunk_start + chunk_size, total)
    prompts = all_prompts[chunk_start:chunk_end]
    logger.info(
        "Chunk %d/%d -> samples [%d, %d) (%d samples)",
        args.chunk_id,
        args.num_chunks,
        chunk_start,
        chunk_end,
        len(prompts),
    )

    checkpoint_path = str(Path(cfg["checkpoint_path"]).expanduser().resolve())
    gemma_root_path = str(Path(cfg["gemma_root"]).expanduser().resolve())
    spatial_upsampler_path = str(Path(cfg["spatial_upsampler_path"]).expanduser().resolve())
    distilled_lora_path = str(Path(cfg["distilled_lora_path"]).expanduser().resolve())
    stage1_model_ledger = ModelLedger(
        dtype=dtype,
        device=device,
        checkpoint_path=checkpoint_path,
        gemma_root_path=gemma_root_path,
        spatial_upsampler_path=spatial_upsampler_path,
        loras=(),
    )
    stage2_model_ledger = stage1_model_ledger.with_additional_loras(
        loras=(
            LoraPathStrengthAndSDOps(
                distilled_lora_path,
                float(cfg.get("distilled_lora_strength", 1.0)),
                LTXV_LORA_COMFY_RENAMING_MAP,
            ),
        )
    )

    detected_defaults = detect_params(checkpoint_path)
    video_guider_params, audio_guider_params = build_guider_params(cfg, detected_defaults)
    logger.info("Detected official defaults from checkpoint: %s", detected_defaults)
    logger.info("Using video guider params: %s", video_guider_params)
    logger.info("Using audio guider params: %s", audio_guider_params)

    negative_prompt = cfg.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    logger.info("Encoding negative prompt...")
    ctx_neg = encode_prompt(negative_prompt, stage1_model_ledger, device)
    prompt_ctx_signature, stage1_signature = build_cache_signatures(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        gemma_root_path=gemma_root_path,
        negative_prompt=negative_prompt,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
        dtype=dtype,
    )

    teacher_steps = int(cfg["teacher_steps"])
    sigmas = LTX2Scheduler().execute(steps=teacher_steps).to(device=device, dtype=torch.float32)
    logger.info(
        "Sigma schedule: %d steps, stage1 save at %s, stage2 save at %s",
        teacher_steps,
        STAGE1_SAVE_INDICES,
        STAGE2_SAVE_INDICES_GLOBAL,
    )

    components = PipelineComponents(dtype=dtype, device=device)
    cache_dir = get_prompt_ctx_cache_dir(cfg, output_dir, args.chunk_id)
    cache_batch_size = get_prompt_ctx_batch_size(cfg)
    prompt_ctx_preencode_mode = get_prompt_ctx_preencode_mode(cfg)
    delete_prompt_ctx_after_use = should_delete_prompt_ctx_after_use(cfg)
    delete_stage1_artifact_after_use = should_delete_stage1_artifact_after_use(cfg)
    cleanup_after_each_sample = should_cleanup_after_each_sample(cfg)
    stage1_artifact_dir = get_stage1_artifact_dir(cache_dir)

    completed = {int(path.stem) for path in output_dir.glob("*.pt")}
    todo = [local_idx for local_idx in range(len(prompts)) if (chunk_start + local_idx) not in completed]
    logger.info(
        "Chunk %d: total=%d done=%d remaining=%d",
        args.chunk_id,
        len(prompts),
        len(prompts) - len(todo),
        len(todo),
    )

    if not todo:
        logger.info("Done. No remaining samples for this chunk.")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    stage1_artifact_dir.mkdir(parents=True, exist_ok=True)
    total_batches = 1 if cache_batch_size == -1 else math.ceil(len(todo) / cache_batch_size)
    logger.info(
        "Prompt cache dir: %s | stage1 artifact dir: %s | batch size: %s | preencode_mode=%s | "
        "delete_prompt_ctx_after_use=%s | delete_stage1_artifact_after_use=%s | cleanup_after_each_sample=%s",
        cache_dir,
        stage1_artifact_dir,
        "all-remaining" if cache_batch_size == -1 else cache_batch_size,
        prompt_ctx_preencode_mode,
        delete_prompt_ctx_after_use,
        delete_stage1_artifact_after_use,
        cleanup_after_each_sample,
    )

    for batch_idx, batch_local_indices in enumerate(iter_batches(todo, cache_batch_size), start=1):
        pending_local_indices = list(batch_local_indices)

        for attempt_idx in range(1, MAX_BATCH_ATTEMPTS + 1):
            if not pending_local_indices:
                break

            prompt_batch = [
                (chunk_start + local_idx, prompts[local_idx])
                for local_idx in pending_local_indices
            ]
            logger.info(
                "Batch %d/%d attempt %d/%d: pre-encode %d prompts before ODE generation",
                batch_idx,
                total_batches,
                attempt_idx,
                MAX_BATCH_ATTEMPTS,
                len(prompt_batch),
            )
            preencode_prompt_batch(
                prompt_batch=prompt_batch,
                model_ledger=stage1_model_ledger,
                cache_dir=cache_dir,
                device=device,
                encode_mode=prompt_ctx_preencode_mode,
                prompt_ctx_signature=prompt_ctx_signature,
            )

            logger.info(
                "Batch %d/%d attempt %d/%d: loading stage1 transformer",
                batch_idx,
                total_batches,
                attempt_idx,
                MAX_BATCH_ATTEMPTS,
            )
            stage1_transformer = stage1_model_ledger.transformer()
            stage1_transformer.eval()
            try:
                for local_idx in tqdm(
                    pending_local_indices,
                    desc=f"GPU{args.chunk_id} Stage1 B{batch_idx} A{attempt_idx}",
                ):
                    global_idx = chunk_start + local_idx
                    prompt = prompts[local_idx]
                    out_path = output_dir / f"{global_idx:05d}.pt"
                    ctx_cache_path = get_prompt_ctx_cache_path(cache_dir, global_idx)
                    stage1_artifact_path = get_stage1_artifact_path(stage1_artifact_dir, global_idx)

                    if out_path.exists():
                        cleanup_sample_cache_files(
                            ctx_cache_path=ctx_cache_path,
                            stage1_artifact_path=stage1_artifact_path,
                            delete_prompt_ctx_after_use=delete_prompt_ctx_after_use,
                            delete_stage1_artifact_after_use=delete_stage1_artifact_after_use,
                        )
                        continue

                    try:
                        ensure_stage1_artifact(
                            prompt=prompt,
                            global_idx=global_idx,
                            stage1_artifact_path=stage1_artifact_path,
                            prompt_ctx_cache_dir=cache_dir,
                            prompt_ctx_signature=prompt_ctx_signature,
                            stage1_signature=stage1_signature,
                            ctx_neg=ctx_neg,
                            stage1_transformer=stage1_transformer,
                            sigmas=sigmas,
                            components=components,
                            cfg=cfg,
                            video_guider_params=video_guider_params,
                            audio_guider_params=audio_guider_params,
                            device=device,
                            dtype=dtype,
                        )
                        if cleanup_after_each_sample:
                            cleanup_memory()
                    except Exception as exc:
                        logger.error("Stage1 global=%d failed: %s", global_idx, exc, exc_info=True)
                        cleanup_memory()
                        continue
            finally:
                del stage1_transformer
                cleanup_memory()

            logger.info(
                "Batch %d/%d attempt %d/%d: loading stage2 transformer",
                batch_idx,
                total_batches,
                attempt_idx,
                MAX_BATCH_ATTEMPTS,
            )
            stage2_transformer = stage2_model_ledger.transformer()
            stage2_transformer.eval()
            try:
                for local_idx in tqdm(
                    pending_local_indices,
                    desc=f"GPU{args.chunk_id} Stage2 B{batch_idx} A{attempt_idx}",
                ):
                    global_idx = chunk_start + local_idx
                    prompt = prompts[local_idx]
                    out_path = output_dir / f"{global_idx:05d}.pt"
                    ctx_cache_path = get_prompt_ctx_cache_path(cache_dir, global_idx)
                    stage1_artifact_path = get_stage1_artifact_path(stage1_artifact_dir, global_idx)
                    artifact_metadata = build_stage1_artifact_metadata(stage1_signature, prompt, global_idx)

                    if out_path.exists():
                        cleanup_sample_cache_files(
                            ctx_cache_path=ctx_cache_path,
                            stage1_artifact_path=stage1_artifact_path,
                            delete_prompt_ctx_after_use=delete_prompt_ctx_after_use,
                            delete_stage1_artifact_after_use=delete_stage1_artifact_after_use,
                        )
                        continue

                    ctx_pos = None
                    stage1_artifact = None
                    try:
                        ctx_pos = try_load_prompt_ctx(
                            ctx_cache_path,
                            expected_metadata=build_prompt_ctx_metadata(prompt_ctx_signature, prompt, global_idx),
                        )
                        if ctx_pos is None:
                            logger.warning(
                                "Stage2 global=%d 缺少有效 prompt ctx，将在本批下一次 attempt 重试。",
                                global_idx,
                            )
                            continue
                        stage1_artifact = try_load_stage1_artifact(
                            stage1_artifact_path,
                            expected_metadata=artifact_metadata,
                        )
                        if stage1_artifact is None:
                            logger.warning(
                                "Stage2 global=%d 缺少有效 stage1 artifact，将在本批下一次 attempt 重试。",
                                global_idx,
                            )
                            continue

                        data = generate_stage2_sample(
                            ctx_pos=ctx_pos,
                            global_idx=global_idx,
                            stage1_artifact=stage1_artifact,
                            stage2_transformer=stage2_transformer,
                            stage1_model_ledger=stage1_model_ledger,
                            stage2_model_ledger=stage2_model_ledger,
                            sigmas=sigmas,
                            components=components,
                            cfg=cfg,
                            device=device,
                            dtype=dtype,
                        )
                        atomic_torch_save(data, out_path)
                        del data
                        cleanup_sample_cache_files(
                            ctx_cache_path=ctx_cache_path,
                            stage1_artifact_path=stage1_artifact_path,
                            delete_prompt_ctx_after_use=delete_prompt_ctx_after_use,
                            delete_stage1_artifact_after_use=delete_stage1_artifact_after_use,
                        )
                        if cleanup_after_each_sample:
                            cleanup_memory()
                    except Exception as exc:
                        logger.error("Stage2 global=%d failed: %s", global_idx, exc, exc_info=True)
                        cleanup_memory()
                        continue
                    finally:
                        del ctx_pos, stage1_artifact
            finally:
                del stage2_transformer
                cleanup_memory()

            next_pending_local_indices = [
                local_idx
                for local_idx in pending_local_indices
                if not (output_dir / f"{chunk_start + local_idx:05d}.pt").exists()
            ]
            if not next_pending_local_indices:
                pending_local_indices = []
                break

            remaining_globals = [chunk_start + local_idx for local_idx in next_pending_local_indices[:10]]
            if attempt_idx < MAX_BATCH_ATTEMPTS:
                logger.warning(
                    "Batch %d/%d attempt %d/%d completed with %d unfinished samples; retrying. "
                    "Example globals=%s",
                    batch_idx,
                    total_batches,
                    attempt_idx,
                    MAX_BATCH_ATTEMPTS,
                    len(next_pending_local_indices),
                    remaining_globals,
                )
            pending_local_indices = next_pending_local_indices

        if pending_local_indices:
            logger.error(
                "Batch %d/%d 仍有 %d 条样本在 %d 次 attempt 后未完成。Example globals=%s",
                batch_idx,
                total_batches,
                len(pending_local_indices),
                MAX_BATCH_ATTEMPTS,
                [chunk_start + local_idx for local_idx in pending_local_indices[:10]],
            )

    logger.info("Done. Data saved to %s", output_dir)


if __name__ == "__main__":
    main()
