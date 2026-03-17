"""
生成 ODE 轨迹数据（蒸馏模型版本）。

蒸馏模型使用固定的 sigma 调度，且不需要 CFG（无负向 prompt）。

Stage 1（半分辨率 512x768）:
  - 使用 DISTILLED_SIGMA_VALUES 共 8 步去噪，保存全部 9 个状态（含初始噪声）
  - sigma 值: [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

Stage 2（全分辨率 1024x1536）:
  - 以 Stage 1 clean latent upsample 后加噪到 sigma=0.909375 作为起点
  - 使用 STAGE_2_DISTILLED_SIGMA_VALUES 共 3 步去噪，保存全部 4 个状态
  - sigma 值: [0.909375, 0.725, 0.421875, 0.0]

每个 sample 保存为 {index:05d}.pt，内容:
  {
    'prompt': str,
    'stage1_video_traj': Tensor [9, 128, 16, 16, 24],  # bfloat16，Stage1 512x768 latent
    'stage1_audio_traj': Tensor [9, 8, 126, 16],        # bfloat16
    'stage1_sigmas':     Tensor [9],                    # float32
    'stage2_video_traj': Tensor [4, 128, 16, 32, 48],  # bfloat16，Stage2 1024x1536 latent
    'stage2_audio_traj': Tensor [4, 8, 126, 16],        # bfloat16
    'stage2_sigmas':     Tensor [4],                    # float32
  }

用法:
    cd /inspire/hdd/project/agileapplication/zhangkaipeng-24043/wgx/myltx
    python ode/gen_ode_data_distilled.py --config ode/configs/gen_ode_data_distilled.yaml
"""

import argparse
import csv
import hashlib
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
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.upsampler import upsample_video
from ltx_core.text_encoders.gemma import EmbeddingsProcessorOutput
from ltx_core.types import VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.constants import (
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from ltx_pipelines.utils.helpers import (
    noise_audio_state,
    noise_video_state,
    post_process_latent,
    simple_denoising_func,
)
from ltx_pipelines.utils.types import PipelineComponents

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 蒸馏模型的 sigma 值（直接来自官方常量）
STAGE1_SIGMAS = DISTILLED_SIGMA_VALUES        # [1.0, ..., 0.0], 9 个值, 8 步
STAGE2_SIGMAS = STAGE_2_DISTILLED_SIGMA_VALUES  # [0.909375, ..., 0.0], 4 个值, 3 步
DISTILLED_CHECKPOINT_BASENAME = "ltx-2.3-22b-distilled.safetensors"
PROMPT_CTX_PREENCODE_MODES = {"staged", "fast", "sequential"}
MAX_TORCH_SEED = (1 << 63) - 1
NOISE_SEED_SCHEME = "per-stage-per-modality-v1"


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def cleanup_memory():
    """在 CPU 环境下跳过 CUDA 清理，避免和 device fallback 冲突。"""
    if torch.cuda.is_available():
        from ltx_pipelines.utils import cleanup_memory as pipeline_cleanup_memory

        pipeline_cleanup_memory()


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype)


def validate_distilled_config(cfg: dict) -> None:
    checkpoint_path = Path(cfg["checkpoint_path"]).expanduser().resolve()
    gemma_root = Path(cfg["gemma_root"]).expanduser().resolve()
    spatial_upsampler_path = Path(cfg["spatial_upsampler_path"]).expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"蒸馏 checkpoint 不存在: {checkpoint_path}")
    if checkpoint_path.name != DISTILLED_CHECKPOINT_BASENAME:
        raise ValueError(
            "当前配置没有指向 full distilled checkpoint。"
            f" 期望文件名: {DISTILLED_CHECKPOINT_BASENAME}，实际: {checkpoint_path.name}"
        )
    if not gemma_root.is_dir():
        raise FileNotFoundError(f"Gemma 目录不存在: {gemma_root}")
    if not spatial_upsampler_path.is_file():
        raise FileNotFoundError(f"Spatial upsampler 不存在: {spatial_upsampler_path}")

    stage1_h = cfg["stage1_height"]
    stage1_w = cfg["stage1_width"]
    if stage1_h <= 0 or stage1_w <= 0:
        raise ValueError("stage1_height 和 stage1_width 必须为正整数")
    if stage1_h % 2 != 0 or stage1_w % 2 != 0:
        raise ValueError("stage1_height 和 stage1_width 必须是偶数，才能严格对齐 two-stage x2 upsample")
    prompt_ctx_batch_size = int(cfg.get("prompt_ctx_batch_size", 512))
    if prompt_ctx_batch_size == 0 or prompt_ctx_batch_size < -1:
        raise ValueError("prompt_ctx_batch_size 必须为正整数，或设为 -1 表示整 chunk 预编码")
    prompt_ctx_preencode_mode = str(cfg.get("prompt_ctx_preencode_mode", "staged")).strip().lower()
    if prompt_ctx_preencode_mode not in PROMPT_CTX_PREENCODE_MODES:
        raise ValueError(
            "prompt_ctx_preencode_mode 必须是 "
            f"{sorted(PROMPT_CTX_PREENCODE_MODES)} 之一，当前值: {prompt_ctx_preencode_mode}"
        )


def get_prompt_ctx_batch_size(cfg: dict) -> int:
    return int(cfg.get("prompt_ctx_batch_size", 512))


def get_prompt_ctx_preencode_mode(cfg: dict) -> str:
    return str(cfg.get("prompt_ctx_preencode_mode", "staged")).strip().lower()


def should_delete_prompt_ctx_after_use(cfg: dict) -> bool:
    return bool(cfg.get("delete_prompt_ctx_after_use", True))


def should_offload_transformer_for_upsample(cfg: dict) -> bool:
    return bool(cfg.get("offload_transformer_for_upsample", True))


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

def load_prompts(csv_path: str, prompt_column: str, max_samples: int) -> list[str]:
    prompts = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_samples > 0 and i >= max_samples:
                break
            prompts.append(row[prompt_column])
    logger.info("Loaded %d prompts from %s", len(prompts), csv_path)
    return prompts


def _raw_to_cpu(raw):
    """将 text_encoder.encode() 的原始输出（含嵌套 tensor）递归移到 CPU。"""
    if isinstance(raw, torch.Tensor):
        return raw.cpu()
    if isinstance(raw, (tuple, list)):
        moved = [_raw_to_cpu(x) for x in raw]
        return type(raw)(moved)
    if hasattr(raw, "keys"):
        return {k: _raw_to_cpu(v) for k, v in raw.items()}
    return raw


def _raw_to_device(raw, device):
    """将 CPU 上的 raw 输出递归移回 GPU。"""
    if isinstance(raw, torch.Tensor):
        return raw.to(device)
    if isinstance(raw, (tuple, list)):
        moved = [_raw_to_device(x, device) for x in raw]
        return type(raw)(moved)
    if hasattr(raw, "keys"):
        return {k: _raw_to_device(v, device) for k, v in raw.items()}
    return raw


def encode_prompt(prompt: str, model_ledger: ModelLedger, device: torch.device):
    """
    编码单条 prompt，严格两阶段：text_encoder → del → embeddings_processor → del。
    返回 EmbeddingsProcessorOutput（所有 tensor 在 CPU）。
    """
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


def _ctx_to_device(ctx, device):
    """将 EmbeddingsProcessorOutput 的 tensor 移到指定设备。"""
    return ctx._replace(
        video_encoding=ctx.video_encoding.to(device),
        audio_encoding=ctx.audio_encoding.to(device) if ctx.audio_encoding is not None else None,
        attention_mask=ctx.attention_mask.to(device),
    )


def _ctx_to_cpu(ctx) -> EmbeddingsProcessorOutput:
    """将 EmbeddingsProcessorOutput 移到 CPU，便于落盘缓存。"""
    return EmbeddingsProcessorOutput(
        video_encoding=ctx.video_encoding.to("cpu", dtype=torch.bfloat16),
        audio_encoding=ctx.audio_encoding.to("cpu", dtype=torch.bfloat16) if ctx.audio_encoding is not None else None,
        attention_mask=ctx.attention_mask.to("cpu", dtype=torch.uint8),
    )


def get_prompt_ctx_cache_path(cache_dir: Path, global_idx: int) -> Path:
    return cache_dir / f"{global_idx:05d}.pt"


def save_prompt_ctx(ctx: EmbeddingsProcessorOutput, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "video_encoding": ctx.video_encoding.contiguous(),
            "audio_encoding": ctx.audio_encoding.contiguous() if ctx.audio_encoding is not None else None,
            "attention_mask": ctx.attention_mask.contiguous(),
        },
        tmp_path,
    )
    tmp_path.replace(path)


def load_prompt_ctx(path: Path) -> EmbeddingsProcessorOutput:
    payload = torch.load(path, map_location="cpu")
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


def preencode_prompt_batch_staged(
    missing: list[tuple[int, str]],
    model_ledger: ModelLedger,
    cache_dir: Path,
    device: torch.device,
) -> None:
    """严格错峰的预编码流程：text_encoder 和 embeddings_processor 不同时驻留 GPU。"""
    raw_batch: list[tuple[int, object]] = []
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
            raw_gpu = _raw_to_device(raw_cpu, device)
            ctx = embeddings_processor.process_hidden_states(*raw_gpu)
            save_prompt_ctx(_ctx_to_cpu(ctx), cache_path)
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
) -> None:
    """
    先单独执行 prompt 编码并落盘，避免与 transformer 峰值显存叠加。

    支持三种模式：
      - staged: 先 text_encoder 编到 CPU，再单独加载 embeddings_processor 落盘
      - fast: 同时常驻 text_encoder + embeddings_processor，速度更快但峰值更高
      - sequential: 单 prompt 严格两阶段编码，最稳但最慢
    """
    missing = [
        (global_idx, prompt)
        for global_idx, prompt in prompt_batch
        if not get_prompt_ctx_cache_path(cache_dir, global_idx).exists()
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
                save_prompt_ctx(ctx, cache_path)
                del ctx
            cleanup_memory()
        return

    if encode_mode == "sequential":
        for global_idx, prompt in tqdm(missing, desc="prompt-preencode-sequential", leave=False):
            cache_path = get_prompt_ctx_cache_path(cache_dir, global_idx)
            ctx = encode_prompt(prompt, model_ledger, device)
            save_prompt_ctx(ctx, cache_path)
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
            save_prompt_ctx(_ctx_to_cpu(ctx), cache_path)
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
            save_prompt_ctx(ctx, cache_path)
            del ctx
    finally:
        text_encoder = None
        embeddings_processor = None
        cleanup_memory()


def _extract_spatial_latent(state, tools) -> torch.Tensor:
    """从 patchified LatentState 中提取空间格式 latent，返回 [1, C, ...] tensor（bfloat16，CPU）。"""
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
) -> tuple[object, object, list[torch.Tensor], list[torch.Tensor]]:
    """
    执行 Euler 去噪循环，保存每一步的 latent 状态。

    蒸馏模型步数很少，全部保存。
    返回: (final_video_state, final_audio_state, video_traj, audio_traj)
    """
    video_traj: list[torch.Tensor] = []
    audio_traj: list[torch.Tensor] = []

    # 保存初始状态（index 0，纯噪声）
    video_traj.append(_extract_spatial_latent(video_state, video_tools))
    audio_traj.append(_extract_spatial_latent(audio_state, audio_tools))

    for step_idx in range(len(sigmas) - 1):
        denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_idx)

        denoised_video = post_process_latent(
            denoised_video, video_state.denoise_mask, video_state.clean_latent
        )
        denoised_audio = post_process_latent(
            denoised_audio, audio_state.denoise_mask, audio_state.clean_latent
        )

        video_state = replace(
            video_state,
            latent=stepper.step(video_state.latent, denoised_video, sigmas, step_idx),
        )
        audio_state = replace(
            audio_state,
            latent=stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx),
        )

        # 保存每一步
        video_traj.append(_extract_spatial_latent(video_state, video_tools))
        audio_traj.append(_extract_spatial_latent(audio_state, audio_tools))

    return video_state, audio_state, video_traj, audio_traj


@torch.inference_mode()
def generate_sample(
    prompt: str,
    global_idx: int,
    ctx_pos,
    transformer,
    model_ledger: ModelLedger,
    components: PipelineComponents,
    cfg: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """
    对单个 prompt 使用蒸馏模型生成 Stage1 + Stage2 ODE 轨迹。
    蒸馏模型不需要 CFG，只用 simple_denoising_func。
    """
    seed = cfg["seed"]
    num_frames = cfg["num_frames"]
    frame_rate = cfg["frame_rate"]
    stage1_h = cfg["stage1_height"]
    stage1_w = cfg["stage1_width"]
    stage2_h = stage1_h * 2
    stage2_w = stage1_w * 2

    noise_seeds = build_noise_seed_metadata(seed, global_idx)
    stage1_noise_seeds = noise_seeds["stage1"]
    stage2_noise_seeds = noise_seeds["stage2"]
    stepper = EulerDiffusionStep()
    offload_transformer_for_upsample = should_offload_transformer_for_upsample(cfg)

    if next(transformer.parameters()).device != device:
        transformer.to(device)
        transformer.eval()
        cleanup_memory()

    v_ctx_pos = ctx_pos.video_encoding.to(dtype=dtype, device=device)
    a_ctx_pos = ctx_pos.audio_encoding.to(dtype=dtype, device=device)

    # 蒸馏模型：无 CFG，单次 forward pass
    denoise_fn = simple_denoising_func(
        video_context=v_ctx_pos,
        audio_context=a_ctx_pos,
        transformer=transformer,
    )

    # Stage 1 sigmas
    s1_sigmas = torch.tensor(STAGE1_SIGMAS, dtype=torch.float32, device=device)

    # ── Stage 1：半分辨率 ────────────────────────────────────────────
    stage1_shape = VideoPixelShape(
        batch=1, frames=num_frames, height=stage1_h, width=stage1_w, fps=frame_rate
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
            sigmas=s1_sigmas,
            stepper=stepper,
            denoise_fn=denoise_fn,
        )

    s1_final_video = _extract_spatial_latent(video_state_s1, video_tools_s1)
    s1_final_audio = _extract_spatial_latent(audio_state_s1, audio_tools_s1)

    del denoise_fn, video_state_s1, audio_state_s1, video_tools_s1, audio_tools_s1
    cleanup_memory()

    if offload_transformer_for_upsample and device.type == "cuda":
        del v_ctx_pos, a_ctx_pos
        transformer.cpu()
        cleanup_memory()

    # ── Stage 2：upsample ────────────────────────────────────────────
    video_encoder = model_ledger.video_encoder()
    upsampler = model_ledger.spatial_upsampler()
    upsampled_video = upsample_video(
        latent=s1_final_video.to(dtype=dtype, device=device),
        video_encoder=video_encoder,
        upsampler=upsampler,
    )
    del video_encoder, upsampler
    cleanup_memory()

    # Stage 2 sigmas
    s2_sigmas = torch.tensor(STAGE2_SIGMAS, dtype=torch.float32, device=device)
    stage2_start_sigma = s2_sigmas[0].item()

    stage2_shape = VideoPixelShape(
        batch=1, frames=num_frames, height=stage2_h, width=stage2_w, fps=frame_rate
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

    if offload_transformer_for_upsample and device.type == "cuda":
        transformer.to(device)
        transformer.eval()
        v_ctx_pos = ctx_pos.video_encoding.to(dtype=dtype, device=device)
        a_ctx_pos = ctx_pos.audio_encoding.to(dtype=dtype, device=device)
        cleanup_memory()

    # Stage 2 也无 CFG
    denoise_fn_s2 = simple_denoising_func(
        video_context=v_ctx_pos,
        audio_context=a_ctx_pos,
        transformer=transformer,
    )
    with autocast_context(device, dtype):
        _, _, s2_video_traj, s2_audio_traj = run_ode_loop(
            video_state=video_state_s2,
            audio_state=audio_state_s2,
            video_tools=video_tools_s2,
            audio_tools=audio_tools_s2,
            sigmas=s2_sigmas,
            stepper=stepper,
            denoise_fn=denoise_fn_s2,
        )

    s1_sigma_vals = torch.tensor(STAGE1_SIGMAS, dtype=torch.float32)
    s2_sigma_vals = torch.tensor(STAGE2_SIGMAS, dtype=torch.float32)
    del denoise_fn_s2, video_state_s2, audio_state_s2, video_tools_s2, audio_tools_s2
    del v_ctx_pos, a_ctx_pos, s1_final_video, s1_final_audio
    cleanup_memory()

    return {
        "prompt": prompt,
        "stage1_video_traj": torch.stack(s1_video_traj, dim=0).squeeze(1),  # [9, C, F, H1, W1]
        "stage1_audio_traj": torch.stack(s1_audio_traj, dim=0).squeeze(1),  # [9, 8, 126, 16]
        "stage1_sigmas": s1_sigma_vals,
        "stage2_video_traj": torch.stack(s2_video_traj, dim=0).squeeze(1),  # [4, C, F, H2, W2]
        "stage2_audio_traj": torch.stack(s2_audio_traj, dim=0).squeeze(1),  # [4, 8, 126, 16]
        "stage2_sigmas": s2_sigma_vals,
        "noise_seeds": noise_seeds,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate ODE trajectory data using distilled LTX-2 model")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--chunk_id", type=int, default=0, help="当前 GPU 处理的分片编号（0-indexed）")
    parser.add_argument("--num_chunks", type=int, default=1, help="总分片数（等于 GPU 数量）")
    args = parser.parse_args()

    if args.chunk_id < 0 or args.chunk_id >= args.num_chunks:
        raise ValueError(f"chunk_id={args.chunk_id} 超出范围 [0, {args.num_chunks - 1}]")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    validate_distilled_config(cfg)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # ── 加载全量 prompts，然后按 chunk 切分 ───────────────────────────────
    all_prompts = load_prompts(cfg["caption_path"], cfg["prompt_column"], cfg["max_samples"])
    total = len(all_prompts)
    chunk_size = math.ceil(total / args.num_chunks)
    chunk_start = args.chunk_id * chunk_size
    chunk_end = min(chunk_start + chunk_size, total)
    prompts = all_prompts[chunk_start:chunk_end]
    logger.info(
        "Chunk %d/%d  →  samples [%d, %d)  (%d samples)",
        args.chunk_id, args.num_chunks, chunk_start, chunk_end, len(prompts),
    )

    model_ledger = ModelLedger(
        dtype=dtype,
        device=device,
        checkpoint_path=cfg["checkpoint_path"],
        gemma_root_path=cfg["gemma_root"],
        spatial_upsampler_path=cfg["spatial_upsampler_path"],
        loras=(),
    )

    # 蒸馏模型使用固定 sigma 调度
    logger.info(
        "Distilled sigma schedule: Stage1 %d values, Stage2 %d values",
        len(STAGE1_SIGMAS), len(STAGE2_SIGMAS),
    )
    logger.info("Using distilled checkpoint: %s", Path(cfg["checkpoint_path"]).resolve())

    components = PipelineComponents(dtype=dtype, device=device)
    cache_dir = get_prompt_ctx_cache_dir(cfg, output_dir, args.chunk_id)
    cache_batch_size = get_prompt_ctx_batch_size(cfg)
    prompt_ctx_preencode_mode = get_prompt_ctx_preencode_mode(cfg)
    delete_prompt_ctx_after_use = should_delete_prompt_ctx_after_use(cfg)
    cleanup_after_each_sample = should_cleanup_after_each_sample(cfg)

    # ── 主循环 ──────────────────────────────────────────────────────────────
    completed = {int(p.stem) for p in output_dir.glob("*.pt")}
    todo = [
        local_idx for local_idx in range(len(prompts))
        if (chunk_start + local_idx) not in completed
    ]
    logger.info(
        "Chunk %d: total=%d  done=%d  remaining=%d",
        args.chunk_id, len(prompts), len(prompts) - len(todo), len(todo),
    )

    if not todo:
        logger.info("Done. No remaining samples for this chunk.")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    total_batches = 1 if cache_batch_size == -1 else math.ceil(len(todo) / cache_batch_size)
    logger.info(
        "Prompt cache dir: %s | batch size: %s | preencode_mode=%s | delete_after_use=%s | "
        "offload_transformer_for_upsample=%s | cleanup_after_each_sample=%s",
        cache_dir,
        "all-remaining" if cache_batch_size == -1 else cache_batch_size,
        prompt_ctx_preencode_mode,
        delete_prompt_ctx_after_use,
        should_offload_transformer_for_upsample(cfg),
        cleanup_after_each_sample,
    )

    for batch_idx, batch_local_indices in enumerate(iter_batches(todo, cache_batch_size), start=1):
        prompt_batch = [
            (chunk_start + local_idx, prompts[local_idx])
            for local_idx in batch_local_indices
        ]
        logger.info(
            "Batch %d/%d: pre-encode %d prompts before loading transformer",
            batch_idx, total_batches, len(prompt_batch),
        )
        preencode_prompt_batch(
            prompt_batch=prompt_batch,
            model_ledger=model_ledger,
            cache_dir=cache_dir,
            device=device,
            encode_mode=prompt_ctx_preencode_mode,
        )

        logger.info("Batch %d/%d: loading transformer for ODE generation", batch_idx, total_batches)
        transformer = model_ledger.transformer()
        transformer.eval()

        try:
            for local_idx in tqdm(batch_local_indices, desc=f"GPU{args.chunk_id} ODE-distilled B{batch_idx}"):
                global_idx = chunk_start + local_idx
                out_path = output_dir / f"{global_idx:05d}.pt"
                prompt = prompts[local_idx]
                ctx_cache_path = get_prompt_ctx_cache_path(cache_dir, global_idx)

                if out_path.exists():
                    if delete_prompt_ctx_after_use and ctx_cache_path.exists():
                        ctx_cache_path.unlink()
                    continue

                try:
                    try:
                        ctx_pos = load_prompt_ctx(ctx_cache_path)
                    except Exception as cache_error:
                        if ctx_cache_path.exists():
                            ctx_cache_path.unlink()
                        logger.error(
                            "Prompt cache for global=%d is invalid and has been removed: %s",
                            global_idx,
                            cache_error,
                            exc_info=True,
                        )
                        continue

                    data = generate_sample(
                        prompt=prompt,
                        global_idx=global_idx,
                        ctx_pos=ctx_pos,
                        transformer=transformer,
                        model_ledger=model_ledger,
                        components=components,
                        cfg=cfg,
                        device=device,
                        dtype=dtype,
                    )
                    del ctx_pos
                    atomic_torch_save(data, out_path)
                    del data
                    if delete_prompt_ctx_after_use and ctx_cache_path.exists():
                        ctx_cache_path.unlink()
                    if cleanup_after_each_sample:
                        cleanup_memory()
                except Exception as e:
                    logger.error("Sample global=%d failed: %s", global_idx, e, exc_info=True)
                    if next(transformer.parameters()).device != device:
                        transformer.to(device)
                        transformer.eval()
                    cleanup_memory()
                    continue
        finally:
            del transformer
            cleanup_memory()

    logger.info("Done. Data saved to %s", output_dir)


if __name__ == "__main__":
    main()
