"""
生成 ODE 轨迹数据，用于将蒸馏模型训练成 causal 版本。

教师模型（非蒸馏）以 N=40 步运行，在与蒸馏模型对齐的 sigma 值处保存中间 latent 状态。

Stage 1（半分辨率 512x768）:
  - 完整运行 40 步去噪，保存 9 个 checkpoint（对应 DISTILLED_SIGMA_VALUES）
  - 保存索引: [0, 2, 3, 5, 6, 17, 29, 36, 40]

Stage 2（全分辨率 1024x1536）:
  - 以 Stage 1 clean latent upsample 后加噪到 sigma[17] 作为起点
  - 继续跑 sigma[17] → sigma[40]（23 步），保存 4 个 checkpoint
  - 保存索引（全局）: [17, 29, 36, 40]

每个 sample 保存为 {index:05d}.pt，内容:
  {
    'prompt': str,
    'stage1_video_traj': Tensor [9, 128, 16, 16, 24],  # bfloat16，Stage1 512x768 latent
    'stage1_audio_traj': Tensor [9, 8, 126, 16],        # bfloat16，121帧@24fps → 音频126帧
    'stage1_sigmas':     Tensor [9],                    # float32，checkpoint 处的 sigma 值
    'stage2_video_traj': Tensor [4, 128, 16, 32, 48],  # bfloat16，Stage2 1024x1536 latent
    'stage2_audio_traj': Tensor [4, 8, 126, 16],        # bfloat16
    'stage2_sigmas':     Tensor [4],                       # float32
  }

用法:
    cd /mnt/shared-storage-user/worldmodel-shared/wgx/LTX-2
    python ode/gen_ode_data.py --config ode/configs/gen_ode_data.yaml
"""

import argparse
import csv
import logging
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, "packages/ltx-core/src")
sys.path.insert(0, "packages/ltx-pipelines/src")

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.upsampler import upsample_video
from ltx_core.types import VideoPixelShape
from ltx_pipelines.utils import ModelLedger, cleanup_memory
from ltx_pipelines.utils.helpers import (
    modality_from_latent_state,
    noise_audio_state,
    noise_video_state,
    post_process_latent,
    simple_denoising_func,
)
from ltx_pipelines.utils.types import PipelineComponents

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── 与蒸馏模型对齐的保存索引 ──────────────────────────────────────────────────
# 对应 DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
# 在 N=40 步的 LTX2Scheduler 中的最近邻索引（max_diff=0.003）
STAGE1_SAVE_INDICES = [0, 2, 3, 5, 6, 17, 29, 36, 40]

# 对应 STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]
STAGE2_SAVE_INDICES_GLOBAL = [17, 29, 36, 40]  # 全局索引
STAGE2_START_IDX = 17                            # Stage 2 从该全局索引处开始


# ── 工具函数 ───────────────────────────────────────────────────────────────────

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
    # transformers ModelOutput（dict-like）
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
    # 阶段 1：text_encoder
    text_encoder = model_ledger.text_encoder()
    raw = text_encoder.encode(prompt)
    raw_cpu = _raw_to_cpu(raw)
    del raw, text_encoder
    cleanup_memory()

    # 阶段 2：embeddings_processor
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
    save_indices: list[int],
) -> tuple[object, object, list[torch.Tensor], list[torch.Tensor]]:
    """
    执行 Euler 去噪循环，在指定的全局 sigma 索引处保存 latent。

    save_indices: 相对于 sigmas 的索引列表（包含 0 和最后一步）。
    返回: (final_video_state, final_audio_state, video_traj, audio_traj)
      video_traj / audio_traj: list of CPU tensors, 长度 = len(save_indices)
    """
    save_set = set(save_indices)
    video_traj: list[torch.Tensor] = []
    audio_traj: list[torch.Tensor] = []

    # 保存初始状态（索引 0）
    if 0 in save_set:
        video_traj.append(_extract_spatial_latent(video_state, video_tools))
        audio_traj.append(_extract_spatial_latent(audio_state, audio_tools))

    for step_idx in range(len(sigmas) - 1):
        # 去噪预测
        denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_idx)

        # 应用 denoise mask（保留条件 latent 不变）
        denoised_video = post_process_latent(
            denoised_video, video_state.denoise_mask, video_state.clean_latent
        )
        denoised_audio = post_process_latent(
            denoised_audio, audio_state.denoise_mask, audio_state.clean_latent
        )

        # Euler 步进：latent 更新到 sigmas[step_idx + 1]
        video_state = replace(
            video_state,
            latent=stepper.step(video_state.latent, denoised_video, sigmas, step_idx),
        )
        audio_state = replace(
            audio_state,
            latent=stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx),
        )

        # 保存（step_idx + 1 对应 sigmas[step_idx + 1]）
        global_idx = step_idx + 1
        if global_idx in save_set:
            video_traj.append(_extract_spatial_latent(video_state, video_tools))
            audio_traj.append(_extract_spatial_latent(audio_state, audio_tools))

    return video_state, audio_state, video_traj, audio_traj


def make_cfg_denoise_fn(transformer, v_ctx_pos, v_ctx_neg, a_ctx_pos, a_ctx_neg, v_cfg, a_cfg):
    """返回带 CFG 的去噪函数（每步两次 forward pass：正向 + 负向）。"""

    def denoise_fn(video_state, audio_state, sigmas, step_idx):
        sigma = sigmas[step_idx]

        pos_v = modality_from_latent_state(video_state, v_ctx_pos, sigma)
        pos_a = modality_from_latent_state(audio_state, a_ctx_pos, sigma)
        denoised_v_pos, denoised_a_pos = transformer(video=pos_v, audio=pos_a, perturbations=None)

        neg_v = modality_from_latent_state(video_state, v_ctx_neg, sigma)
        neg_a = modality_from_latent_state(audio_state, a_ctx_neg, sigma)
        denoised_v_neg, denoised_a_neg = transformer(video=neg_v, audio=neg_a, perturbations=None)

        # CFG: x_neg + scale * (x_pos - x_neg)
        denoised_v = denoised_v_neg + v_cfg * (denoised_v_pos - denoised_v_neg)
        denoised_a = denoised_a_neg + a_cfg * (denoised_a_pos - denoised_a_neg)
        return denoised_v, denoised_a

    return denoise_fn


@torch.inference_mode()
def generate_sample(
    prompt: str,
    ctx_pos,
    ctx_neg,
    transformer,
    model_ledger: ModelLedger,
    sigmas: torch.Tensor,
    components: PipelineComponents,
    cfg: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """
    对单个 prompt 生成 Stage1 + Stage2 ODE 轨迹。
    返回包含两阶段轨迹的 dict，可直接 torch.save。
    """
    seed = cfg["seed"]
    v_cfg = cfg["video_cfg_scale"]
    a_cfg = cfg["audio_cfg_scale"]
    num_frames = cfg["num_frames"]
    frame_rate = cfg["frame_rate"]
    stage1_h = cfg["stage1_height"]
    stage1_w = cfg["stage1_width"]
    stage2_h = stage1_h * 2
    stage2_w = stage1_w * 2

    generator = torch.Generator(device=device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator)
    stepper = EulerDiffusionStep()

    # 将 context 移到 GPU
    v_ctx_pos = ctx_pos.video_encoding.to(dtype=dtype, device=device)
    a_ctx_pos = ctx_pos.audio_encoding.to(dtype=dtype, device=device)
    v_ctx_neg = ctx_neg.video_encoding.to(dtype=dtype, device=device)
    a_ctx_neg = ctx_neg.audio_encoding.to(dtype=dtype, device=device)

    denoise_fn = make_cfg_denoise_fn(transformer, v_ctx_pos, v_ctx_neg, a_ctx_pos, a_ctx_neg, v_cfg, a_cfg)

    # ── Stage 1：半分辨率完整去噪 ────────────────────────────────────────────
    stage1_shape = VideoPixelShape(
        batch=1, frames=num_frames, height=stage1_h, width=stage1_w, fps=frame_rate
    )
    video_state_s1, video_tools_s1 = noise_video_state(
        output_shape=stage1_shape,
        noiser=noiser,
        conditionings=[],
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=1.0,
    )
    audio_state_s1, audio_tools_s1 = noise_audio_state(
        output_shape=stage1_shape,
        noiser=noiser,
        conditionings=[],
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=1.0,
    )

    with torch.amp.autocast("cuda", dtype=dtype):
        video_state_s1, audio_state_s1, s1_video_traj, s1_audio_traj = run_ode_loop(
            video_state=video_state_s1,
            audio_state=audio_state_s1,
            video_tools=video_tools_s1,
            audio_tools=audio_tools_s1,
            sigmas=sigmas,
            stepper=stepper,
            denoise_fn=denoise_fn,
            save_indices=STAGE1_SAVE_INDICES,
        )

    # Stage 1 最终 clean latent（空间格式，CPU）
    s1_final_video = _extract_spatial_latent(video_state_s1, video_tools_s1)
    s1_final_audio = _extract_spatial_latent(audio_state_s1, audio_tools_s1)

    # ── 释放 Stage 1 GPU 状态 ────────────────────────────────────────────────
    del video_state_s1, audio_state_s1, video_tools_s1, audio_tools_s1
    cleanup_memory()

    # ── Stage 2：upsample（临时加载 video_encoder + upsampler）────────────────
    video_encoder = model_ledger.video_encoder()
    upsampler = model_ledger.spatial_upsampler()
    upsampled_video = upsample_video(
        latent=s1_final_video.to(dtype=dtype, device=device),
        video_encoder=video_encoder,
        upsampler=upsampler,
    )
    del video_encoder, upsampler
    cleanup_memory()

    stage2_start_sigma = sigmas[STAGE2_START_IDX].item()
    sub_sigmas = sigmas[STAGE2_START_IDX:]
    stage2_save_local = [g - STAGE2_START_IDX for g in STAGE2_SAVE_INDICES_GLOBAL]

    stage2_shape = VideoPixelShape(
        batch=1, frames=num_frames, height=stage2_h, width=stage2_w, fps=frame_rate
    )
    video_state_s2, video_tools_s2 = noise_video_state(
        output_shape=stage2_shape,
        noiser=noiser,
        conditionings=[],
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=stage2_start_sigma,
        initial_latent=upsampled_video,
    )
    audio_state_s2, audio_tools_s2 = noise_audio_state(
        output_shape=stage2_shape,
        noiser=noiser,
        conditionings=[],
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=stage2_start_sigma,
        initial_latent=s1_final_audio.to(dtype=dtype, device=device),
    )
    del upsampled_video
    cleanup_memory()

    # Stage 2 与官方一致：无 CFG，单次 forward pass（simple_denoising_func）
    denoise_fn_s2 = simple_denoising_func(
        video_context=v_ctx_pos,
        audio_context=a_ctx_pos,
        transformer=transformer,
    )
    with torch.amp.autocast("cuda", dtype=dtype):
        _, _, s2_video_traj, s2_audio_traj = run_ode_loop(
            video_state=video_state_s2,
            audio_state=audio_state_s2,
            video_tools=video_tools_s2,
            audio_tools=audio_tools_s2,
            sigmas=sub_sigmas,
            stepper=stepper,
            denoise_fn=denoise_fn_s2,
            save_indices=stage2_save_local,
        )

    # sigma 值（float32）
    s1_sigma_vals = torch.tensor(
        [sigmas[i].item() for i in STAGE1_SAVE_INDICES], dtype=torch.float32
    )
    s2_sigma_vals = torch.tensor(
        [sigmas[i].item() for i in STAGE2_SAVE_INDICES_GLOBAL], dtype=torch.float32
    )

    return {
        "prompt": prompt,
        "stage1_video_traj": torch.stack(s1_video_traj, dim=0).squeeze(1),  # [9, C, F, H1, W1]
        "stage1_audio_traj": torch.stack(s1_audio_traj, dim=0).squeeze(1),  # [9, 8, 126, 16]
        "stage1_sigmas": s1_sigma_vals,
        "stage2_video_traj": torch.stack(s2_video_traj, dim=0).squeeze(1),  # [4, C, F, H2, W2]
        "stage2_audio_traj": torch.stack(s2_audio_traj, dim=0).squeeze(1),  # [4, 8, 126, 16]
        "stage2_sigmas": s2_sigma_vals,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate ODE trajectory data for LTX-2 distillation")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--chunk_id", type=int, default=0, help="当前 GPU 处理的分片编号（0-indexed）")
    parser.add_argument("--num_chunks", type=int, default=1, help="总分片数（等于 GPU 数量）")
    args = parser.parse_args()

    if args.chunk_id < 0 or args.chunk_id >= args.num_chunks:
        raise ValueError(f"chunk_id={args.chunk_id} 超出范围 [0, {args.num_chunks - 1}]")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

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

    # ── 负向 prompt 编码（此时 GPU 空，text_encoder 独占）─────────────────────
    logger.info("Encoding negative prompt...")
    ctx_neg = encode_prompt(cfg["negative_prompt"], model_ledger, device)

    # ── 加载 transformer（唯一常驻 GPU 的大模型）──────────────────────────────
    logger.info("Loading transformer...")
    transformer = model_ledger.transformer()
    transformer.eval()
    logger.info("Transformer loaded.")

    # ── sigma 调度（N=40，固定）─────────────────────────────────────────────
    sigmas = LTX2Scheduler().execute(steps=40).to(device=device, dtype=torch.float32)
    logger.info(
        "Sigma schedule: %d steps, stage1 save at %s, stage2 save at %s",
        40, STAGE1_SAVE_INDICES, STAGE2_SAVE_INDICES_GLOBAL,
    )

    components = PipelineComponents(dtype=dtype, device=device)

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

    for local_idx in tqdm(todo, desc=f"GPU{args.chunk_id} ODE"):
        global_idx = chunk_start + local_idx
        out_path = output_dir / f"{global_idx:05d}.pt"
        prompt = prompts[local_idx]

        try:
            # ── 编码 prompt：transformer 卸载到 CPU，让 text_encoder 独占 GPU ──
            transformer.cpu()
            cleanup_memory()
            ctx_pos = encode_prompt(prompt, model_ledger, device)
            transformer.to(device)

            data = generate_sample(
                prompt=prompt,
                ctx_pos=ctx_pos,
                ctx_neg=ctx_neg,
                transformer=transformer,
                model_ledger=model_ledger,
                sigmas=sigmas,
                components=components,
                cfg=cfg,
                device=device,
                dtype=dtype,
            )
            del ctx_pos
            torch.save(data, out_path)
        except Exception as e:
            logger.error("Sample global=%d failed: %s", global_idx, e, exc_info=True)
            # 确保 transformer 在 GPU 上以便继续
            if next(transformer.parameters()).device.type != "cuda":
                transformer.to(device)
            cleanup_memory()
            continue

    logger.info("Done. Data saved to %s", output_dir)


if __name__ == "__main__":
    main()
