"""ODE regression training strategy.

This strategy consumes precomputed ODE trajectory samples where each sample
already contains:
- the full video/audio latent trajectory across denoising steps
- the clean video/audio target latent
- the sigma schedule for that trajectory

Because the LTX transformer is a velocity model, the clean target is converted
to the corresponding velocity target during training:
    velocity = (x_t - x0) / sigma
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask
except ImportError:
    BlockMask = None  # type: ignore[assignment]
    create_block_mask = None

from ltx_core.model.transformer.modality import CrossAttentionChunkPlan, Modality
from ltx_trainer import logger
from ltx_trainer.ode_block_layout import (
    build_audio_block_ends,
    build_video_block_ranges,
    expand_block_values_by_ends,
    expand_video_block_values,
)
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)


@dataclass(frozen=True)
class PreparedAudioInputs:
    latents: Tensor
    targets: Tensor
    sigmas: Tensor
    token_sigmas: Tensor
    timesteps: Tensor
    positions: Tensor
    loss_mask: Tensor


@dataclass(frozen=True)
class BlockCausalMasks:
    video_self: Tensor | BlockMask | None
    audio_self: Tensor | BlockMask | None
    video_to_audio: Tensor | BlockMask | None
    audio_to_video: Tensor | BlockMask | None
    video_to_audio_plan: CrossAttentionChunkPlan | None = None
    audio_to_video_plan: CrossAttentionChunkPlan | None = None


@dataclass(frozen=True)
class PreparedVideoInputs:
    latents: Tensor
    targets: Tensor
    sigmas: Tensor
    token_sigmas: Tensor
    timesteps: Tensor
    positions: Tensor
    loss_mask: Tensor


class ODERegressionConfig(TrainingStrategyConfigBase):
    """Configuration for ODE regression training."""

    name: Literal["ode_regression"] = "ode_regression"

    with_audio: bool = Field(
        default=True,
        description="Whether to include audio supervision in ODE regression training.",
    )

    video_trajectories_dir: str = Field(
        default="ode_video_trajectories",
        description="Directory name for ODE video trajectory payloads.",
    )

    audio_trajectories_dir: str = Field(
        default="ode_audio_trajectories",
        description="Directory name for ODE audio trajectory payloads when with_audio is True.",
    )

    dual_stage_training: bool = Field(
        default=False,
        description="Whether to mix stage1 and stage2 ODE batches in a single optimizer step.",
    )

    stage1_loss_weight: float = Field(
        default=1.0,
        description="Weight applied to the stage1 ODE loss during dual-stage training.",
        ge=0.0,
    )

    stage2_loss_weight: float = Field(
        default=1.0,
        description="Weight applied to the stage2 ODE loss during dual-stage training.",
        ge=0.0,
    )

    sigma_epsilon: float = Field(
        default=1e-6,
        description="Minimum sigma treated as a valid ODE denoising step.",
        gt=0.0,
    )

    loss_reweight_mode: Literal["manual", "auto"] = Field(
        default="manual",
        description="How to combine audio/video losses when with_audio is True.",
    )

    video_loss_weight: float = Field(
        default=1.0,
        description="Manual weight applied to the video loss when loss_reweight_mode='manual'.",
        ge=0.0,
    )

    audio_loss_weight: float = Field(
        default=1.0,
        description="Manual weight applied to the audio loss when loss_reweight_mode='manual'.",
        ge=0.0,
    )

    use_block_causal_mask: bool = Field(
        default=True,
        description="Whether to apply block-causal self/cross attention masks during ODE regression.",
    )

    block_size: int = Field(
        default=6,
        description="Number of latent frames per causal block after the independent first frame.",
        ge=1,
    )

    independent_first_frame: bool = Field(
        default=True,
        description="Whether to place the first latent frame in its own causal block.",
    )

    audio_boundary_mode: Literal["center", "left", "right"] = Field(
        default="left",
        description="How audio tokens that straddle a video block boundary are assigned to blocks.",
    )

    local_attn_size: int = Field(
        default=-1,
        description="Optional local attention window in block units. -1 disables local cropping.",
        ge=-1,
    )

    validate_audio_sigma_match: bool = Field(
        default=True,
        description="Whether to require audio and video ODE sigma values to match within tolerance.",
    )

    sigma_match_atol: float = Field(
        default=1e-6,
        description="Absolute tolerance used when comparing audio/video sigma values.",
        ge=0.0,
    )

    sigma_match_rtol: float = Field(
        default=1e-5,
        description="Relative tolerance used when comparing audio/video sigma values.",
        ge=0.0,
    )

    ode_layout_mode: Literal["legacy", "blockwise"] = Field(
        default="legacy",
        description="How ODE trajectory supervision is organized.",
    )

    audio_sink_token_count: int = Field(
        default=0,
        description="Number of audio sink tokens to prepend during training.",
        ge=0,
    )

    audio_sink_identity_rope: bool = Field(
        default=False,
        description="Whether prepended audio sink tokens should use zero positions for identity RoPE.",
    )


class ODERegressionStrategy(TrainingStrategy):
    """ODE regression strategy for precomputed trajectory states."""

    config: ODERegressionConfig

    def __init__(self, config: ODERegressionConfig):
        super().__init__(config)
        self._warned_zero_sigma = False
        self._logged_noise_metadata = False

    @property
    def requires_audio(self) -> bool:
        return self.config.with_audio

    def get_data_sources(self) -> list[str] | dict[str, str]:
        sources = {
            self.config.video_trajectories_dir: "latents",
            "conditions": "conditions",
        }

        if self.config.with_audio:
            sources[self.config.audio_trajectories_dir] = "audio_latents"

        return sources

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,  # noqa: ARG002 - kept for interface compatibility
    ) -> ModelInputs:
        latents = batch["latents"]
        if self.config.ode_layout_mode != "blockwise":
            raise ValueError("ODE regression now requires blockwise full-trajectory training; set ode_layout_mode=blockwise.")

        video_trajectory_ref = latents.get("ode_video_trajectory")
        if video_trajectory_ref is None:
            raise KeyError('ODE regression now requires "ode_video_trajectory" in latents payload.')
        if not isinstance(video_trajectory_ref, torch.Tensor):
            video_trajectory_ref = torch.as_tensor(video_trajectory_ref)

        num_frames = int(latents["num_frames"][0].item())
        height = int(latents["height"][0].item())
        width = int(latents["width"][0].item())

        fps_values = latents.get("fps", None)
        if fps_values is not None and not torch.all(fps_values == fps_values[0]):
            logger.warning(
                "Different FPS values found in the batch. Found: %s, using the first one: %.4f",
                fps_values.tolist(),
                fps_values[0].item(),
            )
        fps = float(fps_values[0].item()) if fps_values is not None else DEFAULT_FPS

        batch_size = video_trajectory_ref.shape[0]
        device = video_trajectory_ref.device
        dtype = video_trajectory_ref.dtype

        video_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=torch.float32,
        )

        video_trajectory = self._load_video_trajectory(latents, device=device, dtype=dtype)
        video_trajectory_sigmas = self._load_trajectory_sigmas(latents, device=device)
        block_ranges = self._build_video_block_ranges(num_frames)
        block_step_indices = self._load_or_sample_block_step_indices(
            latents=latents,
            batch_size=batch_size,
            num_blocks=len(block_ranges),
            trajectory_length=video_trajectory.shape[1],
            device=device,
        )
        prepared_video = self._prepare_blockwise_video_inputs(
            latents=latents,
            num_frames=num_frames,
            height=height,
            width=width,
            video_positions=video_positions,
            video_trajectory=video_trajectory,
            trajectory_sigmas=video_trajectory_sigmas,
            block_step_indices=block_step_indices,
            device=device,
            dtype=dtype,
        )

        video_noise_metadata = self._extract_noise_metadata(latents, batch_size)
        valid_video_sigma_mask = prepared_video.token_sigmas > self.config.sigma_epsilon
        if not self._warned_zero_sigma and not torch.all(valid_video_sigma_mask.any(dim=1)):
            zero_count = int((~valid_video_sigma_mask.any(dim=1)).sum().item())
            logger.warning(
                "ODE regression batch contains %d sample(s) with sigma <= %.2e; they will be ignored in the loss.",
                zero_count,
                self.config.sigma_epsilon,
            )
            self._warned_zero_sigma = True

        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        audio_inputs = None
        audio_prompt_embeds = None
        if self.config.with_audio:
            audio_prompt_embeds = conditions["audio_prompt_embeds"]
            audio_inputs = self._prepare_blockwise_audio_inputs(
                batch=batch,
                video_positions=video_positions,
                expected_trajectory_sigmas=video_trajectory_sigmas,
                block_step_indices=block_step_indices,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        if audio_inputs is not None:
            audio_noise_metadata = self._extract_noise_metadata(batch["audio_latents"], batch_size)
            self._validate_noise_metadata_match(video_noise_metadata, audio_noise_metadata)

        self._log_noise_metadata_once(
            latents=latents,
            sigmas=prepared_video.sigmas,
            noise_metadata=video_noise_metadata,
        )

        causal_masks = None
        if self.config.use_block_causal_mask:
            causal_masks = self._build_block_causal_masks(
                num_frames=num_frames,
                height=height,
                width=width,
                video_positions=video_positions,
                device=device,
                audio_positions=audio_inputs.positions if audio_inputs is not None else None,
                audio_sink_token_count=self.config.audio_sink_token_count,
            )

        video_modality = Modality(
            enabled=True,
            sigma=prepared_video.sigmas,
            latent=prepared_video.latents,
            timesteps=prepared_video.timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
            attention_mask=causal_masks.video_self if causal_masks is not None else None,
            cross_attention_chunk_plan=causal_masks.video_to_audio_plan if causal_masks is not None else None,
            cross_attention_mask=causal_masks.video_to_audio if causal_masks is not None else None,
            prompt_sigma=prepared_video.token_sigmas,
        )
        video_loss_mask = prepared_video.loss_mask

        audio_modality = None
        audio_targets = None
        audio_loss_mask = None
        if audio_inputs is not None and audio_prompt_embeds is not None:
            audio_modality = Modality(
                enabled=True,
                sigma=audio_inputs.sigmas,
                latent=audio_inputs.latents,
                timesteps=audio_inputs.timesteps,
                positions=audio_inputs.positions,
                context=audio_prompt_embeds,
                context_mask=prompt_attention_mask,
                attention_mask=causal_masks.audio_self if causal_masks is not None else None,
                cross_attention_chunk_plan=causal_masks.audio_to_video_plan if causal_masks is not None else None,
                cross_attention_mask=causal_masks.audio_to_video if causal_masks is not None else None,
                sink_token_count=self.config.audio_sink_token_count,
                prompt_sigma=audio_inputs.token_sigmas,
            )
            audio_targets = audio_inputs.targets
            audio_loss_mask = audio_inputs.loss_mask

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=prepared_video.targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
        )

    def _prepare_legacy_video_inputs(
        self,
        latents: dict[str, Any],
        num_frames: int,
        height: int,
        width: int,
        fps: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PreparedVideoInputs:
        video_latents = self._video_patchifier.patchify(latents["latents"])
        video_targets_x0 = self._video_patchifier.patchify(latents["ode_target_latents"])
        video_seq_len = video_latents.shape[1]
        video_sigmas = self._load_sigmas(latents, device=device, dtype=torch.float32)
        video_model_sigmas = video_sigmas.to(dtype=dtype)
        sigma_denom = video_sigmas.clamp_min(self.config.sigma_epsilon).view(-1, 1, 1)
        video_targets = (video_latents - video_targets_x0) / sigma_denom
        video_timesteps = video_model_sigmas.view(-1, 1).expand(-1, video_seq_len)
        video_loss_mask = (video_sigmas > self.config.sigma_epsilon).unsqueeze(1).expand(-1, video_seq_len)
        video_token_sigmas = video_model_sigmas.view(-1, 1).expand(-1, video_seq_len)
        video_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=video_latents.shape[0],
            fps=fps,
            device=device,
            dtype=torch.float32,
        )
        return PreparedVideoInputs(
            latents=video_latents,
            targets=video_targets,
            sigmas=video_model_sigmas,
            token_sigmas=video_token_sigmas,
            timesteps=video_timesteps,
            positions=video_positions,
            loss_mask=video_loss_mask,
        )

    def _prepare_blockwise_video_inputs(
        self,
        latents: dict[str, Any],
        num_frames: int,
        height: int,
        width: int,
        video_positions: Tensor,
        video_trajectory: Tensor,
        trajectory_sigmas: Tensor,
        block_step_indices: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PreparedVideoInputs:
        block_ranges = self._build_video_block_ranges(num_frames)

        mixed_video = torch.empty_like(video_trajectory[:, -1])
        frame_sigmas = torch.empty((video_trajectory.shape[0], num_frames), device=device, dtype=torch.float32)

        for batch_index in range(video_trajectory.shape[0]):
            per_block_indices = block_step_indices[batch_index].tolist()
            per_frame_indices = expand_video_block_values(block_ranges, per_block_indices)
            per_frame_sigmas = [float(trajectory_sigmas[batch_index, step_index].item()) for step_index in per_frame_indices]
            frame_sigmas[batch_index] = torch.tensor(per_frame_sigmas, device=device, dtype=torch.float32)
            for frame_index, step_index in enumerate(per_frame_indices):
                mixed_video[batch_index, :, frame_index] = video_trajectory[batch_index, step_index, :, frame_index]

        video_latents = self._video_patchifier.patchify(mixed_video)
        video_targets_x0 = self._video_patchifier.patchify(latents["ode_target_latents"].to(device=device, dtype=dtype))
        video_token_sigmas = frame_sigmas.unsqueeze(-1).expand(-1, -1, height * width).reshape(video_latents.shape[0], -1)
        sigma_denom = video_token_sigmas.clamp_min(self.config.sigma_epsilon).unsqueeze(-1)
        video_targets = (video_latents - video_targets_x0) / sigma_denom
        video_timesteps = video_token_sigmas.to(dtype=dtype)
        video_loss_mask = video_token_sigmas > self.config.sigma_epsilon
        video_model_sigmas = video_token_sigmas.max(dim=1).values.to(dtype=dtype)

        return PreparedVideoInputs(
            latents=video_latents,
            targets=video_targets,
            sigmas=video_model_sigmas,
            token_sigmas=video_token_sigmas,
            timesteps=video_timesteps,
            positions=video_positions,
            loss_mask=video_loss_mask,
        )

    def _prepare_audio_inputs(
        self,
        batch: dict[str, Any],
        video_sigmas: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PreparedAudioInputs:
        audio_data = batch["audio_latents"]
        audio_latents = self._audio_patchifier.patchify(audio_data["latents"])
        audio_targets_x0 = self._audio_patchifier.patchify(audio_data["ode_target_latents"])

        audio_sigmas = self._load_sigmas(audio_data, device=device, dtype=torch.float32)
        if self.config.validate_audio_sigma_match and not torch.allclose(
            audio_sigmas,
            video_sigmas,
            atol=self.config.sigma_match_atol,
            rtol=self.config.sigma_match_rtol,
        ):
            raise ValueError(
                "Audio/video ode_sigma mismatch detected in ODE regression batch. "
                f"video={video_sigmas.tolist()}, audio={audio_sigmas.tolist()}"
            )

        audio_seq_len = audio_latents.shape[1]
        audio_model_sigmas = audio_sigmas.to(dtype=dtype)
        sigma_denom = audio_sigmas.clamp_min(self.config.sigma_epsilon).view(-1, 1, 1)
        audio_targets = (audio_latents - audio_targets_x0) / sigma_denom
        audio_timesteps = audio_model_sigmas.view(-1, 1).expand(-1, audio_seq_len)
        audio_positions = self._get_audio_positions(
            num_time_steps=audio_seq_len,
            batch_size=batch_size,
            device=device,
            dtype=torch.float32,
        )
        audio_loss_mask = (audio_sigmas > self.config.sigma_epsilon).unsqueeze(1).expand(-1, audio_seq_len)

        prepared = PreparedAudioInputs(
            latents=audio_latents,
            targets=audio_targets,
            sigmas=audio_model_sigmas,
            token_sigmas=audio_model_sigmas.view(-1, 1).expand(-1, audio_seq_len),
            timesteps=audio_timesteps,
            positions=audio_positions,
            loss_mask=audio_loss_mask,
        )
        return self._prepend_audio_sink_tokens(prepared)

    def _prepare_blockwise_audio_inputs(
        self,
        batch: dict[str, Any],
        video_positions: Tensor,
        expected_trajectory_sigmas: Tensor,
        block_step_indices: Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PreparedAudioInputs:
        audio_data = batch["audio_latents"]
        audio_trajectory = self._load_audio_trajectory(audio_data, device=device, dtype=dtype)
        trajectory_sigmas = self._load_trajectory_sigmas(audio_data, device=device)
        self._validate_blockwise_sigma_match(expected_trajectory_sigmas, trajectory_sigmas)
        num_frames = int(batch["latents"]["num_frames"][0].item())
        block_ranges = self._build_video_block_ranges(num_frames)
        if audio_trajectory.shape[1] != expected_trajectory_sigmas.shape[1]:
            raise ValueError(
                "Audio/video trajectory length mismatch detected in blockwise ODE regression. "
                f"video={expected_trajectory_sigmas.shape[1]}, audio={audio_trajectory.shape[1]}"
            )

        audio_seq_len = int(audio_data["num_time_steps"][0].item())
        audio_positions = self._get_audio_positions(
            num_time_steps=audio_seq_len,
            batch_size=batch_size,
            device=device,
            dtype=torch.float32,
        )
        video_frame_end_times = self._get_video_frame_end_times(
            video_positions=video_positions,
            video_tokens_per_frame=int(batch["latents"]["height"][0].item() * batch["latents"]["width"][0].item()),
            num_frames=num_frames,
        )
        audio_block_ends = self._build_audio_block_ends(
            block_ranges=block_ranges,
            video_frame_end_times=video_frame_end_times,
            audio_positions=audio_positions,
            device=device,
        )

        mixed_audio = torch.empty_like(audio_trajectory[:, -1])
        token_sigmas = torch.empty((batch_size, audio_seq_len), device=device, dtype=torch.float32)

        for batch_index in range(batch_size):
            per_block_indices = block_step_indices[batch_index].tolist()
            per_block_sigmas = [float(trajectory_sigmas[batch_index, step_index].item()) for step_index in per_block_indices]
            token_sigmas[batch_index] = torch.tensor(
                expand_block_values_by_ends(
                    block_ends=audio_block_ends.tolist(),
                    block_values=per_block_sigmas,
                    total_len=audio_seq_len,
                ),
                device=device,
                dtype=torch.float32,
            )
            prev_end = 0
            for block_end, step_index in zip(audio_block_ends.tolist(), per_block_indices, strict=True):
                mixed_audio[batch_index, :, prev_end:block_end] = audio_trajectory[batch_index, step_index, :, prev_end:block_end]
                prev_end = block_end

        audio_latents = self._audio_patchifier.patchify(mixed_audio)
        audio_targets_x0 = self._audio_patchifier.patchify(audio_data["ode_target_latents"].to(device=device, dtype=dtype))
        sigma_denom = token_sigmas.clamp_min(self.config.sigma_epsilon).unsqueeze(-1)
        audio_targets = (audio_latents - audio_targets_x0) / sigma_denom
        audio_timesteps = token_sigmas.to(dtype=dtype)
        audio_loss_mask = token_sigmas > self.config.sigma_epsilon
        audio_model_sigmas = token_sigmas.max(dim=1).values.to(dtype=dtype)

        prepared = PreparedAudioInputs(
            latents=audio_latents,
            targets=audio_targets,
            sigmas=audio_model_sigmas,
            token_sigmas=token_sigmas,
            timesteps=audio_timesteps,
            positions=audio_positions,
            loss_mask=audio_loss_mask,
        )
        return self._prepend_audio_sink_tokens(prepared)

    def _prepend_audio_sink_tokens(self, audio_inputs: PreparedAudioInputs) -> PreparedAudioInputs:
        if self.config.audio_sink_token_count == 0:
            return audio_inputs

        sink_token_count = self.config.audio_sink_token_count
        batch_size, _, hidden_dim = audio_inputs.latents.shape
        zero_latents = torch.zeros(
            batch_size,
            sink_token_count,
            hidden_dim,
            device=audio_inputs.latents.device,
            dtype=audio_inputs.latents.dtype,
        )
        zero_targets = torch.zeros_like(zero_latents)
        zero_token_sigmas = torch.zeros(
            batch_size,
            sink_token_count,
            device=audio_inputs.token_sigmas.device,
            dtype=audio_inputs.token_sigmas.dtype,
        )
        zero_timesteps = torch.zeros(
            batch_size,
            sink_token_count,
            device=audio_inputs.timesteps.device,
            dtype=audio_inputs.timesteps.dtype,
        )
        zero_loss_mask = torch.zeros(
            batch_size,
            sink_token_count,
            device=audio_inputs.loss_mask.device,
            dtype=audio_inputs.loss_mask.dtype,
        )
        if self.config.audio_sink_identity_rope:
            sink_positions = torch.zeros(
                batch_size,
                audio_inputs.positions.shape[1],
                sink_token_count,
                audio_inputs.positions.shape[-1],
                device=audio_inputs.positions.device,
                dtype=audio_inputs.positions.dtype,
            )
        else:
            sink_positions = audio_inputs.positions[:, :, :1, :].expand(-1, -1, sink_token_count, -1).clone()

        return PreparedAudioInputs(
            latents=torch.cat([zero_latents, audio_inputs.latents], dim=1),
            targets=torch.cat([zero_targets, audio_inputs.targets], dim=1),
            sigmas=audio_inputs.sigmas,
            token_sigmas=torch.cat([zero_token_sigmas, audio_inputs.token_sigmas], dim=1),
            timesteps=torch.cat([zero_timesteps, audio_inputs.timesteps], dim=1),
            positions=torch.cat([sink_positions, audio_inputs.positions], dim=2),
            loss_mask=torch.cat([zero_loss_mask, audio_inputs.loss_mask], dim=1),
        )

    def compute_loss(
        self,
        video_pred: Tensor,
        audio_pred: Tensor | None,
        inputs: ModelInputs,
    ) -> Tensor:
        video_loss = self._masked_mse(video_pred, inputs.video_targets, inputs.video_loss_mask)

        if (
            not self.config.with_audio
            or audio_pred is None
            or inputs.audio_targets is None
            or inputs.audio_loss_mask is None
        ):
            return video_loss

        audio_loss = self._masked_mse(audio_pred, inputs.audio_targets, inputs.audio_loss_mask)
        if self.config.loss_reweight_mode == "manual":
            return self.config.video_loss_weight * video_loss + self.config.audio_loss_weight * audio_loss
        return video_loss + audio_loss

    def _build_block_causal_masks(
        self,
        num_frames: int,
        height: int,
        width: int,
        video_positions: Tensor,
        device: torch.device,
        audio_positions: Tensor | None = None,
        audio_sink_token_count: int = 0,
    ) -> BlockCausalMasks:
        self._require_block_mask_support()

        video_tokens_per_frame = height * width
        video_seq_len = video_positions.shape[2]
        expected_video_seq_len = num_frames * video_tokens_per_frame
        if video_seq_len != expected_video_seq_len:
            raise ValueError(
                f"Expected video sequence length {expected_video_seq_len}, got {video_seq_len}. "
                "Block-causal masking assumes patch_size=1 video tokens."
            )

        block_ranges = self._build_video_block_ranges(num_frames)
        video_block_ends = torch.tensor(
            [(frame_end + 1) * video_tokens_per_frame for _, frame_end in block_ranges],
            device=device,
            dtype=torch.long,
        )
        video_self_mask = self._create_self_block_mask(
            total_len=video_seq_len,
            block_ends=video_block_ends,
            device=device,
            unit_size=video_tokens_per_frame,
        )

        if audio_positions is None:
            return BlockCausalMasks(
                video_self=video_self_mask,
                audio_self=None,
                video_to_audio=None,
                audio_to_video=None,
            )

        audio_seq_len = audio_positions.shape[2]
        layout_audio_positions = (
            audio_positions[:, :, audio_sink_token_count:, :]
            if audio_sink_token_count > 0
            else audio_positions
        )
        video_frame_end_times = self._get_video_frame_end_times(
            video_positions=video_positions,
            video_tokens_per_frame=video_tokens_per_frame,
            num_frames=num_frames,
        )
        audio_block_ends = self._build_audio_block_ends(
            block_ranges=block_ranges,
            video_frame_end_times=video_frame_end_times,
            audio_positions=layout_audio_positions,
            device=device,
        )
        if audio_sink_token_count > 0:
            audio_block_ends = audio_block_ends + audio_sink_token_count
        avg_audio_tokens_per_block = self._average_audio_tokens_per_block(audio_seq_len, len(block_ranges))
        audio_self_mask = self._create_self_block_mask(
            total_len=audio_seq_len,
            block_ends=audio_block_ends,
            device=device,
            unit_size=avg_audio_tokens_per_block,
        )
        video_to_audio_mask = self._create_cross_block_mask(
            src_len=video_seq_len,
            target_len=audio_seq_len,
            src_block_ends=video_block_ends,
            target_block_ends=audio_block_ends,
            device=device,
            target_unit_size=avg_audio_tokens_per_block,
        )
        video_to_audio_plan = self._create_cross_attention_chunk_plan(
            src_len=video_seq_len,
            target_len=audio_seq_len,
            src_block_ends=video_block_ends,
            target_block_ends=audio_block_ends,
            device=device,
            target_unit_size=avg_audio_tokens_per_block,
        )
        audio_to_video_mask = self._create_cross_block_mask(
            src_len=audio_seq_len,
            target_len=video_seq_len,
            src_block_ends=audio_block_ends,
            target_block_ends=video_block_ends,
            device=device,
            target_unit_size=video_tokens_per_frame,
        )
        audio_to_video_plan = self._create_cross_attention_chunk_plan(
            src_len=audio_seq_len,
            target_len=video_seq_len,
            src_block_ends=audio_block_ends,
            target_block_ends=video_block_ends,
            device=device,
            target_unit_size=video_tokens_per_frame,
        )

        return BlockCausalMasks(
            video_self=video_self_mask,
            audio_self=audio_self_mask,
            video_to_audio=video_to_audio_mask,
            audio_to_video=audio_to_video_mask,
            video_to_audio_plan=video_to_audio_plan,
            audio_to_video_plan=audio_to_video_plan,
        )

    def _build_video_block_ranges(self, num_frames: int) -> list[tuple[int, int]]:
        return build_video_block_ranges(
            num_frames,
            block_size=self.config.block_size,
            independent_first_frame=self.config.independent_first_frame,
        )

    def _build_audio_block_ends(
        self,
        block_ranges: list[tuple[int, int]],
        video_frame_end_times: Tensor,
        audio_positions: Tensor,
        device: torch.device,
    ) -> Tensor:
        block_ends = build_audio_block_ends(
            block_ranges=block_ranges,
            video_frame_end_times=video_frame_end_times.tolist(),
            audio_starts=audio_positions[0, 0, :, 0].to(dtype=torch.float32).tolist(),
            audio_ends=audio_positions[0, 0, :, 1].to(dtype=torch.float32).tolist(),
            audio_boundary_mode=self.config.audio_boundary_mode,
        )
        return torch.tensor(block_ends, device=device, dtype=torch.long)

    def _create_self_block_mask(
        self,
        total_len: int,
        block_ends: Tensor,
        device: torch.device,
        unit_size: float,
    ) -> BlockMask:
        expanded_ends = self._expand_block_ends(block_ends=block_ends, total_len=total_len, device=device)
        local_window = self._resolve_local_window_tokens(unit_size)

        def mask_fn(b, h, q_idx, kv_idx):
            visible = kv_idx < expanded_ends[q_idx]
            if local_window is None:
                return visible
            return (visible & (kv_idx >= (expanded_ends[q_idx] - local_window))) | (q_idx == kv_idx)

        return create_block_mask(mask_fn, B=None, H=None, Q_LEN=total_len, KV_LEN=total_len, device=device, _compile=False)

    def _create_cross_block_mask(
        self,
        src_len: int,
        target_len: int,
        src_block_ends: Tensor,
        target_block_ends: Tensor,
        device: torch.device,
        target_unit_size: float,
    ) -> BlockMask:
        expanded_target_ends = self._expand_block_ends(
            block_ends=src_block_ends,
            total_len=src_len,
            device=device,
            target_block_ends=target_block_ends,
            target_len=target_len,
        )
        local_window = self._resolve_local_window_tokens(target_unit_size)

        def mask_fn(b, h, q_idx, kv_idx):
            visible = kv_idx < expanded_target_ends[q_idx]
            if local_window is None:
                return visible
            return visible & (kv_idx >= (expanded_target_ends[q_idx] - local_window))

        return create_block_mask(mask_fn, B=None, H=None, Q_LEN=src_len, KV_LEN=target_len, device=device, _compile=False)

    def _create_cross_attention_chunk_plan(
        self,
        src_len: int,
        target_len: int,
        src_block_ends: Tensor,
        target_block_ends: Tensor,
        device: torch.device,
        target_unit_size: float,
    ) -> CrossAttentionChunkPlan:
        query_block_ends = src_block_ends.to(device=device, dtype=torch.long)
        target_ends = target_block_ends.to(device=device, dtype=torch.long).clamp(min=0, max=target_len)
        if query_block_ends.numel() == 0 or target_ends.numel() == 0:
            raise ValueError('Cross-attention chunk plan requires at least one source and target block.')
        if query_block_ends.shape != target_ends.shape:
            raise ValueError(
                'Cross-attention chunk plan expects matching source/target block layouts. '
                f'Got source {tuple(query_block_ends.shape)} and target {tuple(target_ends.shape)}.'
            )
        if int(query_block_ends[-1].item()) != src_len:
            raise ValueError(f'Cross-attention chunk plan must cover the full source length {src_len}.')
        if int(target_ends[-1].item()) != target_len:
            raise ValueError(f'Cross-attention chunk plan must cover the full target length {target_len}.')

        local_window = self._resolve_local_window_tokens(target_unit_size)
        if local_window is None:
            target_starts = torch.zeros_like(target_ends)
        else:
            target_starts = torch.clamp(target_ends - local_window, min=0)

        return CrossAttentionChunkPlan(query_block_ends=query_block_ends, target_starts=target_starts, target_ends=target_ends)

    @staticmethod
    def _expand_block_ends(
        block_ends: Tensor,
        total_len: int,
        device: torch.device,
        target_block_ends: Tensor | None = None,
        target_len: int | None = None,
    ) -> Tensor:
        expanded = torch.zeros(total_len, device=device, dtype=torch.int32)
        prev_end = 0
        mapped_ends = target_block_ends if target_block_ends is not None else block_ends
        final_limit = int(target_len if target_len is not None else total_len)

        for src_end, mapped_end in zip(block_ends.tolist(), mapped_ends.tolist(), strict=True):
            expanded[prev_end:src_end] = int(mapped_end)
            prev_end = src_end

        if prev_end < total_len:
            expanded[prev_end:] = final_limit

        return expanded

    def _resolve_local_window_tokens(self, unit_size: float) -> int | None:
        if self.config.local_attn_size == -1:
            return None
        return max(1, int(math.ceil(self.config.local_attn_size * unit_size)))

    @staticmethod
    def _get_video_frame_end_times(video_positions: Tensor, video_tokens_per_frame: int, num_frames: int) -> Tensor:
        frame_end_times = video_positions[0, 0, ::video_tokens_per_frame, 1].to(dtype=torch.float32).contiguous()
        if frame_end_times.shape[0] != num_frames:
            raise ValueError(
                f"Expected {num_frames} video frame time bounds, found {frame_end_times.shape[0]}."
            )
        return frame_end_times

    @staticmethod
    def _average_audio_tokens_per_block(audio_seq_len: int, num_blocks: int) -> float:
        if num_blocks <= 0:
            return 1.0
        return max(audio_seq_len / num_blocks, 1.0)

    @staticmethod
    def _require_block_mask_support() -> None:
        if BlockMask is None or create_block_mask is None:
            raise RuntimeError(
                "Block-causal ODE regression requires torch.nn.attention.flex_attention.BlockMask support."
            )

    @staticmethod
    def _masked_mse(pred: Tensor, target: Tensor, loss_mask: Tensor) -> Tensor:
        if loss_mask.ndim != 2:
            raise ValueError(f"Expected loss mask with shape [B, seq_len], got {tuple(loss_mask.shape)}")

        if not torch.any(loss_mask):
            return pred.sum() * 0

        loss = (pred - target).pow(2)
        expanded_mask = loss_mask.unsqueeze(-1).to(dtype=loss.dtype)
        return loss.mul(expanded_mask).div(expanded_mask.mean()).mean()

    def _load_or_sample_block_step_indices(
        self,
        latents: dict[str, Any],
        batch_size: int,
        num_blocks: int,
        trajectory_length: int,
        device: torch.device,
    ) -> Tensor:
        block_step_indices = latents.get("ode_block_step_indices")
        if block_step_indices is None:
            return self._sample_block_step_indices(
                batch_size=batch_size,
                num_blocks=num_blocks,
                trajectory_length=trajectory_length,
                device=device,
            )
        if not isinstance(block_step_indices, torch.Tensor):
            block_step_indices = torch.as_tensor(block_step_indices)
        if block_step_indices.ndim == 1:
            block_step_indices = block_step_indices.unsqueeze(0)
        return block_step_indices.to(device=device, dtype=torch.long)

    @staticmethod
    def _sample_block_step_indices(
        batch_size: int,
        num_blocks: int,
        trajectory_length: int,
        device: torch.device,
    ) -> Tensor:
        return torch.randint(0, trajectory_length, (batch_size, num_blocks), device=device, dtype=torch.long)

    def _validate_blockwise_sigma_match(self, video_sigmas: Tensor, audio_sigmas: Tensor) -> None:
        if not self.config.validate_audio_sigma_match:
            return
        if not torch.allclose(
            video_sigmas,
            audio_sigmas,
            atol=self.config.sigma_match_atol,
            rtol=self.config.sigma_match_rtol,
        ):
            raise ValueError(
                "Audio/video blockwise ode_trajectory_sigmas mismatch detected in ODE regression batch. "
                f"video={video_sigmas.tolist()}, audio={audio_sigmas.tolist()}"
            )

    @staticmethod
    def _load_sigmas(latents: dict[str, Any], device: torch.device, dtype: torch.dtype) -> Tensor:
        if "ode_sigma" not in latents:
            raise KeyError(
                'ODE regression requires "ode_sigma" in latents payload. '
                "Re-export the dataset with convert_ode_pt_to_precomputed.py --export-mode ode_regression."
            )

        sigmas = latents["ode_sigma"]
        if not isinstance(sigmas, torch.Tensor):
            sigmas = torch.as_tensor(sigmas)

        return sigmas.to(device=device, dtype=dtype).flatten()

    def _use_blockwise_layout(self, latents: dict[str, Any]) -> bool:
        return self.config.ode_layout_mode == "blockwise" and "ode_video_trajectory" in latents

    @staticmethod
    def _load_video_trajectory(latents: dict[str, Any], device: torch.device, dtype: torch.dtype) -> Tensor:
        if "ode_video_trajectory" not in latents:
            raise KeyError('Blockwise ODE regression requires "ode_video_trajectory" in latents payload.')
        return latents["ode_video_trajectory"].to(device=device, dtype=dtype)

    @staticmethod
    def _load_audio_trajectory(latents: dict[str, Any], device: torch.device, dtype: torch.dtype) -> Tensor:
        if "ode_audio_trajectory" not in latents:
            raise KeyError('Blockwise ODE regression requires "ode_audio_trajectory" in audio payload.')
        return latents["ode_audio_trajectory"].to(device=device, dtype=dtype)

    @staticmethod
    def _load_trajectory_sigmas(latents: dict[str, Any], device: torch.device) -> Tensor:
        if "ode_trajectory_sigmas" not in latents:
            raise KeyError('Blockwise ODE regression requires "ode_trajectory_sigmas" in payload.')
        sigmas = latents["ode_trajectory_sigmas"]
        if not isinstance(sigmas, torch.Tensor):
            sigmas = torch.as_tensor(sigmas)
        if sigmas.ndim == 1:
            sigmas = sigmas.unsqueeze(0)
        return sigmas.to(device=device, dtype=torch.float32)

    def _log_noise_metadata_once(
        self,
        latents: dict[str, Any],
        sigmas: Tensor,
        noise_metadata: list[dict[str, Any] | None],
    ) -> None:
        if self._logged_noise_metadata:
            return
        if not noise_metadata or noise_metadata[0] is None:
            return

        step_indices = self._extract_optional_batch_values(latents.get("ode_step_index"))
        clean_step_indices = self._extract_optional_batch_values(latents.get("ode_clean_step_index"))
        logger.debug(
            "ODE regression noise metadata sample[0]: sigma=%s step=%s clean_step=%s seeds=%s",
            float(sigmas[0].item()),
            step_indices[0] if step_indices is not None else None,
            clean_step_indices[0] if clean_step_indices is not None else None,
            noise_metadata[0],
        )
        self._logged_noise_metadata = True

    def _validate_noise_metadata_match(
        self,
        video_noise_metadata: list[dict[str, Any] | None],
        audio_noise_metadata: list[dict[str, Any] | None],
    ) -> None:
        if not video_noise_metadata or not audio_noise_metadata:
            return
        if len(video_noise_metadata) != len(audio_noise_metadata):
            raise ValueError(
                "Video/audio ode_noise_seeds batch size mismatch detected in ODE regression batch."
            )

        for batch_index, (video_meta, audio_meta) in enumerate(zip(video_noise_metadata, audio_noise_metadata, strict=True)):
            if video_meta is None or audio_meta is None:
                continue
            if video_meta != audio_meta:
                raise ValueError(
                    "Video/audio ode_noise_seeds mismatch detected in ODE regression batch. "
                    f"batch_index={batch_index}, video={video_meta}, audio={audio_meta}"
                )

    @classmethod
    def _extract_noise_metadata(cls, latents: dict[str, Any], batch_size: int) -> list[dict[str, Any] | None]:
        metadata = latents.get("ode_noise_seeds")
        if metadata is None:
            return [None] * batch_size
        return [cls._extract_batch_item(metadata, batch_index) for batch_index in range(batch_size)]

    @staticmethod
    def _extract_optional_batch_values(value: Any) -> list[Any] | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            flat = value.flatten().tolist()
            return [int(item) if isinstance(item, (int, float)) else item for item in flat]
        if isinstance(value, list | tuple):
            return list(value)
        return [value]

    @classmethod
    def _extract_batch_item(cls, value: Any, batch_index: int) -> Any:
        if isinstance(value, dict):
            return {key: cls._extract_batch_item(item, batch_index) for key, item in value.items()}
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.item()
            return cls._extract_batch_item(value[batch_index], batch_index)
        if isinstance(value, tuple):
            return tuple(cls._extract_batch_item(item, batch_index) for item in value)
        if isinstance(value, list):
            if value and isinstance(value[0], str):
                return value[batch_index]
            return [cls._extract_batch_item(item, batch_index) for item in value]
        return value
