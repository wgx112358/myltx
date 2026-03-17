"""ODE regression training strategy.

This strategy consumes precomputed ODE trajectory samples where each sample
already contains:
- the current video/audio latent state at a known sigma
- the clean video/audio target latent
- the sigma value for that trajectory step

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

from ltx_core.model.transformer.modality import Modality
from ltx_trainer import logger
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
    timesteps: Tensor
    positions: Tensor
    loss_mask: Tensor


@dataclass(frozen=True)
class BlockCausalMasks:
    video_self: Tensor | BlockMask | None
    audio_self: Tensor | BlockMask | None
    video_to_audio: Tensor | BlockMask | None
    audio_to_video: Tensor | BlockMask | None


class ODERegressionConfig(TrainingStrategyConfigBase):
    """Configuration for ODE regression training."""

    name: Literal["ode_regression"] = "ode_regression"

    with_audio: bool = Field(
        default=True,
        description="Whether to include audio supervision in ODE regression training.",
    )

    audio_latents_dir: str = Field(
        default="audio_latents",
        description="Directory name for audio latents when with_audio is True.",
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
            "latents": "latents",
            "conditions": "conditions",
        }

        if self.config.with_audio:
            sources[self.config.audio_latents_dir] = "audio_latents"

        return sources

    def prepare_training_inputs(
        self,
        batch: dict[str, Any],
        timestep_sampler: TimestepSampler,  # noqa: ARG002 - kept for interface compatibility
    ) -> ModelInputs:
        latents = batch["latents"]
        video_latents = self._video_patchifier.patchify(latents["latents"])
        video_targets_x0 = self._video_patchifier.patchify(latents["ode_target_latents"])

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

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]
        device = video_latents.device
        dtype = video_latents.dtype

        video_sigmas = self._load_sigmas(latents, device=device, dtype=torch.float32)
        video_model_sigmas = video_sigmas.to(dtype=dtype)
        video_noise_metadata = self._extract_noise_metadata(latents, batch_size)
        valid_video_sigma_mask = video_sigmas > self.config.sigma_epsilon
        if not self._warned_zero_sigma and not torch.all(valid_video_sigma_mask):
            zero_count = int((~valid_video_sigma_mask).sum().item())
            logger.warning(
                "ODE regression batch contains %d sample(s) with sigma <= %.2e; they will be ignored in the loss.",
                zero_count,
                self.config.sigma_epsilon,
            )
            self._warned_zero_sigma = True

        sigma_denom = video_sigmas.clamp_min(self.config.sigma_epsilon).view(-1, 1, 1)
        video_targets = (video_latents - video_targets_x0) / sigma_denom
        video_timesteps = video_model_sigmas.view(-1, 1).expand(-1, video_seq_len)
        video_positions = self._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=torch.float32,
        )

        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        audio_inputs = None
        audio_prompt_embeds = None
        if self.config.with_audio:
            audio_prompt_embeds = conditions["audio_prompt_embeds"]
            audio_inputs = self._prepare_audio_inputs(
                batch=batch,
                video_sigmas=video_sigmas,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        if audio_inputs is not None:
            audio_noise_metadata = self._extract_noise_metadata(batch["audio_latents"], batch_size)
            self._validate_noise_metadata_match(video_noise_metadata, audio_noise_metadata)

        self._log_noise_metadata_once(
            latents=latents,
            sigmas=video_sigmas,
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
            )

        video_modality = Modality(
            enabled=True,
            sigma=video_model_sigmas,
            latent=video_latents,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
            attention_mask=causal_masks.video_self if causal_masks is not None else None,
            cross_attention_mask=causal_masks.video_to_audio if causal_masks is not None else None,
        )
        video_loss_mask = valid_video_sigma_mask.unsqueeze(1).expand(-1, video_seq_len)

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
                cross_attention_mask=causal_masks.audio_to_video if causal_masks is not None else None,
            )
            audio_targets = audio_inputs.targets
            audio_loss_mask = audio_inputs.loss_mask

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
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

        return PreparedAudioInputs(
            latents=audio_latents,
            targets=audio_targets,
            sigmas=audio_model_sigmas,
            timesteps=audio_timesteps,
            positions=audio_positions,
            loss_mask=audio_loss_mask,
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
        if self.config.loss_reweight_mode == "auto":
            scale_ratio = (video_loss / (audio_loss + 1e-8)).detach()
            return video_loss + audio_loss * scale_ratio

        return self.config.video_loss_weight * video_loss + self.config.audio_loss_weight * audio_loss

    def _build_block_causal_masks(
        self,
        num_frames: int,
        height: int,
        width: int,
        video_positions: Tensor,
        device: torch.device,
        audio_positions: Tensor | None = None,
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
        video_frame_end_times = self._get_video_frame_end_times(
            video_positions=video_positions,
            video_tokens_per_frame=video_tokens_per_frame,
            num_frames=num_frames,
        )
        audio_block_ends = self._build_audio_block_ends(
            block_ranges=block_ranges,
            video_frame_end_times=video_frame_end_times,
            audio_positions=audio_positions,
            device=device,
        )
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
        audio_to_video_mask = self._create_cross_block_mask(
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
        )

    def _build_video_block_ranges(self, num_frames: int) -> list[tuple[int, int]]:
        if num_frames <= 0:
            raise ValueError(f"Expected num_frames > 0, got {num_frames}")

        block_ranges: list[tuple[int, int]] = []
        next_start = 0
        if self.config.independent_first_frame:
            block_ranges.append((0, 0))
            next_start = 1

        while next_start < num_frames:
            end = min(next_start + self.config.block_size - 1, num_frames - 1)
            block_ranges.append((next_start, end))
            next_start = end + 1

        return block_ranges

    def _build_audio_block_ends(
        self,
        block_ranges: list[tuple[int, int]],
        video_frame_end_times: Tensor,
        audio_positions: Tensor,
        device: torch.device,
    ) -> Tensor:
        audio_starts = audio_positions[0, 0, :, 0].to(dtype=torch.float32).contiguous()
        audio_ends = audio_positions[0, 0, :, 1].to(dtype=torch.float32).contiguous()
        audio_centers = ((audio_starts + audio_ends) / 2.0).contiguous()
        audio_seq_len = audio_starts.shape[0]

        frame_end_indices = torch.tensor(
            [frame_end for _, frame_end in block_ranges],
            device=video_frame_end_times.device,
            dtype=torch.long,
        )
        boundary_times = video_frame_end_times.index_select(0, frame_end_indices).contiguous()

        if self.config.audio_boundary_mode == "left":
            block_ends = torch.searchsorted(audio_starts, boundary_times, right=False)
        elif self.config.audio_boundary_mode == "center":
            block_ends = torch.searchsorted(audio_centers, boundary_times, right=True)
        else:
            block_ends = torch.searchsorted(audio_ends, boundary_times, right=True)

        block_ends = block_ends.clamp_(0, audio_seq_len).to(device=device, dtype=torch.long)
        block_ends[-1] = audio_seq_len
        return block_ends

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
        loss = loss.mul(expanded_mask)
        return loss.sum() / expanded_mask.sum().clamp_min(1.0)

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
