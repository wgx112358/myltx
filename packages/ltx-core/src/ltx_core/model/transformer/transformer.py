from dataclasses import dataclass, replace

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core.model.transformer.adaln import adaln_embedding_coefficient
from ltx_core.model.transformer.attention import Attention, AttentionCallable, AttentionFunction
from ltx_core.model.transformer.feed_forward import FeedForward
from ltx_core.model.transformer.modality import CrossAttentionChunkPlan
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.utils import rms_norm

PROMPT_QUERY_CHUNK_SIZE = 256


@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int
    apply_gated_attention: bool = False
    cross_attention_adaln: bool = False


def apply_checkpointed_feed_forward(
    ff: torch.nn.Module,
    x: torch.Tensor,
    checkpoint_ff: bool = False,
) -> torch.Tensor:
    if checkpoint_ff and torch.is_grad_enabled() and x.requires_grad:
        return torch.utils.checkpoint.checkpoint(ff, x, use_reentrant=False)
    return ff(x)


def _slice_rope(
    pe: tuple[torch.Tensor, torch.Tensor] | None,
    start: int,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if pe is None:
        return None
    cos_freqs, sin_freqs = pe
    if cos_freqs.ndim == 3:
        return cos_freqs[:, start:end], sin_freqs[:, start:end]
    if cos_freqs.ndim == 4:
        return cos_freqs[:, :, start:end], sin_freqs[:, :, start:end]
    raise ValueError(f'Unsupported RoPE rank {cos_freqs.ndim}; expected 3 or 4 dimensions.')


def apply_chunked_cross_attention(
    x: torch.Tensor,
    context: torch.Tensor,
    attn: AttentionCallable,
    chunk_plan: CrossAttentionChunkPlan | None,
    mask: object | None = None,
    pe: tuple[torch.Tensor, torch.Tensor] | None = None,
    k_pe: tuple[torch.Tensor, torch.Tensor] | None = None,
    checkpoint_chunks: bool = False,
) -> torch.Tensor:
    if chunk_plan is None:
        return attn(x, context=context, mask=mask, pe=pe, k_pe=k_pe)

    block_ends = chunk_plan.query_block_ends.tolist()
    target_starts = chunk_plan.target_starts.tolist()
    target_ends = chunk_plan.target_ends.tolist()
    if len(block_ends) != len(target_starts) or len(block_ends) != len(target_ends):
        raise ValueError('Cross-attention chunk plan must provide aligned query and target ranges.')

    projected_context: tuple[torch.Tensor, torch.Tensor] | None = None
    if isinstance(attn, Attention):
        projected_context = attn.project_context(context, k_pe=k_pe)

    output = torch.empty_like(x)
    query_start = 0
    for query_end, target_start, target_end in zip(block_ends, target_starts, target_ends, strict=True):
        if query_end < query_start:
            raise ValueError('Cross-attention chunk plan query block ends must be non-decreasing.')
        if target_start < 0 or target_end < target_start or target_end > context.shape[1]:
            raise ValueError(
                'Cross-attention chunk plan target range is out of bounds. '
                f'Got [{target_start}, {target_end}) for context length {context.shape[1]}.'
            )
        if query_end == query_start:
            continue

        query_chunk = x[:, query_start:query_end]
        pe_chunk = _slice_rope(pe, query_start, query_end)

        if projected_context is not None:
            projected_k, projected_v = projected_context
            key_chunk = projected_k[:, target_start:target_end]
            value_chunk = projected_v[:, target_start:target_end]

            if checkpoint_chunks and torch.is_grad_enabled() and (
                query_chunk.requires_grad or key_chunk.requires_grad or value_chunk.requires_grad
            ):
                current_pe_chunk = pe_chunk

                def run_attn(
                    query_slice: torch.Tensor,
                    key_slice: torch.Tensor,
                    value_slice: torch.Tensor,
                    current_pe_chunk: tuple[torch.Tensor, torch.Tensor] | None = current_pe_chunk,
                ) -> torch.Tensor:
                    return attn.forward_with_preprojected_context(
                        query_slice,
                        projected_k=key_slice,
                        projected_v=value_slice,
                        mask=None,
                        pe=current_pe_chunk,
                    )

                chunk_output = torch.utils.checkpoint.checkpoint(
                    run_attn,
                    query_chunk,
                    key_chunk,
                    value_chunk,
                    use_reentrant=False,
                )
            else:
                chunk_output = attn.forward_with_preprojected_context(
                    query_chunk,
                    projected_k=key_chunk,
                    projected_v=value_chunk,
                    mask=None,
                    pe=pe_chunk,
                )
        else:
            context_chunk = context[:, target_start:target_end]
            k_pe_chunk = _slice_rope(k_pe, target_start, target_end)

            if checkpoint_chunks and torch.is_grad_enabled() and (
                query_chunk.requires_grad or context_chunk.requires_grad
            ):
                current_pe_chunk = pe_chunk
                current_k_pe_chunk = k_pe_chunk

                def run_attn(
                    query_slice: torch.Tensor,
                    context_slice: torch.Tensor,
                    current_pe_chunk: tuple[torch.Tensor, torch.Tensor] | None = current_pe_chunk,
                    current_k_pe_chunk: tuple[torch.Tensor, torch.Tensor] | None = current_k_pe_chunk,
                ) -> torch.Tensor:
                    return attn(
                        query_slice,
                        context=context_slice,
                        mask=None,
                        pe=current_pe_chunk,
                        k_pe=current_k_pe_chunk,
                    )

                chunk_output = torch.utils.checkpoint.checkpoint(
                    run_attn,
                    query_chunk,
                    context_chunk,
                    use_reentrant=False,
                )
            else:
                chunk_output = attn(
                    query_chunk,
                    context=context_chunk,
                    mask=None,
                    pe=pe_chunk,
                    k_pe=k_pe_chunk,
                )

        output[:, query_start:query_end] = chunk_output
        query_start = query_end

    if query_start != x.shape[1]:
        raise ValueError(f'Cross-attention chunk plan covered {query_start} query tokens, expected {x.shape[1]}.')

    return output


class BasicAVTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        idx: int,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
        attention_function: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
    ):
        super().__init__()

        self.idx = idx
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            video_sst_size = adaln_embedding_coefficient(video.cross_attention_adaln)
            self.scale_shift_table = torch.nn.Parameter(torch.empty(video_sst_size, video.dim))

        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            audio_sst_size = adaln_embedding_coefficient(audio.cross_attention_adaln)
            self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(audio_sst_size, audio.dim))

        if audio is not None and video is not None:
            # Q: Video, K,V: Audio
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )

            # Q: Audio, K,V: Video
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )

            self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))

        self.cross_attention_adaln = (video is not None and video.cross_attention_adaln) or (
            audio is not None and audio.cross_attention_adaln
        )

        if self.cross_attention_adaln and video is not None:
            self.prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, video.dim))
        if self.cross_attention_adaln and audio is not None:
            self.audio_prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, audio.dim))

        self.norm_eps = norm_eps

    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]

        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        scale_shift_indices: slice,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :], batch_size, scale_shift_timestep, scale_shift_indices
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :], batch_size, gate_timestep, slice(None, None)
        )

        scale, shift = (t.squeeze(2) for t in scale_shift_ada_values)
        (gate,) = (t.squeeze(2) for t in gate_ada_values)

        return scale, shift, gate

    def _apply_text_cross_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn: AttentionCallable,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep: torch.Tensor,
        prompt_timestep: torch.Tensor | None,
        prompt_timestep_is_per_query: bool,
        prompt_timestep_run_lengths: torch.Tensor | None,
        context_mask: torch.Tensor | None,
        cross_attention_adaln: bool = False,
    ) -> torch.Tensor:
        """Apply text cross-attention, with optional AdaLN modulation."""
        if cross_attention_adaln:
            shift_q, scale_q, gate = self.get_ada_values(scale_shift_table, x.shape[0], timestep, slice(6, 9))
            return apply_cross_attention_adaln(
                x,
                context,
                attn,
                shift_q,
                scale_q,
                gate,
                prompt_scale_shift_table,
                prompt_timestep,
                prompt_timestep_is_per_query,
                prompt_timestep_run_lengths,
                context_mask,
                self.norm_eps,
            )
        return attn(rms_norm(x, eps=self.norm_eps), context=context, mask=context_mask)

    def forward(  # noqa: PLR0915
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        batch_size = (video or audio).x.shape[0]

        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(batch_size)

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and vx.numel() > 0)

        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )
            norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
            del vshift_msa, vscale_msa

            all_perturbed = perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx)
            none_perturbed = not perturbations.any_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx)
            v_mask = (
                perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, vx)
                if not all_perturbed and not none_perturbed
                else None
            )
            vx = (
                vx
                + self.attn1(
                    norm_vx,
                    pe=video.positional_embeddings,
                    mask=video.self_attention_mask,
                    perturbation_mask=v_mask,
                    all_perturbed=all_perturbed,
                )
                * vgate_msa
            )
            del vgate_msa, norm_vx, v_mask
            vx = vx + self._apply_text_cross_attention(
                vx,
                video.context,
                self.attn2,
                self.scale_shift_table,
                getattr(self, "prompt_scale_shift_table", None),
                video.timesteps,
                video.prompt_timestep,
                video.prompt_timestep_is_per_query,
                video.prompt_timestep_run_lengths,
                video.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
            )

        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )

            norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
            del ashift_msa, ascale_msa
            all_perturbed = perturbations.all_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx)
            none_perturbed = not perturbations.any_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx)
            a_mask = (
                perturbations.mask_like(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx, ax)
                if not all_perturbed and not none_perturbed
                else None
            )
            ax = (
                ax
                + self.audio_attn1(
                    norm_ax,
                    pe=audio.positional_embeddings,
                    mask=audio.self_attention_mask,
                    perturbation_mask=a_mask,
                    all_perturbed=all_perturbed,
                )
                * agate_msa
            )
            del agate_msa, norm_ax, a_mask
            ax = ax + self._apply_text_cross_attention(
                ax,
                audio.context,
                self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio.timesteps,
                audio.prompt_timestep,
                audio.prompt_timestep_is_per_query,
                audio.prompt_timestep_run_lengths,
                audio.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
            )

        # Audio - Video cross attention.
        if run_a2v or run_v2a:
            vx_cross_input = vx
            ax_cross_input = ax

            if run_a2v and not perturbations.all_in_batch(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx):
                scale_ca_video_a2v, shift_ca_video_a2v, gate_out_a2v = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx_cross_input.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(0, 2),
                )
                vx_scaled = rms_norm(vx_cross_input, eps=self.norm_eps) * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
                del scale_ca_video_a2v, shift_ca_video_a2v

                scale_ca_audio_a2v, shift_ca_audio_a2v, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax_cross_input.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(0, 2),
                )
                ax_scaled = rms_norm(ax_cross_input, eps=self.norm_eps) * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v
                del scale_ca_audio_a2v, shift_ca_audio_a2v
                a2v_mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx, vx)
                vx = vx + (
                    apply_chunked_cross_attention(
                        vx_scaled,
                        context=ax_scaled,
                        attn=self.audio_to_video_attn,
                        chunk_plan=video.cross_attention_chunk_plan,
                        mask=video.cross_attention_mask,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                        checkpoint_chunks=self.training,
                    )
                    * gate_out_a2v
                    * a2v_mask
                )
                del gate_out_a2v, a2v_mask, vx_scaled, ax_scaled

            if run_v2a and not perturbations.all_in_batch(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx):
                scale_ca_audio_v2a, shift_ca_audio_v2a, gate_out_v2a = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax_cross_input.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(2, 4),
                )
                ax_scaled = rms_norm(ax_cross_input, eps=self.norm_eps) * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
                del scale_ca_audio_v2a, shift_ca_audio_v2a
                scale_ca_video_v2a, shift_ca_video_v2a, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx_cross_input.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(2, 4),
                )
                vx_scaled = rms_norm(vx_cross_input, eps=self.norm_eps) * (1 + scale_ca_video_v2a) + shift_ca_video_v2a
                del scale_ca_video_v2a, shift_ca_video_v2a
                v2a_mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx, ax)
                ax = ax + (
                    apply_chunked_cross_attention(
                        ax_scaled,
                        context=vx_scaled,
                        attn=self.video_to_audio_attn,
                        chunk_plan=audio.cross_attention_chunk_plan,
                        mask=audio.cross_attention_mask,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                        checkpoint_chunks=self.training,
                    )
                    * gate_out_v2a
                    * v2a_mask
                )
                del gate_out_v2a, v2a_mask, ax_scaled, vx_scaled

        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6)
            )
            vx_scaled = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
            vx = vx + apply_checkpointed_feed_forward(self.ff, vx_scaled, checkpoint_ff=self.training) * vgate_mlp

            del vshift_mlp, vscale_mlp, vgate_mlp, vx_scaled

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
            )
            ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
            ax = ax + apply_checkpointed_feed_forward(self.audio_ff, ax_scaled, checkpoint_ff=self.training) * agate_mlp

            del ashift_mlp, ascale_mlp, agate_mlp, ax_scaled

        return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None


def apply_cross_attention_adaln(
    x: torch.Tensor,
    context: torch.Tensor,
    attn: AttentionCallable,
    q_shift: torch.Tensor,
    q_scale: torch.Tensor,
    q_gate: torch.Tensor,
    prompt_scale_shift_table: torch.Tensor,
    prompt_timestep: torch.Tensor,
    prompt_timestep_is_per_query: bool,
    prompt_timestep_run_lengths: torch.Tensor | None = None,
    context_mask: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    batch_size = x.shape[0]
    attn_input = rms_norm(x, eps=norm_eps) * (1 + q_scale) + q_shift
    prompt_scale_shift = prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)

    def apply_prompt_run(
        query_slice: torch.Tensor,
        conditioned_context: torch.Tensor,
        query_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(attn, Attention):
            chunk_outputs: list[torch.Tensor] = []
            for chunk_start in range(0, query_slice.shape[1], PROMPT_QUERY_CHUNK_SIZE):
                chunk_end = min(chunk_start + PROMPT_QUERY_CHUNK_SIZE, query_slice.shape[1])
                query_chunk = query_slice[:, chunk_start:chunk_end]
                mask_chunk = None
                if query_mask is not None:
                    mask_chunk = query_mask[:, :, chunk_start:chunk_end, :]

                if torch.is_grad_enabled() and (
                    query_chunk.requires_grad or conditioned_context.requires_grad
                ):
                    if mask_chunk is not None:
                        def run_prompt_chunk(
                            query_chunk_input: torch.Tensor,
                            context_input: torch.Tensor,
                            mask_input: torch.Tensor,
                        ) -> torch.Tensor:
                            projected_k, projected_v = attn.project_context(context_input)
                            return attn.forward_with_preprojected_context(
                                query_chunk_input,
                                projected_k=projected_k,
                                projected_v=projected_v,
                                mask=mask_input,
                            )

                        chunk_outputs.append(
                            torch.utils.checkpoint.checkpoint(
                                run_prompt_chunk,
                                query_chunk,
                                conditioned_context,
                                mask_chunk,
                                use_reentrant=False,
                            )
                        )
                    else:
                        def run_prompt_chunk(
                            query_chunk_input: torch.Tensor,
                            context_input: torch.Tensor,
                        ) -> torch.Tensor:
                            projected_k, projected_v = attn.project_context(context_input)
                            return attn.forward_with_preprojected_context(
                                query_chunk_input,
                                projected_k=projected_k,
                                projected_v=projected_v,
                                mask=None,
                            )

                        chunk_outputs.append(
                            torch.utils.checkpoint.checkpoint(
                                run_prompt_chunk,
                                query_chunk,
                                conditioned_context,
                                use_reentrant=False,
                            )
                        )
                    continue

                projected_k, projected_v = attn.project_context(conditioned_context)
                chunk_outputs.append(
                    attn.forward_with_preprojected_context(
                        query_chunk,
                        projected_k=projected_k,
                        projected_v=projected_v,
                        mask=mask_chunk,
                    )
                )
            return torch.cat(chunk_outputs, dim=1)

        return attn(query_slice, context=conditioned_context, mask=query_mask)

    if not prompt_timestep_is_per_query:
        shift_kv, scale_kv = (
            prompt_scale_shift + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
        ).unbind(dim=2)
        if scale_kv.shape[1] not in {1, context.shape[1]}:
            raise ValueError(
                "Prompt timestep must align with either the batch or prompt token axis when not using per-query prompt conditioning. "
                f"Got prompt length {scale_kv.shape[1]} for context length {context.shape[1]}."
            )
        encoder_hidden_states = context * (1 + scale_kv) + shift_kv
        return apply_prompt_run(attn_input, encoder_hidden_states, context_mask) * q_gate

    query_len = attn_input.shape[1]
    output = torch.empty_like(attn_input)

    for batch_index in range(batch_size):
        run_start = 0
        if prompt_timestep_run_lengths is not None:
            sample_prompt_timestep = prompt_timestep[batch_index]
            for run_index, run_length_tensor in enumerate(prompt_timestep_run_lengths[batch_index]):
                run_length = int(run_length_tensor.item())
                if run_length <= 0:
                    break

                run_end = run_start + run_length
                if run_end > query_len:
                    raise ValueError(
                        'Per-query prompt conditioning run lengths exceed the query length. '
                        f'Got run_end={run_end} for query length {query_len}.'
                    )

                shift_kv, scale_kv = (
                    prompt_scale_shift + sample_prompt_timestep[run_index : run_index + 1].reshape(1, 1, 2, -1)
                ).unbind(dim=2)
                encoder_hidden_states = context[batch_index : batch_index + 1] * (1 + scale_kv) + shift_kv

                chunk_mask = None
                if context_mask is not None:
                    chunk_mask = context_mask[batch_index : batch_index + 1].expand(-1, -1, run_length, -1)

                output[batch_index : batch_index + 1, run_start:run_end] = apply_prompt_run(
                    attn_input[batch_index : batch_index + 1, run_start:run_end],
                    encoder_hidden_states,
                    chunk_mask,
                )
                run_start = run_end

            if run_start != query_len:
                raise ValueError(
                    'Per-query prompt conditioning run lengths must cover the full query length. '
                    f'Covered {run_start} tokens for query length {query_len}.'
                )
            continue

        if prompt_timestep.shape[1] != query_len:
            raise ValueError(
                "Per-query prompt conditioning requires one prompt timestep per query token. "
                f"Got prompt length {prompt_timestep.shape[1]} for query length {query_len}."
            )

        sample_prompt_timestep = prompt_timestep[batch_index]
        while run_start < query_len:
            run_end = run_start + 1
            while run_end < query_len and torch.equal(
                sample_prompt_timestep[run_end],
                sample_prompt_timestep[run_start],
            ):
                run_end += 1

            shift_kv, scale_kv = (
                prompt_scale_shift + sample_prompt_timestep[run_start : run_start + 1].reshape(1, 1, 2, -1)
            ).unbind(dim=2)
            encoder_hidden_states = context[batch_index : batch_index + 1] * (1 + scale_kv) + shift_kv

            chunk_mask = None
            if context_mask is not None:
                chunk_mask = context_mask[batch_index : batch_index + 1].expand(-1, -1, run_end - run_start, -1)

            output[batch_index : batch_index + 1, run_start:run_end] = apply_prompt_run(
                attn_input[batch_index : batch_index + 1, run_start:run_end],
                encoder_hidden_states,
                chunk_mask,
            )
            run_start = run_end

    return output * q_gate
