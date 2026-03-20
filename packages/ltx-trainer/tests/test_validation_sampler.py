from __future__ import annotations

import pytest
import torch

from ltx_core.types import LatentState
from ltx_trainer.training_strategies.ode_regression import create_block_mask
from ltx_trainer.validation_sampler import CachedPromptEmbeddings, GenerationConfig, ValidationSampler


def _make_sampler() -> ValidationSampler:
    return ValidationSampler(
        transformer=object(),  # type: ignore[arg-type]
        vae_decoder=object(),  # type: ignore[arg-type]
        vae_encoder=None,
    )


def test_prepend_audio_sink_tokens_uses_zero_prefix_and_identity_rope() -> None:
    sampler = _make_sampler()
    state = LatentState(
        latent=torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float32),
        denoise_mask=torch.tensor([[[1.0], [1.0], [1.0]]], dtype=torch.float32),
        positions=torch.tensor([[[[0.0, 0.5], [0.5, 1.0], [1.0, 1.5]]]], dtype=torch.float32),
        clean_latent=torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float32),
    )
    config = GenerationConfig(
        prompt="test",
        height=32,
        width=32,
        num_frames=9,
        frame_rate=8.0,
        generate_audio=True,
        cached_embeddings=CachedPromptEmbeddings(
            video_context_positive=torch.zeros(1, 1, 4),
            audio_context_positive=torch.zeros(1, 1, 4),
        ),
        audio_sink_token_count=2,
        audio_sink_identity_rope=True,
    )

    prefixed = sampler._prepend_audio_sink_tokens(state, config)

    assert torch.equal(prefixed.latent[0, :, 0], torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(prefixed.clean_latent[0, :, 0], torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(prefixed.denoise_mask[0, :, 0], torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]))
    assert torch.equal(prefixed.positions[0, 0, :2], torch.zeros(2, 2))


@pytest.mark.skipif(create_block_mask is None, reason="flex attention block mask support is unavailable")
def test_build_denoising_modalities_applies_block_masks_and_sink_token_count() -> None:
    sampler = _make_sampler()
    config = GenerationConfig(
        prompt="test",
        height=32,
        width=32,
        num_frames=9,
        frame_rate=8.0,
        generate_audio=True,
        cached_embeddings=CachedPromptEmbeddings(
            video_context_positive=torch.zeros(1, 1, 4),
            audio_context_positive=torch.zeros(1, 1, 4),
        ),
        use_block_causal_mask=True,
        block_size=2,
        independent_first_frame=True,
        audio_boundary_mode="left",
        audio_sink_token_count=2,
        audio_sink_identity_rope=True,
    )
    video_tools = sampler._create_video_latent_tools(config)
    audio_tools = sampler._create_audio_latent_tools(config)
    video_state = video_tools.create_initial_state(device="cpu", dtype=torch.float32)
    audio_state = sampler._prepend_audio_sink_tokens(
        audio_tools.create_initial_state(device="cpu", dtype=torch.float32),
        config,
    )

    video, audio = sampler._build_denoising_modalities(
        config=config,
        video_state=video_state,
        audio_state=audio_state,
        sigma=torch.tensor(1.0),
        v_ctx=torch.zeros(1, 1, 4),
        a_ctx=torch.zeros(1, 1, 4),
    )

    assert video.attention_mask is not None
    assert video.cross_attention_mask is not None
    assert audio is not None
    assert audio.attention_mask is not None
    assert audio.cross_attention_mask is not None
    assert audio.sink_token_count == 2
    assert torch.equal(audio.timesteps[0, :2, 0], torch.zeros(2))


def test_strip_audio_sink_tokens_removes_prefix_slots_before_decode() -> None:
    sampler = _make_sampler()
    state = LatentState(
        latent=torch.tensor([[[10.0], [20.0], [1.0], [2.0], [3.0]]], dtype=torch.float32),
        denoise_mask=torch.tensor([[[0.0], [0.0], [1.0], [1.0], [1.0]]], dtype=torch.float32),
        positions=torch.tensor(
            [[[[0.0, 0.0], [0.0, 0.0], [0.0, 0.5], [0.5, 1.0], [1.0, 1.5]]]],
            dtype=torch.float32,
        ),
        clean_latent=torch.tensor([[[10.0], [20.0], [1.0], [2.0], [3.0]]], dtype=torch.float32),
    )

    stripped = sampler._strip_audio_sink_tokens(state, sink_token_count=2)

    assert torch.equal(stripped.latent[0, :, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(stripped.clean_latent[0, :, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(stripped.denoise_mask[0, :, 0], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(stripped.positions[0, 0, :, 0], torch.tensor([0.0, 0.5, 1.0]))
