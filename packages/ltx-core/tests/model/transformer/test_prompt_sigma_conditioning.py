from __future__ import annotations

import pytest
import torch

from ltx_core.model.transformer.adaln import AdaLayerNormSingle
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer import apply_cross_attention_adaln
from ltx_core.model.transformer.transformer_args import TransformerArgsPreprocessor


def test_preprocessor_uses_prompt_sigma_for_prompt_timestep() -> None:
    preprocessor = TransformerArgsPreprocessor(
        patchify_proj=torch.nn.Linear(1, 4, bias=False),
        adaln=AdaLayerNormSingle(4),
        inner_dim=4,
        max_pos=[20],
        num_attention_heads=1,
        use_middle_indices_grid=True,
        timestep_scale_multiplier=1,
        double_precision_rope=False,
        positional_embedding_theta=10000.0,
        rope_type=LTXRopeType.INTERLEAVED,
        prompt_adaln=AdaLayerNormSingle(4),
    )

    modality = Modality(
        latent=torch.zeros(1, 2, 1),
        sigma=torch.tensor([9.0]),
        timesteps=torch.zeros(1, 2),
        positions=torch.tensor([[[[0.0, 0.5], [0.5, 1.0]]]], dtype=torch.float32),
        context=torch.zeros(1, 3, 4),
        prompt_sigma=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    )

    args = preprocessor.prepare(modality)

    assert args.prompt_timestep is not None
    assert args.prompt_timestep.shape[1] == 2
    assert args.prompt_timestep_is_per_query is True


def test_apply_cross_attention_adaln_supports_per_query_prompt_conditioning() -> None:
    observed_x_shapes: list[torch.Size] = []
    observed_context_shapes: list[torch.Size] = []
    observed_mask_shapes: list[torch.Size | None] = []

    def fake_attn(x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        observed_x_shapes.append(x.shape)
        observed_context_shapes.append(context.shape)
        observed_mask_shapes.append(None if mask is None else mask.shape)
        return context.mean(dim=1, keepdim=True)

    output = apply_cross_attention_adaln(
        x=torch.ones(1, 2, 1),
        context=torch.ones(1, 3, 1),
        attn=fake_attn,
        q_shift=torch.zeros(1, 2, 1),
        q_scale=torch.zeros(1, 2, 1),
        q_gate=torch.ones(1, 2, 1),
        prompt_scale_shift_table=torch.zeros(2, 1),
        prompt_timestep=torch.tensor([[[1.0, 0.0], [2.0, 0.5]]], dtype=torch.float32),
        prompt_timestep_is_per_query=True,
        context_mask=None,
    )

    assert observed_x_shapes == [torch.Size([1, 1, 1]), torch.Size([1, 1, 1])]
    assert observed_context_shapes == [torch.Size([1, 3, 1]), torch.Size([1, 3, 1])]
    assert observed_mask_shapes == [None, None]
    assert torch.allclose(output, torch.tensor([[[2.0], [3.5]]], dtype=torch.float32))


def test_preprocessor_rejects_ambiguous_prompt_sigma_shape() -> None:
    preprocessor = TransformerArgsPreprocessor(
        patchify_proj=torch.nn.Linear(1, 4, bias=False),
        adaln=AdaLayerNormSingle(4),
        inner_dim=4,
        max_pos=[20],
        num_attention_heads=1,
        use_middle_indices_grid=True,
        timestep_scale_multiplier=1,
        double_precision_rope=False,
        positional_embedding_theta=10000.0,
        rope_type=LTXRopeType.INTERLEAVED,
        prompt_adaln=AdaLayerNormSingle(4),
    )

    modality = Modality(
        latent=torch.zeros(1, 2, 1),
        sigma=torch.tensor([9.0]),
        timesteps=torch.zeros(1, 2),
        positions=torch.tensor([[[[0.0, 0.5], [0.5, 1.0]]]], dtype=torch.float32),
        context=torch.zeros(1, 3, 4),
        prompt_sigma=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="Expected prompt sigma"):
        preprocessor.prepare(modality)
