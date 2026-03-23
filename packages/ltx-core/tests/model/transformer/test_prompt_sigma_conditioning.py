from __future__ import annotations

import pytest
import torch

import ltx_core.model.transformer.transformer as transformer_module
from ltx_core.model.transformer.adaln import AdaLayerNormSingle
from ltx_core.model.transformer.attention import Attention, AttentionFunction
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer import PROMPT_QUERY_CHUNK_SIZE, apply_cross_attention_adaln
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


def test_preprocessor_compresses_repeated_per_query_prompt_sigma_into_runs() -> None:
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
        prompt_adaln=AdaLayerNormSingle(4, embedding_coefficient=2),
    )

    modality = Modality(
        latent=torch.zeros(1, 6, 1),
        sigma=torch.tensor([9.0]),
        timesteps=torch.zeros(1, 6),
        positions=torch.tensor(
            [[[[0.0, 0.16666667], [0.16666667, 0.33333334], [0.33333334, 0.5], [0.5, 0.6666667], [0.6666667, 0.8333333], [0.8333333, 1.0]]]],
            dtype=torch.float32,
        ),
        context=torch.zeros(1, 3, 4),
        prompt_sigma=torch.tensor([[1.0, 1.0, 2.0, 2.0, 2.0, 3.0]], dtype=torch.float32),
    )

    args = preprocessor.prepare(modality)

    assert args.prompt_timestep_is_per_query is True
    assert args.prompt_timestep.shape == (1, 3, 8)
    assert hasattr(args, 'prompt_timestep_run_lengths')
    assert torch.equal(getattr(args, 'prompt_timestep_run_lengths'), torch.tensor([[2, 3, 1]], dtype=torch.long))


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


def test_apply_cross_attention_adaln_supports_run_length_encoded_per_query_prompt_conditioning() -> None:
    observed_x_shapes: list[torch.Size] = []
    observed_context_shapes: list[torch.Size] = []

    def fake_attn(x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        observed_x_shapes.append(x.shape)
        observed_context_shapes.append(context.shape)
        return context.mean(dim=1, keepdim=True)

    output = apply_cross_attention_adaln(
        x=torch.ones(1, 3, 1),
        context=torch.ones(1, 3, 1),
        attn=fake_attn,
        q_shift=torch.zeros(1, 3, 1),
        q_scale=torch.zeros(1, 3, 1),
        q_gate=torch.ones(1, 3, 1),
        prompt_scale_shift_table=torch.zeros(2, 1),
        prompt_timestep=torch.tensor([[[1.0, 0.0], [2.0, 0.5]]], dtype=torch.float32),
        prompt_timestep_is_per_query=True,
        prompt_timestep_run_lengths=torch.tensor([[2, 1]], dtype=torch.long),
        context_mask=None,
    )

    assert observed_x_shapes == [torch.Size([1, 2, 1]), torch.Size([1, 1, 1])]
    assert observed_context_shapes == [torch.Size([1, 3, 1]), torch.Size([1, 3, 1])]
    assert torch.allclose(output, torch.tensor([[[2.0], [2.0], [3.5]]], dtype=torch.float32))


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


def test_apply_cross_attention_adaln_attention_path_matches_baseline_under_query_chunking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)

    attn = Attention(
        query_dim=4,
        context_dim=4,
        heads=1,
        dim_head=4,
        attention_function=AttentionFunction.PYTORCH,
    )

    x = torch.randn(1, PROMPT_QUERY_CHUNK_SIZE + 32, 4, dtype=torch.float32)
    context = torch.randn(1, 6, 4, dtype=torch.float32)
    q_shift = torch.randn_like(x)
    q_scale = torch.randn_like(x) * 0.1
    q_gate = torch.ones_like(x)
    prompt_scale_shift_table = torch.randn(2, 4, dtype=torch.float32)
    prompt_timestep = torch.randn(1, 2, 8, dtype=torch.float32)
    prompt_timestep_run_lengths = torch.tensor([[PROMPT_QUERY_CHUNK_SIZE, 32]], dtype=torch.long)

    # Use a second attention instance to avoid mutating the module used for the baseline.
    attn_chunked = Attention(
        query_dim=4,
        context_dim=4,
        heads=1,
        dim_head=4,
        attention_function=AttentionFunction.PYTORCH,
    )
    attn_chunked.load_state_dict(attn.state_dict())

    x_baseline = x.clone().requires_grad_(True)
    context_baseline = context.clone().requires_grad_(True)
    q_shift_baseline = q_shift.clone().requires_grad_(True)
    q_scale_baseline = q_scale.clone().requires_grad_(True)

    # Disable query chunking for the baseline call so we can compare exact outputs.
    monkeypatch.setattr(transformer_module, 'PROMPT_QUERY_CHUNK_SIZE', x.shape[1] + 1)
    output_baseline = apply_cross_attention_adaln(
        x=x_baseline,
        context=context_baseline,
        attn=attn,
        q_shift=q_shift_baseline,
        q_scale=q_scale_baseline,
        q_gate=q_gate,
        prompt_scale_shift_table=prompt_scale_shift_table,
        prompt_timestep=prompt_timestep,
        prompt_timestep_is_per_query=True,
        prompt_timestep_run_lengths=prompt_timestep_run_lengths,
        context_mask=None,
    )
    output_baseline.sum().backward()

    observed_shapes: list[torch.Size] = []
    original_chunked = attn_chunked.forward_with_preprojected_context

    def recorder(
        x: torch.Tensor,
        projected_k: torch.Tensor,
        projected_v: torch.Tensor,
        mask: object | None = None,
        pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        observed_shapes.append(x.shape)
        return original_chunked(x, projected_k, projected_v, mask=mask, pe=pe)

    attn_chunked.forward_with_preprojected_context = recorder  # type: ignore[method-assign]
    monkeypatch.setattr(transformer_module, 'PROMPT_QUERY_CHUNK_SIZE', PROMPT_QUERY_CHUNK_SIZE)

    x_chunked = x.clone().requires_grad_(True)
    context_chunked = context.clone().requires_grad_(True)
    q_shift_chunked = q_shift.clone().requires_grad_(True)
    q_scale_chunked = q_scale.clone().requires_grad_(True)
    output_chunked = apply_cross_attention_adaln(
        x=x_chunked,
        context=context_chunked,
        attn=attn_chunked,
        q_shift=q_shift_chunked,
        q_scale=q_scale_chunked,
        q_gate=q_gate,
        prompt_scale_shift_table=prompt_scale_shift_table,
        prompt_timestep=prompt_timestep,
        prompt_timestep_is_per_query=True,
        prompt_timestep_run_lengths=prompt_timestep_run_lengths,
        context_mask=None,
    )
    output_chunked.sum().backward()

    assert torch.allclose(output_chunked, output_baseline, atol=1e-5, rtol=1e-5)
    assert torch.allclose(x_chunked.grad, x_baseline.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(context_chunked.grad, context_baseline.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(q_shift_chunked.grad, q_shift_baseline.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(q_scale_chunked.grad, q_scale_baseline.grad, atol=1e-5, rtol=1e-5)
    for param_chunked, param_baseline in zip(attn_chunked.parameters(), attn.parameters(), strict=True):
        assert torch.allclose(param_chunked.grad, param_baseline.grad, atol=1e-5, rtol=1e-5)
    assert observed_shapes == [
        torch.Size([1, PROMPT_QUERY_CHUNK_SIZE, 4]),
        torch.Size([1, 32, 4]),
        torch.Size([1, 32, 4]),
        torch.Size([1, PROMPT_QUERY_CHUNK_SIZE, 4]),
    ]


def test_apply_cross_attention_adaln_query_chunk_checkpoint_keeps_chunk_local_masks_in_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1)

    attn = Attention(
        query_dim=4,
        context_dim=4,
        heads=1,
        dim_head=4,
        attention_function=AttentionFunction.PYTORCH,
    )
    attn_chunked = Attention(
        query_dim=4,
        context_dim=4,
        heads=1,
        dim_head=4,
        attention_function=AttentionFunction.PYTORCH,
    )
    attn_chunked.load_state_dict(attn.state_dict())

    x = torch.randn(1, PROMPT_QUERY_CHUNK_SIZE + 128, 4, dtype=torch.float32)
    context = torch.randn(1, 6, 4, dtype=torch.float32)
    q_shift = torch.randn_like(x)
    q_scale = torch.randn_like(x) * 0.1
    q_gate = torch.ones_like(x)
    prompt_scale_shift_table = torch.randn(2, 4, dtype=torch.float32)
    prompt_timestep = torch.randn(1, 1, 8, dtype=torch.float32)
    prompt_timestep_run_lengths = torch.tensor([[x.shape[1]]], dtype=torch.long)
    context_mask = torch.zeros(1, 1, 1, context.shape[1], dtype=torch.float32)

    x_baseline = x.clone().requires_grad_(True)
    context_baseline = context.clone().requires_grad_(True)
    q_shift_baseline = q_shift.clone().requires_grad_(True)
    q_scale_baseline = q_scale.clone().requires_grad_(True)

    monkeypatch.setattr(transformer_module, 'PROMPT_QUERY_CHUNK_SIZE', x.shape[1] + 1)
    output_baseline = apply_cross_attention_adaln(
        x=x_baseline,
        context=context_baseline,
        attn=attn,
        q_shift=q_shift_baseline,
        q_scale=q_scale_baseline,
        q_gate=q_gate,
        prompt_scale_shift_table=prompt_scale_shift_table,
        prompt_timestep=prompt_timestep,
        prompt_timestep_is_per_query=True,
        prompt_timestep_run_lengths=prompt_timestep_run_lengths,
        context_mask=context_mask,
    )
    output_baseline.sum().backward()

    x_chunked = x.clone().requires_grad_(True)
    context_chunked = context.clone().requires_grad_(True)
    q_shift_chunked = q_shift.clone().requires_grad_(True)
    q_scale_chunked = q_scale.clone().requires_grad_(True)

    monkeypatch.setattr(transformer_module, 'PROMPT_QUERY_CHUNK_SIZE', PROMPT_QUERY_CHUNK_SIZE)
    output_chunked = apply_cross_attention_adaln(
        x=x_chunked,
        context=context_chunked,
        attn=attn_chunked,
        q_shift=q_shift_chunked,
        q_scale=q_scale_chunked,
        q_gate=q_gate,
        prompt_scale_shift_table=prompt_scale_shift_table,
        prompt_timestep=prompt_timestep,
        prompt_timestep_is_per_query=True,
        prompt_timestep_run_lengths=prompt_timestep_run_lengths,
        context_mask=context_mask,
    )
    output_chunked.sum().backward()

    assert torch.allclose(output_chunked, output_baseline, atol=1e-5, rtol=1e-5)
    assert torch.allclose(x_chunked.grad, x_baseline.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(context_chunked.grad, context_baseline.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(q_shift_chunked.grad, q_shift_baseline.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(q_scale_chunked.grad, q_scale_baseline.grad, atol=1e-5, rtol=1e-5)
    for param_chunked, param_baseline in zip(attn_chunked.parameters(), attn.parameters(), strict=True):
        assert torch.allclose(param_chunked.grad, param_baseline.grad, atol=1e-5, rtol=1e-5)
