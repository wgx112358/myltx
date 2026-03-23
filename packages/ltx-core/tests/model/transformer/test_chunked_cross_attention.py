from __future__ import annotations

import copy

import torch

from ltx_core.model.transformer.attention import Attention, AttentionFunction
from ltx_core.model.transformer.modality import CrossAttentionChunkPlan
from ltx_core.model.transformer.transformer import apply_chunked_cross_attention


def _make_rope(seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(seq_len, dtype=torch.float32).view(1, seq_len, 1)
    return base.clone(), (base + 100).clone()


def test_apply_chunked_cross_attention_matches_exact_token_ranges_across_batch() -> None:
    query = torch.zeros(2, 5, 1)
    context = torch.arange(1, 9, dtype=torch.float32).view(1, 8, 1).expand(2, -1, -1).clone()
    plan = CrossAttentionChunkPlan(
        query_block_ends=torch.tensor([2, 5], dtype=torch.long),
        target_starts=torch.tensor([0, 3], dtype=torch.long),
        target_ends=torch.tensor([4, 7], dtype=torch.long),
    )
    observed_shapes: list[tuple[torch.Size, torch.Size, torch.Size, torch.Size]] = []

    def fake_attn(
        x: torch.Tensor,
        context: torch.Tensor,
        mask: object | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        k_pe: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        assert mask is None
        assert pe is not None and k_pe is not None
        observed_shapes.append((x.shape, context.shape, pe[0].shape, k_pe[0].shape))
        return context.mean(dim=1, keepdim=True).expand(x.shape[0], x.shape[1], context.shape[-1])

    output = apply_chunked_cross_attention(query, context, fake_attn, plan, pe=_make_rope(5), k_pe=_make_rope(8))

    expected = torch.tensor([2.5, 2.5, 5.5, 5.5, 5.5], dtype=torch.float32).view(1, 5, 1).expand(2, -1, -1)
    assert torch.allclose(output, expected)
    assert observed_shapes == [
        (torch.Size([2, 2, 1]), torch.Size([2, 4, 1]), torch.Size([1, 2, 1]), torch.Size([1, 4, 1])),
        (torch.Size([2, 3, 1]), torch.Size([2, 4, 1]), torch.Size([1, 3, 1]), torch.Size([1, 4, 1])),
    ]


def test_apply_chunked_cross_attention_supports_local_windows_that_cut_through_target_blocks() -> None:
    query = torch.zeros(1, 5, 1)
    context = torch.arange(1, 8, dtype=torch.float32).view(1, 7, 1)
    plan = CrossAttentionChunkPlan(
        query_block_ends=torch.tensor([1, 3, 5], dtype=torch.long),
        target_starts=torch.tensor([0, 2, 5], dtype=torch.long),
        target_ends=torch.tensor([2, 5, 7], dtype=torch.long),
    )

    def fake_attn(
        x: torch.Tensor,
        context: torch.Tensor,
        mask: object | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        k_pe: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        del mask, pe, k_pe
        return context.sum(dim=1, keepdim=True).expand(x.shape[0], x.shape[1], context.shape[-1])

    output = apply_chunked_cross_attention(query, context, fake_attn, plan)

    expected = torch.tensor([3.0, 12.0, 12.0, 13.0, 13.0], dtype=torch.float32).view(1, 5, 1)
    assert torch.allclose(output, expected)


def test_apply_chunked_cross_attention_falls_back_to_masked_attention_without_plan() -> None:
    seen_masks: list[object | None] = []
    sentinel_mask = object()

    def fake_attn(x: torch.Tensor, context: torch.Tensor, mask: object | None = None, **_: object) -> torch.Tensor:
        del context
        seen_masks.append(mask)
        return x + 1

    output = apply_chunked_cross_attention(torch.zeros(1, 2, 1), torch.zeros(1, 3, 1), fake_attn, None, mask=sentinel_mask)
    assert torch.equal(output, torch.ones(1, 2, 1))
    assert seen_masks == [sentinel_mask]


def test_apply_chunked_cross_attention_slices_split_rope_on_the_time_axis() -> None:
    query = torch.zeros(1, 5, 1)
    context = torch.zeros(1, 7, 1)
    plan = CrossAttentionChunkPlan(
        query_block_ends=torch.tensor([2, 5], dtype=torch.long),
        target_starts=torch.tensor([0, 3], dtype=torch.long),
        target_ends=torch.tensor([4, 7], dtype=torch.long),
    )
    pe = (
        torch.zeros(1, 2, 5, 4, dtype=torch.float32),
        torch.zeros(1, 2, 5, 4, dtype=torch.float32),
    )
    k_pe = (
        torch.zeros(1, 2, 7, 4, dtype=torch.float32),
        torch.zeros(1, 2, 7, 4, dtype=torch.float32),
    )
    seen_pe_shapes: list[tuple[torch.Size, torch.Size]] = []

    def fake_attn(
        x: torch.Tensor,
        context: torch.Tensor,
        mask: object | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        k_pe: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        del context, mask
        assert pe is not None and k_pe is not None
        seen_pe_shapes.append((pe[0].shape, k_pe[0].shape))
        return x

    output = apply_chunked_cross_attention(query, context, fake_attn, plan, pe=pe, k_pe=k_pe)

    assert torch.equal(output, query)
    assert seen_pe_shapes == [
        (torch.Size([1, 2, 2, 4]), torch.Size([1, 2, 4, 4])),
        (torch.Size([1, 2, 3, 4]), torch.Size([1, 2, 4, 4])),
    ]


def test_apply_chunked_cross_attention_checkpointed_chunks_match_outputs_and_gradients() -> None:
    class FakeAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query_proj = torch.nn.Linear(3, 3, bias=False)
            self.context_proj = torch.nn.Linear(3, 3, bias=False)

        def forward(
            self,
            x: torch.Tensor,
            context: torch.Tensor,
            mask: object | None = None,
            pe: tuple[torch.Tensor, torch.Tensor] | None = None,
            k_pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        ) -> torch.Tensor:
            del mask, pe, k_pe
            pooled_context = context.mean(dim=1, keepdim=True)
            return self.query_proj(x) + self.context_proj(pooled_context).expand(x.shape[0], x.shape[1], -1)

    torch.manual_seed(0)
    plan = CrossAttentionChunkPlan(
        query_block_ends=torch.tensor([2, 5], dtype=torch.long),
        target_starts=torch.tensor([0, 3], dtype=torch.long),
        target_ends=torch.tensor([4, 7], dtype=torch.long),
    )
    base_attn = FakeAttention()
    checkpointed_attn = copy.deepcopy(base_attn)

    query = torch.randn(2, 5, 3, dtype=torch.float32)
    context = torch.randn(2, 7, 3, dtype=torch.float32)

    query_ref = query.clone().requires_grad_(True)
    context_ref = context.clone().requires_grad_(True)
    output_ref = apply_chunked_cross_attention(query_ref, context_ref, base_attn, plan, checkpoint_chunks=False)
    loss_ref = output_ref.sum()
    loss_ref.backward()

    query_ckpt = query.clone().requires_grad_(True)
    context_ckpt = context.clone().requires_grad_(True)
    output_ckpt = apply_chunked_cross_attention(
        query_ckpt,
        context_ckpt,
        checkpointed_attn,
        plan,
        checkpoint_chunks=True,
    )
    loss_ckpt = output_ckpt.sum()
    loss_ckpt.backward()

    assert torch.allclose(output_ckpt, output_ref)
    assert torch.allclose(query_ckpt.grad, query_ref.grad)
    assert torch.allclose(context_ckpt.grad, context_ref.grad)
    assert torch.allclose(checkpointed_attn.query_proj.weight.grad, base_attn.query_proj.weight.grad)
    assert torch.allclose(checkpointed_attn.context_proj.weight.grad, base_attn.context_proj.weight.grad)


def test_apply_chunked_cross_attention_reuses_attention_kv_without_changing_outputs_or_gradients() -> None:
    torch.manual_seed(1)
    plan = CrossAttentionChunkPlan(
        query_block_ends=torch.tensor([2, 5], dtype=torch.long),
        target_starts=torch.tensor([0, 3], dtype=torch.long),
        target_ends=torch.tensor([4, 7], dtype=torch.long),
    )
    baseline_attn = Attention(
        query_dim=4,
        context_dim=4,
        heads=1,
        dim_head=4,
        attention_function=AttentionFunction.PYTORCH,
    )
    optimized_attn = copy.deepcopy(baseline_attn)

    query = torch.randn(2, 5, 4, dtype=torch.float32)
    context = torch.randn(2, 7, 4, dtype=torch.float32)

    query_ref = query.clone().requires_grad_(True)
    context_ref = context.clone().requires_grad_(True)

    def baseline_wrapper(
        x: torch.Tensor,
        context: torch.Tensor,
        mask: object | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        k_pe: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return baseline_attn(x, context=context, mask=mask, pe=pe, k_pe=k_pe)

    output_ref = apply_chunked_cross_attention(
        query_ref,
        context_ref,
        baseline_wrapper,
        plan,
    )
    output_ref.sum().backward()

    query_opt = query.clone().requires_grad_(True)
    context_opt = context.clone().requires_grad_(True)
    output_opt = apply_chunked_cross_attention(
        query_opt,
        context_opt,
        optimized_attn,
        plan,
    )
    output_opt.sum().backward()

    assert torch.allclose(output_opt, output_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(query_opt.grad, query_ref.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(context_opt.grad, context_ref.grad, atol=1e-5, rtol=1e-5)
    for param_opt, param_ref in zip(optimized_attn.parameters(), baseline_attn.parameters(), strict=True):
        assert torch.allclose(param_opt.grad, param_ref.grad, atol=1e-5, rtol=1e-5)


def test_apply_chunked_cross_attention_checkpointed_chunks_keep_chunk_local_rope_during_backward() -> None:
    torch.manual_seed(2)
    plan = CrossAttentionChunkPlan(
        query_block_ends=torch.tensor([2, 5], dtype=torch.long),
        target_starts=torch.tensor([0, 3], dtype=torch.long),
        target_ends=torch.tensor([4, 7], dtype=torch.long),
    )
    baseline_attn = Attention(
        query_dim=4,
        context_dim=4,
        heads=1,
        dim_head=4,
        attention_function=AttentionFunction.PYTORCH,
    )
    checkpointed_attn = copy.deepcopy(baseline_attn)

    query = torch.randn(1, 5, 4, dtype=torch.float32)
    context = torch.randn(1, 7, 4, dtype=torch.float32)
    pe = _make_rope(5)
    k_pe = _make_rope(7)

    query_ref = query.clone().requires_grad_(True)
    context_ref = context.clone().requires_grad_(True)
    output_ref = apply_chunked_cross_attention(
        query_ref,
        context_ref,
        baseline_attn,
        plan,
        pe=pe,
        k_pe=k_pe,
        checkpoint_chunks=False,
    )
    output_ref.sum().backward()

    query_ckpt = query.clone().requires_grad_(True)
    context_ckpt = context.clone().requires_grad_(True)
    output_ckpt = apply_chunked_cross_attention(
        query_ckpt,
        context_ckpt,
        checkpointed_attn,
        plan,
        pe=pe,
        k_pe=k_pe,
        checkpoint_chunks=True,
    )
    output_ckpt.sum().backward()

    assert torch.allclose(output_ckpt, output_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(query_ckpt.grad, query_ref.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(context_ckpt.grad, context_ref.grad, atol=1e-5, rtol=1e-5)
    for param_ckpt, param_ref in zip(checkpointed_attn.parameters(), baseline_attn.parameters(), strict=True):
        assert torch.allclose(param_ckpt.grad, param_ref.grad, atol=1e-5, rtol=1e-5)
