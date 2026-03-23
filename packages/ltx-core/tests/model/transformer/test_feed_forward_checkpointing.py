import torch

from ltx_core.model.transformer.feed_forward import FeedForward
from ltx_core.model.transformer.transformer import apply_checkpointed_feed_forward


def test_apply_checkpointed_feed_forward_matches_baseline() -> None:
    torch.manual_seed(0)

    ff = FeedForward(dim=4, dim_out=4, mult=2)
    ff_checkpointed = FeedForward(dim=4, dim_out=4, mult=2)
    ff_checkpointed.load_state_dict(ff.state_dict())

    x_baseline = torch.randn(2, 8, 4, dtype=torch.float32, requires_grad=True)
    x_checkpointed = x_baseline.detach().clone().requires_grad_(True)

    baseline_output = ff(x_baseline)
    baseline_output.sum().backward()

    observed_shapes: list[torch.Size] = []
    original_forward = ff_checkpointed.forward

    def recorder(x: torch.Tensor) -> torch.Tensor:
        observed_shapes.append(x.shape)
        return original_forward(x)

    ff_checkpointed.forward = recorder  # type: ignore[method-assign]

    checkpointed_output = apply_checkpointed_feed_forward(
        ff_checkpointed,
        x_checkpointed,
        checkpoint_ff=True,
    )
    checkpointed_output.sum().backward()

    assert torch.allclose(checkpointed_output, baseline_output, atol=1e-5, rtol=1e-5)
    assert torch.allclose(x_checkpointed.grad, x_baseline.grad, atol=1e-5, rtol=1e-5)
    for checkpointed_param, baseline_param in zip(ff_checkpointed.parameters(), ff.parameters(), strict=True):
        assert torch.allclose(checkpointed_param.grad, baseline_param.grad, atol=1e-5, rtol=1e-5)
    assert observed_shapes == [
        torch.Size([2, 8, 4]),
        torch.Size([2, 8, 4]),
    ]


def test_apply_checkpointed_feed_forward_skips_checkpoint_when_disabled() -> None:
    torch.manual_seed(0)

    ff = FeedForward(dim=4, dim_out=4, mult=2)
    x = torch.randn(2, 8, 4, dtype=torch.float32, requires_grad=True)

    observed_shapes: list[torch.Size] = []
    original_forward = ff.forward

    def recorder(x: torch.Tensor) -> torch.Tensor:
        observed_shapes.append(x.shape)
        return original_forward(x)

    ff.forward = recorder  # type: ignore[method-assign]

    output = apply_checkpointed_feed_forward(ff, x, checkpoint_ff=False)
    output.sum().backward()

    assert observed_shapes == [torch.Size([2, 8, 4])]
