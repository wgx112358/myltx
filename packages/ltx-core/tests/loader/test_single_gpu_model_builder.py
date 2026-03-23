from __future__ import annotations

import torch

from ltx_core.loader.single_gpu_model_builder import _materialize_meta_tensors


class _AliasLeaf:
    def __init__(self, sink_token_embedding: torch.nn.Parameter) -> None:
        self.sink_token_embedding = sink_token_embedding


class _AliasHolder:
    def __init__(self, sink_token_embedding: torch.nn.Parameter) -> None:
        self.simple_preprocessor = _AliasLeaf(sink_token_embedding)


class _ToyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 3, bias=False)
        with torch.device("meta"):
            self.audio_sink_token = torch.nn.Parameter(torch.zeros(1, 4))
            self.register_buffer("meta_buffer", torch.ones(2))
        self.audio_args_preprocessor = _AliasHolder(self.audio_sink_token)


def test_materialize_meta_tensors_keeps_loaded_weights_and_zero_initializes_missing_state() -> None:
    model = _ToyModule()
    with torch.no_grad():
        model.linear.weight.fill_(7.0)

    materialized = _materialize_meta_tensors(model, torch.device("cpu"))

    assert materialized == ["audio_sink_token", "meta_buffer"]
    assert model.audio_sink_token.device.type == "cpu"
    assert model.meta_buffer.device.type == "cpu"
    assert torch.equal(model.audio_sink_token, torch.zeros_like(model.audio_sink_token))
    assert torch.equal(model.meta_buffer, torch.zeros_like(model.meta_buffer))
    assert torch.equal(model.linear.weight, torch.full_like(model.linear.weight, 7.0))
    assert model.audio_args_preprocessor.simple_preprocessor.sink_token_embedding is model.audio_sink_token
