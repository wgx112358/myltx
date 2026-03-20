from __future__ import annotations

import torch

from ltx_core.model.transformer.adaln import AdaLayerNormSingle
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer_args import TransformerArgsPreprocessor


def test_audio_sink_tokens_replace_prefix_and_keep_identity_rope() -> None:
    patchify_proj = torch.nn.Linear(1, 4, bias=False)
    with torch.no_grad():
        patchify_proj.weight.zero_()

    sink_token_embedding = torch.nn.Parameter(torch.ones(1, 4))

    preprocessor = TransformerArgsPreprocessor(
        patchify_proj=patchify_proj,
        adaln=AdaLayerNormSingle(4),
        inner_dim=4,
        max_pos=[20],
        num_attention_heads=1,
        use_middle_indices_grid=True,
        timestep_scale_multiplier=1,
        double_precision_rope=False,
        positional_embedding_theta=10000.0,
        rope_type=LTXRopeType.INTERLEAVED,
        sink_token_embedding=sink_token_embedding,
    )

    modality = Modality(
        latent=torch.zeros(1, 4, 1),
        sigma=torch.zeros(1),
        timesteps=torch.zeros(1, 4),
        positions=torch.tensor(
            [
                [
                    [[0.0, 0.0], [0.0, 0.0], [0.8, 1.2], [1.2, 1.6]],
                ]
            ],
            dtype=torch.float32,
        ),
        context=torch.zeros(1, 1, 4),
        sink_token_count=2,
    )

    args = preprocessor.prepare(modality)
    cos_freqs, sin_freqs = args.positional_embeddings

    assert torch.equal(args.x[0, :2], torch.ones(2, 4))
    assert torch.equal(args.x[0, 2:], torch.zeros(2, 4))
    assert torch.allclose(cos_freqs[0, :2], torch.ones_like(cos_freqs[0, :2]))
    assert torch.allclose(sin_freqs[0, :2], torch.zeros_like(sin_freqs[0, :2]))
