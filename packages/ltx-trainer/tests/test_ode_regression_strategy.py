from __future__ import annotations

import torch

from ltx_trainer.training_strategies.ode_regression import ODERegressionConfig, ODERegressionStrategy


def _make_video_positions() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [[0.0, 0.5], [0.5, 1.0], [1.0, 1.5], [1.5, 2.0], [2.0, 2.5]],
                [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            ]
        ],
        dtype=torch.float32,
    )


def _make_audio_positions() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [[0.0, 0.4], [0.4, 0.8], [0.8, 1.2], [1.2, 1.6], [1.6, 2.0], [2.0, 2.4]],
            ]
        ],
        dtype=torch.float32,
    )


def test_strategy_uses_ode_trajectory_directories_by_default() -> None:
    strategy = ODERegressionStrategy(ODERegressionConfig(name="ode_regression", with_audio=True))

    assert strategy.get_data_sources() == {
        "ode_video_trajectories": "latents",
        "conditions": "conditions",
        "ode_audio_trajectories": "audio_latents",
    }


def test_blockwise_strategy_builds_mixed_video_and_audio_inputs() -> None:
    strategy = ODERegressionStrategy(
        ODERegressionConfig(
            name="ode_regression",
            with_audio=True,
            use_block_causal_mask=False,
            block_size=2,
            independent_first_frame=True,
            audio_boundary_mode="left",
            ode_layout_mode="blockwise",
        )
    )

    strategy._get_video_positions = lambda **_: _make_video_positions()  # type: ignore[method-assign]
    strategy._get_audio_positions = lambda **_: _make_audio_positions()  # type: ignore[method-assign]

    video_trajectory = torch.tensor(
        [
            [
                [[[10.0]], [[10.0]], [[10.0]], [[10.0]], [[10.0]]],
            ],
            [
                [[[20.0]], [[20.0]], [[20.0]], [[20.0]], [[20.0]]],
            ],
            [
                [[[30.0]], [[30.0]], [[30.0]], [[30.0]], [[30.0]]],
            ],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    audio_trajectory = torch.tensor(
        [
            [
                [[100.0], [100.0], [100.0], [100.0], [100.0], [100.0]],
            ],
            [
                [[200.0], [200.0], [200.0], [200.0], [200.0], [200.0]],
            ],
            [
                [[300.0], [300.0], [300.0], [300.0], [300.0], [300.0]],
            ],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    batch = {
        "latents": {
            "latents": torch.full((1, 1, 5, 1, 1), -999.0),
            "ode_target_latents": video_trajectory[:, -1],
            "ode_sigma": torch.tensor([1.0]),
            "ode_video_trajectory": video_trajectory,
            "ode_trajectory_sigmas": torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float32),
            "ode_block_step_indices": torch.tensor([[0, 1, 2]], dtype=torch.long),
            "num_frames": torch.tensor([5]),
            "height": torch.tensor([1]),
            "width": torch.tensor([1]),
            "fps": torch.tensor([24.0]),
        },
        "audio_latents": {
            "latents": torch.full((1, 1, 6, 1), -999.0),
            "ode_target_latents": audio_trajectory[:, -1],
            "ode_sigma": torch.tensor([1.0]),
            "ode_audio_trajectory": audio_trajectory,
            "ode_trajectory_sigmas": torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float32),
            "num_time_steps": torch.tensor([6]),
            "frequency_bins": torch.tensor([1]),
        },
        "conditions": {
            "video_prompt_embeds": torch.zeros(1, 2, 4),
            "audio_prompt_embeds": torch.zeros(1, 2, 4),
            "prompt_attention_mask": torch.ones(1, 2, dtype=torch.bool),
        },
    }

    inputs = strategy.prepare_training_inputs(batch, timestep_sampler=None)

    assert torch.equal(inputs.video.latent[0, :, 0], torch.tensor([10.0, 20.0, 20.0, 30.0, 30.0]))
    assert torch.equal(inputs.video.timesteps[0], torch.tensor([1.0, 0.5, 0.5, 0.0, 0.0]))
    assert torch.equal(inputs.video_loss_mask[0], torch.tensor([True, True, True, False, False]))

    assert torch.equal(inputs.audio.latent[0, :, 0], torch.tensor([100.0, 100.0, 200.0, 200.0, 300.0, 300.0]))
    assert torch.equal(inputs.audio.timesteps[0], torch.tensor([1.0, 1.0, 0.5, 0.5, 0.0, 0.0]))
    assert torch.equal(inputs.audio_loss_mask[0], torch.tensor([True, True, True, True, False, False]))


def test_blockwise_strategy_prepends_audio_sink_tokens() -> None:
    strategy = ODERegressionStrategy(
        ODERegressionConfig(
            name="ode_regression",
            with_audio=True,
            use_block_causal_mask=False,
            block_size=2,
            independent_first_frame=True,
            audio_boundary_mode="left",
            ode_layout_mode="blockwise",
            audio_sink_token_count=2,
            audio_sink_identity_rope=True,
        )
    )

    strategy._get_video_positions = lambda **_: _make_video_positions()  # type: ignore[method-assign]
    strategy._get_audio_positions = lambda **_: _make_audio_positions()  # type: ignore[method-assign]

    video_trajectory = torch.tensor(
        [
            [
                [[[10.0]], [[10.0]], [[10.0]], [[10.0]], [[10.0]]],
            ],
            [
                [[[20.0]], [[20.0]], [[20.0]], [[20.0]], [[20.0]]],
            ],
            [
                [[[30.0]], [[30.0]], [[30.0]], [[30.0]], [[30.0]]],
            ],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    audio_trajectory = torch.tensor(
        [
            [
                [[100.0], [100.0], [100.0], [100.0], [100.0], [100.0]],
            ],
            [
                [[200.0], [200.0], [200.0], [200.0], [200.0], [200.0]],
            ],
            [
                [[300.0], [300.0], [300.0], [300.0], [300.0], [300.0]],
            ],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    batch = {
        "latents": {
            "latents": torch.full((1, 1, 5, 1, 1), -999.0),
            "ode_target_latents": video_trajectory[:, -1],
            "ode_sigma": torch.tensor([1.0]),
            "ode_video_trajectory": video_trajectory,
            "ode_trajectory_sigmas": torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float32),
            "ode_block_step_indices": torch.tensor([[0, 1, 2]], dtype=torch.long),
            "num_frames": torch.tensor([5]),
            "height": torch.tensor([1]),
            "width": torch.tensor([1]),
            "fps": torch.tensor([24.0]),
        },
        "audio_latents": {
            "latents": torch.full((1, 1, 6, 1), -999.0),
            "ode_target_latents": audio_trajectory[:, -1],
            "ode_sigma": torch.tensor([1.0]),
            "ode_audio_trajectory": audio_trajectory,
            "ode_trajectory_sigmas": torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float32),
            "num_time_steps": torch.tensor([6]),
            "frequency_bins": torch.tensor([1]),
        },
        "conditions": {
            "video_prompt_embeds": torch.zeros(1, 2, 4),
            "audio_prompt_embeds": torch.zeros(1, 2, 4),
            "prompt_attention_mask": torch.ones(1, 2, dtype=torch.bool),
        },
    }

    inputs = strategy.prepare_training_inputs(batch, timestep_sampler=None)

    assert torch.equal(inputs.audio.latent[0, :, 0], torch.tensor([0.0, 0.0, 100.0, 100.0, 200.0, 200.0, 300.0, 300.0]))
    assert torch.equal(inputs.audio.timesteps[0], torch.tensor([0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0]))
    assert torch.equal(inputs.audio_loss_mask[0], torch.tensor([False, False, True, True, True, True, False, False]))
    assert torch.equal(inputs.audio.positions[0, 0, :2], torch.zeros(2, 2))


def test_blockwise_strategy_shares_block_sigmas_between_video_and_audio() -> None:
    strategy = ODERegressionStrategy(
        ODERegressionConfig(
            name="ode_regression",
            with_audio=True,
            use_block_causal_mask=False,
            block_size=2,
            independent_first_frame=True,
            audio_boundary_mode="left",
            ode_layout_mode="blockwise",
        )
    )

    strategy._get_video_positions = lambda **_: _make_video_positions()  # type: ignore[method-assign]
    strategy._get_audio_positions = lambda **_: _make_audio_positions()  # type: ignore[method-assign]
    strategy._sample_block_step_indices = lambda **_: torch.tensor([[0, 2, 1]], dtype=torch.long)  # type: ignore[method-assign]

    video_trajectory = torch.tensor(
        [
            [
                [[[10.0]], [[10.0]], [[10.0]], [[10.0]], [[10.0]]],
            ],
            [
                [[[20.0]], [[20.0]], [[20.0]], [[20.0]], [[20.0]]],
            ],
            [
                [[[30.0]], [[30.0]], [[30.0]], [[30.0]], [[30.0]]],
            ],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    audio_trajectory = torch.tensor(
        [
            [
                [[100.0], [100.0], [100.0], [100.0], [100.0], [100.0]],
            ],
            [
                [[200.0], [200.0], [200.0], [200.0], [200.0], [200.0]],
            ],
            [
                [[300.0], [300.0], [300.0], [300.0], [300.0], [300.0]],
            ],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    batch = {
        "latents": {
            "latents": torch.full((1, 1, 5, 1, 1), -999.0),
            "ode_target_latents": video_trajectory[:, -1],
            "ode_sigma": torch.tensor([1.0]),
            "ode_video_trajectory": video_trajectory,
            "ode_trajectory_sigmas": torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float32),
            "num_frames": torch.tensor([5]),
            "height": torch.tensor([1]),
            "width": torch.tensor([1]),
            "fps": torch.tensor([24.0]),
        },
        "audio_latents": {
            "latents": torch.full((1, 1, 6, 1), -999.0),
            "ode_target_latents": audio_trajectory[:, -1],
            "ode_sigma": torch.tensor([1.0]),
            "ode_audio_trajectory": audio_trajectory,
            "ode_trajectory_sigmas": torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float32),
            "num_time_steps": torch.tensor([6]),
            "frequency_bins": torch.tensor([1]),
        },
        "conditions": {
            "video_prompt_embeds": torch.zeros(1, 2, 4),
            "audio_prompt_embeds": torch.zeros(1, 2, 4),
            "prompt_attention_mask": torch.ones(1, 2, dtype=torch.bool),
        },
    }

    inputs = strategy.prepare_training_inputs(batch, timestep_sampler=None)

    assert torch.equal(inputs.video.timesteps[0], torch.tensor([1.0, 0.0, 0.0, 0.5, 0.5]))
    assert torch.equal(inputs.audio.timesteps[0], torch.tensor([1.0, 1.0, 0.0, 0.0, 0.5, 0.5]))
    assert torch.equal(inputs.video.prompt_sigma[0], inputs.video.timesteps[0])
    assert torch.equal(inputs.audio.prompt_sigma[0], inputs.audio.timesteps[0])


def test_blockwise_strategy_uses_trajectory_without_top_level_current_latents() -> None:
    strategy = ODERegressionStrategy(
        ODERegressionConfig(
            name="ode_regression",
            with_audio=True,
            use_block_causal_mask=False,
            block_size=2,
            independent_first_frame=True,
            audio_boundary_mode="left",
            ode_layout_mode="blockwise",
        )
    )

    strategy._get_video_positions = lambda **_: _make_video_positions()  # type: ignore[method-assign]
    strategy._get_audio_positions = lambda **_: _make_audio_positions()  # type: ignore[method-assign]
    strategy._sample_block_step_indices = lambda **_: torch.tensor([[0, 2, 1]], dtype=torch.long)  # type: ignore[method-assign]

    video_trajectory = torch.tensor(
        [
            [
                [[[10.0]], [[10.0]], [[10.0]], [[10.0]], [[10.0]]],
            ],
            [
                [[[20.0]], [[20.0]], [[20.0]], [[20.0]], [[20.0]]],
            ],
            [
                [[[30.0]], [[30.0]], [[30.0]], [[30.0]], [[30.0]]],
            ],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    audio_trajectory = torch.tensor(
        [
            [
                [[100.0], [100.0], [100.0], [100.0], [100.0], [100.0]],
            ],
            [
                [[200.0], [200.0], [200.0], [200.0], [200.0], [200.0]],
            ],
            [
                [[300.0], [300.0], [300.0], [300.0], [300.0], [300.0]],
            ],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    batch = {
        "latents": {
            "ode_target_latents": video_trajectory[:, -1],
            "ode_video_trajectory": video_trajectory,
            "ode_trajectory_sigmas": torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float32),
            "num_frames": torch.tensor([5]),
            "height": torch.tensor([1]),
            "width": torch.tensor([1]),
            "fps": torch.tensor([24.0]),
        },
        "audio_latents": {
            "ode_target_latents": audio_trajectory[:, -1],
            "ode_audio_trajectory": audio_trajectory,
            "ode_trajectory_sigmas": torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float32),
            "num_time_steps": torch.tensor([6]),
            "frequency_bins": torch.tensor([1]),
        },
        "conditions": {
            "video_prompt_embeds": torch.zeros(1, 2, 4),
            "audio_prompt_embeds": torch.zeros(1, 2, 4),
            "prompt_attention_mask": torch.ones(1, 2, dtype=torch.bool),
        },
    }

    inputs = strategy.prepare_training_inputs(batch, timestep_sampler=None)

    assert torch.equal(inputs.video.timesteps[0], torch.tensor([1.0, 0.0, 0.0, 0.5, 0.5]))
    assert torch.equal(inputs.audio.timesteps[0], torch.tensor([1.0, 1.0, 0.0, 0.0, 0.5, 0.5]))


def test_compute_loss_matches_ltx_masked_mse_for_video_and_audio() -> None:
    strategy = ODERegressionStrategy(
        ODERegressionConfig(
            name="ode_regression",
            with_audio=True,
            use_block_causal_mask=False,
            ode_layout_mode="legacy",
            loss_reweight_mode="manual",
            video_loss_weight=1.0,
            audio_loss_weight=1.0,
        )
    )

    video_pred = torch.tensor([[[3.0], [0.0], [5.0]]], dtype=torch.float32)
    video_target = torch.tensor([[[1.0], [7.0], [1.0]]], dtype=torch.float32)
    video_mask = torch.tensor([[True, False, True]])

    audio_pred = torch.tensor([[[3.0], [4.0], [9.0]]], dtype=torch.float32)
    audio_target = torch.tensor([[[1.0], [8.0], [3.0]]], dtype=torch.float32)
    audio_mask = torch.tensor([[True, False, True]])

    inputs = type("Inputs", (), {})()
    inputs.video_targets = video_target
    inputs.audio_targets = audio_target
    inputs.video_loss_mask = video_mask
    inputs.audio_loss_mask = audio_mask

    loss = strategy.compute_loss(video_pred=video_pred, audio_pred=audio_pred, inputs=inputs)

    expected_video = ((((video_pred - video_target) ** 2) * video_mask.unsqueeze(-1).float()) / video_mask.unsqueeze(-1).float().mean()).mean()
    expected_audio = ((((audio_pred - audio_target) ** 2) * audio_mask.unsqueeze(-1).float()) / audio_mask.unsqueeze(-1).float().mean()).mean()

    assert torch.isclose(loss, expected_video + expected_audio)


def test_compute_loss_ignores_legacy_reweight_config_and_keeps_ltx_sum() -> None:
    strategy = ODERegressionStrategy(
        ODERegressionConfig(
            name="ode_regression",
            with_audio=True,
            use_block_causal_mask=False,
            ode_layout_mode="legacy",
            loss_reweight_mode="auto",
            video_loss_weight=7.0,
            audio_loss_weight=11.0,
        )
    )

    video_pred = torch.tensor([[[3.0], [0.0], [5.0]]], dtype=torch.float32)
    video_target = torch.tensor([[[1.0], [7.0], [1.0]]], dtype=torch.float32)
    video_mask = torch.tensor([[True, False, True]])

    audio_pred = torch.tensor([[[3.0], [4.0], [9.0]]], dtype=torch.float32)
    audio_target = torch.tensor([[[1.0], [8.0], [3.0]]], dtype=torch.float32)
    audio_mask = torch.tensor([[True, False, True]])

    inputs = type("Inputs", (), {})()
    inputs.video_targets = video_target
    inputs.audio_targets = audio_target
    inputs.video_loss_mask = video_mask
    inputs.audio_loss_mask = audio_mask

    loss = strategy.compute_loss(video_pred=video_pred, audio_pred=audio_pred, inputs=inputs)

    expected_video = ((((video_pred - video_target) ** 2) * video_mask.unsqueeze(-1).float()) / video_mask.unsqueeze(-1).float().mean()).mean()
    expected_audio = ((((audio_pred - audio_target) ** 2) * audio_mask.unsqueeze(-1).float()) / audio_mask.unsqueeze(-1).float().mean()).mean()

    assert torch.isclose(loss, expected_video + expected_audio)
