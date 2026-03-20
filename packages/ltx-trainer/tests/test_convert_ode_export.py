from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import torch

from ltx_trainer.datasets import PrecomputedDataset


ODE_VIDEO_DIR = "ode_video_trajectories"
ODE_AUDIO_DIR = "ode_audio_trajectories"


def test_ode_export_writes_blockwise_metadata(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    sample = {
        "prompt": "test prompt",
        "stage2_video_traj": torch.arange(3 * 1 * 5 * 1 * 1, dtype=torch.float32).reshape(3, 1, 5, 1, 1),
        "stage2_audio_traj": torch.arange(3 * 1 * 6 * 1, dtype=torch.float32).reshape(3, 1, 6, 1),
        "stage2_sigmas": torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32),
    }
    torch.save(sample, input_dir / "sample.pt")

    env = os.environ.copy()
    env["PYTHONPATH"] = "packages/ltx-trainer/src:packages/ltx-core/src:packages/ltx-pipelines/src"
    subprocess.run(
        [
            sys.executable,
            "ode/convert_ode_pt_to_precomputed.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--stage",
            "stage2",
            "--export-mode",
            "ode_regression",
            "--no-write-conditions",
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )

    latent_path = output_dir / ".precomputed" / ODE_VIDEO_DIR / "sample.pt"
    audio_path = output_dir / ".precomputed" / ODE_AUDIO_DIR / "sample.pt"

    latent_payload = torch.load(latent_path, map_location="cpu")
    audio_payload = torch.load(audio_path, map_location="cpu")

    assert "latents" not in latent_payload
    assert "ode_step_index" in latent_payload
    assert "ode_sigma" in latent_payload
    assert "ode_video_trajectory" in latent_payload
    assert "ode_trajectory_sigmas" in latent_payload
    assert tuple(latent_payload["ode_video_trajectory"].shape) == (3, 1, 5, 1, 1)
    assert torch.equal(latent_payload["ode_trajectory_sigmas"], sample["stage2_sigmas"])

    assert "latents" not in audio_payload
    assert "ode_audio_trajectory" in audio_payload
    assert "ode_trajectory_sigmas" in audio_payload
    assert tuple(audio_payload["ode_audio_trajectory"].shape) == (3, 1, 6, 1)
    assert torch.equal(audio_payload["ode_trajectory_sigmas"], sample["stage2_sigmas"])

    dataset = PrecomputedDataset(
        str(output_dir),
        {
            ODE_VIDEO_DIR: "latents",
            ODE_AUDIO_DIR: "audio_latents",
        },
    )
    loaded = dataset[0]
    assert "ode_video_trajectory" in loaded["latents"]
    assert "ode_audio_trajectory" in loaded["audio_latents"]
    assert loaded["idx"] == 0


def test_ode_export_rejects_multistep_expansion_for_ode_regression(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    sample = {
        "prompt": "test prompt",
        "stage2_video_traj": torch.arange(3 * 1 * 5 * 1 * 1, dtype=torch.float32).reshape(3, 1, 5, 1, 1),
        "stage2_audio_traj": torch.arange(3 * 1 * 6 * 1, dtype=torch.float32).reshape(3, 1, 6, 1),
        "stage2_sigmas": torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32),
    }
    torch.save(sample, input_dir / "sample.pt")

    env = os.environ.copy()
    env["PYTHONPATH"] = "packages/ltx-trainer/src:packages/ltx-core/src:packages/ltx-pipelines/src"
    result = subprocess.run(
        [
            sys.executable,
            "ode/convert_ode_pt_to_precomputed.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--stage",
            "stage2",
            "--export-mode",
            "ode_regression",
            "--trajectory-step",
            "all_non_last",
            "--no-write-conditions",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unrecognized arguments: --trajectory-step all_non_last" in result.stderr


def test_standard_export_still_supports_explicit_multistep_selection(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    sample = {
        "prompt": "test prompt",
        "stage2_video_traj": torch.arange(3 * 1 * 5 * 1 * 1, dtype=torch.float32).reshape(3, 1, 5, 1, 1),
        "stage2_audio_traj": torch.arange(3 * 1 * 6 * 1, dtype=torch.float32).reshape(3, 1, 6, 1),
        "stage2_sigmas": torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32),
    }
    torch.save(sample, input_dir / "sample.pt")

    env = os.environ.copy()
    env["PYTHONPATH"] = "packages/ltx-trainer/src:packages/ltx-core/src:packages/ltx-pipelines/src"
    subprocess.run(
        [
            sys.executable,
            "ode/convert_ode_pt_to_precomputed.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--stage",
            "stage2",
            "--export-mode",
            "standard",
            "--standard-trajectory-step",
            "all_non_last",
            "--no-write-conditions",
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )

    assert (output_dir / ".precomputed" / "latents" / "sample__step_000.pt").exists()
    assert (output_dir / ".precomputed" / "latents" / "sample__step_001.pt").exists()


def test_dataset_normalizes_custom_mapped_trajectory_dirs(tmp_path: Path) -> None:
    precomputed_root = tmp_path / ".precomputed"
    video_dir = precomputed_root / ODE_VIDEO_DIR
    audio_dir = precomputed_root / ODE_AUDIO_DIR
    conditions_dir = precomputed_root / "conditions"

    video_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)
    conditions_dir.mkdir(parents=True)

    torch.save(
        {
            "latents": torch.arange(6, dtype=torch.float32).reshape(3, 2),
            "num_frames": 3,
            "height": 1,
            "width": 1,
        },
        video_dir / "sample.pt",
    )
    torch.save(
        {
            "latents": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "num_time_steps": 2,
            "frequency_bins": 2,
        },
        audio_dir / "sample.pt",
    )
    torch.save(
        {
            "video_prompt_embeds": torch.zeros(1, 4),
            "audio_prompt_embeds": torch.zeros(1, 4),
            "prompt_attention_mask": torch.ones(1, dtype=torch.bool),
        },
        conditions_dir / "sample.pt",
    )

    dataset = PrecomputedDataset(
        str(tmp_path),
        {
            ODE_VIDEO_DIR: "latents",
            ODE_AUDIO_DIR: "audio_latents",
            "conditions": "conditions",
        },
    )

    loaded = dataset[0]
    assert tuple(loaded["latents"]["latents"].shape) == (2, 3, 1, 1)
    assert tuple(loaded["audio_latents"]["latents"].shape) == (2, 2, 2)
