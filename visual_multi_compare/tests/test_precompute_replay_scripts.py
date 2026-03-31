from __future__ import annotations

import json
import sys
from importlib import util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for rel_path in (
    "packages/ltx-core/src",
    "packages/ltx-pipelines/src",
    "packages/ltx-trainer/src",
):
    resolved = str(REPO_ROOT / rel_path)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _load_module(relative_path: str, module_name: str) -> ModuleType:
    script_path = REPO_ROOT / relative_path
    spec = util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {script_path}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decode_gt_from_precompute_muxes_audio_from_paired_payload(tmp_path: Path, monkeypatch) -> None:
    script = _load_module("ode/scripts/decode_gt_from_precompute.py", "decode_gt_from_precompute_for_test")

    samples_path = tmp_path / "samples.json"
    samples_path.write_text(json.dumps([{"index": 12, "prompt": "sample prompt"}]), encoding="utf-8")
    precompute_root = tmp_path / "precompute"
    (precompute_root / ".precomputed" / "ode_video_trajectories").mkdir(parents=True)
    (precompute_root / ".precomputed" / "ode_audio_trajectories").mkdir(parents=True)

    observed: dict[str, object] = {}
    fake_audio = SimpleNamespace(waveform=torch.zeros(1, 1600), sampling_rate=24000)

    video_payload = {
        "ode_target_latents": torch.zeros(128, 16, 32, 48, dtype=torch.bfloat16),
    }
    audio_payload = {
        "ode_target_latents": torch.zeros(8, 126, 16, dtype=torch.bfloat16),
    }

    def fake_torch_load(path, map_location=None, weights_only=None):  # noqa: ANN001
        path = Path(path)
        if "ode_audio_trajectories" in str(path):
            return audio_payload
        return video_payload

    class FakeModelLedger:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs

        def video_decoder(self):
            return object()

        def audio_decoder(self):
            return object()

        def vocoder(self):
            return object()

    monkeypatch.setattr(script.torch, "load", fake_torch_load)
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(script, "ModelLedger", FakeModelLedger)
    monkeypatch.setattr(script, "vae_decode_video", lambda latent, decoder, tiling_config, generator: iter([latent]))
    monkeypatch.setattr(script, "vae_decode_audio", lambda latent, audio_decoder, vocoder: fake_audio, raising=False)
    monkeypatch.setattr(script, "get_video_chunks_number", lambda num_frames, tiling_config: 1)
    monkeypatch.setattr(
        script,
        "encode_video",
        lambda **kwargs: observed.setdefault("encode_video_kwargs", kwargs),
    )

    old_argv = sys.argv
    sys.argv = [
        "decode_gt_from_precompute.py",
        "--samples-json",
        str(samples_path),
        "--precompute-root",
        str(precompute_root),
        "--checkpoint-path",
        str(tmp_path / "distilled.safetensors"),
        "--output-dir",
        str(tmp_path / "outputs"),
    ]
    try:
        script.main()
    finally:
        sys.argv = old_argv

    encode_kwargs = observed["encode_video_kwargs"]
    assert isinstance(encode_kwargs, dict)
    assert encode_kwargs["audio"] is fake_audio


def test_replay_precompute_exact_uses_sample_index_and_outputs_audio(tmp_path: Path, monkeypatch) -> None:
    script = _load_module("visual_multi_compare/replay_precompute_exact.py", "replay_precompute_exact_for_test")

    samples_path = tmp_path / "samples.json"
    samples_path.write_text(json.dumps([{"index": 34, "prompt": "sample prompt"}]), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "checkpoint_path: model/ltx-2.3-22b-distilled.safetensors",
                "gemma_root: model/gemma",
                "spatial_upsampler_path: model/ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
                "stage1_height: 512",
                "stage1_width: 768",
                "num_frames: 121",
                "frame_rate: 24.0",
                "seed: 42",
            ]
        ),
        encoding="utf-8",
    )
    precompute_root = tmp_path / "precompute"
    (precompute_root / ".precomputed" / "ode_video_trajectories").mkdir(parents=True)
    (precompute_root / ".precomputed" / "ode_audio_trajectories").mkdir(parents=True)

    video_payload = {
        "ode_target_latents": torch.zeros(128, 16, 32, 48, dtype=torch.bfloat16),
        "fps": 24.0,
    }
    audio_payload = {
        "ode_target_latents": torch.zeros(8, 126, 16, dtype=torch.bfloat16),
    }

    observed: dict[str, object] = {"saved_paths": []}
    fake_audio = SimpleNamespace(waveform=torch.zeros(1, 1600), sampling_rate=24000)

    def fake_torch_load(path, map_location=None, weights_only=None):  # noqa: ANN001
        path = Path(path)
        if "ode_audio_trajectories" in str(path):
            return audio_payload
        return video_payload

    class FakeModelLedger:
        def __init__(self, **kwargs):  # noqa: ANN003
            observed["ledger_kwargs"] = kwargs

        def video_decoder(self):
            return object()

        def audio_decoder(self):
            return object()

        def vocoder(self):
            return object()

        def transformer(self):
            return SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]), to=lambda device: self, eval=lambda: self)

        def text_encoder(self):
            return object()

        def gemma_embeddings_processor(self):
            return object()

        def video_encoder(self):
            return object()

        def spatial_upsampler(self):
            return object()

    monkeypatch.setattr(script.torch, "load", fake_torch_load)
    monkeypatch.setattr(script.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(script, "ModelLedger", FakeModelLedger)
    monkeypatch.setattr(script, "PipelineComponents", lambda dtype, device: object())
    monkeypatch.setattr(
        script,
        "encode_prompt",
        lambda prompt, model_ledger, device: SimpleNamespace(
            video_encoding=torch.zeros(1, 1, 1),
            audio_encoding=torch.zeros(1, 1, 1),
        ),
    )

    def fake_generate_sample(prompt, global_idx, ctx_pos, transformer, model_ledger, components, cfg, device, dtype):  # noqa: ANN001
        observed["generate_sample_args"] = {
            "prompt": prompt,
            "global_idx": global_idx,
            "cfg_seed": cfg["seed"],
            "num_frames": cfg["num_frames"],
            "frame_rate": cfg["frame_rate"],
        }
        return {
            "stage2_video_traj": torch.zeros(4, 128, 16, 32, 48, dtype=torch.bfloat16),
            "stage2_audio_traj": torch.zeros(4, 8, 126, 16, dtype=torch.bfloat16),
        }

    monkeypatch.setattr(script, "generate_sample", fake_generate_sample)
    monkeypatch.setattr(script, "vae_decode_video", lambda latent, decoder, tiling_config, generator: iter([latent]))
    monkeypatch.setattr(script, "vae_decode_audio", lambda latent, audio_decoder, vocoder: fake_audio)
    monkeypatch.setattr(script, "get_video_chunks_number", lambda num_frames, tiling_config: 1)
    monkeypatch.setattr(
        script,
        "encode_video",
        lambda **kwargs: observed["saved_paths"].append((kwargs["output_path"], kwargs["audio"])),
    )

    exit_code = script.main(
        [
            "--samples-json",
            str(samples_path),
            "--precompute-root",
            str(precompute_root),
            "--config",
            str(config_path),
            "--output-root",
            str(tmp_path / "outputs"),
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert observed["generate_sample_args"] == {
        "prompt": "sample prompt",
        "global_idx": 34,
        "cfg_seed": 42,
        "num_frames": 121,
        "frame_rate": 24.0,
    }
    assert len(observed["saved_paths"]) == 2
    assert all(audio is fake_audio for _, audio in observed["saved_paths"])
