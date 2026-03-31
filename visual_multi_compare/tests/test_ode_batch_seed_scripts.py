from __future__ import annotations

import json
import sys
from importlib import util
from pathlib import Path
from types import ModuleType

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


def test_sample_builder_uses_stage2_video_seed(tmp_path: Path) -> None:
    script = _load_module("ode/scripts/sample_5_indices.py", "ode_sample_indices_script_for_test")

    csv_path = tmp_path / "prompts.csv"
    csv_path.write_text("text_prompt\nprompt zero\n", encoding="utf-8")

    precompute_root = tmp_path / ".precomputed" / "ode_video_trajectories"
    precompute_root.mkdir(parents=True)
    torch.save(
        {
            "ode_noise_seeds": {
                "base_seed": 42,
                "stage2": {
                    "video": 5534811387362042849,
                    "audio": 2925523964658678304,
                },
            }
        },
        precompute_root / "00000.pt",
    )

    samples = script.build_samples(
        csv_path=csv_path,
        precompute_root=tmp_path / ".precomputed",
        indices=[0],
    )

    assert samples == [
        {
            "index": 0,
            "prompt": "prompt zero",
            "seed": 5534811387362042849,
            "noise_seeds": {
                "base_seed": 42,
                "stage2": {
                    "video": 5534811387362042849,
                    "audio": 2925523964658678304,
                },
            },
        }
    ]


def test_ode_batch_inference_prefers_per_sample_seed(tmp_path: Path, monkeypatch) -> None:
    script = _load_module("ode/scripts/infer_5prompt_ode.py", "ode_batch_infer_script_for_test")

    samples_path = tmp_path / "samples.json"
    samples_path.write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "prompt": "first prompt",
                    "seed": 101,
                },
                {
                    "index": 2,
                    "prompt": "second prompt",
                    "noise_seeds": {"stage2": {"video": 202}},
                },
            ]
        ),
        encoding="utf-8",
    )

    observed_seeds: list[int] = []

    class FakePipeline:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs

        def __call__(self, **kwargs):  # noqa: ANN003
            observed_seeds.append(kwargs["seed"])
            return iter([torch.zeros(1, 3, 9, 32, 32, dtype=torch.float32)]), None

    monkeypatch.setattr(script, "CausalDistilledODEPipeline", FakePipeline)
    monkeypatch.setattr(script, "encode_video", lambda **kwargs: None)
    monkeypatch.setattr(script, "get_video_chunks_number", lambda num_frames, tiling_config: 1)

    old_argv = sys.argv
    sys.argv = [
        "infer_5prompt_ode.py",
        "--samples-json",
        str(samples_path),
        "--ode-checkpoint-path",
        str(tmp_path / "ode.safetensors"),
        "--distilled-checkpoint-path",
        str(tmp_path / "distilled.safetensors"),
        "--spatial-upsampler-path",
        str(tmp_path / "upsampler.safetensors"),
        "--gemma-root",
        str(tmp_path / "gemma"),
        "--tag",
        "test",
        "--output-root",
        str(tmp_path / "outputs"),
        "--seed",
        "999",
        "--height",
        "32",
        "--width",
        "32",
        "--num-frames",
        "9",
        "--frame-rate",
        "8",
    ]
    try:
        script.main()
    finally:
        sys.argv = old_argv

    assert observed_seeds == [101, 202]


def test_distilled_batch_inference_prefers_per_sample_seed(tmp_path: Path, monkeypatch) -> None:
    script = _load_module("ode/scripts/infer_5prompt_distilled.py", "distilled_batch_infer_script_for_test")

    samples_path = tmp_path / "samples.json"
    samples_path.write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "prompt": "first prompt",
                    "seed": 303,
                },
                {
                    "index": 2,
                    "prompt": "second prompt",
                    "noise_seeds": {"stage2": {"video": 404}},
                },
            ]
        ),
        encoding="utf-8",
    )

    observed_seeds: list[int] = []

    class FakePipeline:
        def __init__(self, **kwargs):  # noqa: ANN003
            self.kwargs = kwargs

        def __call__(self, **kwargs):  # noqa: ANN003
            observed_seeds.append(kwargs["seed"])
            return iter([torch.zeros(1, 3, 9, 32, 32, dtype=torch.float32)]), None

    monkeypatch.setattr(script, "DistilledPipeline", FakePipeline)
    monkeypatch.setattr(script, "encode_video", lambda **kwargs: None)
    monkeypatch.setattr(script, "get_video_chunks_number", lambda num_frames, tiling_config: 1)

    old_argv = sys.argv
    sys.argv = [
        "infer_5prompt_distilled.py",
        "--samples-json",
        str(samples_path),
        "--distilled-checkpoint-path",
        str(tmp_path / "distilled.safetensors"),
        "--spatial-upsampler-path",
        str(tmp_path / "upsampler.safetensors"),
        "--gemma-root",
        str(tmp_path / "gemma"),
        "--tag",
        "test",
        "--output-root",
        str(tmp_path / "outputs"),
        "--seed",
        "999",
        "--height",
        "32",
        "--width",
        "32",
        "--num-frames",
        "9",
        "--frame-rate",
        "8",
    ]
    try:
        script.main()
    finally:
        sys.argv = old_argv

    assert observed_seeds == [303, 404]
